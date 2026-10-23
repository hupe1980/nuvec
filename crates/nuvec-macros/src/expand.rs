//! The one expansion engine behind `derive(Tree)`, `derive(Module)` and `derive(Static)`.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{
    Data, DeriveInput, Error, Fields, Ident, Index, LitStr, Path, Result, Type, WherePredicate,
};

use crate::attrs::{FieldAttrs, Namespaces, Role, crate_path, field_attrs};
use crate::lint::{has_trait_object, is_array_like, mentions};

/// Which derive is expanding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Tree,
    Module,
}

/// One field: how to reach it, its name in paths and its role.
struct Field {
    /// The binding in a `match` arm, or `self.field` for structs.
    access: TokenStream,
    access_mut: TokenStream,
    /// The path segment (empty: transparent).
    name: String,
    ty: Type,
    role: Role,
    span: Span,
}

/// A struct's fields or one enum variant's.
struct Group {
    fields: Vec<Field>,
}

fn role(attrs: &FieldAttrs, ty: &Type) -> Role {
    if attrs.buffer {
        Role::Buffer
    } else if attrs.mode {
        Role::Mode
    } else if attrs.skip && (attrs.static_unchecked || is_array_like(ty)) {
        Role::Ignored
    } else if attrs.skip {
        Role::Static
    } else {
        Role::Visit
    }
}

/// The fields of `fields` with their names (`prefix` prepended for renamed variants;
/// `transparent_newtype`: a single unnamed field without a name is transparent).
fn group(
    fields: &Fields,
    kind: Kind,
    ns: Namespaces,
    in_enum: bool,
    prefix: Option<&str>,
) -> Result<Group> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let newtype = matches!(fields, Fields::Unnamed(u) if u.unnamed.len() == 1);
    for (i, f) in fields.iter().enumerate() {
        let attrs = field_attrs(&f.attrs, ns)?;
        let base = match (&attrs.rename, &f.ident) {
            (Some(r), _) => r.value(),
            (None, Some(id)) => id.to_string().trim_start_matches("r#").to_owned(),
            (None, None) if in_enum && newtype => String::new(),
            (None, None) => i.to_string(),
        };
        let name = match prefix {
            Some(p) if base.is_empty() => p.to_owned(),
            Some(p) => format!("{p}.{base}"),
            None => base,
        };
        let r = role(&attrs, &f.ty);
        if !matches!(r, Role::Ignored | Role::Static | Role::Mode) && !seen.insert(name.clone()) {
            return Err(Error::new(
                f.span(),
                format!("two fields have the path segment {name:?}: rename one"),
            ));
        }
        if r == Role::Visit && has_trait_object(&f.ty) {
            return Err(Error::new(
                f.ty.span(),
                "trait objects are not trees; use an enum of the concrete types (or `#[tree(skip, static_unchecked)]` for host-side data)",
            ));
        }
        if kind == Kind::Module && r == Role::Visit && is_array_like(&f.ty) {
            return Err(Error::new(
                f.ty.span(),
                "a module field of type `Array` needs a role: use `Param` for a parameter, `#[tree(buffer)]` for a buffer, or `#[tree(skip)]` for a computed constant",
            ));
        }
        let (access, access_mut) = if in_enum {
            let b = binding(f.ident.as_ref(), i);
            (quote!(#b), quote!(#b))
        } else {
            match &f.ident {
                Some(id) => (quote!(&self.#id), quote!(&mut self.#id)),
                None => {
                    let idx = Index::from(i);
                    (quote!(&self.#idx), quote!(&mut self.#idx))
                }
            }
        };
        out.push(Field {
            access,
            access_mut,
            name,
            ty: f.ty.clone(),
            role: r,
            span: f.ty.span(),
        });
    }
    Ok(Group { fields: out })
}

fn binding(ident: Option<&Ident>, i: usize) -> Ident {
    match ident {
        Some(id) => format_ident!("__f_{}", id),
        None => format_ident!("__f{}", i),
    }
}

/// The pattern binding every field of a variant.
fn pattern(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(n) => {
            let binds = n.named.iter().enumerate().map(|(i, f)| {
                let id = f.ident.as_ref().expect("named");
                let b = binding(Some(id), i);
                quote!(#id: #b)
            });
            quote!({ #(#binds),* })
        }
        Fields::Unnamed(u) => {
            let binds = (0..u.unnamed.len()).map(|i| binding(None, i));
            quote!(( #(#binds),* ))
        }
        Fields::Unit => quote!(),
    }
}

fn visit_body(g: &Group, krate: &Path) -> TokenStream {
    let d = quote!(#krate::__derive);
    let stmts = g.fields.iter().map(|f| {
        let name = LitStr::new(&f.name, f.span);
        let access = &f.access;
        match f.role {
            Role::Visit => quote_spanned!(f.span=> __v.child(#name, #access);),
            Role::Buffer => quote_spanned!(f.span=> __v.buffer(#name, #access);),
            Role::Mode => quote_spanned!(f.span=> __v.mode(*#access);),
            Role::Static => {
                let ty = &f.ty;
                let check = quote_spanned!(f.span=> #d::assert_static::<#ty>(););
                quote!(#check __v.static_field(#name, #access as &dyn #d::Static);)
            }
            Role::Ignored => quote!(),
        }
    });
    quote!(#(#stmts)*)
}

fn visit_mut_body(g: &Group) -> TokenStream {
    let stmts = g.fields.iter().map(|f| {
        let name = LitStr::new(&f.name, f.span);
        let access = &f.access_mut;
        match f.role {
            Role::Visit => quote_spanned!(f.span=> __v.child(#name, #access);),
            Role::Buffer => quote_spanned!(f.span=> __v.buffer(#name, #access);),
            Role::Mode => quote_spanned!(f.span=> __v.training(#access);),
            Role::Static | Role::Ignored => quote!(),
        }
    });
    quote!(#(#stmts)*)
}

/// Where-clause predicates: field types that mention a type parameter get the trait their
/// role needs (visited: `bound`; hashed: `Static`); every type parameter is `'static` when
/// `statik` (trees and statics are `'static`).
fn bounds(
    groups: &[&Group],
    params: &[Ident],
    visited: &TokenStream,
    krate: &Path,
    statik: bool,
) -> Vec<WherePredicate> {
    let d = quote!(#krate::__derive);
    let mut out: Vec<WherePredicate> = Vec::new();
    let mut seen = HashSet::new();
    for g in groups {
        for f in &g.fields {
            if !mentions(&f.ty, params) {
                continue;
            }
            let ty = &f.ty;
            let pred: Option<WherePredicate> = match f.role {
                Role::Visit => Some(syn::parse_quote!(#ty: #visited)),
                Role::Static => Some(syn::parse_quote!(#ty: #d::Static)),
                _ => None,
            };
            if let Some(p) = pred
                && seen.insert(quote!(#p).to_string())
            {
                out.push(p);
            }
        }
    }
    if statik {
        for p in params {
            out.push(syn::parse_quote!(#p: 'static));
        }
    }
    out
}

pub(crate) fn derive_tree(input: &DeriveInput, kind: Kind) -> Result<TokenStream> {
    let ns = match kind {
        Kind::Tree => Namespaces::Tree,
        Kind::Module => Namespaces::Module,
    };
    let krate = crate_path(&input.attrs, ns)?;
    let d = quote!(#krate::__derive);
    let name = &input.ident;
    let params: Vec<Ident> = input
        .generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect();

    let (visit, visit_mut, groups): (TokenStream, TokenStream, Vec<Group>) = match &input.data {
        Data::Struct(s) => {
            let g = group(&s.fields, kind, ns, false, None)?;
            (visit_body(&g, &krate), visit_mut_body(&g), vec![g])
        }
        Data::Enum(e) => {
            let mut arms = Vec::new();
            let mut arms_mut = Vec::new();
            let mut groups = Vec::new();
            for (i, v) in e.variants.iter().enumerate() {
                let attrs = field_attrs(&v.attrs, ns)?;
                if attrs.skip || attrs.buffer || attrs.mode || attrs.static_unchecked {
                    return Err(Error::new(
                        v.span(),
                        "only `rename` applies to enum variants",
                    ));
                }
                let prefix = attrs.rename.as_ref().map(LitStr::value);
                let g = group(&v.fields, kind, ns, true, prefix.as_deref())?;
                let vid = &v.ident;
                let pat = pattern(&v.fields);
                let index = i as u32;
                let body = visit_body(&g, &krate);
                let body_mut = visit_mut_body(&g);
                arms.push(quote!(Self::#vid #pat => { __v.variant(#index); #body }));
                arms_mut.push(quote!(Self::#vid #pat => { #body_mut }));
                groups.push(g);
            }
            let visit = if arms.is_empty() {
                quote!()
            } else {
                quote!(match self { #(#arms)* })
            };
            let visit_mut = if arms_mut.is_empty() {
                quote!()
            } else {
                quote!(match self { #(#arms_mut)* })
            };
            (visit, visit_mut, groups)
        }
        Data::Union(u) => {
            return Err(Error::new(
                u.union_token.span(),
                "unions are not trees; use a struct or an enum",
            ));
        }
    };

    let refs: Vec<&Group> = groups.iter().collect();
    let (impl_g, ty_g, where_c) = input.generics.split_for_impl();
    let base: Vec<WherePredicate> = where_c
        .map(|w| w.predicates.iter().cloned().collect())
        .unwrap_or_default();
    let leaves_bounds = bounds(&refs, &params, &quote!(#d::Leaves), &krate, false);
    let tree_bounds = bounds(&refs, &params, &quote!(#d::Tree), &krate, true);
    let leaves_where = base.iter().chain(&leaves_bounds);
    let clone: WherePredicate = syn::parse_quote!(Self: #d::Clone);
    let tree_where = base
        .iter()
        .chain(&tree_bounds)
        .chain(std::iter::once(&clone));
    Ok(quote! {
        #[automatically_derived]
        impl #impl_g #d::Leaves for #name #ty_g where #(#leaves_where,)* {
            #[allow(unused_variables)]
            fn visit(&self, __v: &mut #d::Visit<'_>) {
                #visit
            }
        }
        #[automatically_derived]
        impl #impl_g #d::Tree for #name #ty_g where #(#tree_where,)* {
            #[allow(unused_variables)]
            fn visit_mut(&mut self, __v: &mut #d::VisitMut<'_>) {
                #visit_mut
            }
        }
    })
}

pub(crate) fn derive_static(input: &DeriveInput) -> Result<TokenStream> {
    let krate = crate_path(&input.attrs, Namespaces::Tree)?;
    let d = quote!(#krate::__derive);
    let name = &input.ident;
    let params: Vec<Ident> = input
        .generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect();
    let hash = |ty: &Type, access: TokenStream| quote_spanned!(ty.span()=> #d::Static::hash_static(#access, __h););
    let mut tys: Vec<Type> = Vec::new();
    let body = match &input.data {
        Data::Struct(s) => {
            let stmts: Vec<TokenStream> = s
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    tys.push(f.ty.clone());
                    let access = match &f.ident {
                        Some(id) => quote!(&self.#id),
                        None => {
                            let idx = Index::from(i);
                            quote!(&self.#idx)
                        }
                    };
                    hash(&f.ty, access)
                })
                .collect();
            quote!(#(#stmts)*)
        }
        Data::Enum(e) => {
            let arms: Vec<TokenStream> = e
                .variants
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let vid = &v.ident;
                    let pat = pattern(&v.fields);
                    let index = i as u32;
                    let stmts: Vec<TokenStream> = v
                        .fields
                        .iter()
                        .enumerate()
                        .map(|(j, f)| {
                            tys.push(f.ty.clone());
                            let b = binding(f.ident.as_ref(), j);
                            hash(&f.ty, quote!(#b))
                        })
                        .collect();
                    quote!(Self::#vid #pat => { __h.write_u32(#index); #(#stmts)* })
                })
                .collect();
            if arms.is_empty() {
                quote!()
            } else {
                quote!(match self { #(#arms)* })
            }
        }
        Data::Union(u) => {
            return Err(Error::new(
                u.union_token.span(),
                "unions cannot derive `Static`",
            ));
        }
    };
    let (impl_g, ty_g, where_c) = input.generics.split_for_impl();
    let mut preds: Vec<WherePredicate> = where_c
        .map(|w| w.predicates.iter().cloned().collect())
        .unwrap_or_default();
    let mut seen = HashSet::new();
    for ty in &tys {
        if mentions(ty, &params) && seen.insert(quote!(#ty).to_string()) {
            preds.push(syn::parse_quote!(#ty: #d::Static));
        }
    }
    for p in &params {
        preds.push(syn::parse_quote!(#p: 'static));
    }
    Ok(quote! {
        #[automatically_derived]
        impl #impl_g #d::Static for #name #ty_g where #(#preds,)* {
            #[allow(unused_variables)]
            fn hash_static(&self, __h: &mut dyn #d::Hasher) {
                #body
            }
        }
    })
}
