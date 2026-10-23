//! The `#[tree(..)]` attributes (and their `#[module(..)]` alias under `derive(Module)`).

use syn::spanned::Spanned;
use syn::{Attribute, Error, LitStr, Path, Result};

/// What the derive does with a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// A child tree (the default).
    Visit,
    /// An `Array` leaf of kind buffer.
    Buffer,
    /// A mode flag (`bool`): reported to `Visit::mode` and `VisitMut::training`.
    Mode,
    /// Not a leaf, hashed into the structure (`Static`).
    Static,
    /// Not a leaf, not hashed (`static_unchecked`, or a computed `Array` constant).
    Ignored,
}

/// The attributes of one field or variant.
#[derive(Default)]
pub(crate) struct FieldAttrs {
    pub(crate) skip: bool,
    pub(crate) static_unchecked: bool,
    pub(crate) buffer: bool,
    pub(crate) mode: bool,
    pub(crate) rename: Option<LitStr>,
}

/// The attribute namespaces a derive accepts.
#[derive(Clone, Copy)]
pub(crate) enum Namespaces {
    /// `#[tree(..)]` only (`derive(Tree)`).
    Tree,
    /// `#[tree(..)]` or `#[module(..)]`, not both on one item (`derive(Module)`).
    Module,
}

impl Namespaces {
    fn accepts(self, attr: &Attribute) -> bool {
        attr.path().is_ident("tree")
            || (matches!(self, Namespaces::Module) && attr.path().is_ident("module"))
    }
}

/// Checks that one item does not mix `#[tree(..)]` and `#[module(..)]`.
fn check_mixing(attrs: &[Attribute], ns: Namespaces) -> Result<()> {
    if matches!(ns, Namespaces::Module) {
        let tree = attrs.iter().find(|a| a.path().is_ident("tree"));
        let module = attrs.iter().find(|a| a.path().is_ident("module"));
        if let (Some(_), Some(m)) = (tree, module) {
            return Err(Error::new(
                m.span(),
                "use one attribute namespace per field: `#[tree(..)]` or its alias `#[module(..)]`, not both",
            ));
        }
    }
    Ok(())
}

pub(crate) fn field_attrs(attrs: &[Attribute], ns: Namespaces) -> Result<FieldAttrs> {
    check_mixing(attrs, ns)?;
    let mut out = FieldAttrs::default();
    for attr in attrs.iter().filter(|a| ns.accepts(a)) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                out.skip = true;
            } else if meta.path.is_ident("static_unchecked") {
                out.static_unchecked = true;
            } else if meta.path.is_ident("buffer") {
                out.buffer = true;
            } else if meta.path.is_ident("mode") {
                out.mode = true;
            } else if meta.path.is_ident("rename") {
                if !meta.input.peek(syn::Token![=]) {
                    return Err(meta.error(
                        "`rename` needs a value: `#[tree(rename = \"name\")]`",
                    ));
                }
                out.rename = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("crate") {
                return Err(meta.error("`crate` is a container attribute: put it on the type"));
            } else {
                return Err(meta.error(
                    "unknown tree attribute; expected `skip`, `static_unchecked`, `buffer`, `mode` or `rename = \"..\"`",
                ));
            }
            Ok(())
        })?;
    }
    if out.static_unchecked && !out.skip {
        return Err(Error::new(
            attrs
                .first()
                .map_or_else(proc_macro2::Span::call_site, Spanned::span),
            "`static_unchecked` qualifies `skip`: write `#[tree(skip, static_unchecked)]`",
        ));
    }
    let roles = [out.skip, out.buffer, out.mode]
        .iter()
        .filter(|&&b| b)
        .count();
    if roles > 1 {
        return Err(Error::new(
            attrs
                .first()
                .map_or_else(proc_macro2::Span::call_site, Spanned::span),
            "`skip`, `buffer` and `mode` exclude each other",
        ));
    }
    Ok(out)
}

/// The container's `#[tree(crate = "..")]`: where `__derive` lives (default `::nuvec`).
pub(crate) fn crate_path(attrs: &[Attribute], ns: Namespaces) -> Result<Path> {
    check_mixing(attrs, ns)?;
    let mut path: Option<Path> = None;
    for attr in attrs.iter().filter(|a| ns.accepts(a)) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("crate") {
                let lit: LitStr = meta.value()?.parse()?;
                path = Some(lit.parse()?);
                Ok(())
            } else {
                Err(meta.error(
                    "unknown container attribute; expected `crate = \"path\"` (field attributes go on fields)",
                ))
            }
        })?;
    }
    Ok(path.unwrap_or_else(|| syn::parse_quote!(::nuvec)))
}
