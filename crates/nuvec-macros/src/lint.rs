//! Syntactic type checks: the bare-`Array` lint of `derive(Module)`, computed constants,
//! trait objects and type-parameter use. The macro sees tokens, not types, so a type alias
//! of `Array` is not recognised (documented).

use syn::visit::Visit;
use syn::{GenericArgument, Ident, PathArguments, Type};

/// Whether `ty` is `Array` (last path segment), directly or inside `Option`, `Vec` or `Box`.
pub(crate) fn is_array_like(ty: &Type) -> bool {
    match ty {
        Type::Path(p) if p.qself.is_none() => {
            let Some(last) = p.path.segments.last() else {
                return false;
            };
            if last.ident == "Array" {
                return true;
            }
            if ["Option", "Vec", "Box"].iter().any(|w| last.ident == w)
                && let PathArguments::AngleBracketed(args) = &last.arguments
                && let Some(GenericArgument::Type(inner)) = args.args.first()
            {
                return is_array_like(inner);
            }
            false
        }
        Type::Paren(p) => is_array_like(&p.elem),
        Type::Group(g) => is_array_like(&g.elem),
        _ => false,
    }
}

/// Whether `ty` contains a trait object (`dyn Trait`, or a bare trait in edition 2015 style).
pub(crate) fn has_trait_object(ty: &Type) -> bool {
    struct Finder(bool);
    impl<'ast> Visit<'ast> for Finder {
        fn visit_type_trait_object(&mut self, _: &'ast syn::TypeTraitObject) {
            self.0 = true;
        }
    }
    let mut f = Finder(false);
    f.visit_type(ty);
    f.0
}

/// Whether `ty` mentions one of `params` (ignoring `PhantomData<..>`, which needs no bound).
pub(crate) fn mentions(ty: &Type, params: &[Ident]) -> bool {
    struct Finder<'a> {
        params: &'a [Ident],
        found: bool,
    }
    impl<'ast> Visit<'ast> for Finder<'_> {
        fn visit_type_path(&mut self, p: &'ast syn::TypePath) {
            if p.qself.is_none()
                && p.path.segments.len() == 1
                && self.params.iter().any(|i| p.path.segments[0].ident == *i)
            {
                self.found = true;
                return;
            }
            if p.path
                .segments
                .last()
                .is_some_and(|s| s.ident == "PhantomData")
            {
                return;
            }
            syn::visit::visit_type_path(self, p);
        }
    }
    let mut f = Finder {
        params,
        found: false,
    };
    f.visit_type(ty);
    f.found
}
