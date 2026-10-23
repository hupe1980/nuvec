//! Derive macros for nuvec, re-exported as `nuvec::Tree`, `nuvec::Static` and
//! `nuvec_nn::Module`: use them through those crates.
//!
//! One expansion engine generates `Leaves` and `Tree` impls; `derive(Module)` is
//! `derive(Tree)` plus a lint against bare `Array` fields. The generated code names only
//! `::nuvec::__derive` (or the module given by `#[tree(crate = "..")]` on the type).
//!
//! Field attributes (`#[tree(..)]`; `#[module(..)]` is an alias under `derive(Module)`):
//!
//! | attribute | meaning |
//! |---|---|
//! | (none) | a child tree (its leaves are visited) |
//! | `skip` | not a leaf; hashed into the structure (the type must be `Static`); on an `Array` field (last path segment `Array`): a computed constant, neither hashed nor saved |
//! | `skip, static_unchecked` | not a leaf, not hashed: baked into compiled traces |
//! | `buffer` | an `Array` leaf that is saved and loaded but never trained |
//! | `mode` | a `bool` mode flag (`true` = training), part of the structure |
//! | `rename = ".."` | the path segment (fields and enum variants) |
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod attrs;
mod expand;
mod lint;

use expand::Kind;

/// Derives `Leaves` and `Tree` for a struct (named, tuple or unit) or an enum: the type's
/// array leaves get paths from its field names, as MLX's `tree_flatten`.
///
/// ```ignore
/// #[derive(Clone, Tree)]
/// struct Batch { x: Array, y: Array, #[tree(skip)] len: usize }
///
/// #[derive(Clone, Tree)]
/// enum Either { A(Array), #[tree(rename = "b")] B(Vec<Array>), Empty }
/// ```
///
/// A newtype enum variant is transparent unless renamed; struct variants name their
/// fields; every variant reports its index, so a leaf-less variant changes the structure.
/// Type parameters are bounded only where a field uses them. Trait-object fields are
/// rejected (use an enum of the concrete types).
#[proc_macro_derive(Tree, attributes(tree))]
pub fn derive_tree(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::derive_tree(&input, Kind::Tree)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives `Tree` for a model (it emits no `impl Module`: `Module` is implemented for every
/// tree), with one more check: a field typed `Array` (directly or in `Option`, `Vec` or
/// `Box`) without `buffer` or `skip` is an error suggesting `Param`. The check is
/// syntactic: a type alias of `Array` is not caught.
#[proc_macro_derive(Module, attributes(tree, module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::derive_tree(&input, Kind::Module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives `Static` (the type of a hashed `#[tree(skip)]` field) by hashing every field in
/// order (enums: the variant index, then its fields).
#[proc_macro_derive(Static, attributes(tree))]
pub fn derive_static(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::derive_static(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
