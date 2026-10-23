//! Trees of arrays: structured values (tuples, vectors, maps, options, user structs and
//! enums) whose array *leaves* have paths and a fixed order, as MLX's `tree_flatten`.
//!
//! - [`Leaves`] is the read-only, object-safe half: anything whose leaves can be visited,
//!   including borrowed views such as `&T`, `(&A, &B)` and `[T]`. [`crate::eval`] takes it,
//!   so `eval(&(&loss, &grads))` evaluates both without cloning.
//! - [`Tree`] is an owned tree that can be mutated in place and rebuilt with new leaves of
//!   the same structure ([`Tree::unflatten`]); transforms differentiate, batch and compile
//!   trees.
//!
//! **Paths and order** (MLX's `tree_flatten` names): struct fields in declaration order,
//! sequences and tuples by index (`0`, `1`, …), maps by sorted key, `Option`, `Box` and
//! newtype enum variants transparent, joined with `.`; a root array has the empty path.
//!
//! **Leaf kinds** ([`LeafKind`]): a plain [`Array`] (always differentiable), a [`Param`]
//! (a module parameter, trainable unless frozen) or a buffer (an array field marked
//! `#[tree(buffer)]`: saved and loaded, never trained).
//!
//! **Structure** ([`TreeDef`]): the root type, the ordered `(path, kind, trainable)` of every
//! leaf, the active enum variants, the mode flags (train/eval) and a hash of the *static*
//! fields (`#[tree(skip)]` fields of a [`Static`] type, such as a layer's `eps`). Two trees
//! with equal `TreeDef`s have the same structure; a compiled function retraces when the
//! structure changes.
//!
//! User types implement the traits with `#[derive(Tree)]` (`nuvec::Tree`; code that uses
//! `nuvec_core` directly names it with `#[tree(crate = "nuvec_core")]`):
//!
//! ```
//! use nuvec_core::tree::Tree;
//! use nuvec_core::{Array, array};
//!
//! #[derive(Clone, nuvec_macros::Tree)]
//! #[tree(crate = "nuvec_core")]
//! struct Batch {
//!     x: Array,
//!     y: Array,
//!     #[tree(skip)]
//!     len: usize, // not a leaf: hashed into the structure
//! }
//!
//! let b = Batch { x: array![1.0, 2.0], y: array![0.0], len: 2 };
//! assert_eq!(b.paths(), ["x", "y"]);
//! let doubled = b.map(|_, a| a.multiply(2.0));
//! assert_eq!(doubled.x.to_vec::<f32>().unwrap(), [2.0, 4.0]);
//! assert_eq!(doubled.len, 2);
//! # let _ = b.len;
//! ```
//!
//! Static fields and retracing: everything in a `TreeDef` is part of a compiled
//! function's cache key, so switching train/eval, freezing, a different enum variant or a
//! changed hashed field retraces once. A `#[tree(skip, static_unchecked)]` field is not
//! hashed: its value is baked into compiled traces.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use crate::scalar::{ScalarOperand, ScalarOrArray};
use crate::{Array, Category, DType, Device, Error, Result};

/// What a leaf is (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeafKind {
    /// A plain array: always differentiable.
    Array,
    /// A module parameter: trainable unless frozen.
    Param,
    /// A buffer: saved and loaded, never trainable.
    Buffer,
}

/// A module parameter: an array and a frozen flag (trainable unless frozen).
///
/// It reads as an [`Array`] (`Deref`), so methods, `&Param` arguments where `&Array` is
/// expected, operators (`&x * &p`, `&p + 1.0`) and free functions (`ops::add(&p, &x)`)
/// work as on the array. Gradients, [`Tree::map`] and clones keep the flag.
#[derive(Debug, Clone, Default)]
pub struct Param {
    value: Array,
    frozen: bool,
}

impl Param {
    /// A trainable parameter.
    pub fn new(value: Array) -> Self {
        Self {
            value,
            frozen: false,
        }
    }

    /// A parameter frozen at construction (a LoRA base, say).
    pub fn frozen(value: Array) -> Self {
        Self {
            value,
            frozen: true,
        }
    }

    /// Whether the parameter is frozen (not trained).
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Freezes (`true`) or unfreezes (`false`) the parameter.
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    /// The array.
    pub fn value(&self) -> &Array {
        &self.value
    }

    /// The array, mutably (the frozen flag is kept).
    pub fn value_mut(&mut self) -> &mut Array {
        &mut self.value
    }

    /// The array, consuming the parameter.
    pub fn into_inner(self) -> Array {
        self.value
    }
}

impl std::ops::Deref for Param {
    type Target = Array;
    fn deref(&self) -> &Array {
        &self.value
    }
}

impl AsRef<Array> for Param {
    fn as_ref(&self) -> &Array {
        &self.value
    }
}

impl From<Array> for Param {
    fn from(value: Array) -> Self {
        Self::new(value)
    }
}

impl From<Param> for Array {
    fn from(p: Param) -> Self {
        p.value
    }
}

impl From<&Param> for ScalarOrArray {
    fn from(p: &Param) -> Self {
        Self::Array(p.value.clone())
    }
}

impl From<Param> for ScalarOrArray {
    fn from(p: Param) -> Self {
        Self::Array(p.value)
    }
}

/// A parameter where an optional array is expected (`fast::rms_norm(x, &p, eps)`).
impl<'a> From<&'a Param> for Option<&'a Array> {
    fn from(p: &'a Param) -> Self {
        Some(&p.value)
    }
}

impl std::fmt::Display for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.value, f)
    }
}

/// Operators with a [`Param`] on either side, delegating to the arrays' (no blanket over
/// `AsRef<Array>`, which would overlap the weak-scalar operators).
macro_rules! param_operators {
    ($($trait:ident $method:ident),*) => {$(
        impl std::ops::$trait<&Param> for &Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Param) -> Array {
                std::ops::$trait::$method(self, &rhs.value)
            }
        }
        impl std::ops::$trait<Param> for &Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Param) -> Array {
                std::ops::$trait::$method(self, &rhs.value)
            }
        }
        impl std::ops::$trait<&Param> for Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Param) -> Array {
                std::ops::$trait::$method(&self, &rhs.value)
            }
        }
        impl std::ops::$trait<Param> for Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Param) -> Array {
                std::ops::$trait::$method(&self, &rhs.value)
            }
        }
        impl std::ops::$trait<&Array> for &Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Array) -> Array {
                std::ops::$trait::$method(&self.value, rhs)
            }
        }
        impl std::ops::$trait<Array> for &Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Array) -> Array {
                std::ops::$trait::$method(&self.value, &rhs)
            }
        }
        impl std::ops::$trait<&Array> for Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Array) -> Array {
                std::ops::$trait::$method(&self.value, rhs)
            }
        }
        impl std::ops::$trait<Array> for Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Array) -> Array {
                std::ops::$trait::$method(&self.value, &rhs)
            }
        }
        impl std::ops::$trait<&Param> for &Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Param) -> Array {
                std::ops::$trait::$method(&self.value, &rhs.value)
            }
        }
        impl std::ops::$trait<Param> for Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Param) -> Array {
                std::ops::$trait::$method(&self.value, &rhs.value)
            }
        }
        impl<S: ScalarOperand> std::ops::$trait<S> for &Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: S) -> Array {
                std::ops::$trait::$method(&self.value, rhs)
            }
        }
        impl<S: ScalarOperand> std::ops::$trait<S> for Param {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: S) -> Array {
                std::ops::$trait::$method(&self.value, rhs)
            }
        }
    )*};
}

param_operators!(Add add, Sub sub, Mul mul, Div div);

impl std::ops::Neg for &Param {
    type Output = Array;
    #[track_caller]
    fn neg(self) -> Array {
        -&self.value
    }
}

// ── Static fields ───────────────────────────────────────────────────────────

/// A value hashed into a [`TreeDef`]: the type of a `#[tree(skip)]` field (a
/// configuration, a hyperparameter). A different value is a different structure, so a
/// compiled function retraces instead of replaying a stale constant.
///
/// Implemented for integers, `bool`, `char`, `()`, strings, `f32`/`f64` (by bit pattern:
/// `-0.0` differs from `0.0`), [`DType`], [`Device`], `Duration` and `Option`, `Vec`,
/// `Box`, `Arc`, arrays and tuples (up to 8) of them; `#[derive(Static)]` implements it for
/// user types. (Not a blanket over `Hash`: that would rule out the float impls.)
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Static`, so a `#[tree(skip)]` field of this type cannot be hashed into the tree's structure",
    note = "derive `Static` for it (`#[derive(Static)]`), or mark the field `#[tree(skip, static_unchecked)]` (its value is then baked into compiled traces)"
)]
pub trait Static: 'static {
    /// Feeds the value to `h`.
    fn hash_static(&self, h: &mut dyn Hasher);
}

macro_rules! static_by_hash {
    ($($t:ty),*) => {$(
        impl Static for $t {
            fn hash_static(&self, mut h: &mut dyn Hasher) {
                Hash::hash(self, &mut h);
            }
        }
    )*};
}

static_by_hash!(
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    bool,
    char,
    (),
    String,
    &'static str,
    DType,
    Device,
    Duration
);

impl Static for f32 {
    fn hash_static(&self, h: &mut dyn Hasher) {
        h.write_u32(self.to_bits());
    }
}

impl Static for f64 {
    fn hash_static(&self, h: &mut dyn Hasher) {
        h.write_u64(self.to_bits());
    }
}

impl<T: Static> Static for Option<T> {
    fn hash_static(&self, h: &mut dyn Hasher) {
        match self {
            None => h.write_u8(0),
            Some(v) => {
                h.write_u8(1);
                v.hash_static(h);
            }
        }
    }
}

impl<T: Static> Static for Vec<T> {
    fn hash_static(&self, h: &mut dyn Hasher) {
        h.write_usize(self.len());
        for v in self {
            v.hash_static(h);
        }
    }
}

impl<T: Static, const N: usize> Static for [T; N] {
    fn hash_static(&self, h: &mut dyn Hasher) {
        for v in self {
            v.hash_static(h);
        }
    }
}

impl<T: Static + ?Sized> Static for Box<T> {
    fn hash_static(&self, h: &mut dyn Hasher) {
        (**self).hash_static(h);
    }
}

impl<T: Static + ?Sized> Static for Arc<T> {
    fn hash_static(&self, h: &mut dyn Hasher) {
        (**self).hash_static(h);
    }
}

impl<T: 'static> Static for PhantomData<T> {
    fn hash_static(&self, _: &mut dyn Hasher) {}
}

macro_rules! static_tuples {
    ($(($($t:ident $i:tt),+)),*) => {$(
        impl<$($t: Static),+> Static for ($($t,)+) {
            fn hash_static(&self, h: &mut dyn Hasher) {
                $(self.$i.hash_static(h);)+
            }
        }
    )*};
}

static_tuples!(
    (A 0),
    (A 0, B 1),
    (A 0, B 1, C 2),
    (A 0, B 1, C 2, D 3),
    (A 0, B 1, C 2, D 3, E 4),
    (A 0, B 1, C 2, D 3, E 4, F 5),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7)
);

// ── Visitors ────────────────────────────────────────────────────────────────

/// Receives what a [`Visit`] reports, with full paths. Every method but [`Visitor::leaf`]
/// has an empty default.
pub trait Visitor {
    /// A leaf: its array, kind and whether it is trainable.
    fn leaf(&mut self, path: &str, array: &Array, kind: LeafKind, trainable: bool);
    /// Whether the visitor reads paths (`true` by default); without them, every path it
    /// receives is empty and the visit builds none (faster).
    fn paths(&self) -> bool {
        true
    }
    /// A child tree of type `type_name` starts at `path` (`name` is its own segment; empty
    /// for transparent wrappers such as `Option`).
    fn enter(&mut self, path: &str, name: &str, type_name: &'static str) {
        let _ = (path, name, type_name);
    }
    /// The child tree entered last at `path` ends.
    fn exit(&mut self, path: &str) {
        let _ = path;
    }
    /// An enum's active variant (or `Option`'s: 0 for `None`, 1 for `Some`).
    fn variant(&mut self, path: &str, index: u32) {
        let _ = (path, index);
    }
    /// A mode flag (train/eval).
    fn mode(&mut self, path: &str, training: bool) {
        let _ = (path, training);
    }
    /// A static field (a hashed `#[tree(skip)]` field).
    fn static_field(&mut self, path: &str, value: &dyn Static) {
        let _ = (path, value);
    }
}

/// Appends `name` to `path` (`.`-joined; an empty name is transparent) and returns the
/// previous length.
fn push(path: &mut String, name: &str) -> usize {
    let len = path.len();
    if !name.is_empty() {
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(name);
    }
    len
}

/// The read-only visitor that [`Leaves::visit`] reports to: it builds paths and dispatches
/// leaves, child trees, enum variants, mode flags and static fields to a [`Visitor`].
pub struct Visit<'a> {
    path: String,
    paths: bool,
    visitor: &'a mut dyn Visitor,
}

impl<'a> Visit<'a> {
    /// A visit reporting to `visitor`, starting at the root (empty path).
    pub fn new(visitor: &'a mut dyn Visitor) -> Self {
        Self {
            path: String::new(),
            paths: visitor.paths(),
            visitor,
        }
    }

    fn push(&mut self, name: &str) -> usize {
        if self.paths {
            push(&mut self.path, name)
        } else {
            0
        }
    }

    /// The current path.
    pub fn path(&self) -> &str {
        &self.path
    }

    fn leaf(&mut self, name: &str, a: &Array, kind: LeafKind, trainable: bool) {
        let len = self.push(name);
        self.visitor.leaf(&self.path, a, kind, trainable);
        self.path.truncate(len);
    }

    /// A plain array leaf (always differentiable).
    pub fn array(&mut self, name: &str, a: &Array) {
        self.leaf(name, a, LeafKind::Array, true);
    }

    /// A parameter leaf (trainable unless frozen).
    pub fn param(&mut self, name: &str, p: &Param) {
        self.leaf(name, &p.value, LeafKind::Param, !p.frozen);
    }

    /// A buffer leaf (never trainable).
    pub fn buffer(&mut self, name: &str, a: &Array) {
        self.leaf(name, a, LeafKind::Buffer, false);
    }

    /// A child tree named `name` (empty: transparent).
    pub fn child<T: Leaves + ?Sized>(&mut self, name: &str, t: &T) {
        let len = self.push(name);
        self.visitor
            .enter(&self.path, name, std::any::type_name::<T>());
        t.visit(self);
        self.visitor.exit(&self.path);
        self.path.truncate(len);
    }

    /// The active variant of an enum (its index), part of the structure.
    pub fn variant(&mut self, index: u32) {
        self.visitor.variant(&self.path, index);
    }

    /// A mode flag (`true` in training mode), part of the structure.
    pub fn mode(&mut self, training: bool) {
        self.visitor.mode(&self.path, training);
    }

    /// A static field, hashed into the structure.
    pub fn static_field(&mut self, name: &str, value: &dyn Static) {
        let len = self.push(name);
        self.visitor.static_field(&self.path, value);
        self.path.truncate(len);
    }
}

/// Receives what a [`VisitMut`] reports, with full paths.
pub trait VisitorMut {
    /// A leaf, mutably: its array, kind and, for a parameter, its frozen flag.
    fn leaf(&mut self, path: &str, array: &mut Array, kind: LeafKind, frozen: Option<&mut bool>);
    /// Whether the visitor reads paths (`true` by default); see [`Visitor::paths`].
    fn paths(&self) -> bool {
        true
    }
    /// A child tree at `path`, as `&mut dyn Any` (to find or replace submodules of a given
    /// type); returns whether to visit inside it (the default).
    fn child(&mut self, path: &str, child: &mut dyn Any) -> bool {
        let _ = (path, child);
        true
    }
    /// A mode flag (train/eval), mutably.
    fn training(&mut self, path: &str, flag: &mut bool) {
        let _ = (path, flag);
    }
}

/// The mutable visitor that [`Tree::visit_mut`] reports to.
pub struct VisitMut<'a> {
    path: String,
    paths: bool,
    visitor: &'a mut dyn VisitorMut,
}

impl<'a> VisitMut<'a> {
    /// A visit reporting to `visitor`, starting at the root (empty path).
    pub fn new(visitor: &'a mut dyn VisitorMut) -> Self {
        Self {
            path: String::new(),
            paths: visitor.paths(),
            visitor,
        }
    }

    fn push(&mut self, name: &str) -> usize {
        if self.paths {
            push(&mut self.path, name)
        } else {
            0
        }
    }

    /// The current path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// A plain array leaf.
    pub fn array(&mut self, name: &str, a: &mut Array) {
        let len = self.push(name);
        self.visitor.leaf(&self.path, a, LeafKind::Array, None);
        self.path.truncate(len);
    }

    /// A parameter leaf.
    pub fn param(&mut self, name: &str, p: &mut Param) {
        let len = self.push(name);
        self.visitor.leaf(
            &self.path,
            &mut p.value,
            LeafKind::Param,
            Some(&mut p.frozen),
        );
        self.path.truncate(len);
    }

    /// A buffer leaf.
    pub fn buffer(&mut self, name: &str, a: &mut Array) {
        let len = self.push(name);
        self.visitor.leaf(&self.path, a, LeafKind::Buffer, None);
        self.path.truncate(len);
    }

    /// A child tree named `name` (empty: transparent).
    pub fn child<T: Tree>(&mut self, name: &str, t: &mut T) {
        let len = self.push(name);
        if self.visitor.child(&self.path, t) {
            t.visit_mut(self);
        }
        self.path.truncate(len);
    }

    /// A mode flag (`true` in training mode).
    pub fn training(&mut self, flag: &mut bool) {
        self.visitor.training(&self.path, flag);
    }
}

// ── The traits ──────────────────────────────────────────────────────────────

/// Anything whose array leaves can be visited in a fixed order with their paths: owned
/// trees and borrowed views (`&T`, `(&A, &B)`, `[T]`). Object-safe.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a tree of arrays",
    note = "derive `Tree` for it (`#[derive(Clone, Tree)]`), or use a tuple, `Vec`, map or `Option` of arrays"
)]
pub trait Leaves {
    /// Reports every leaf (and child tree, enum variant, mode flag and static field) to
    /// `v`, in order.
    fn visit(&self, v: &mut Visit<'_>);
}

/// An owned tree of arrays: rebuilt with new leaves of the same structure, mutated in place.
///
/// Implemented for [`Array`], [`Param`], `()`, `Option<T>`, `Box<T>`, `Vec<T>`, `[T; N]`,
/// tuples up to 8 elements, `BTreeMap<String, T>` and `HashMap<String, T>` (visited in
/// sorted key order), and user types with `#[derive(Tree)]`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a tree of arrays",
    note = "derive `Tree` for it (`#[derive(Clone, Tree)]`), or use a tuple, `Vec`, map or `Option` of arrays"
)]
pub trait Tree: Leaves + Clone + 'static {
    /// Reports every leaf (and child tree and mode flag) to `v` mutably, in the order of
    /// [`Leaves::visit`].
    fn visit_mut(&mut self, v: &mut VisitMut<'_>);

    /// Calls `f(path, leaf)` for every leaf, in order.
    fn for_each(&self, mut f: impl FnMut(&str, &Array)) {
        struct ForEach<'f>(&'f mut dyn FnMut(&str, &Array));
        impl Visitor for ForEach<'_> {
            fn leaf(&mut self, path: &str, a: &Array, _: LeafKind, _: bool) {
                (self.0)(path, a);
            }
        }
        self.visit(&mut Visit::new(&mut ForEach(&mut f)));
    }

    /// The tree with every leaf replaced by `f(path, leaf)` (same structure; parameters keep
    /// their frozen flag).
    fn map(&self, mut f: impl FnMut(&str, &Array) -> Array) -> Self {
        struct Map<'f>(&'f mut dyn FnMut(&str, &Array) -> Array);
        impl VisitorMut for Map<'_> {
            fn leaf(&mut self, path: &str, a: &mut Array, _: LeafKind, _: Option<&mut bool>) {
                *a = (self.0)(path, a);
            }
        }
        let mut out = self.clone();
        out.visit_mut(&mut VisitMut::new(&mut Map(&mut f)));
        out
    }

    /// The leaves, in order.
    fn leaves(&self) -> Vec<Array> {
        collect_leaves(self)
    }

    /// The leaves' paths, in order.
    fn paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.for_each(|p, _| out.push(p.to_owned()));
        out
    }

    /// The leaves with their paths, in order (MLX's `tree_flatten`).
    fn flatten(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        self.for_each(|p, a| out.push((p.to_owned(), a.clone())));
        out
    }

    /// A tree of this structure (`self` is the template) with `leaves` in order: exactly
    /// [`Tree::len`] of them, else an error naming both counts.
    fn unflatten(&self, leaves: impl IntoIterator<Item = Array>) -> Result<Self> {
        struct Assign<I>(I, usize, usize);
        impl<I: Iterator<Item = Array>> VisitorMut for Assign<I> {
            fn leaf(&mut self, _: &str, a: &mut Array, _: LeafKind, _: Option<&mut bool>) {
                self.2 += 1;
                if let Some(next) = self.0.next() {
                    *a = next;
                    self.1 += 1;
                }
            }
            fn paths(&self) -> bool {
                false
            }
        }
        let mut out = self.clone();
        let mut assign = Assign(leaves.into_iter(), 0, 0);
        out.visit_mut(&mut VisitMut::new(&mut assign));
        let (given, expected) = (assign.1 + assign.0.count(), assign.2);
        if given != expected {
            return Err(Error::InvalidArgument(format!(
                "unflatten: {given} leaves for a tree of {expected}"
            )));
        }
        Ok(out)
    }

    /// Replaces leaves by path. [`LoadMode::Strict`] requires exactly the tree's paths with
    /// matching shapes; [`LoadMode::Partial`] assigns the matching names and ignores
    /// unknown ones (shapes must still match). Loaded dtypes are kept. On any problem the
    /// error lists every one (missing, unknown and mismatched names, sorted) and the tree is
    /// unchanged. Duplicate paths (in the tree or in `named`) are an error.
    fn load_flat(
        &mut self,
        named: impl IntoIterator<Item = (String, Array)>,
        mode: LoadMode,
    ) -> Result<()> {
        load_named(self, named.into_iter().collect(), mode)
    }

    /// The number of leaves.
    fn len(&self) -> usize {
        struct Count(usize);
        impl Visitor for Count {
            fn leaf(&mut self, _: &str, _: &Array, _: LeafKind, _: bool) {
                self.0 += 1;
            }
            fn paths(&self) -> bool {
                false
            }
        }
        let mut c = Count(0);
        self.visit(&mut Visit::new(&mut c));
        c.0
    }

    /// Whether the tree has no leaves.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The tree's structure.
    fn def(&self) -> TreeDef {
        TreeDef::of(self)
    }
}

/// The leaves of any [`Leaves`] value, in order.
pub fn collect_leaves<T: Leaves + ?Sized>(tree: &T) -> Vec<Array> {
    struct Collect(Vec<Array>);
    impl Visitor for Collect {
        fn leaf(&mut self, _: &str, a: &Array, _: LeafKind, _: bool) {
            self.0.push(a.clone());
        }
        fn paths(&self) -> bool {
            false
        }
    }
    let mut c = Collect(Vec::new());
    tree.visit(&mut Visit::new(&mut c));
    c.0
}

/// How [`Tree::load_flat`] treats names it does not find.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LoadMode {
    /// Every path must be given, and nothing else.
    #[default]
    Strict,
    /// Matching names are assigned; missing paths keep their values, unknown names are
    /// ignored.
    Partial,
}

fn load_named<T: Tree>(tree: &mut T, named: Vec<(String, Array)>, mode: LoadMode) -> Result<()> {
    let mut given: HashMap<String, Array> = HashMap::with_capacity(named.len());
    let mut problems: Vec<(u8, String)> = Vec::new();
    for (name, a) in named {
        if given.contains_key(&name) {
            return Err(Error::InvalidArgument(format!(
                "load_flat: {name:?} is given twice"
            )));
        }
        given.insert(name, a);
    }
    let mut shapes: Vec<(String, Vec<i32>)> = Vec::new();
    {
        struct Shapes<'a>(&'a mut Vec<(String, Vec<i32>)>);
        impl Visitor for Shapes<'_> {
            fn leaf(&mut self, path: &str, a: &Array, _: LeafKind, _: bool) {
                self.0.push((path.to_owned(), a.shape().to_vec()));
            }
        }
        tree.visit(&mut Visit::new(&mut Shapes(&mut shapes)));
    }
    let mut seen = std::collections::HashSet::new();
    for (path, shape) in &shapes {
        if !seen.insert(path.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "load_flat: the tree has the path {path:?} twice"
            )));
        }
        match given.get(path) {
            Some(a) if a.shape() != shape.as_slice() => problems.push((
                2,
                format!("shape: {path} {:?}, expected {shape:?}", a.shape()),
            )),
            Some(_) => {}
            None if mode == LoadMode::Strict => problems.push((0, format!("missing: {path}"))),
            None => {}
        }
    }
    if mode == LoadMode::Strict {
        for name in given.keys() {
            if !seen.contains(name.as_str()) {
                problems.push((1, format!("unknown: {name}")));
            }
        }
    }
    if !problems.is_empty() {
        problems.sort();
        let n = problems.len();
        let mut msg = format!("load_flat: {n} problem{}", if n == 1 { "" } else { "s" });
        for (_, p) in problems {
            msg.push_str("\n  ");
            msg.push_str(&p);
        }
        return Err(Error::InvalidArgument(msg));
    }
    struct Assign<'a>(&'a mut HashMap<String, Array>);
    impl VisitorMut for Assign<'_> {
        fn leaf(&mut self, path: &str, a: &mut Array, _: LeafKind, _: Option<&mut bool>) {
            if let Some(v) = self.0.remove(path) {
                *a = v;
            }
        }
    }
    tree.visit_mut(&mut VisitMut::new(&mut Assign(&mut given)));
    Ok(())
}

// ── Structure ───────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum DefNode {
    Leaf {
        path: String,
        kind: LeafKind,
        trainable: bool,
    },
    Variant {
        path: String,
        index: u32,
    },
    Mode {
        path: String,
        training: bool,
    },
}

/// The structure of a tree: its root type, the ordered `(path, kind, trainable)` of every
/// leaf, the active enum variants, the mode flags and a hash of the static fields. Equal
/// structures have equal `TreeDef`s (see the [module documentation](self)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TreeDef {
    root: TypeId,
    root_name: &'static str,
    nodes: Vec<DefNode>,
    statics: u64,
}

impl TreeDef {
    /// The structure of `tree`.
    pub fn of<T: Leaves + ?Sized + 'static>(tree: &T) -> Self {
        struct Collect {
            nodes: Vec<DefNode>,
            hasher: std::hash::DefaultHasher,
        }
        impl Visitor for Collect {
            fn leaf(&mut self, path: &str, _: &Array, kind: LeafKind, trainable: bool) {
                self.nodes.push(DefNode::Leaf {
                    path: path.to_owned(),
                    kind,
                    trainable,
                });
            }
            fn variant(&mut self, path: &str, index: u32) {
                self.nodes.push(DefNode::Variant {
                    path: path.to_owned(),
                    index,
                });
            }
            fn mode(&mut self, path: &str, training: bool) {
                self.nodes.push(DefNode::Mode {
                    path: path.to_owned(),
                    training,
                });
            }
            fn static_field(&mut self, path: &str, value: &dyn Static) {
                self.hasher.write(path.as_bytes());
                self.hasher.write_u8(0xff);
                value.hash_static(&mut self.hasher);
            }
        }
        let mut c = Collect {
            nodes: Vec::new(),
            hasher: std::hash::DefaultHasher::new(),
        };
        tree.visit(&mut Visit::new(&mut c));
        Self {
            root: TypeId::of::<T>(),
            root_name: std::any::type_name::<T>(),
            nodes: c.nodes,
            statics: c.hasher.finish(),
        }
    }

    /// The name of the root type.
    pub fn type_name(&self) -> &'static str {
        self.root_name
    }

    /// The leaves: path, kind and whether trainable, in order.
    pub fn leaves(&self) -> impl Iterator<Item = (&str, LeafKind, bool)> {
        self.nodes.iter().filter_map(|n| match n {
            DefNode::Leaf {
                path,
                kind,
                trainable,
            } => Some((path.as_str(), *kind, *trainable)),
            _ => None,
        })
    }

    /// The number of leaves.
    pub fn len(&self) -> usize {
        self.leaves().count()
    }

    /// Whether the tree has no leaves.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The first difference from `other`, described for error messages (`None` if equal).
    pub fn difference(&self, other: &TreeDef) -> Option<String> {
        if self == other {
            return None;
        }
        if self.root != other.root {
            return Some(format!("type {} vs {}", self.root_name, other.root_name));
        }
        let name = |n: &DefNode| match n {
            DefNode::Leaf {
                path,
                kind,
                trainable,
            } => format!(
                "leaf {path:?} ({kind:?}{})",
                if *trainable { "" } else { ", not trainable" }
            ),
            DefNode::Variant { path, index } => format!("variant {index} at {path:?}"),
            DefNode::Mode { path, training } => {
                format!(
                    "{} mode at {path:?}",
                    if *training { "train" } else { "eval" }
                )
            }
        };
        for i in 0..self.nodes.len().max(other.nodes.len()) {
            match (self.nodes.get(i), other.nodes.get(i)) {
                (Some(a), Some(b)) if a == b => {}
                (Some(a), Some(b)) => return Some(format!("{} vs {}", name(a), name(b))),
                (Some(a), None) => return Some(format!("{} vs nothing", name(a))),
                (None, Some(b)) => return Some(format!("nothing vs {}", name(b))),
                (None, None) => {}
            }
        }
        Some("different static fields (configuration or hyperparameters)".into())
    }
}

// ── Zero gradients ──────────────────────────────────────────────────────────

/// A zero of `like`'s dtype broadcast to its shape: a stride-0 view of one element, so
/// evaluating it allocates no full-size memory (gradients of frozen leaves).
pub(crate) fn broadcast_zero(like: &Array) -> Array {
    Array::from(0u8)
        .astype(like.dtype())
        .broadcast_to(like.shape())
}

/// Whether a leaf of this dtype can be differentiated.
pub(crate) fn differentiable(dtype: DType) -> bool {
    matches!(dtype.category(), Category::Float | Category::Complex)
}

// ── Impls ───────────────────────────────────────────────────────────────────

impl Leaves for Array {
    fn visit(&self, v: &mut Visit<'_>) {
        v.array("", self);
    }
}

impl Tree for Array {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        v.array("", self);
    }
}

impl Leaves for Param {
    fn visit(&self, v: &mut Visit<'_>) {
        v.param("", self);
    }
}

impl Tree for Param {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        v.param("", self);
    }
}

impl Leaves for () {
    fn visit(&self, _: &mut Visit<'_>) {}
}

impl Tree for () {
    fn visit_mut(&mut self, _: &mut VisitMut<'_>) {}
}

impl<T: ?Sized> Leaves for PhantomData<T> {
    fn visit(&self, _: &mut Visit<'_>) {}
}

impl<T: ?Sized + 'static> Tree for PhantomData<T> {
    fn visit_mut(&mut self, _: &mut VisitMut<'_>) {}
}

impl<T: Leaves + ?Sized> Leaves for &T {
    fn visit(&self, v: &mut Visit<'_>) {
        (**self).visit(v);
    }
}

impl<T: Leaves + ?Sized> Leaves for &mut T {
    fn visit(&self, v: &mut Visit<'_>) {
        (**self).visit(v);
    }
}

impl<T: Leaves> Leaves for Option<T> {
    fn visit(&self, v: &mut Visit<'_>) {
        match self {
            None => v.variant(0),
            Some(x) => {
                v.variant(1);
                v.child("", x);
            }
        }
    }
}

impl<T: Tree> Tree for Option<T> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        if let Some(x) = self {
            v.child("", x);
        }
    }
}

impl<T: Leaves> Leaves for Box<T> {
    fn visit(&self, v: &mut Visit<'_>) {
        v.child("", &**self);
    }
}

impl<T: Tree> Tree for Box<T> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        v.child("", &mut **self);
    }
}

impl<T: Leaves> Leaves for [T] {
    fn visit(&self, v: &mut Visit<'_>) {
        for (i, x) in self.iter().enumerate() {
            v.child(&i.to_string(), x);
        }
    }
}

impl<T: Leaves> Leaves for Vec<T> {
    fn visit(&self, v: &mut Visit<'_>) {
        self.as_slice().visit(v);
    }
}

impl<T: Tree> Tree for Vec<T> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        for (i, x) in self.iter_mut().enumerate() {
            v.child(&i.to_string(), x);
        }
    }
}

impl<T: Leaves, const N: usize> Leaves for [T; N] {
    fn visit(&self, v: &mut Visit<'_>) {
        self.as_slice().visit(v);
    }
}

impl<T: Tree, const N: usize> Tree for [T; N] {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        for (i, x) in self.iter_mut().enumerate() {
            v.child(&i.to_string(), x);
        }
    }
}

impl<T: Leaves> Leaves for BTreeMap<String, T> {
    fn visit(&self, v: &mut Visit<'_>) {
        for (k, x) in self {
            v.child(k, x);
        }
    }
}

impl<T: Tree> Tree for BTreeMap<String, T> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        for (k, x) in self.iter_mut() {
            v.child(k, x);
        }
    }
}

impl<T: Leaves, S: 'static> Leaves for HashMap<String, T, S> {
    fn visit(&self, v: &mut Visit<'_>) {
        let mut entries: Vec<(&String, &T)> = self.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        for (k, x) in entries {
            v.child(k, x);
        }
    }
}

impl<T: Tree, S: std::hash::BuildHasher + Clone + 'static> Tree for HashMap<String, T, S> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        let mut entries: Vec<(&String, &mut T)> = self.iter_mut().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        for (k, x) in entries {
            v.child(k, x);
        }
    }
}

macro_rules! tuples {
    ($(($($t:ident $i:tt),+)),*) => {$(
        impl<$($t: Leaves),+> Leaves for ($($t,)+) {
            fn visit(&self, v: &mut Visit<'_>) {
                $(v.child(stringify!($i), &self.$i);)+
            }
        }
        impl<$($t: Tree),+> Tree for ($($t,)+) {
            fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
                $(v.child(stringify!($i), &mut self.$i);)+
            }
        }
    )*};
}

tuples!(
    (A 0),
    (A 0, B 1),
    (A 0, B 1, C 2),
    (A 0, B 1, C 2, D 3),
    (A 0, B 1, C 2, D 3, E 4),
    (A 0, B 1, C 2, D 3, E 4, F 5),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7)
);

/// What `#[derive(Tree)]`, `#[derive(Module)]` and `#[derive(Static)]` generate code
/// against (re-exported as `nuvec::__derive`). Not a public API.
#[doc(hidden)]
pub mod __derive {
    pub use std::any::TypeId;
    pub use std::clone::Clone;
    pub use std::hash::Hasher;
    pub use std::marker::PhantomData;

    pub use super::{LeafKind, Leaves, Param, Static, Tree, Visit, VisitMut};
    pub use crate::Array;

    /// Fails to compile unless `T` is [`Static`] (the field's span carries the error).
    pub const fn assert_static<T: Static + ?Sized>() {}
}
