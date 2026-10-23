//! Neural-network building blocks on nuvec, following MLX's `mlx.nn` and `mlx.optimizers`.
//!
//! A model is a plain struct of layers with `#[derive(Module)]`: its parameters get MLX's
//! names from the field names (`layers.0.weight`), and every model is a [`Tree`], so the
//! core transforms differentiate it and return gradients of the model's own type
//! (`grads.layers[0].weight`).
//!
//! - [`Module`]: names and counts ([`Module::parameters`], [`Module::num_params`],
//!   [`Module::summary`], [`Module::display`]), freezing by path ([`Module::freeze`],
//!   [`Module::unfreeze`]), train/eval mode ([`Module::train`]), loading and saving by name
//!   ([`Module::load_weights`], [`Module::save_weights`]) and replacing submodules
//!   ([`Module::map_modules`]).
//! - Layers ([`layers`]): [`Linear`], [`QuantizedLinear`], [`Embedding`],
//!   [`QuantizedEmbedding`], [`RmsNorm`], [`LayerNorm`], [`Rope`], [`LoraLinear`] and
//!   [`Dropout`]; activations [`relu`], [`silu`], [`gelu_approx`] and [`gelu_fast_approx`].
//!   Constructors take no `bool` and return no `Result`: `Linear::new(784, 32)`,
//!   `Linear::new(32, 10).without_bias()`.
//! - [`losses::cross_entropy`] and [`optim`]: SGD, Adam, AdamW, Lion and Muon with
//!   learning-rate schedules, parameter groups ([`optim::MultiOptimizer`]) and gradient
//!   clipping; the optimizer state is a tree keyed by parameter path.
//!
//! Layers' `forward`s, activations and losses return an [`Array`], as every nuvec op: an
//! invalid input gives a failed array whose error surfaces at a boundary (`check()`,
//! `eval()`, `item()`, a transform) and reports where `forward` was called.
//!
//! ```
//! use nuvec::{Array, Result, array, random, value_and_grad};
//! use nuvec_nn::optim::{Optimizer, Sgd};
//! use nuvec_nn::{Linear, Module, Tree, losses, relu};
//!
//! #[derive(Clone, Module)]
//! struct Mlp {
//!     hidden: Linear,
//!     out: Linear,
//! }
//!
//! impl Mlp {
//!     fn forward(&self, x: &Array) -> Array {
//!         self.out.forward(&relu(&self.hidden.forward(x)))
//!     }
//! }
//!
//! # fn main() -> Result<()> {
//! random::seed(0);
//! let mut model = Mlp { hidden: Linear::new(2, 8), out: Linear::new(8, 2) };
//! assert_eq!(model.paths(), ["hidden.weight", "hidden.bias", "out.weight", "out.bias"]);
//!
//! let (x, y) = (array![[1.0, -1.0], [0.5, 2.0]], array![0u32, 1]);
//! let loss = |m: &Mlp| losses::cross_entropy(&m.forward(&x), &y, losses::Reduction::Mean);
//! let mut opt = Sgd::new(0.5);
//! let (before, _) = value_and_grad(loss)(&model)?;
//! for _ in 0..20 {
//!     let (_, grads) = value_and_grad(loss)(&model)?;   // grads: Mlp
//!     opt.update(&mut model, &grads)?;
//! }
//! let (after, _) = value_and_grad(loss)(&model)?;
//! assert!(after.item::<f32>()? < before.item::<f32>()?);
//! # Ok(())
//! # }
//! ```
//!
//! The derive is syntactic: a field typed `Array` (or `Option<Array>`, `Vec<Array>`)
//! needs a role (`Param` for a parameter, `#[tree(buffer)]`, `#[tree(skip)]` for a computed
//! constant); a type alias of `Array` is not recognised. Trait-object fields are not
//! trees: use an enum of the concrete layer types.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::any::Any;
use std::path::Path;

use nuvec::tree::{LeafKind, LoadMode, Visit, VisitMut, Visitor, VisitorMut};
use nuvec::{Array, Category, Error, Result};

mod display;
pub mod layers;
pub mod losses;
pub mod optim;
mod weights;

pub use display::{ModuleDisplay, Summary, SummaryRow};
pub use layers::{
    Dropout, Embedding, LayerNorm, Linear, LoraBase, LoraLinear, QuantizedEmbedding,
    QuantizedLinear, RmsNorm, Rope,
};
/// Derives a model's [`Tree`] impl, with a check against bare `Array` fields (see the
/// crate documentation). Every tree is a [`Module`] (a blanket impl).
pub use nuvec::macros::Module;
pub use nuvec::{Param, Tree};
pub use weights::WeightSource;

/// The Rust snippets of the user guide's neural-network page, run as doctests.
#[cfg(doctest)]
mod site_doctests {
    #[doc = include_str!("../../../site/content/docs/neural-networks.md")]
    pub struct NeuralNetworks;
}

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Model operations, as MLX's `nn.Module`, for every [`Tree`] (a blanket impl): a model is
/// a struct of layers with `#[derive(Module)]`, and its leaves' paths are MLX's names.
///
/// Leaves are parameters ([`Param`], trainable unless frozen), buffers (never trained,
/// such as quantized weights) or plain arrays.
pub trait Module: Tree {
    /// Every leaf (parameters, buffers and plain arrays) with its path, in order.
    fn parameters(&self) -> Vec<(String, Array)> {
        self.flatten()
    }

    /// The trainable leaves with their paths, in order.
    fn trainable_parameters(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        visit_leaves(self, |p, a, _, trainable| {
            if trainable {
                out.push((p.to_owned(), a.clone()));
            }
        });
        out
    }

    /// The number of elements over every leaf (MLX's `tree_flatten` total).
    fn num_params(&self) -> usize {
        let mut n = 0;
        visit_leaves(self, |_, a, _, _| n += a.size());
        n
    }

    /// The number of elements over the trainable leaves.
    fn num_trainable(&self) -> usize {
        let mut n = 0;
        visit_leaves(self, |_, a, _, trainable| {
            if trainable {
                n += a.size();
            }
        });
        n
    }

    /// A table of the leaves (path, shape, dtype, trainable, bytes) with totals.
    fn summary(&self) -> Summary {
        Summary::of(self)
    }

    /// The nested modules with their parameter counts, for printing:
    /// `println!("{}", model.display())`.
    fn display(&self) -> ModuleDisplay<'_, Self> {
        ModuleDisplay::new(self)
    }

    /// Freezes the parameters whose path matches `filter`, returning how many changed.
    /// A matching plain `Array` leaf cannot be frozen (only [`Param`]s can): that is an
    /// error naming the paths, and nothing changes. Buffers are never trainable anyway.
    fn freeze(&mut self, filter: impl Fn(&str) -> bool) -> Result<usize> {
        set_frozen(self, &filter, true)
    }

    /// Unfreezes the parameters whose path matches `filter`, returning how many changed.
    /// Buffers (quantized weights) and integer parameters stay non-trainable.
    fn unfreeze(&mut self, filter: impl Fn(&str) -> bool) -> Result<usize> {
        set_frozen(self, &filter, false)
    }

    /// Sets every mode flag of the tree: training (`true`, the default) or evaluation
    /// (`false`, where dropout is the identity), as MLX's `Module.train(mode)`. (There is
    /// no `eval()`: it would read like `x.eval()`, which evaluates.)
    fn train(&mut self, on: bool) {
        struct Train(bool);
        impl VisitorMut for Train {
            fn leaf(&mut self, _: &str, _: &mut Array, _: LeafKind, _: Option<&mut bool>) {}
            fn training(&mut self, _: &str, flag: &mut bool) {
                *flag = self.0;
            }
        }
        self.visit_mut(&mut VisitMut::new(&mut Train(on)));
    }

    /// Whether the tree is in training mode (its first mode flag; `true` without any).
    fn training(&self) -> bool {
        struct Mode(Option<bool>);
        impl Visitor for Mode {
            fn leaf(&mut self, _: &str, _: &Array, _: LeafKind, _: bool) {}
            fn mode(&mut self, _: &str, training: bool) {
                self.0.get_or_insert(training);
            }
        }
        let mut m = Mode(None);
        self.visit(&mut Visit::new(&mut m));
        m.0.unwrap_or(true)
    }

    /// Replaces leaves by name, strictly (MLX's `load_weights(strict=True)`): every leaf
    /// must be given with its shape and nothing else (loaded dtypes are kept). On any
    /// problem the error lists every missing, unknown and mismatched name, and the model
    /// is unchanged. `src` is a safetensors path or named arrays.
    fn load_weights(&mut self, src: impl WeightSource) -> Result<()> {
        weights::load(self, src, LoadMode::Strict)
    }

    /// Replaces the leaves given by name (MLX's `load_weights(strict=False)`): unknown
    /// names are ignored, missing leaves keep their values, shapes must match.
    fn load_weights_partial(&mut self, src: impl WeightSource) -> Result<()> {
        weights::load(self, src, LoadMode::Partial)
    }

    /// Writes every leaf as safetensors with MLX's names and dtypes (MLX's
    /// `save_weights`), evaluating them. Unloaded placeholders are an error naming them.
    fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        weights::save(self, path.as_ref(), false)
    }

    /// Writes only the trainable leaves (an adapter's weights, say).
    fn save_trainable(&self, path: impl AsRef<Path>) -> Result<()> {
        weights::save(self, path.as_ref(), true)
    }

    /// Calls `f(path, submodule)` on every submodule of type `T` whose path matches
    /// `filter` (`f` may replace it, as when applying LoRA), returning how many. It stops at
    /// the first error from `f`; submodules changed before it keep their change. Matching
    /// submodules are not searched for further matches.
    fn map_modules<T: Tree>(
        &mut self,
        filter: impl Fn(&str) -> bool,
        mut f: impl FnMut(&str, &mut T) -> Result<()>,
    ) -> Result<usize> {
        struct Map<'a, T> {
            filter: &'a dyn Fn(&str) -> bool,
            f: &'a mut dyn FnMut(&str, &mut T) -> Result<()>,
            count: usize,
            error: Option<Error>,
        }
        impl<T: 'static> VisitorMut for Map<'_, T> {
            fn leaf(&mut self, _: &str, _: &mut Array, _: LeafKind, _: Option<&mut bool>) {}
            fn child(&mut self, path: &str, child: &mut dyn Any) -> bool {
                if self.error.is_some() {
                    return false;
                }
                match child.downcast_mut::<T>() {
                    Some(m) if (self.filter)(path) => {
                        match (self.f)(path, m) {
                            Ok(()) => self.count += 1,
                            Err(e) => self.error = Some(e),
                        }
                        false
                    }
                    _ => true,
                }
            }
        }
        let mut map = Map {
            filter: &filter,
            f: &mut f,
            count: 0,
            error: None,
        };
        self.visit_mut(&mut VisitMut::new(&mut map));
        match map.error {
            Some(e) => Err(e),
            None => Ok(map.count),
        }
    }
}

impl<T: Tree> Module for T {}

/// Calls `f(path, leaf, kind, trainable)` for every leaf.
fn visit_leaves<T: Tree>(tree: &T, f: impl FnMut(&str, &Array, LeafKind, bool)) {
    struct V<F>(F);
    impl<F: FnMut(&str, &Array, LeafKind, bool)> Visitor for V<F> {
        fn leaf(&mut self, path: &str, a: &Array, kind: LeafKind, trainable: bool) {
            (self.0)(path, a, kind, trainable);
        }
    }
    tree.visit(&mut Visit::new(&mut V(f)));
}

fn set_frozen<T: Tree>(tree: &mut T, filter: &dyn Fn(&str) -> bool, frozen: bool) -> Result<usize> {
    if frozen {
        let mut plain = Vec::new();
        visit_leaves(tree, |p, _, kind, _| {
            if kind == LeafKind::Array && filter(p) {
                plain.push(p.to_owned());
            }
        });
        if !plain.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "freeze: plain arrays cannot be frozen (make them `Param`s): {}",
                plain.join(", ")
            )));
        }
    }
    struct Set<'a> {
        filter: &'a dyn Fn(&str) -> bool,
        frozen: bool,
        changed: usize,
    }
    impl VisitorMut for Set<'_> {
        fn leaf(&mut self, path: &str, a: &mut Array, kind: LeafKind, flag: Option<&mut bool>) {
            let Some(flag) = flag else { return };
            if kind != LeafKind::Param || !(self.filter)(path) || *flag == self.frozen {
                return;
            }
            // Integer parameters (packed weights, say) are never trained.
            if !self.frozen && !matches!(a.dtype().category(), Category::Float | Category::Complex)
            {
                return;
            }
            *flag = self.frozen;
            self.changed += 1;
        }
    }
    let mut set = Set {
        filter,
        frozen,
        changed: 0,
    };
    tree.visit_mut(&mut VisitMut::new(&mut set));
    Ok(set.changed)
}

/// `max(x, 0)`, as MLX's `nn.relu`.
#[track_caller]
pub fn relu(x: &Array) -> Array {
    x.maximum(0.0)
}

/// `x · sigmoid(x)`, as MLX's `nn.silu`.
#[track_caller]
pub fn silu(x: &Array) -> Array {
    x.multiply(x.sigmoid())
}

/// GELU with the tanh approximation, as MLX's `nn.gelu_approx`:
/// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
#[track_caller]
pub fn gelu_approx(x: &Array) -> Array {
    let cube = x.multiply(x).multiply(x);
    let inner = x
        .add(cube.multiply(0.044715))
        .multiply((2.0 / std::f64::consts::PI).sqrt());
    x.multiply(0.5).multiply(inner.tanh().add(1.0))
}

/// `x · sigmoid(1.702 x)`, as MLX's `nn.gelu_fast_approx`.
#[track_caller]
pub fn gelu_fast_approx(x: &Array) -> Array {
    x.multiply(x.multiply(1.702).sigmoid())
}
