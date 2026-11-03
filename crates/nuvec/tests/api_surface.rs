//! Compile-time pins of the public surface of `nuvec` (and of `nuvec-nn` and `nuvec-lm`).
//!
//! One module per spec that defines or changes the surface (`mod spec_062` for the array API;
//! later specs add `mod spec_NNN`). A change that breaks this file is a semver event: record it
//! in the CHANGELOG under "Breaking" and update the pin in the same change. `nuvec::graph` is
//! deliberately not pinned (it is the graph-level API for compilers). Items that must *not* be
//! at the crate root are pinned by `compile_fail` doc tests in `src/lib.rs`.

#![allow(dead_code, clippy::type_complexity)]

mod spec_062 {
    /// `Error::Op`, `OpErrorKind`, and the record-time failure queries.
    mod errors {
        use std::panic::Location;

        use nuvec::{Array, Error, OpErrorKind, Result};

        const _: fn(Array) -> Result<Array> = Array::check;
        const _: fn(&Array) -> bool = Array::is_failed;
        const _: for<'a> fn(&'a Array) -> Option<&'a Error> = Array::error;
        const _: fn(&Error) -> Option<&OpErrorKind> = Error::kind;
        const _: fn(&Error) -> Option<&'static Location<'static>> = Error::location;
        const _: fn(&Error) -> Option<&'static str> = Error::op_name;
        const _: fn(&Error) -> bool = Error::is_oom;

        type Fields = (
            &'static str,
            Option<&'static Location<'static>>,
            OpErrorKind,
        );
        fn fields(e: Error) -> Option<Fields> {
            match e {
                Error::Op { op, at, kind } => Some((op, at, kind)),
                _ => None,
            }
        }

        fn kinds(k: &OpErrorKind) -> u8 {
            match k {
                OpErrorKind::ShapeMismatch { lhs, rhs } => (lhs.len() + rhs.len()) as u8,
                OpErrorKind::DTypeMismatch { lhs: _, rhs: _ } => 1,
                OpErrorKind::SizeMismatch {
                    expected: _,
                    actual: _,
                } => 2,
                OpErrorKind::UnsupportedDType(_) => 3,
                OpErrorKind::InvalidAxis { axis: _, ndim: _ } => 4,
                OpErrorKind::InvalidShape(_) => 5,
                OpErrorKind::InvalidIndex(_) => 6,
                OpErrorKind::OutOfMemory(_) => 7,
                OpErrorKind::InvalidArgument(_) => 8,
                _ => 9,
            }
        }

        #[test]
        fn display_shape() {
            let e = nuvec::array![1.0, 2.0].reshape([3]).check().unwrap_err();
            let at = e.location().unwrap();
            assert!(
                e.to_string()
                    .starts_with(&format!("reshape at {}:", at.file()))
            );
        }
    }

    /// `array!`, `From`/`FromIterator`, `from_slice` and the constructors.
    mod construction {
        use nuvec::{Array, DType, Scalar, ScalarOrArray, array, ops};

        const _: fn(&[f32], [i32; 2]) -> Array = Array::from_slice;
        const _: fn([i32; 2], DType) -> Array = ops::zeros;
        const _: fn((), DType) -> Array = ops::ones;
        const _: fn([usize; 1], f64) -> Array = ops::full;
        const _: fn(i32, i32, i32) -> Array = ops::arange;
        const _: fn(&Array) -> Array = Array::zeros_like;
        const _: fn(&Array) -> Array = Array::ones_like;
        const _: fn(&Array, f32) -> Array = Array::full_like;
        const _: fn(&Array) -> Array = ops::zeros_like;
        const _: fn(&Array, ScalarOrArray) -> Array = ops::full_like;
        const _: fn(Scalar) -> ScalarOrArray = ScalarOrArray::from;

        fn conversions() -> [Array; 6] {
            [
                Array::from(1.0f32),
                Array::from([1u8, 2]),
                Array::from(vec![1i64]),
                Array::from(&[1.0f64][..]),
                (0..3).map(|i| i as f32).collect(),
                array![[1, 2], [3, 4]],
            ]
        }

        fn default_is_cheap() -> Array {
            Array::default()
        }
    }

    /// The methods, one closure per family (generic methods have no single fn type).
    mod methods {
        use nuvec::{Array, DType, ScanOptions, VarOptions};

        type M = fn(&Array, &Array) -> Array;

        const BINARY: [M; 13] = [
            |x, y| x.add(y),
            |x, y| x.subtract(y),
            |x, y| x.multiply(y),
            |x, y| x.divide(y),
            |x, y| x.maximum(y),
            |x, y| x.minimum(y),
            |x, y| x.logaddexp(y),
            |x, y| x.equal(y),
            |x, y| x.not_equal(y),
            |x, y| x.less(y),
            |x, y| x.less_equal(y),
            |x, y| x.greater(y),
            |x, y| x.greater_equal(y),
        ];
        const WEAK: [fn(&Array) -> Array; 2] = [|x| x.add(1.0), |x| x.multiply(2)];
        const UNARY: [fn(&Array) -> Array; 14] = [
            Array::negative,
            Array::abs,
            Array::exp,
            Array::log,
            Array::sqrt,
            Array::sin,
            Array::cos,
            Array::tanh,
            Array::sigmoid,
            Array::erfinv,
            Array::sign,
            Array::conjugate,
            Array::stop_gradient,
            Array::contiguous,
        ];
        const OTHER: [M; 4] = [
            |x, y| x.select(y, 0.0),
            |x, _| x.astype(DType::F16),
            |x, y| x.matmul(y),
            |x, y| x.addmm(y, y, 1.0, 1.0),
        ];
        const REDUCE: [fn(&Array) -> Array; 24] = [
            |x| x.sum(-1),
            |x| x.sum_keepdims([0, 1]),
            |x| x.prod(..),
            |x| x.prod_keepdims(1..),
            |x| x.max(0),
            |x| x.max_keepdims(-1),
            |x| x.min(..),
            |x| x.min_keepdims(0),
            |x| x.all(..),
            |x| x.all_keepdims(0),
            |x| x.any(..),
            |x| x.any_keepdims(0),
            |x| x.mean(..),
            |x| x.mean_keepdims(..),
            |x| x.logsumexp(-1),
            |x| x.logsumexp_keepdims(-1),
            |x| x.var(..),
            |x| x.var_keepdims(..),
            |x| {
                x.var_with(
                    -1,
                    VarOptions {
                        ddof: 1,
                        keepdims: false,
                    },
                )
            },
            |x| x.std(..),
            |x| x.std_keepdims(..),
            |x| x.std_with(.., VarOptions::default()),
            |x| x.softmax(-1),
            |x| x.argmax(-1),
        ];
        const SINGLE_AXIS: [fn(&Array) -> Array; 17] = [
            |x| x.argmax_keepdims(..),
            |x| x.argmin(0),
            |x| x.argmin_keepdims(0usize),
            |x| x.cumsum(-1),
            |x| {
                x.cumsum_with(
                    ..,
                    ScanOptions {
                        reverse: true,
                        exclusive: true,
                    },
                )
            },
            |x| x.cumprod(0),
            |x| x.cumprod_with(0, ScanOptions::default()),
            |x| x.cummax(0),
            |x| x.cummax_with(0, ScanOptions::default()),
            |x| x.cummin(0),
            |x| x.cummin_with(0, ScanOptions::default()),
            |x| x.logcumsumexp(0),
            |x| x.logcumsumexp_with(0, ScanOptions::default()),
            |x| x.sort(-1),
            |x| x.argsort(-1),
            |x| x.partition(1, -1),
            |x| x.argpartition(1, -1),
        ];
        const SHAPE: [fn(&Array) -> Array; 8] = [
            |x| x.topk(2, -1),
            |x| x.reshape([2, -1]),
            |x| x.transpose([1, 0]),
            |x| x.t(),
            |x| x.broadcast_to([2, 3]),
            |x| x.expand_dims([0, -1]),
            |x| x.squeeze(..),
            |x| x.repeat(2, ..),
        ];
        const SPLIT: [fn(&Array) -> Vec<Array>; 2] = [|x| x.split(2, -1), |x| x.split_at(&[1], 0)];
        const SPLIT_N: fn(&Array) -> [Array; 2] = |x| x.split_n::<2>(-1);
        const SHAPE_N: fn(&Array) -> nuvec::Result<[i32; 2]> = Array::shape_n;
        const SHAPE_OF: for<'a> fn(&'a Array) -> &'a [i32] = Array::shape;
        const GATHER: [M; 4] = [
            |x, i| x.take(i, ..),
            |x, i| x.take_along_axis(i, 0),
            |x, i| x.put_along_axis(i, 0.0, 0),
            |x, i| x.scatter_add_along_axis(i, 1.0, -1),
        ];
        const QUANT: fn(&Array) -> nuvec::Quantized = |w| w.quantize(nuvec::Quant::affine(64, 4));
        const QUANT_OPS: [fn(&Array, &nuvec::Quantized) -> Array; 3] = [
            |_, q| q.dequantize(None),
            |x, q| x.quantized_matmul(q, true),
            |x, q| x.gather_qmm(q, None, None, nuvec::GatherQmm::default()),
        ];
        const MM: M = |a, b| a.gather_mm(b, None, None, false);
        const CONV: M = |x, w| x.conv1d(w, 1, 0, 1, 1);
        const LIKE: [fn(&Array) -> Array; 3] =
            [Array::zeros_like, Array::ones_like, |x| x.full_like(7)];
    }

    /// The free functions: MLX's names in `ops`, the same parameters as the methods.
    mod free_functions {
        use nuvec::{Array, ScalarOrArray, VarOptions, ops};

        const _: fn(ScalarOrArray, ScalarOrArray) -> Array = ops::add;
        const _: fn(&Array) -> Array = ops::exp;
        const _: fn(&Array, i32) -> Array = ops::sum;
        const _: fn(&Array, [i32; 2]) -> Array = ops::mean_keepdims;
        const _: fn(&Array, std::ops::RangeFull) -> Array = ops::std;
        const _: fn(&Array, i32, VarOptions) -> Array = ops::var_with;
        const _: fn(&Array, i32) -> Array = ops::argmax;
        const _: fn(&Array, &Array) -> Array = ops::matmul;
        const _: fn(&Array, &Array, &Array, f32, f32) -> Array = ops::addmm;
        const _: fn(&[Array], i32) -> Array = ops::concatenate;
        const _: fn(&[Array], i32) -> Array = ops::stack;
        const _: fn(&Array, i32, i32) -> Vec<Array> = ops::split;
        const _: fn(&Array, i32) -> [Array; 3] = ops::split_n;
        const _: fn(ScalarOrArray, ScalarOrArray, ScalarOrArray) -> Array = ops::where_;
        const _: fn(&Array, nuvec::Quant) -> nuvec::Quantized = ops::quantize;
        const _: fn(&nuvec::Quantized, Option<nuvec::DType>) -> Array = ops::dequantize;
        const _: fn(&Array, &nuvec::Quantized, bool) -> Array = ops::quantized_matmul;
        const _: fn() -> &'static [&'static str] = ops::__op_names;

        #[test]
        fn every_method_has_a_mirror() {
            assert!(ops::__op_names().len() >= 94, "{}", ops::__op_names().len());
        }
    }

    /// `Axes` (one, a list, `..`, `1..`; empty = none), `Axis`, `IntoShape`.
    mod axes_shapes {
        use std::ops::{RangeFrom, RangeFull};

        use nuvec::{Axes, Axis, IntoShape};

        fn axes<A: Axes>() {}
        fn axis<A: Axis>() {}
        fn shape<S: IntoShape>() {}

        fn pins() {
            axes::<i32>();
            axes::<usize>();
            axes::<[i32; 2]>();
            axes::<[i32; 0]>();
            axes::<&[i32]>();
            axes::<&[usize]>();
            axes::<Vec<i32>>();
            axes::<Vec<usize>>();
            axes::<&Vec<usize>>();
            axes::<RangeFull>();
            axes::<RangeFrom<i32>>();
            axes::<RangeFrom<usize>>();
            axis::<i32>();
            axis::<usize>();
            axis::<RangeFull>();
            shape::<[i32; 3]>();
            shape::<[usize; 2]>();
            shape::<&[i32]>();
            shape::<&[usize]>();
            shape::<Vec<i32>>();
            shape::<Vec<usize>>();
            shape::<&[usize; 2]>();
            shape::<()>();
        }
    }

    /// `x.i(..)` for every index form, `at(..).set/add`, `Ellipsis`, `NewAxis`, `step`.
    mod indexing {
        use nuvec::{Array, ArrayIndex, At, Ellipsis, NewAxis, Step};

        fn index<I: ArrayIndex>() {}

        const AT: fn(Array) -> At = |x| x.at(0);
        const SET: fn(At, f32) -> Array = At::set;
        const ADD: fn(At, &Array) -> Array = |at, v| at.add(v);

        fn forms(x: &Array, idx: &Array) -> [Array; 9] {
            [
                x.i(0),
                x.i(1usize),
                x.i((.., 1..3)),
                x.i((Ellipsis, -1)),
                x.i((.., NewAxis)),
                x.i((..).step(-1)),
                x.i((1.., ..=2, (0..6).step(2))),
                x.i(idx),
                x.i((idx, 1usize..=2usize, ..3, ..=4usize, 0..)),
            ]
        }

        fn types() {
            index::<(i32, i32, i32, i32, i32, i32, i32, i32)>();
            index::<Array>();
            index::<nuvec::Stepped<std::ops::Range<usize>>>();
            index::<&[std::ops::Range<i32>]>();
            index::<Vec<std::ops::RangeFull>>();
            index::<&Vec<std::ops::Range<usize>>>();
        }

        const SLICE_UPDATE: fn(&Array, &Array) -> Array = |a, u| a.slice_update(u, (.., 0..2));
    }

    /// Scoped default streams.
    mod streams {
        use nuvec::{Array, Device, Stream, with_stream};

        const _: fn(Device, fn() -> Array) -> Array = with_stream;
        const _: fn(Device) -> Stream = Stream::from;

        #[test]
        fn restores() {
            let before = nuvec::default_stream();
            let y = with_stream(Device::cpu(), || nuvec::array![1.0].exp());
            assert_eq!(y.device(), Device::cpu());
            assert_eq!(nuvec::default_stream(), before);
        }
    }

    /// The everyday names in one import.
    mod prelude {
        use nuvec::prelude::*;

        fn everyday() -> Result<Array> {
            let x: Array = array![[1.0, 2.0]];
            let _ = (DType::F32, Ellipsis, NewAxis, (..).step(1));
            let y = with_stream(nuvec::Device::cpu(), || x.i((Ellipsis, 0)));
            y.check().map_err(|e: Error| e)
        }
    }
}

/// Trees, modules and tree-generic transforms.
mod spec_063 {
    use std::collections::BTreeMap;

    use nuvec::{
        Array, CustomFunction, CustomFunctionFlat, GradOutput, InAxes, InAxis, IntoResult,
        LeafKind, Leaves, LoadMode, OutAxes, Param, Result, Static, Tree, TreeDef, Visit, VisitMut,
        Visitor, VisitorMut, Vmap,
    };

    /// The tree traits, `Param`, `Static`, `TreeDef` and the visitors.
    mod trees {
        use super::*;

        fn leaves<T: Leaves + ?Sized>(_: &T) {}
        fn tree<T: Tree>(_: &T) {}
        fn statik<T: Static>(_: &T) {}
        fn kinds(k: LeafKind) -> u8 {
            match k {
                LeafKind::Array => 0,
                LeafKind::Param => 1,
                LeafKind::Buffer => 2,
            }
        }

        const _: fn(Array) -> Param = Param::new;
        const _: fn(Array) -> Param = Param::frozen;
        const _: fn(&Param) -> bool = Param::is_frozen;
        const _: fn(&mut Param, bool) = Param::set_frozen;
        const _: for<'a> fn(&'a Param) -> &'a Array = Param::value;
        const _: for<'a> fn(&'a mut Param) -> &'a mut Array = Param::value_mut;
        const _: fn(Param) -> Array = Param::into_inner;
        const _: fn(&TreeDef) -> usize = TreeDef::len;
        const _: fn(&TreeDef) -> bool = TreeDef::is_empty;
        const _: fn(&TreeDef) -> &'static str = TreeDef::type_name;
        const _: fn(&TreeDef, &TreeDef) -> Option<String> = TreeDef::difference;
        const _: fn(&Array) -> TreeDef = TreeDef::of::<Array>;
        fn visit<'a>(v: &'a mut dyn Visitor) -> Visit<'a> {
            Visit::new(v)
        }
        fn visit_mut<'a>(v: &'a mut dyn VisitorMut) -> VisitMut<'a> {
            VisitMut::new(v)
        }
        const _: LoadMode = LoadMode::Strict;

        fn provided(t: &(Array, Vec<Param>), named: Vec<(String, Array)>) -> Result<()> {
            let _: Vec<Array> = t.leaves();
            let _: Vec<String> = t.paths();
            let _: Vec<(String, Array)> = t.flatten();
            let _: usize = t.len();
            let _: bool = t.is_empty();
            let _: TreeDef = t.def();
            t.for_each(|_: &str, _: &Array| {});
            let _: (Array, Vec<Param>) = t.map(|_: &str, a: &Array| a.clone());
            let _: (Array, Vec<Param>) = t.unflatten(t.leaves())?;
            let mut t = t.clone();
            t.load_flat(named, LoadMode::Partial)?;
            Ok(())
        }

        fn impls(a: Array, p: Param) {
            let map: BTreeMap<String, Array> = BTreeMap::new();
            let hash: std::collections::HashMap<String, Array> = Default::default();
            tree(&a);
            tree(&p);
            tree(&());
            tree(&Some(a.clone()));
            tree(&Box::new(a.clone()));
            tree(&vec![a.clone()]);
            tree(&[a.clone(), a.clone()]);
            tree(&(
                a.clone(),
                a.clone(),
                a.clone(),
                a.clone(),
                a.clone(),
                a.clone(),
                a.clone(),
                a.clone(),
            ));
            tree(&map);
            tree(&hash);
            leaves(&(&a, &p));
            leaves(&[a.clone()][..]);
            statik(&1.0f32);
            statik(&"name");
            statik(&(1u8, vec![2i64], Some(nuvec::DType::F16)));
            let _ = kinds(LeafKind::Buffer);
        }

        /// `&Param` works where `&Array` does.
        fn params(x: &Array, p: &Param) -> Array {
            let a = x * p;
            let b = p + 1.0;
            let c = x.matmul(p);
            a.add(p)
                .add(b)
                .add(c)
                .add(p.sum(..))
                .add(nuvec::ops::add(p, x))
        }
    }

    /// The derives, re-exported under the traits' names.
    mod derives {
        use super::*;

        #[derive(Clone, Tree)]
        struct Batch {
            x: Array,
            #[tree(skip)]
            len: usize,
        }

        #[derive(Clone, nuvec::Static)]
        struct Config {
            heads: i32,
        }

        #[derive(Clone, nuvec_nn::Module)]
        struct Model {
            w: Param,
            #[tree(skip)]
            config: Config,
        }

        fn used(b: &Batch, m: &Model) -> usize {
            b.len + m.config.heads as usize + b.x.ndim() + m.w.ndim()
        }
    }

    /// Evaluation of any tree; checkpoints of any tree.
    mod eval_io {
        use super::*;

        fn eval(loss: &Array, grads: &Vec<Array>) -> Result<()> {
            nuvec::eval(&(loss, grads))?;
            nuvec::async_eval(std::slice::from_ref(loss))?;
            nuvec::save("ckpt.safetensors", &(loss, grads))?;
            let mut fresh = (loss.clone(), grads.clone());
            nuvec::load_into("ckpt.safetensors", &mut fresh)
        }
        fn monomorphic(path: &std::path::Path, a: &mut Array) -> Result<()> {
            nuvec::save::<Array>(path, a)?;
            nuvec::load_into::<Array>(path, a)
        }
    }

    /// Tree-generic transforms and their flat forms.
    mod transforms {
        use super::*;

        fn tree_forms(x: &Array, p: &(Array, Array)) -> Result<()> {
            let _: Array = nuvec::grad(|x: &Array| x.multiply(x).sum(..))(x)?;
            let (_, _): (Array, (Array, Array)) =
                nuvec::value_and_grad(|p: &(Array, Array)| p.0.multiply(&p.1).sum(..))(p)?;
            let ((_, _), _): ((Array, Array), Array) =
                nuvec::value_and_grad(|x: &Array| (x.sum(..), x.exp()))(x)?;
            let (_, _): (Array, Array) = nuvec::vjp(|x: &Array| x.exp(), x, x)?;
            let (_, _): (Array, Array) = nuvec::jvp(|x: &Array| x.exp(), x, x)?;
            let v: Vmap<(Array, Array), Array, _> =
                nuvec::vmap(|(m, v): &(Array, Array)| m.matmul(v))
                    .in_axes((None, 0))
                    .out_axes(0);
            let _: Array = v.call(p)?;
            let f = v.into_fn();
            let _: Array = f(p)?;
            let c = nuvec::checkpoint(|x: &Array| x.sin());
            let _: Array = c(x)?;
            let cf: CustomFunction<Array, Array> = nuvec::custom_function(|x: &Array| x.sin())
                .with_vjp(|_: &Array, _: &Array, c: &Array| Ok(c.clone()))
                .with_jvp(|_: &Array, t: &Array| Ok(t.clone()))
                .with_vmap(|x: &Array, _: &[Option<i32>]| Ok((x.sin(), 0)));
            let _: Array = cf.call(x)?;
            Ok(())
        }

        fn flat_forms(xs: &[Array]) -> Result<()> {
            let _: Vec<Array> = nuvec::grad_flat(|xs: &[Array]| Ok(xs[0].sum(..)), &[0])(xs)?;
            let _: (Array, Vec<Array>) =
                nuvec::value_and_grad_flat(|xs: &[Array]| Ok(xs[0].sum(..)), &[0])(xs)?;
            let _ = nuvec::vjp_flat(|xs| Ok(xs.to_vec()), xs, xs)?;
            let _ = nuvec::jvp_flat(|xs| Ok(xs.to_vec()), xs, xs)?;
            let _ = nuvec::vmap_flat(|xs| Ok(xs.to_vec()), &[Some(0)], &[0])(xs)?;
            let _ = nuvec::checkpoint_flat(|xs| Ok(xs.to_vec()))(xs)?;
            let f: CustomFunctionFlat = nuvec::custom_function_flat(|xs| Ok(xs.to_vec()));
            let _ = f.call(xs)?;
            Ok(())
        }

        fn traits<R: IntoResult, G: GradOutput, A: InAxes, B: InAxis, O: OutAxes>() {}
        fn impls() {
            traits::<Result<Array>, (Array, Vec<Array>), (Option<i32>, i32), i32, (i32, i32)>();
            traits::<Array, Array, Option<i32>, Option<i32>, i32>();
        }
    }

    /// `compile` over trees, `compile_mut`, `compile_flat` and the options.
    mod compile {
        use super::*;

        fn forms(x: &Array, w: &Array) -> Result<()> {
            let f = nuvec::compile(|x: &Array| x.matmul(w));
            let _: Array = f(x)?;
            let step = nuvec::compile_mut(|s: &mut Array, x: &Array| {
                *s = s.add(x);
                s.sum(..)
            });
            let mut s = x.clone();
            let _: Array = step(&mut s, x)?;
            let opts = nuvec::compile::Options::default().cache_capacity(8);
            let _: usize = opts.cache_capacity;
            let g = opts.clone().compile(|x: &Array| x.exp());
            let _: Array = g(x)?;
            let h = opts.compile_mut(|s: &mut Array, x: &Array| s.add(x));
            let _: Array = h(&mut s, x)?;
            let flat = nuvec::compile_flat(|xs: &[Array]| Ok(xs.to_vec()));
            let _: Vec<Array> = flat(std::slice::from_ref(x))?;
            let shapeless = nuvec::compile::Options::default().shapeless(true);
            let _: bool = shapeless.shapeless;
            let flat = nuvec::compile::compile_flat_with(shapeless, |xs: &[Array]| Ok(xs.to_vec()));
            let _: Vec<Array> = flat(std::slice::from_ref(x))?;
            Ok(())
        }
    }

    /// Speculation: on-device acceptance, capture-all Gated DeltaNet with
    /// storage rounding, and nuvec-lm's drafters, policy and MTP head.
    mod speculation {
        use nuvec::fast::{self, StepOptions};
        use nuvec::{Array, DType, Result, speculative};
        use nuvec_lm::{
            Auto, Draft, DraftPolicy, Drafter, Model, MtpDrafter, PromptLookup, SpeculativeStats,
            SuffixDecoding,
        };

        #[test]
        fn pinned() -> Result<()> {
            const _: fn(&Array, &Array, &Array) -> Array = speculative::accept_greedy;
            const _: fn(&Array, &Array, Option<&Array>, &Array, &Array) -> Array =
                speculative::accept_sampled;
            const _: fn(&Array) -> Result<(usize, u32, bool)> = speculative::read;
            let _: StepOptions = StepOptions::default().storage(DType::BF16);
            const _: fn(
                &Array,
                &Array,
                &Array,
                &Array,
                &Array,
                &Array,
                &StepOptions,
            ) -> (Array, Array) = fast::gated_delta_steps_with;
            let _: Draft = Draft::from_tokens(&[1, 2]);
            let _: Draft = Draft::empty();
            let _: DraftPolicy = DraftPolicy::default();
            let _ = [
                DraftPolicy::Auto,
                DraftPolicy::Fixed(2),
                DraftPolicy::Adaptive(4),
            ];
            let _: Auto = Auto {
                len: 1,
                break_even: 0.5,
            };
            let _: PromptLookup = PromptLookup::default();
            let mut s = SuffixDecoding::default();
            s.remember(&[1, 2, 3]);
            let _: Vec<u32> = s.suggest(&[1, 2], 1);
            const _: fn(&Model) -> Result<MtpDrafter> = MtpDrafter::new;
            const _: fn(&Model) -> usize = nuvec_lm::verify_rows;
            const _: fn(&Model) -> bool = Model::has_mtp;
            const _: fn(&Model, &Array, &Array, &mut [nuvec_lm::KvCache]) -> Array =
                Model::mtp_forward;
            const _: fn(&Model, &Array) -> Array = Model::logits;
            let _: fn(&mut Model, &str) -> Result<()> = |m, p| m.attach_mtp(p);
            let s = SpeculativeStats::default();
            let _: [usize; 5] = [s.steps, s.verifies, s.drafted, s.accepted, s.pauses];
            fn takes_drafter<D: Drafter>(_: D) {}
            takes_drafter(PromptLookup::default());
            takes_drafter(&mut SuffixDecoding::default());
            Ok(())
        }
    }

    /// The Neural Accelerator status.
    #[test]
    fn nax_status_is_pinned() {
        use nuvec::metal::{NaxReason, NaxStatus};
        let _: fn() -> NaxStatus = nuvec::metal::nax;
        let s = NaxStatus::Off(NaxReason::Unmeasured);
        assert!(!s.is_active());
        assert!(NaxStatus::Active { forced: true }.is_active());
        let _ = [
            NaxReason::Override,
            NaxReason::Os,
            NaxReason::Generation,
            NaxReason::NoLibrary,
            NaxReason::PipelineError,
            NaxReason::NoDevice,
        ];
    }

    /// `distributed`: the group, the collectives and the backends' entry points.
    mod distributed {
        use nuvec::distributed::{self, Backend, Group, Options};
        use nuvec::{Array, DType, Result};

        #[test]
        fn pinned() -> Result<()> {
            let g: Group = Group::single();
            let _: (usize, usize) = (g.rank(), g.size());
            let _: Result<Group> = g.split(0, 0);
            let _: Result<()> = g.health();
            g.close();
            let _: Group = distributed::default_group();
            let x = Array::from(1.0f32);
            let _: Array = distributed::all_sum(&x, None);
            let _: Array = distributed::all_max(&x, Some(&g));
            let _: Array = distributed::all_min(&x, None);
            let _: Array = distributed::all_gather(&x, None);
            let _: Array = distributed::sum_gradients(&x, None);
            let _: Array = distributed::sum_scatter(&x, None);
            const _: fn(&Array, usize, Option<&Group>) -> Array = distributed::send;
            const _: fn(&Array, usize, Option<&Group>) -> Array = distributed::recv_like;
            let _: Array = distributed::recv([2], DType::F32, 1, None);
            const _: fn(bool) -> Result<Group> = distributed::init;
            const _: fn(&[Vec<String>], usize, &Options) -> Result<Group> = distributed::init_with;
            let _: Options = Options::default()
                .timeout(std::time::Duration::from_secs(1))
                .key("a secret of sixteen bytes");
            const _: fn(bool, Backend) -> Result<Group> = distributed::init_backend;
            const _: fn(Backend) -> bool = distributed::is_available;
            let _: Backend = "ring".parse::<Backend>()?;
            let _ = [Backend::Any, Backend::Ring, Backend::Jaccl];
            Ok(())
        }

        /// The experimental JACCL backend (feature `jaccl`).
        #[cfg(feature = "jaccl")]
        #[test]
        fn jaccl_pinned() -> Result<()> {
            use nuvec::distributed::jaccl::{
                self, Config, Options, PortState, RdmaDevice, Topology, parse_devices,
            };
            let devices = parse_devices(r#"[[null, "rdma_en2"], ["rdma_en2", null]]"#)?;
            let c: Config = Config::new(0, devices)?
                .coordinator("127.0.0.1:5000")
                .prefer_ring(false);
            let _: (usize, usize) = (c.rank(), c.size());
            let _: Topology = c.topology()?;
            const _: fn() -> Result<Option<Config>> = Config::from_env;
            let _: Options = Options::default()
                .timeout(std::time::Duration::from_secs(1))
                .side_channel(|d: &[u8]| Ok(d.to_vec()))
                .key("a secret of sixteen bytes");
            const _: fn(&Config, &Options) -> Result<Group> = jaccl::init_with;
            const _: fn() -> bool = jaccl::is_available;
            const _: fn() -> Result<Vec<RdmaDevice>> = jaccl::devices;
            let _ = |d: RdmaDevice| -> (String, PortState, Option<std::net::Ipv4Addr>) {
                (d.name, d.port_state, d.ipv4)
            };
            Ok(())
        }
    }

    /// nuvec-nn: `Module`, introspection, weights, optimizers (reached with the facade).
    mod nn {
        use nuvec_nn::optim::{
            Adam, AdamSlot, AdamW, LearningRate, Lion, MomentumSlot, MultiOptimizer, Muon,
            Optimizer, OptimizerState, Sgd, SgdSlot, clip_grad_norm, schedules,
        };
        use nuvec_nn::{
            Dropout, LayerNorm, Linear, LoraBase, LoraLinear, Module, ModuleDisplay,
            QuantizedEmbedding, QuantizedLinear, RmsNorm, Rope, Summary, SummaryRow, WeightSource,
        };

        use super::*;

        fn module<M: Module>(m: &mut M) -> Result<()> {
            let _: Vec<(String, Array)> = m.parameters();
            let _: Vec<(String, Array)> = m.trainable_parameters();
            let _: usize = m.num_params() + m.num_trainable();
            let s: Summary = m.summary();
            let _: &Vec<SummaryRow> = &s.rows;
            let d: ModuleDisplay<'_, M> = m.display();
            let _ = d.to_string();
            let _: usize = m.freeze(|p: &str| p.contains("weight"))?;
            let _: usize = m.unfreeze(|_: &str| true)?;
            m.train(false);
            let _: bool = m.training();
            m.load_weights(Vec::<(String, Array)>::new())?;
            m.load_weights_partial("weights.safetensors")?;
            m.save_weights("weights.safetensors")?;
            m.save_trainable("adapter.safetensors")?;
            let _: usize = m.map_modules(|_: &str| true, |_: &str, _: &mut Linear| Ok(()))?;
            Ok(())
        }

        fn sources<S: WeightSource>() {}
        fn layers() {
            sources::<&str>();
            sources::<std::path::PathBuf>();
            sources::<Vec<(String, Array)>>();
            sources::<BTreeMap<String, Array>>();
            let l: Linear = Linear::new(4, 2).without_bias();
            let q: QuantizedLinear = QuantizedLinear::from_linear(&l);
            let _: QuantizedLinear =
                QuantizedLinear::from_linear_with(&l, nuvec::Quant::affine(32, 4));
            let _: QuantizedLinear = QuantizedLinear::from_linear_with(&l, nuvec::QuantMode::Mxfp4);
            let e = nuvec_nn::Embedding::new(10, 64);
            let _: QuantizedEmbedding = QuantizedEmbedding::from_embedding(&e);
            let _: RmsNorm = RmsNorm::new(4).eps(1e-6);
            let _: LayerNorm = LayerNorm::new(4).eps(1e-6).affine(false).without_bias();
            let _: Rope = Rope::new(8).traditional().base(1e4).scale(1.0);
            let _: Dropout = Dropout::new(0.1);
            let lora: LoraLinear = LoraLinear::from_base(q, 4)
                .scale(10.0)
                .dropout(0.1)
                .rslora();
            let _: &LoraBase = &lora.base;
            let _: Linear = lora.fuse();
        }

        fn optimizers(m: &mut (Linear, Param), g: &(Linear, Param)) -> Result<()> {
            let mut opt: Box<dyn Optimizer> = Box::new(AdamW::new(1e-3).betas(0.9, 0.99).eps(1e-6));
            opt.init(m);
            opt.update(m, g)?;
            let _: Array = opt.learning_rate();
            opt.set_learning_rate(1e-4);
            let _ = opt.clone().paths();
            let mut multi = MultiOptimizer::new(Sgd::new(1e-2).momentum(0.9).nesterov()).group(
                |p: &str| p.contains("lora_a"),
                Adam::new(1e-3).bias_correction(),
            );
            multi.update(m, g)?;
            let _ = (
                Lion::new(1e-3),
                Muon::new(1e-3).without_nesterov().ns_steps(3),
            );
            let _: (Vec<Array>, Array) = clip_grad_norm(&vec![m.1.value().clone()], 1.0);
            let _: LearningRate = schedules::join_schedules(
                vec![
                    schedules::linear_schedule(0.0, 0.1, 5),
                    schedules::cosine_decay(0.1, 10, 0.0),
                ],
                vec![5],
            )?;
            let _: Array = LearningRate::from(0.1f32).at(&Array::from(0u64));
            Ok(())
        }

        fn states(s: &OptimizerState<AdamSlot>, a: &SgdSlot, b: &MomentumSlot) -> usize {
            s.slots.len() + s.step.ndim() + s.learning_rate.ndim() + a.v.ndim() + b.m.ndim()
        }
    }

    /// nuvec-lm: LoRA by path and mlx-lm adapter directories.
    mod lm {
        use nuvec_lm::lora::{
            AdapterConfig, LoraParams, linear_to_lora_layers, load_adapters, save_adapters,
        };
        use nuvec_lm::model::Projection;
        use nuvec_lm::{Batcher, Model, Split, Stage, load_model_pipelined};

        use super::*;

        const _: fn(&mut Model, usize, Option<&[&str]>, LoraParams) -> Result<usize> =
            linear_to_lora_layers;

        /// Pipeline parallelism.
        fn pipeline(m: &mut Model, g: &nuvec::distributed::Group) -> Result<()> {
            m.pipeline(g, Split::Even)?;
            let _ = load_model_pipelined("dir", g, Split::memory());
            let _ = [
                Split::Layers(vec![1]),
                Split::Memory { reserve: 0.1 },
                Split::Budgets(vec![1], 0.1),
            ];
            if let Some(s) = m.stage() {
                let s: &Stage = s;
                let _: (std::ops::Range<usize>, bool, bool) =
                    (s.layers(), s.is_first(), s.is_last());
                let _: &nuvec::distributed::Group = s.group();
            }
            let _: Array = m.share_tokens(&Array::from(1u32));
            // Tensor parallelism.
            m.shard(g)?;
            let _: Option<&nuvec::distributed::Group> = m.sharding();
            let _ = nuvec_lm::load_model_sharded("dir", g);
            let options = nuvec_lm::BatcherOptions {
                micro_batches: Some(2),
                ..Default::default()
            };
            let _: Result<Batcher<'_>> = Batcher::new(m, options);
            Ok(())
        }

        /// Generation controls and batcher options.
        fn controls(m: &Model, t: &nuvec_lm::Tokenizer) -> Result<()> {
            use nuvec_lm::{
                BatcherOptions, FinishReason, GenerateOptions, Stop, Stopped, TextStream,
            };
            let stop: Stop = Stop::strings(t, ["\n\n"])?.with_tokens([7]);
            let _: (&[u32], bool) = (stop.stop_tokens(), stop.is_empty());
            let _ = Stop::tokens([1, 2]);
            let options = GenerateOptions {
                stop,
                ignore_eos: true,
                min_tokens: 4,
                ..Default::default()
            };
            options.validate()?;
            let mut stream = TextStream::new(t, &options);
            let _: String = stream.push(1) + &stream.finish();
            let mut g = nuvec_lm::generate(m, &[1], options.clone())?;
            let _ = g.next();
            let reason: Option<&FinishReason> = g.finish_reason();
            let _ = match reason {
                Some(FinishReason::Stop(Stopped::String { index, cut })) => index + cut,
                Some(FinishReason::Stop(Stopped::Token(t))) => *t as usize,
                Some(FinishReason::Error(e)) => usize::from(e.is_oom()),
                Some(
                    FinishReason::EndOfSequence
                    | FinishReason::Length
                    | FinishReason::CacheFull
                    | FinishReason::Cancelled,
                ) => 0,
                Some(_) | None => 0,
            };
            let b = BatcherOptions {
                blocks: Some(64),
                block_size: 16,
                max_batch: 8,
                micro_batches: None,
                batch_invariant: true,
                memory_fraction: 0.9,
            };
            b.validate()?;
            let _: Batcher<'_> = Batcher::new(m, b)?;
            let _: usize = m.kv_bytes_per_position();
            let _ = (
                nuvec_lm::stop::MAX_STOP_STRINGS,
                nuvec_lm::stop::MAX_STOP_BYTES,
            );
            Ok(())
        }
        fn adapters(dir: &str, m: &mut Model, c: &AdapterConfig) -> Result<AdapterConfig> {
            save_adapters(dir, m, c)?;
            load_adapters(dir, m)
        }

        fn config() -> AdapterConfig {
            let p = LoraParams {
                rank: 8,
                scale: 20.0,
                dropout: 0.0,
            };
            let c = AdapterConfig::new(2, p).keys(["self_attn.q_proj"]);
            let _: (usize, &Option<Vec<String>>, LoraParams) = (c.num_layers, &c.keys, c.params);
            c
        }

        /// `Projection` is non-exhaustive: matches need a wildcard.
        fn projection(p: &Projection) -> u8 {
            match p {
                Projection::Dense(_) => 0,
                Projection::Quantized(_) => 1,
                Projection::Lora(_) => 2,
                _ => 3,
            }
        }
    }
}

/// Floating-point quantization modes (MXFP4, MXFP8, NVFP4), bit views and E4M3
/// conversions, and the checkpoint side in nuvec-nn and nuvec-lm.
mod spec_055 {
    use std::str::FromStr;

    use nuvec::{Array, DType, GatherQmm, Quant, QuantMode, Quantized, ops};

    const MODES: [QuantMode; 4] = [
        QuantMode::Affine,
        QuantMode::Mxfp4,
        QuantMode::Mxfp8,
        QuantMode::Nvfp4,
    ];
    const NAME: fn(QuantMode) -> &'static str = QuantMode::name;
    const QUANT: [Quant; 3] = [
        Quant::new(QuantMode::Nvfp4),
        Quant::affine(64, 4).group_size(32).bits(8),
        Quant {
            mode: QuantMode::Mxfp8,
            group_size: 32,
            bits: 8,
        },
    ];

    fn quant() {
        let _: Quant = Quant::default();
        let _: Quant = QuantMode::Mxfp4.into();
        let _: Result<QuantMode, _> = QuantMode::from_str("mxfp4");
        let _: String = QuantMode::Nvfp4.to_string();
    }

    fn quantized(w: Array, s: Array) -> nuvec::Result<()> {
        let q = Quantized::new(w, s, None, QuantMode::Mxfp4);
        let _: (&Array, &Array, &Option<Array>, Quant) = (&q.weight, &q.scales, &q.biases, q.quant);
        let _: Array = q.dequantize(DType::BF16);
        const _: fn(Array, Array, Array, i32, i32) -> Quantized = Quantized::affine;
        const _: fn(Array, Array, QuantMode) -> Quantized = Quantized::fp;
        q.check()
    }

    const OPS: [fn(&Array) -> Array; 3] = [
        |x| x.view(DType::U32),
        |x| x.to_fp8(),
        |x| x.from_fp8(DType::F16),
    ];
    const _: fn(&Array, DType) -> Array = ops::view;
    const _: fn(&Array) -> Array = ops::to_fp8;
    const _: fn(&Array, DType) -> Array = ops::from_fp8;
    const _: fn(&Array, &Quantized, Option<&Array>, Option<&Array>, GatherQmm<'_>) -> Array =
        ops::gather_qmm;

    fn gather(x: &Array, q: &Quantized, rhs: &Array, scale: &Array) -> Array {
        let o = GatherQmm::default()
            .transpose(true)
            .sorted_indices(true)
            .global_scale(scale);
        let _: (bool, bool, Option<&Array>) = (o.transpose, o.sorted_indices, o.global_scale);
        x.gather_qmm(q, None, Some(rhs), o)
    }

    #[test]
    fn op_names() {
        for name in [
            "view",
            "to_fp8",
            "from_fp8",
            "quantize",
            "quantized_matmul",
            "gather_qmm",
        ] {
            assert!(ops::__op_names().contains(&name), "{name}");
        }
    }

    fn nn(l: &nuvec_nn::Linear, e: &nuvec_nn::Embedding) {
        use nuvec_nn::{QuantizedEmbedding, QuantizedLinear};
        let q = QuantizedLinear::from_linear_with(l, Quant::new(QuantMode::Nvfp4));
        let _: (&Option<Array>, Quant) = (&q.biases, q.quant);
        let _: Quantized = q.quantized();
        let _: Array = q.dequantized_weight(DType::BF16);
        let qe = QuantizedEmbedding::from_embedding_with(e, QuantMode::Mxfp8);
        let _: (&Option<Array>, Quant, Quantized) = (&qe.biases, qe.quant, qe.quantized());
    }

    fn lm(q: &nuvec_lm::Quantization) {
        let _: Quant = q.default;
        let _: &Vec<nuvec_lm::LayerQuant> = &q.layers;
        let l: Option<&nuvec_lm::LayerQuant> = q.for_layer("model.layers.0.mlp.gate");
        let _ = l.map(|l| (&l.path, l.quant, l.global_scale));
    }
    const _: fn(&nuvec_lm::Config) -> Option<&nuvec_lm::Quantization> = |c| c.quantization.as_ref();
    const _: fn(&Array) -> Array = nuvec_lm::quant::mxfp4_from_hf_blocks;
    type Weights = std::collections::BTreeMap<String, Array>;
    const _: fn(Weights, nuvec_lm::PackedFormat, Quant) -> nuvec::Result<Weights> =
        nuvec_lm::quant::convert_packed;
    fn packed(q: &nuvec_lm::Quantization) -> u8 {
        match q.packed {
            Some(nuvec_lm::PackedFormat::Awq) => 1,
            Some(nuvec_lm::PackedFormat::Gptq { v1 }) => 2 + u8::from(v1),
            Some(_) => 4,
            None => 0,
        }
    }
}

/// Gpt-oss and Gemma 4: windows and sinks in the attention options (paged too),
/// the blocked head-512 plan, `norm_rope_kv`, `power`, RoPE scalings, the activations and
/// the model settings.
mod spec_056 {
    use nuvec::fast::{self, AttentionKernel, AttentionOptions, Rotation};
    use nuvec::{Array, ops};
    use nuvec_nn::{Llama3, LongRope, Rope, RopeScaling, Yarn};

    fn attention(q: &Array, k: &Array, sinks: &Array, tables: &Array, lengths: &Array) {
        let o = AttentionOptions::causal().window(128).sinks(Some(sinks));
        let _: Option<i32> = o.window;
        let _: Array = fast::scaled_dot_product_attention(q, k, k, 0.125, &o);
        let _: Array = fast::paged_attention(q, k, k, tables, lengths, 0.125, &o);
    }

    fn kernel(k: AttentionKernel) -> i32 {
        match k {
            AttentionKernel::BlockedD512 { rows } => rows,
            _ => 0,
        }
    }

    fn norm_rope_kv(x: &Array, w: &Array, r: Rotation<'_>) -> (Array, Array) {
        fast::norm_rope_kv(x, Some((w, 1e-6)), 1e-6, r, 0, None)
    }

    const _: fn(&Array, &Array) -> Array = |a, b| a.power(b);
    const _: fn(&Array, &Array) -> Array = |a, b| ops::power(a, b);
    const _: fn(&Array, &Array, f32, f32) -> Array = nuvec_nn::swiglu_oai;
    const _: fn(&Array, &Array) -> Array = nuvec_nn::geglu;
    const _: fn(&Array) -> Array = nuvec_nn::gelu_approx;
    const _: fn(&Array, f32) -> Array = nuvec_nn::logit_softcap;

    fn rope() {
        let scalings = [
            RopeScaling::None,
            RopeScaling::Linear { factor: 2.0 },
            RopeScaling::Llama3(Llama3::default()),
            RopeScaling::Yarn(Yarn::default()),
            RopeScaling::LongRope(LongRope::default()),
            RopeScaling::Proportional {
                rotated_dims: 128,
                factor: 1.0,
            },
        ];
        for s in &scalings {
            let _: Option<Array> = s.frequencies(512, 1e6);
            let _: f32 = s.mscale();
            let r = Rope::scaled(512, 1e6, s);
            let _: (f32, Rotation<'_>) = (r.mscale(), r.rotation());
        }
    }

    fn lm(c: &nuvec_lm::Config) {
        let _: &Vec<Option<usize>> = &c.layer_windows;
        let _: bool = c.attention_sinks;
        if let Some(g) = &c.gemma4 {
            let _: &nuvec_lm::Gemma4 = g;
            let _ = (
                g.global_head_dim,
                g.global_kv_heads,
                g.k_eq_v,
                g.kv_shared_layers,
            );
            let _ = (g.double_wide_mlp, g.per_layer_input, g.per_layer_vocab);
            let _: (Option<f32>, &(f32, RopeScaling), &(f32, RopeScaling)) =
                (g.final_logit_softcap, &g.sliding_rope, &g.full_rope);
            let _: usize = g.first_shared(c.num_hidden_layers);
            let _ = g.moe.as_ref().map(|m| m.experts_per_token);
        }
        let _ = nuvec_lm::Activation::GeGlu;
        let _ = nuvec_lm::Activation::SwiGluOai {
            alpha: 1.702,
            limit: 7.0,
        };
    }

    #[test]
    fn op_names() {
        assert!(ops::__op_names().contains(&"power"));
    }
}

/// The engine surface: every `nuvec::` item a runtime built on nuvec (such as an inference
/// server's tensor adapter) uses, and the engine items (attention plan, memory statistics, checkpoint footprint, snapshots, the
/// row-independence constant). Changing any of these is a semver event for runtimes.
mod engine_surface {
    use nuvec::fast::{
        AttentionKernel, AttentionMask, AttentionOptions, AttentionPlan, KernelConfig, MetalKernel,
        TemplateArg,
    };
    use nuvec::io::{Footprint, SafeTensors, SaveHandle};
    use nuvec::{
        Array, DType, Device, Element, Error, Result, Stream, async_eval, default_device, eval,
        fast, io, ops, promote_types, random, set_default_stream,
    };

    type M = fn(&Array, &Array) -> Array;

    // Arrays, conversion and creation.
    const _: fn(&[f32], [i32; 2]) -> Array = Array::from_slice;
    const _: fn(f32) -> Array = Array::from;
    const _: fn([i32; 2], DType) -> Array = ops::zeros;
    const _: fn(&Array, DType) -> Array = Array::astype;
    const _: fn(DType, DType) -> DType = promote_types;
    const _: fn(&Array) -> Array = Array::contiguous;
    const _: fn(&Array) -> Array = Array::copy;
    const _: fn(&Array) -> Array = ops::copy;
    const _: fn(&Array) -> Result<Vec<f32>> = Array::to_vec::<f32>;
    fn element<T: Element>() {}
    // Operations, in the forms the adapter calls.
    const OPS: [fn(&Array) -> Array; 12] = [
        |x| x.reshape([2, -1]),
        |x| x.transpose([1, 0]),
        |x| x.squeeze(0),
        |x| x.sum(-1),
        |x| x.max(-1),
        |x| x.logsumexp(-1),
        |x| x.softmax(-1),
        |x| x.argmax(-1),
        |x| x.argsort(-1),
        |x| x.sort(-1),
        |x| x.argpartition(3, -1),
        |x| x.cumsum(-1),
    ];
    const BINARY: [M; 6] = [
        |a, b| a.subtract(b),
        |a, b| a.divide(b),
        |a, b| a.greater(b),
        |a, b| a.take(b, 0),
        |a, b| a.take_along_axis(b, -1),
        |a, b| a.put_along_axis(b, 0.0, -1),
    ];
    const SELECT: fn(&Array, &Array, &Array) -> Array = |m, a, b| m.select(a, b);
    const WHERE: fn(&Array, &Array, &Array) -> Array = |m, a, b| ops::where_(m, a, b);
    const SLICE: fn(&Array) -> Array = |x| x.i((.., 1..3));
    const UPDATE: fn(Array, &Array) -> Array = |x, u| x.at((.., 1..2)).set(u);
    const REPEAT: fn(&Array) -> Array = |x| x.repeat(2, 0);
    const SPLIT: [fn(&Array) -> Vec<Array>; 2] = [|x| x.split(2, -1), |x| x.split_at(&[1], 0)];
    const JOIN: [fn(&[Array]) -> Array; 2] = [|xs| ops::concatenate(xs, 0), |xs| ops::stack(xs, 0)];
    const CONV: M = |x, w| x.conv1d(w, 1, 0, 1, 4);
    // Quantized and gathered (mixture-of-experts) products.
    const QUANT: fn(&Array) -> nuvec::Quantized = |w| w.quantize(nuvec::Quant::affine(64, 4));
    const DEQUANT: fn(&nuvec::Quantized) -> Array = |q| q.dequantize(None);
    const QMM: fn(&Array, &nuvec::Quantized) -> Array = |x, q| x.quantized_matmul(q, true);
    const GQMM: fn(&Array, &nuvec::Quantized, &Array) -> Array = |x, q, i| {
        ops::gather_qmm(
            x,
            q,
            None,
            Some(i),
            nuvec::GatherQmm::default().sorted_indices(true),
        )
    };
    const GMM: fn(&Array, &Array, &Array) -> Array =
        |x, w, i| ops::gather_mm(x, w, None, Some(i), true);
    const ROWS: usize = nuvec::ROW_INDEPENDENT_ROWS;
    // Row-exact products and the grouped-gather rule.
    const ROW_EXACT: fn(fn() -> Array) -> Array = |f| nuvec::with_row_exact(f);
    const IS_ROW_EXACT: fn() -> bool = nuvec::row_exact;
    const GROUPS: fn(usize, usize) -> bool = nuvec::GatherQmm::groups;
    // Fused kernels and attention.
    const _: fn(&Array, &Array, &Array, f32, &AttentionOptions<'_>) -> Array =
        fast::scaled_dot_product_attention;
    const _: fn(&[i32], &[i32], &[i32], DType, &AttentionOptions<'_>) -> Result<AttentionPlan> =
        fast::plan_attention;
    const NORMS: [fn(&Array, &Array) -> Array; 1] = [|x, w| fast::rms_norm(x, w, 1e-6)];
    const ADD_NORM: fn(&Array, &Array, &Array) -> (Array, Array) =
        |x, r, w| fast::add_rms_norm(x, r, w, 1e-6);
    const ROPE: fn(&Array, &Array) -> Array =
        |x, o| fast::rope(x, 64, false, Some(1e4), 1.0, o, None);

    fn options(m: &Array, s: &Array) -> [AttentionOptions<'static>; 0] {
        let _ = AttentionOptions::causal()
            .mask(m)
            .sinks(s)
            .window(128)
            .max_score_bytes(512 << 20);
        let _: AttentionOptions<'_> = AttentionMask::None.into();
        let o = AttentionOptions::default();
        let _: (
            AttentionMask<'_>,
            Option<&Array>,
            Option<i32>,
            Option<usize>,
        ) = (o.mask, o.sinks, o.window, o.max_score_bytes);
        []
    }

    fn plan_fields(p: AttentionPlan) -> (AttentionKernel, usize, usize, bool) {
        (
            p.kernel,
            p.scratch_bytes,
            p.max_intermediate_bytes,
            p.capped,
        )
    }

    fn kernels(k: AttentionKernel) -> i32 {
        match k {
            AttentionKernel::Vector {
                passes,
                blocks,
                dispatches,
            } => i32::from(passes) + blocks + dispatches,
            AttentionKernel::Tiled => 1,
            AttentionKernel::Composed => 2,
            AttentionKernel::ComposedBlocks { rows } => rows,
            _ => 0,
        }
    }

    // Custom kernels.
    const _: fn(&str, &[&str], &[&str], &str) -> MetalKernel = MetalKernel::new;
    const _: fn(&MetalKernel, &[Array], &[(&[i32], DType)], &KernelConfig) -> Vec<Array> =
        MetalKernel::call;
    fn template_args() -> [TemplateArg; 2] {
        [TemplateArg::Int(4), TemplateArg::DType(DType::F16)]
    }

    // Compilation, evaluation, streams and random numbers.
    fn compiled() -> impl Fn(&Array) -> Result<Array> {
        nuvec::compile(|x: &Array| x.exp())
    }
    const _: fn(&[Array]) -> Result<()> = |xs| eval(xs);
    const _: fn(&[Array]) -> Result<()> = |xs| async_eval(xs);
    const _: fn(Stream) = set_default_stream;
    const _: fn() -> Device = default_device;
    const _: fn(Device) -> Stream = Stream::default_for;
    const _: fn() -> Device = Device::cpu;
    const _: fn(u64) -> Array = random::key;
    const _: fn(&Array, &Array) -> Array = |l, k| random::categorical(l, -1, k);

    // Memory controls and statistics, device facts, captures, the kernel profile.
    #[allow(dead_code)]
    fn memory_and_metal() -> Result<()> {
        use nuvec::memory::{self, MemoryStats};
        use nuvec::metal::{self, DeviceInfo};
        let m: MemoryStats = memory::stats();
        let _: [usize; 12] = [
            m.active,
            m.peak,
            m.active_with_mapped,
            m.peak_with_mapped,
            m.cache,
            m.memory_limit,
            m.cache_limit,
            m.wired_limit,
            m.wired,
            m.mapped,
            m.wired_mapped,
            m.system_allocations as usize,
        ];
        let _: [usize; 3] = [
            memory::active_memory(),
            memory::peak_memory(),
            memory::cache_memory(),
        ];
        let _: usize = memory::set_memory_limit(1);
        let _: usize = memory::set_cache_limit(1);
        let _: usize = memory::set_wired_limit(0)?;
        let _: bool = memory::can_wire();
        let _: u64 = memory::physical_memory();
        memory::clear_cache();
        memory::reset_peak_memory();
        let _: bool = metal::is_available();
        let info: Option<&'static DeviceInfo> = metal::device_info();
        if let Some(info) = info {
            let _: (&str, &str, Option<u32>, Option<char>, u64, u64, usize) = (
                &info.name,
                &info.architecture,
                info.generation,
                info.class,
                info.recommended_working_set,
                info.physical_memory,
                info.max_threads_per_threadgroup,
            );
        }
        metal::start_capture("run.gputrace")?;
        metal::stop_capture();
        let _: bool = metal::profile::enabled();
        let _: Vec<(String, u64, f64)> = metal::profile::report();
        metal::profile::reset();
        Ok(())
    }

    // Checkpoints: loading, saving, footprints and asynchronous saves.
    const _: fn(&str) -> Result<SafeTensors> = |p| io::load_safetensors(p);
    const _: fn(&str, fn(&str) -> bool) -> Result<SafeTensors> =
        |p, keep| io::load_safetensors_filtered(p, keep);
    const _: fn(&str, &str) -> Result<()> = |s, d| io::align_safetensors(s, d);
    const _: fn(&str, &str, fn(&str) -> bool) -> Result<SafeTensors> =
        |p, c, keep| io::load_safetensors_cached(p, c, keep);
    const _: fn(&str, &[(&str, &Array)], &[(&str, &str)]) -> Result<()> =
        |p, t, m| io::save_safetensors(p, t, m);
    const _: fn(&str) -> Result<Footprint> = |p| io::safetensors_footprint(p);
    const _: fn(&str, fn(&str) -> bool) -> Result<Footprint> =
        |p, keep| io::safetensors_footprint_filtered(p, keep);
    const _: fn(&str) -> Result<Vec<io::TensorInfo>> = |p| io::safetensors_tensors(p);
    const _: fn(io::TensorInfo) -> (String, DType, Vec<usize>, usize) =
        |t| (t.name, t.dtype, t.shape, t.bytes);
    // MLX's `depends`.
    const _: fn(&Array, &[Array]) -> Array = |x, d| nuvec::ops::depends(x, d);
    // Storage identity (aliasing checks) and floor division (Python's `//`).
    const _: fn(&Array) -> Option<usize> = Array::buffer_id;
    const _: fn(&Array, &Array) -> Array = |a, b| a.floor_divide(b);
    const _: fn(&Array, &Array) -> Array = |a, b| nuvec::ops::floor_divide(a, b);
    const _: fn(&Array, &Array) -> Array = |a, b| a.power(b);
    const _: fn(&Array, &Array) -> Array = |a, b| nuvec::ops::power(a, b);
    const _: fn(&str, &[(&str, &Array)], &[(&str, &str)]) -> SaveHandle =
        |p, t, m| io::save_safetensors_async(p, t, m);
    const _: fn(SaveHandle) -> Result<()> = SaveHandle::wait;
    fn footprint(f: Footprint) -> (u64, u64, u64) {
        (f.mapped, f.copied, f.transient)
    }

    // Errors the adapter maps.
    fn error(e: Error) -> bool {
        matches!(e, Error::InvalidArgument(_) | Error::OutOfMemory(_))
    }

    #[test]
    fn the_engine_items_work_together() {
        let x = nuvec::array![[1.0, 2.0], [3.0, 4.0]];
        let plan = fast::plan_attention(
            &[1, 2, 4, 64],
            &[1, 1, 9, 64],
            &[1, 1, 9, 64],
            DType::F32,
            &AttentionOptions::causal(),
        )
        .unwrap();
        assert!(plan_fields(plan).1 < usize::MAX);
        assert_eq!(x.copy().to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
        const { assert!(ROWS >= 8) };
        let _ = (
            OPS, BINARY, SELECT, WHERE, SLICE, UPDATE, REPEAT, SPLIT, JOIN, CONV,
        );
        let _ = (
            QUANT, DEQUANT, QMM, GQMM, GMM, NORMS, ADD_NORM, ROPE, kernels, footprint,
        );
        let _ = (options, template_args, compiled, error, element::<f32>);
    }
}

/// Paged serving: `norm_rope_at` with and without a pool, and `paged_write` of
/// several sequences' rows.
mod spec_076 {
    use nuvec::Array;
    use nuvec::fast::{self, Rotation};

    fn norm_rope_at(x: &Array, w: &Array, r: Rotation<'_>, at: &Array, pool: &Array) -> Array {
        let q = fast::norm_rope_at(x, Some((w, 1e-6)), r, at, None);
        let slots = at.reshape([-1, 1]);
        let _: Array = fast::norm_rope_at(x, None, r, at, (pool, &slots));
        q
    }

    const _: fn(&Array, &Array, &Array) -> Array = fast::paged_write;

    #[test]
    fn the_items_exist() {
        let _ = norm_rope_at;
    }
}

mod spec_077 {
    use nuvec::{Array, fast};

    const _: fn(&Array, i32, bool) -> (Array, Array) = fast::moe_route;
    const _: fn(&Array, &Array) -> Array = fast::weighted_sum;

    #[test]
    fn the_items_exist() {}
}

mod spec_078 {
    use nuvec::{Array, fast};

    const _: fn(&Array, &Array, &Array, &Array, bool) -> (Array, Array) = fast::causal_conv1d_slots;

    #[test]
    fn the_items_exist() {}
}

mod spec_079 {
    use nuvec::{Array, fast};

    fn residual(h: &Array, x: &Array, w: &Array, scale: &Array) -> (Array, Option<Array>) {
        let _: (Array, Option<Array>) = fast::residual_rms_norm(h, x, (w, 1e-6), None, None);
        fast::residual_rms_norm(h, x, (w, 1e-6), scale, Some((w, 1e-6)))
    }

    #[test]
    fn the_items_exist() {
        let _ = residual;
    }
}
