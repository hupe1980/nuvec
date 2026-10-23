//! Random numbers from explicit keys or a per-thread key sequence, bit for bit as in MLX.
//!
//! A key is a uint32 array of shape `[2]`. [`bits`] hashes counters with threefry2x32
//! using MLX's layout, and [`uniform`] and [`normal`] are composed from operations
//! exactly as in MLX's `random.cpp`. For the same seed, bits and uniform samples equal MLX's.
//! Normal samples agree within a few ulps: `erfinv` needs a logarithm, whose last bits
//! differ between implementations.
//!
//! Without a key, functions draw the next key from this thread's sequence. Each thread's
//! sequence starts from a key derived from the process seed (a time seed until [`seed`]
//! is called) and the thread's ordinal, the order in which threads first draw:
//!
//! - ordinal 0 starts from `key(process_seed)`, so a single-threaded program seeded with
//!   `seed(s)` draws exactly MLX's values for `s`;
//! - ordinal `n > 0` starts from `key(splitmix64(process_seed ^ n))`.
//!
//! So threads that never called [`seed`] draw distinct streams. [`seed`] sets the process
//! seed, restarts the calling thread's sequence as ordinal 0 and makes every other thread
//! derive its sequence again (with the next ordinals, in the order they draw next); a
//! thread that called [`seed`] itself keeps its own sequence until it calls [`seed`] again.
//! Draws are reproducible across runs when threads start drawing in a fixed order; for
//! exact reproducibility of worker threads whatever their order, pass explicit keys
//! ([`key`], [`split`], [`split_n`]).
//!
//! Like every operation, these return arrays: invalid arguments give a failed array that
//! carries the error and the caller's location (see [`crate::Array::check`]).

pub(crate) mod imp;

use std::panic::Location;

use crate::axes::{Axis, IntoShape};
use crate::{Array, DType, Error};

pub use imp::{key, seed};

/// The next key of this thread's sequence (`split` the current key; keep the first half),
/// the key that functions called without one use. Useful to pass the sequence's state into
/// a compiled function, where keys drawn inside would be frozen at the first trace.
#[track_caller]
pub fn next_key() -> Array {
    Array::sticky("next_key", Location::caller(), &[], imp::next_key)
}

/// The current key of this thread's sequence, without advancing it. An internal hook for
/// `compile_mut` and `checkpoint`, which thread the sequence through traces and
/// recomputations; not part of the stable API.
#[doc(hidden)]
pub fn sequence_key() -> Array {
    imp::sequence_key()
}

/// Replaces the current key of this thread's sequence (see [`sequence_key`]). Internal.
#[doc(hidden)]
pub fn set_sequence_key(key: Array) {
    imp::set_sequence_key(key);
}

/// Random bits of `shape`, `width` bytes per element (1, 2 or 4 → uint8, uint16, uint32).
#[track_caller]
pub fn bits<'a>(shape: impl IntoShape, width: i32, key: impl Into<Option<&'a Array>>) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    let dtype = match width {
        1 => DType::U8,
        2 => DType::U16,
        _ => DType::U32,
    };
    Array::constructor("bits", Location::caller(), dtype, &spec, |shape| {
        let width = usize::try_from(width)
            .map_err(|_| Error::InvalidArgument(format!("bits: invalid width {width}")))?;
        imp::bits(shape, width, key)
    })
}

/// `num` new keys from `key`, as an array of shape `[num, 2]`.
#[track_caller]
pub fn split(key: &Array, num: usize) -> Array {
    Array::sticky("split", Location::caller(), &[key], || imp::split(key, num))
}

/// `N` new keys from `key`, for destructuring: `let [k1, k2] = random::split_n::<2>(&key);`.
///
/// ```
/// use nuvec_core::random;
///
/// let [a, b] = random::split_n::<2>(&random::key(0));
/// assert_eq!((a.shape(), b.shape()), (&[2][..], &[2][..]));
/// ```
#[track_caller]
pub fn split_n<const N: usize>(key: &Array) -> [Array; N] {
    Array::sticky_n("split", Location::caller(), &[key], || {
        let all = imp::split(key, N)?;
        let mut out = Vec::with_capacity(N);
        for i in 0..N {
            let row = crate::ops::imp::slice(
                &all,
                &[
                    crate::SliceArg::new(Some(i as isize), Some(i as isize + 1), 1),
                    crate::SliceArg::full(),
                ],
            )?;
            out.push(crate::ops::imp::reshape(&row, &[2])?);
        }
        Ok(out
            .try_into()
            .unwrap_or_else(|_| unreachable!("N keys were split")))
    })
}

/// Uniform samples in `[low, high)` of `shape` and floating-point `dtype` (float16,
/// bfloat16 or float32), as MLX: 32 random bits divided by `u32::MAX`, clamped below 1.
///
/// ```
/// use nuvec_core::{DType, random};
///
/// let u = random::uniform(0.0, 1.0, [2, 3], DType::F32, Some(&random::key(1)));
/// assert_eq!(u.shape(), [2, 3]);
/// assert!(u.to_vec::<f32>().unwrap().iter().all(|&x| (0.0..1.0).contains(&x)));
/// ```
#[track_caller]
pub fn uniform<'a>(
    low: f64,
    high: f64,
    shape: impl IntoShape,
    dtype: DType,
    key: impl Into<Option<&'a Array>>,
) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    Array::constructor("uniform", Location::caller(), dtype, &spec, |shape| {
        imp::uniform(low, high, shape, dtype, key)
    })
}

/// Normal samples (mean `loc`, standard deviation `scale`; MLX's defaults are 0 and 1) of
/// `shape` and floating-point `dtype`, as MLX: `√2 · erfinv(u)` for `u` uniform in `(-1, 1)`.
#[track_caller]
pub fn normal<'a>(
    shape: impl IntoShape,
    dtype: DType,
    loc: f64,
    scale: f64,
    key: impl Into<Option<&'a Array>>,
) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    Array::constructor("normal", Location::caller(), dtype, &spec, |shape| {
        imp::normal(shape, dtype, Some(loc), Some(scale), key)
    })
}

/// Gumbel samples of `shape` and floating-point `dtype`, as MLX: `−log(−log(u))` for `u`
/// uniform in `[0, 1)`.
#[track_caller]
pub fn gumbel<'a>(shape: impl IntoShape, dtype: DType, key: impl Into<Option<&'a Array>>) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    Array::constructor("gumbel", Location::caller(), dtype, &spec, |shape| {
        imp::gumbel(shape, dtype, key)
    })
}

/// Indices (uint32) drawn from the categorical distributions whose unnormalised log
/// probabilities run along one `axis` of `logits` (`..`: the flattened array), as MLX: the
/// argmax of `logits` plus Gumbel noise. The result has `logits`' shape without `axis`.
///
/// ```
/// use nuvec_core::{array, random};
///
/// // One certain outcome per row.
/// let logits = array![[0.0, -1e9, -1e9], [-1e9, -1e9, 0.0]];
/// let draws = random::categorical(&logits, -1, Some(&random::key(7)));
/// assert_eq!(draws.to_vec::<u32>().unwrap(), [0, 2]);
/// ```
#[track_caller]
pub fn categorical<'a>(
    logits: &Array,
    axis: impl Axis,
    key: impl Into<Option<&'a Array>>,
) -> Array {
    let key = key.into();
    let spec = axis.axis_spec();
    let mut ins = vec![logits];
    ins.extend(key);
    Array::sticky(
        "categorical",
        Location::caller(),
        &ins,
        || match crate::axes::one_axis(spec, logits.ndim())? {
            Some(axis) => imp::categorical(logits, axis as isize, key),
            None => imp::categorical(&crate::ops::imp::reshape(logits, &[-1])?, 0, key),
        },
    )
}

/// Integers in `[low, high)` of `shape` and integer `dtype`, as MLX: uniform floats in
/// `[low, high)` rounded down.
#[track_caller]
pub fn randint<'a>(
    low: i64,
    high: i64,
    shape: impl IntoShape,
    dtype: DType,
    key: impl Into<Option<&'a Array>>,
) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    Array::constructor("randint", Location::caller(), dtype, &spec, |shape| {
        imp::randint(low, high, shape, dtype, key)
    })
}

/// Booleans of `shape` that are true with probability `p`, as MLX: `uniform < p`.
#[track_caller]
pub fn bernoulli<'a>(p: f64, shape: impl IntoShape, key: impl Into<Option<&'a Array>>) -> Array {
    let key = key.into();
    let spec = shape.shape_spec();
    Array::constructor(
        "bernoulli",
        Location::caller(),
        DType::Bool,
        &spec,
        |shape| imp::bernoulli(p, shape, key),
    )
}
