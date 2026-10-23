//! The implementation of the random number functions (see [`crate::random`]).

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::ops::imp::{add, divide, erfinv, minimum, multiply, slice, subtract};
use crate::primitive::Primitive;
use crate::{Array, DType, Error, Result, SliceArg, bf16, f16};

/// The body of [`crate::random::key`].
pub fn key(seed: u64) -> Array {
    let data = [(seed >> 32) as u32, seed as u32];
    crate::creation::from_data(&data, &[2]).unwrap_or_else(|e| Array::failed(e, DType::U32, &[2]))
}

fn time_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    })
}

/// The last [`seed`], once one was given (else the time seed applies).
static PROCESS_SEED: AtomicU64 = AtomicU64::new(0);
static SEEDED: AtomicBool = AtomicBool::new(false);
/// Bumped by [`seed`]: sequences derived before are stale.
static EPOCH: AtomicU64 = AtomicU64::new(0);
/// The next thread ordinal (reset by [`seed`]).
static THREAD_COUNTER: AtomicU64 = AtomicU64::new(0);

fn process_seed() -> u64 {
    match SEEDED.load(Ordering::Acquire) {
        true => PROCESS_SEED.load(Ordering::Acquire),
        false => time_seed(),
    }
}

/// One step of SplitMix64: spreads nearby seeds and ordinals over unrelated keys.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The first key of the sequence of thread `ordinal` (see the module documentation).
fn derived_key(seed: u64, ordinal: u64) -> Array {
    match ordinal {
        0 => key(seed),
        n => key(splitmix64(seed ^ n)),
    }
}

/// A thread's key sequence.
struct Sequence {
    key: Array,
    /// The [`EPOCH`] it was derived in.
    epoch: u64,
    /// Set by this thread's own [`seed`] call: kept when another thread seeds.
    seeded_here: bool,
}

thread_local!(static SEQUENCE: RefCell<Option<Sequence>> = const { RefCell::new(None) });

/// The body of [`crate::random::seed`].
pub fn seed(seed: u64) {
    PROCESS_SEED.store(seed, Ordering::Release);
    SEEDED.store(true, Ordering::Release);
    THREAD_COUNTER.store(1, Ordering::Release);
    let epoch = EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    SEQUENCE.set(Some(Sequence {
        key: derived_key(seed, 0),
        epoch,
        seeded_here: true,
    }));
}

/// The body of [`crate::random::next_key`].
pub(crate) fn next_key() -> Result<Array> {
    let current = sequence_key();
    let (keep, out) = split_two(&current)?;
    SEQUENCE.with_borrow_mut(|s| {
        if let Some(s) = s {
            s.key = keep;
        }
    });
    Ok(out)
}

fn key_or_next(key: Option<&Array>) -> Result<Array> {
    match key {
        Some(k) => Ok(k.clone()),
        None => next_key(),
    }
}

/// The body of [`crate::random::bits`].
pub(crate) fn bits(shape: &[usize], width: usize, key: Option<&Array>) -> Result<Array> {
    let key = key_or_next(key)?;
    if key.dtype() != DType::U32 || key.dims() != [2] {
        return Err(Error::InvalidArgument(format!(
            "bits: expected a uint32 key of shape [2], got {} {:?}",
            key.dtype(),
            key.dims()
        )));
    }
    let width = u8::try_from(width)
        .map_err(|_| Error::InvalidArgument(format!("bits: width {width} not in {{1, 2, 4}}")))?;
    let primitive = Primitive::RandomBits {
        shape: shape.to_vec(),
        width,
    };
    Ok(Array::from_op(primitive, vec![key])?.swap_remove(0))
}

/// The body of [`crate::random::split`].
pub(crate) fn split(key: &Array, num: usize) -> Result<Array> {
    bits(&[num, 2], 4, Some(key))
}

/// The body of [`crate::random::split_two`].
pub(crate) fn split_two(key: &Array) -> Result<(Array, Array)> {
    let both = split(key, 2)?;
    let row = |i: isize| {
        let args = [SliceArg::new(Some(i), Some(i + 1), 1), SliceArg::full()];
        slice(&both, &args)?.try_reshape(&[2])
    };
    Ok((row(0)?, row(1)?))
}

/// The largest value below 1 in `dtype`, as float32 (samples stay in `[low, high)`).
fn below_one(dtype: DType) -> Result<f32> {
    Ok(match dtype {
        DType::F32 => f32::from_bits(1.0f32.to_bits() - 1),
        DType::F16 => f16::from_bits(f16::ONE.to_bits() - 1).to_f32(),
        DType::BF16 => bf16::from_bits(bf16::ONE.to_bits() - 1).to_f32(),
        _ => {
            return Err(Error::unsupported_dtype("uniform", dtype));
        }
    })
}

/// The body of [`crate::random::uniform`].
pub(crate) fn uniform(
    low: f64,
    high: f64,
    shape: &[usize],
    dtype: DType,
    key: Option<&Array>,
) -> Result<Array> {
    let upper = below_one(dtype)?;
    let lo = Array::from(low).try_astype(dtype)?;
    let hi = Array::from(high).try_astype(dtype)?;
    let range = subtract(&hi, &lo)?;
    let out = divide(&bits(shape, 4, key)?, Array::from(u32::MAX as f32))?;
    let out = minimum(&out, Array::from(upper))?.try_astype(dtype)?;
    add(multiply(&range, &out)?, &lo)
}

/// The body of [`crate::random::normal`].
pub(crate) fn normal(
    shape: &[usize],
    dtype: DType,
    loc: Option<f64>,
    scale: Option<f64>,
    key: Option<&Array>,
) -> Result<Array> {
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        return Err(Error::unsupported_dtype("normal", dtype));
    }
    let low = f64::from(f32::from_bits((-1.0f32).to_bits() - 1));
    let samples = uniform(low, 1.0, shape, DType::F32, key)?;
    let mut applied = Array::from(2f64.sqrt()).try_astype(dtype)?;
    if let Some(scale) = scale {
        applied = multiply(&applied, Array::from(scale).try_astype(dtype)?)?;
    }
    let samples = multiply(&applied, erfinv(&samples)?.try_astype(dtype)?)?;
    match loc {
        Some(loc) => add(Array::from(loc).try_astype(dtype)?, &samples),
        None => Ok(samples),
    }
}

/// The body of [`crate::random::gumbel`].
pub(crate) fn gumbel(shape: &[usize], dtype: DType, key: Option<&Array>) -> Result<Array> {
    let u = uniform(0.0, 1.0, shape, dtype, key)?;
    crate::ops::imp::negative(&crate::ops::imp::log(&crate::ops::imp::negative(
        &crate::ops::imp::log(&u)?,
    )?)?)
}

/// The body of [`crate::random::categorical`].
pub(crate) fn categorical(logits: &Array, axis: isize, key: Option<&Array>) -> Result<Array> {
    let noise = gumbel(logits.dims(), logits.dtype(), key)?;
    crate::ops::imp::argmax(&add(&noise, logits)?, Some(axis), false)
}

/// The body of [`crate::random::randint`].
pub(crate) fn randint(
    low: i64,
    high: i64,
    shape: &[usize],
    dtype: DType,
    key: Option<&Array>,
) -> Result<Array> {
    if dtype.category() == crate::Category::Float || dtype == DType::Complex64 {
        return Err(Error::unsupported_dtype("randint", dtype));
    }
    let u = uniform(low as f64, high as f64, shape, DType::F32, key)?;
    let u = crate::ops::imp::maximum(&u, Array::from(low as f32))?;
    // floor(u): truncation, one less where it rounded a negative value up.
    let t = u.try_astype(DType::I64)?;
    let up = crate::ops::imp::less(&u, &t.try_astype(DType::F32)?)?;
    subtract(&t, up.try_astype(DType::I64)?)?.try_astype(dtype)
}

/// The body of [`crate::random::bernoulli`].
pub(crate) fn bernoulli(p: f64, shape: &[usize], key: Option<&Array>) -> Result<Array> {
    crate::ops::imp::less(&uniform(0.0, 1.0, shape, DType::F32, key)?, p)
}

/// The current key of this thread's sequence (derived first if needed), without advancing
/// it: the body of [`crate::random::sequence_key`].
pub(crate) fn sequence_key() -> Array {
    SEQUENCE.with_borrow_mut(|s| {
        let epoch = EPOCH.load(Ordering::Acquire);
        let stale = s
            .as_ref()
            .is_none_or(|s| !s.seeded_here && s.epoch != epoch);
        if stale {
            let ordinal = THREAD_COUNTER.fetch_add(1, Ordering::AcqRel);
            *s = Some(Sequence {
                key: derived_key(process_seed(), ordinal),
                epoch,
                seeded_here: false,
            });
        }
        s.as_ref().expect("set above").key.clone()
    })
}

/// Replaces the current key of this thread's sequence: the body of
/// [`crate::random::set_sequence_key`].
pub(crate) fn set_sequence_key(key: Array) {
    let _ = sequence_key(); // derive the sequence first, so its epoch is current
    SEQUENCE.with_borrow_mut(|s| {
        if let Some(s) = s {
            s.key = key;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::threefry2x32;

    #[test]
    fn threefry_known_answer() {
        // Random123's known-answer vector for threefry2x32-20 with a zero key and counter.
        assert_eq!(threefry2x32((0, 0), (0, 0)), (0x6b20_0159, 0x99ba_4efe));
    }

    #[test]
    fn keys_and_layout() {
        assert_eq!(key(1 << 32 | 7).to_vec::<u32>().unwrap(), [1, 7]);
        // An odd word count and a partial final word use the tail paths.
        let k = key(3);
        for (n, width) in [(5, 4), (3, 2), (7, 1), (0, 4)] {
            let b = bits(&[n], width, Some(&k)).unwrap();
            b.eval().unwrap();
            assert_eq!(b.size(), n);
        }
        assert!(bits(&[2], 3, Some(&k)).is_err());
        assert!(bits(&[2], 4, Some(&Array::from(1u32))).is_err());
    }

    #[test]
    fn sequences_are_per_thread_and_reproducible() {
        seed(42);
        let a = uniform(0.0, 1.0, &[4], DType::F32, None)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        seed(42);
        let b = uniform(0.0, 1.0, &[4], DType::F32, None)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|&x| (0.0..1.0).contains(&x)));
    }
}
