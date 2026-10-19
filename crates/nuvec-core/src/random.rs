//! Random numbers from explicit keys or a per-thread key sequence, bit for bit as in MLX.
//!
//! A key is a uint32 array of shape `[2]`. [`bits`] hashes counters with threefry2x32
//! using MLX's layout, and [`uniform`] and [`normal`] are composed from operations
//! exactly as in MLX's `random.cpp`. For the same seed, bits and uniform samples equal MLX's.
//! Normal samples agree within a few ulps: `erfinv` needs a logarithm, whose last bits
//! differ between implementations.
//!
//! Without a key, functions draw the next key from this thread's sequence, which starts from
//! a per-process time seed until [`seed`] is called (as in MLX).

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::ops::{add, divide, erfinv, minimum, multiply, slice, subtract};
use crate::primitive::Primitive;
use crate::{Array, DType, Error, Result, SliceArg, bf16, f16};

/// A key for `seed`: `[seed >> 32, seed & 0xffffffff]`.
pub fn key(seed: u64) -> Array {
    Array::from_slice(&[(seed >> 32) as u32, seed as u32], &[2]).expect("two elements")
}

fn time_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    })
}

thread_local!(static SEQUENCE: RefCell<Option<Array>> = const { RefCell::new(None) });

/// Restarts this thread's key sequence from `seed`.
pub fn seed(seed: u64) {
    SEQUENCE.set(Some(key(seed)));
}

/// The next key of this thread's sequence (`split` the current key; keep the first half).
fn next_key() -> Result<Array> {
    let current = SEQUENCE.with_borrow_mut(|k| k.get_or_insert_with(|| key(time_seed())).clone());
    let (keep, out) = split_pair(&current)?;
    SEQUENCE.set(Some(keep));
    Ok(out)
}

fn key_or_next(key: Option<&Array>) -> Result<Array> {
    match key {
        Some(k) => Ok(k.clone()),
        None => next_key(),
    }
}

/// Random bits of `shape`, `width` bytes per element (1, 2 or 4 → uint8, uint16, uint32).
pub fn bits(shape: &[usize], width: usize, key: Option<&Array>) -> Result<Array> {
    let key = key_or_next(key)?;
    if key.dtype() != DType::U32 || key.shape() != [2] {
        return Err(Error::InvalidArgument(format!(
            "bits: expected a uint32 key of shape [2], got {} {:?}",
            key.dtype(),
            key.shape()
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

/// `num` new keys from `key`, as an array of shape `[num, 2]`.
pub fn split(key: &Array, num: usize) -> Result<Array> {
    bits(&[num, 2], 4, Some(key))
}

/// Two new keys from `key`.
pub fn split_pair(key: &Array) -> Result<(Array, Array)> {
    let both = split(key, 2)?;
    let row = |i: isize| {
        let args = [SliceArg::new(Some(i), Some(i + 1), 1), SliceArg::full()];
        slice(&both, &args)?.reshape(&[2])
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
            return Err(Error::UnsupportedDType {
                op: "uniform",
                dtype,
            });
        }
    })
}

/// Uniform samples in `[low, high)` of `shape` and floating-point `dtype` (float16,
/// bfloat16 or float32), as MLX: 32 random bits divided by `u32::MAX`, clamped below 1.
pub fn uniform(
    low: f64,
    high: f64,
    shape: &[usize],
    dtype: DType,
    key: Option<&Array>,
) -> Result<Array> {
    let upper = below_one(dtype)?;
    let lo = Array::scalar(low).astype(dtype)?;
    let hi = Array::scalar(high).astype(dtype)?;
    let range = subtract(&hi, &lo)?;
    let out = divide(&bits(shape, 4, key)?, Array::scalar(u32::MAX as f32))?;
    let out = minimum(&out, Array::scalar(upper))?.astype(dtype)?;
    add(multiply(&range, &out)?, &lo)
}

/// Normal samples (mean `loc`, standard deviation `scale`) of `shape` and floating-point
/// `dtype`, as MLX: `√2 · erfinv(u)` for `u` uniform in `(-1, 1)`.
pub fn normal(
    shape: &[usize],
    dtype: DType,
    loc: Option<f64>,
    scale: Option<f64>,
    key: Option<&Array>,
) -> Result<Array> {
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            op: "normal",
            dtype,
        });
    }
    let low = f64::from(f32::from_bits((-1.0f32).to_bits() - 1));
    let samples = uniform(low, 1.0, shape, DType::F32, key)?;
    let mut applied = Array::scalar(2f64.sqrt()).astype(dtype)?;
    if let Some(scale) = scale {
        applied = multiply(&applied, Array::scalar(scale).astype(dtype)?)?;
    }
    let samples = multiply(&applied, erfinv(&samples)?.astype(dtype)?)?;
    match loc {
        Some(loc) => add(Array::scalar(loc).astype(dtype)?, &samples),
        None => Ok(samples),
    }
}

/// Gumbel samples of `shape` and floating-point `dtype`, as MLX: `−log(−log(u))` for `u`
/// uniform in `[0, 1)`.
pub fn gumbel(shape: &[usize], dtype: DType, key: Option<&Array>) -> Result<Array> {
    let u = uniform(0.0, 1.0, shape, dtype, key)?;
    crate::ops::negative(&crate::ops::log(&crate::ops::negative(&crate::ops::log(
        &u,
    )?)?)?)
}

/// Indices (uint32) drawn from the categorical distributions whose unnormalised log
/// probabilities run along `axis` of `logits`, as MLX: the argmax of `logits` plus Gumbel
/// noise. The result has `logits`' shape without `axis`.
///
/// ```
/// use nuvec_core::{Array, Result, random};
///
/// # fn main() -> Result<()> {
/// // One certain outcome per row.
/// let logits = Array::from_slice(&[0.0f32, -1e9, -1e9, -1e9, -1e9, 0.0], &[2, 3])?;
/// let draws = random::categorical(&logits, -1, Some(&random::key(7)))?;
/// assert_eq!(draws.to_vec::<u32>()?, [0, 2]);
/// # Ok(())
/// # }
/// ```
pub fn categorical(logits: &Array, axis: isize, key: Option<&Array>) -> Result<Array> {
    let noise = gumbel(logits.shape(), logits.dtype(), key)?;
    crate::ops::argmax(&add(&noise, logits)?, Some(axis), false)
}

/// Integers in `[low, high)` of `shape` and integer `dtype`, as MLX: uniform floats in
/// `[low, high)` rounded down.
pub fn randint(
    low: i64,
    high: i64,
    shape: &[usize],
    dtype: DType,
    key: Option<&Array>,
) -> Result<Array> {
    if dtype.category() == crate::Category::Float || dtype == DType::Complex64 {
        return Err(Error::UnsupportedDType {
            op: "randint",
            dtype,
        });
    }
    let u = uniform(low as f64, high as f64, shape, DType::F32, key)?;
    let u = crate::ops::maximum(&u, Array::scalar(low as f32))?;
    // floor(u): truncation, one less where it rounded a negative value up.
    let t = u.astype(DType::I64)?;
    let up = crate::ops::less(&u, &t.astype(DType::F32)?)?;
    subtract(&t, up.astype(DType::I64)?)?.astype(dtype)
}

/// Booleans of `shape` that are true with probability `p`, as MLX: `uniform < p`.
pub fn bernoulli(p: f64, shape: &[usize], key: Option<&Array>) -> Result<Array> {
    crate::ops::less(&uniform(0.0, 1.0, shape, DType::F32, key)?, p)
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
        assert!(bits(&[2], 4, Some(&Array::scalar(1u32))).is_err());
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
