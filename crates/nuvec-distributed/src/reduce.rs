//! Element-wise reductions of raw element bytes, shared by the backends.

use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use nuvec_core::distributed::AllReduce;
use nuvec_core::{DType, Error, Result, bf16, f16};

macro_rules! fold {
    ($acc:ident, $x:ident, $t:ty, $n:expr, $f:expr) => {{
        for (a, b) in $acc.chunks_exact_mut($n).zip($x.chunks_exact($n)) {
            let (p, q) = (
                <$t>::from_le_bytes(a.try_into().unwrap()),
                <$t>::from_le_bytes(b.try_into().unwrap()),
            );
            let f: fn($t, $t) -> $t = $f;
            a.copy_from_slice(&f(p, q).to_le_bytes());
        }
    }};
}

macro_rules! numeric {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        let n = size_of::<$t>();
        match $op {
            AllReduce::Sum => fold!($acc, $x, $t, n, |a, b| a + b),
            AllReduce::Max => fold!($acc, $x, $t, n, |a, b| if a < b { b } else { a }),
            _ => fold!($acc, $x, $t, n, |a, b| if b < a { b } else { a }),
        }
    }};
}

macro_rules! wrapping {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        let n = size_of::<$t>();
        match $op {
            AllReduce::Sum => fold!($acc, $x, $t, n, |a, b| a.wrapping_add(b)),
            AllReduce::Max => fold!($acc, $x, $t, n, |a, b| a.max(b)),
            _ => fold!($acc, $x, $t, n, |a, b| a.min(b)),
        }
    }};
}

macro_rules! half_float {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        if $op == AllReduce::Sum {
            sum_half::<$t>($acc, $x);
            return Ok(());
        }
        for (a, b) in $acc.chunks_exact_mut(2).zip($x.chunks_exact(2)) {
            let p = <$t>::from_bits(u16::from_le_bytes([a[0], a[1]]));
            let q = <$t>::from_bits(u16::from_le_bytes([b[0], b[1]]));
            let r = match $op {
                AllReduce::Max => {
                    if p < q {
                        q
                    } else {
                        p
                    }
                }
                _ => {
                    if q < p {
                        q
                    } else {
                        p
                    }
                }
            };
            a.copy_from_slice(&r.to_bits().to_le_bytes());
        }
    }};
}

/// A 16-bit float whose blocks widen to and narrow from float32.
trait Half: Copy {
    fn widen(bits: &[u16], out: &mut [f32]);
    fn narrow(values: &[f32], out: &mut [u16]);
}

impl Half for f16 {
    fn widen(bits: &[u16], out: &mut [f32]) {
        bits.reinterpret_cast::<f16>().convert_to_f32_slice(out);
    }
    fn narrow(values: &[f32], out: &mut [u16]) {
        out.reinterpret_cast_mut::<f16>()
            .convert_from_f32_slice(values);
    }
}

impl Half for bf16 {
    fn widen(bits: &[u16], out: &mut [f32]) {
        for (o, &b) in out.iter_mut().zip(bits) {
            *o = f32::from_bits(u32::from(b) << 16);
        }
    }
    fn narrow(values: &[f32], out: &mut [u16]) {
        // `bf16::from_f32`, branch-free: round to nearest even, NaN kept quiet.
        for (o, &v) in out.iter_mut().zip(values) {
            let x = v.to_bits();
            let rounded = ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16;
            let nan = ((x >> 16) | 0x0040) as u16;
            *o = if x & 0x7fff_ffff > 0x7f80_0000 {
                nan
            } else {
                rounded
            };
        }
    }
}

/// `acc += x` over 16-bit floats in blocks the compiler vectorizes. Each sum is exact in
/// float32 and rounded once, as MLX's half-precision add.
fn sum_half<T: Half>(acc: &mut [u8], x: &[u8]) {
    const BLOCK: usize = 512;
    let (mut a16, mut b16) = ([0u16; BLOCK], [0u16; BLOCK]);
    let (mut a32, mut b32) = ([0f32; BLOCK], [0f32; BLOCK]);
    for (a, b) in acc.chunks_mut(2 * BLOCK).zip(x.chunks(2 * BLOCK)) {
        let n = a.len() / 2;
        for (i, (p, q)) in a
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_chunks::<2>().0)
            .enumerate()
        {
            a16[i] = u16::from_le_bytes(*p);
            b16[i] = u16::from_le_bytes(*q);
        }
        T::widen(&a16[..n], &mut a32[..n]);
        T::widen(&b16[..n], &mut b32[..n]);
        for (p, q) in a32[..n].iter_mut().zip(&b32[..n]) {
            *p += q;
        }
        T::narrow(&a32[..n], &mut a16[..n]);
        for (p, v) in a.as_chunks_mut::<2>().0.iter_mut().zip(&a16[..n]) {
            *p = v.to_le_bytes();
        }
    }
}

/// `acc = op(acc, x)` element-wise for elements of `dtype`.
pub(crate) fn combine(op: AllReduce, dtype: DType, acc: &mut [u8], x: &[u8]) -> Result<()> {
    match dtype {
        DType::Bool => {
            for (a, b) in acc.iter_mut().zip(x) {
                *a = match op {
                    AllReduce::Min => u8::from(*a != 0 && *b != 0),
                    _ => u8::from(*a != 0 || *b != 0),
                };
            }
        }
        DType::U8 => wrapping!(op, acc, x, u8),
        DType::U16 => wrapping!(op, acc, x, u16),
        DType::U32 => wrapping!(op, acc, x, u32),
        DType::U64 => wrapping!(op, acc, x, u64),
        DType::I8 => wrapping!(op, acc, x, i8),
        DType::I16 => wrapping!(op, acc, x, i16),
        DType::I32 => wrapping!(op, acc, x, i32),
        DType::I64 => wrapping!(op, acc, x, i64),
        DType::F32 => numeric!(op, acc, x, f32),
        DType::F64 => numeric!(op, acc, x, f64),
        DType::F16 => half_float!(op, acc, x, f16),
        DType::BF16 => half_float!(op, acc, x, bf16),
        DType::Complex64 if op == AllReduce::Sum => fold!(acc, x, f32, 4, |a, b| a + b),
        d => {
            return Err(Error::InvalidArgument(format!(
                "distributed: no {op:?} reduction for {}",
                d.name()
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "jaccl")]
/// `out = op(a, b)` element-wise (the same bits as `combine` on a copy of `a`).
pub(crate) fn combine_into(
    op: AllReduce,
    dtype: DType,
    out: &mut [u8],
    a: &[u8],
    b: &[u8],
) -> Result<()> {
    out.copy_from_slice(a);
    combine(op, dtype, out, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian bytes of `v`.
    macro_rules! bytes {
        ($($x:expr),* $(,)?) => {
            [$($x.to_le_bytes()),*].concat()
        };
    }

    #[test]
    fn reduces_every_dtype() {
        let mut acc = bytes![1.0f32, -2.0f32, 3.0f32];
        combine(
            AllReduce::Sum,
            DType::F32,
            &mut acc,
            &bytes![0.5f32, 2.0f32, -1.0f32],
        )
        .unwrap();
        assert_eq!(acc, bytes![1.5f32, 0.0f32, 2.0f32]);
        let mut acc = bytes![200u8, 3u8];
        combine(AllReduce::Sum, DType::U8, &mut acc, &bytes![100u8, 4u8]).unwrap();
        assert_eq!(acc, bytes![44u8, 7u8], "wrapping");
        let mut acc = bytes![f16::from_f32(1.0).to_bits(), f16::from_f32(5.0).to_bits()];
        combine(
            AllReduce::Max,
            DType::F16,
            &mut acc,
            &bytes![f16::from_f32(2.0).to_bits(), f16::from_f32(4.0).to_bits()],
        )
        .unwrap();
        assert_eq!(
            acc,
            bytes![f16::from_f32(2.0).to_bits(), f16::from_f32(5.0).to_bits()]
        );
        let mut acc = vec![1u8, 0, 1];
        combine(AllReduce::Min, DType::Bool, &mut acc, &[1, 1, 0]).unwrap();
        assert_eq!(acc, [1, 0, 0]);
        #[cfg(feature = "jaccl")]
        {
            let (a, b) = (bytes![1i64, 9i64], bytes![5i64, -3i64]);
            let mut out = vec![0u8; 16];
            combine_into(AllReduce::Min, DType::I64, &mut out, &a, &b).unwrap();
            assert_eq!(out, bytes![1i64, -3i64]);
        }
        assert!(combine(AllReduce::Max, DType::Complex64, &mut [0; 8], &[0; 8]).is_err());
    }

    /// The block-wise half sums equal rounding each float32 sum once (`half`'s scalar
    /// conversions), over every bit pattern of one operand: NaNs, infinities, subnormals,
    /// overflow, and a length that ends mid-block.
    #[test]
    fn half_sums_round_once() {
        let a: Vec<u16> = (0..=u16::MAX).collect();
        // A second operand that pairs each pattern with a varied one.
        let b: Vec<u16> = a
            .iter()
            .map(|&x| x.wrapping_mul(40503).rotate_left(5))
            .collect();
        let raw = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for dtype in [DType::F16, DType::BF16] {
            let mut acc = raw(&a);
            combine(AllReduce::Sum, dtype, &mut acc, &raw(&b)).unwrap();
            let want: Vec<u16> = a
                .iter()
                .zip(&b)
                .map(|(&p, &q)| match dtype {
                    DType::F16 => {
                        let s = f16::from_bits(p).to_f32() + f16::from_bits(q).to_f32();
                        f16::from_f32(s).to_bits()
                    }
                    _ => {
                        let s = bf16::from_bits(p).to_f32() + bf16::from_bits(q).to_f32();
                        bf16::from_f32(s).to_bits()
                    }
                })
                .collect();
            for (i, (got, want)) in acc.as_chunks::<2>().0.iter().zip(&want).enumerate() {
                let got = u16::from_le_bytes(*got);
                // NaN payloads may differ between conversions; NaN-ness may not.
                let nan = |x: u16| match dtype {
                    DType::F16 => f16::from_bits(x).is_nan(),
                    _ => bf16::from_bits(x).is_nan(),
                };
                assert!(
                    got == *want || (nan(got) && nan(*want)),
                    "{dtype:?} {i}: {got:#06x} != {want:#06x}"
                );
            }
        }
    }
}
