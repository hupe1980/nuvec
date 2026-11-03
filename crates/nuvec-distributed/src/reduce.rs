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
            <$t as Half>::sum($acc, $x);
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

/// A 16-bit float's sum: each element's float32 sum rounded once, as MLX's half-precision
/// add.
trait Half: Copy {
    fn sum(acc: &mut [u8], x: &[u8]);
}

impl Half for f16 {
    /// In blocks through `half`'s slice conversions, which the compiler vectorizes.
    fn sum(acc: &mut [u8], x: &[u8]) {
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
            a16[..n]
                .reinterpret_cast::<f16>()
                .convert_to_f32_slice(&mut a32[..n]);
            b16[..n]
                .reinterpret_cast::<f16>()
                .convert_to_f32_slice(&mut b32[..n]);
            for (p, q) in a32[..n].iter_mut().zip(&b32[..n]) {
                *p += q;
            }
            a16[..n]
                .reinterpret_cast_mut::<f16>()
                .convert_from_f32_slice(&a32[..n]);
            for (p, v) in a.as_chunks_mut::<2>().0.iter_mut().zip(&a16[..n]) {
                *p = v.to_le_bytes();
            }
        }
    }
}

impl Half for bf16 {
    /// One loop the compiler vectorizes: widening is a shift, and the float32 sum rounds to
    /// nearest even without a NaN check. Both operands widen with zero low bits, so a NaN
    /// sum (an operand's, made quiet, or the default NaN) keeps them zero, and rounding
    /// cannot carry it into an infinity.
    fn sum(acc: &mut [u8], x: &[u8]) {
        for (a, b) in acc
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(x.as_chunks::<2>().0)
        {
            let widen = |h: [u8; 2]| f32::from_bits(u32::from(u16::from_le_bytes(h)) << 16);
            let s = (widen(*a) + widen(*b)).to_bits();
            *a = (((s + 0x7fff + ((s >> 16) & 1)) >> 16) as u16).to_le_bytes();
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

    /// The half sums equal rounding each float32 sum once (`half`'s scalar conversions),
    /// over every bit pattern of one operand paired with varied ones and with each edge
    /// value: NaNs, infinities, subnormals and overflow.
    #[test]
    fn half_sums_round_once() {
        let a: Vec<u16> = (0..=u16::MAX).collect();
        // A second operand that pairs each pattern with a varied one.
        let b: Vec<u16> = a
            .iter()
            .map(|&x| x.wrapping_mul(40503).rotate_left(5))
            .collect();
        let raw = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // And every pattern against the edge values: NaNs of several payloads and both
        // signs, infinities, the largest finite values, the smallest subnormals and zeros.
        let edges: [u16; 12] = [
            0x7fc0, 0xffc0, 0x7f81, 0xff81, 0x7fff, 0x7f80, 0xff80, 0x7f7f, 0xff7f, 0x0001, 0x8001,
            0x8000,
        ];
        let f16_edges: [u16; 12] = [
            0x7e00, 0xfe00, 0x7c01, 0xfc01, 0x7fff, 0x7c00, 0xfc00, 0x7bff, 0xfbff, 0x0001, 0x8001,
            0x8000,
        ];
        for dtype in [DType::F16, DType::BF16] {
            let edges = if dtype == DType::F16 {
                f16_edges
            } else {
                edges
            };
            let a: Vec<u16> = (0..=u16::MAX)
                .chain(edges.iter().flat_map(|_| 0..=u16::MAX))
                .collect();
            let b: Vec<u16> = b
                .iter()
                .copied()
                .chain(edges.iter().flat_map(|&e| std::iter::repeat_n(e, 1 << 16)))
                .collect();
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
