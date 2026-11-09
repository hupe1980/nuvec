//! MLX's `hadamard_transform`: `x` times the Hadamard matrix of its last axis's size
//! `n = m · 2^k` (`m` in 1, 12, 20, 28; Sylvester's construction for `2^k`, Sloane's
//! matrices for `m`), scaled by `1/√n` by default. On the GPU it runs MLX's kernels; the
//! composition (butterflies and one product) is the fallback and gives the derivatives.

use crate::ops::imp::{astype, matmul, multiply, reshape, slice};
use crate::{Array, DType, Error, Result, SliceArg};

/// The Hadamard matrix of order 12 (`+` and `-`, rows in order).
const H12: [&str; 12] = [
    "+-++++++++++",
    "--+-+-+-+-+-",
    "+++-++----++",
    "+---+--+-++-",
    "+++++-++----",
    "+-+---+--+-+",
    "++--+++-++--",
    "+--++---+--+",
    "++----+++-++",
    "+--+-++---+-",
    "++++----+++-",
    "+-+--+-++---",
];

/// The Hadamard matrix of order 20 (`+` and `-`, rows in order).
const H20: [&str; 20] = [
    "+----+----++--++-++-",
    "-+----+---+++---+-++",
    "--+----+---+++-+-+-+",
    "---+----+---+++++-+-",
    "----+----++--++-++-+",
    "-+++++-----+--+++--+",
    "+-+++-+---+-+--+++--",
    "++-++--+---+-+--+++-",
    "+++-+---+---+-+--+++",
    "++++-----++--+-+--++",
    "--++-+-++-+-----++++",
    "---++-+-++-+---+-+++",
    "+---++-+-+--+--++-++",
    "++---++-+----+-+++-+",
    "-++---++-+----+++++-",
    "-+--+--++-+----+----",
    "+-+-----++-+----+---",
    "-+-+-+---+--+----+--",
    "--+-+++------+----+-",
    "+--+--++------+----+",
];

/// The Hadamard matrix of order 28 (`+` and `-`, rows in order).
const H28: [&str; 28] = [
    "+------++----++-+--+-+--++--",
    "-+-----+++-----+-+--+-+--++-",
    "--+-----+++---+-+-+----+--++",
    "---+-----+++---+-+-+-+--+--+",
    "----+-----+++---+-+-+++--+--",
    "-----+-----++++--+-+--++--+-",
    "------++----++-+--+-+--++--+",
    "--++++-+-------++--+++-+--+-",
    "---++++-+-----+-++--+-+-+--+",
    "+---+++--+----++-++--+-+-+--",
    "++---++---+----++-++--+-+-+-",
    "+++---+----+----++-++--+-+-+",
    "++++--------+-+--++-++--+-+-",
    "-++++--------+++--++--+--+-+",
    "-+-++-++--++--+--------++++-",
    "+-+-++--+--++--+--------++++",
    "-+-+-++--+--++--+----+---+++",
    "+-+-+-++--+--+---+---++---++",
    "++-+-+-++--+------+--+++---+",
    "-++-+-+-++--+------+-++++---",
    "+-++-+---++--+------+-++++--",
    "-++--++-+-++-+++----++------",
    "+-++--++-+-++-+++-----+-----",
    "++-++---+-+-++-+++-----+----",
    "-++-++-+-+-+-+--+++-----+---",
    "--++-++++-+-+----+++-----+--",
    "+--++-+-++-+-+----+++-----+-",
    "++--++-+-++-+-+----++------+",
];

/// `n` as `(2^k, m)`, or an error naming `hadamard_transform`.
pub(crate) fn decompose(n: usize) -> Result<(usize, usize)> {
    let invalid = |m: String| Error::InvalidArgument(format!("hadamard_transform: {m}"));
    let (pow2, m) = if n.is_power_of_two() {
        (n, 1)
    } else {
        [12, 20, 28]
            .into_iter()
            .find(|&m| n.is_multiple_of(m) && (n / m).is_power_of_two())
            .map(|m| (n / m, m))
            .ok_or_else(|| {
                invalid(format!(
                    "the size {n} is not m · 2^k with m in 1, 12, 20 or 28"
                ))
            })?
    };
    if pow2 > 1 << 26 {
        return Err(invalid(format!("the size {n} needs 2^k ≤ 2^26")));
    }
    Ok((pow2, m))
}

/// The ±1 matrix of order `m` as an array of `dtype`.
fn matrix(m: usize, dtype: DType) -> Result<Array> {
    let rows: &[&str] = match m {
        12 => &H12,
        20 => &H20,
        _ => &H28,
    };
    let v: Vec<f64> = rows
        .iter()
        .flat_map(|r| r.chars().map(|c| if c == '+' { 1.0 } else { -1.0 }))
        .collect();
    astype(&Array::from_slice(&v, [m, m]), dtype)
}

/// The transform composed of `log2(2^k)` butterfly passes and, for `m > 1`, a product
/// with the order-`m` matrix, in float32 (float64 for float64), rounded once.
pub(crate) fn compose(x: &Array, scale: f32) -> Result<Array> {
    let dtype = x.dtype();
    let acc = if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let n = x.dims().last().copied().unwrap_or(1);
    let (pow2, m) = decompose(n)?;
    let rows = x.size() / n.max(1);
    let mut y = reshape(
        &astype(x, acc)?,
        &[rows as isize, m as isize, pow2 as isize],
    )?;
    let mut h = 1;
    while h < pow2 {
        let split = reshape(
            &y,
            &[
                rows as isize,
                m as isize,
                (pow2 / (2 * h)) as isize,
                2,
                h as isize,
            ],
        )?;
        let half = |i: isize| -> Result<Array> {
            let mut args = vec![SliceArg::full(); 5];
            args[3] = SliceArg::new(Some(i), Some(i + 1), 1);
            slice(&split, &args)
        };
        let (a, b) = (half(0)?, half(1)?);
        let both = crate::indexing::concatenate(
            &[
                crate::ops::imp::add(&a, &b)?,
                crate::ops::imp::subtract(&a, &b)?,
            ],
            3,
        )?;
        y = reshape(&both, &[rows as isize, m as isize, pow2 as isize])?;
        h *= 2;
    }
    if m > 1 {
        y = matmul(&matrix(m, acc)?, &y)?;
    }
    let y = multiply(&y, &astype(&Array::from(scale), acc)?)?;
    astype(
        &reshape(
            &y,
            &x.dims().iter().map(|&d| d as isize).collect::<Vec<_>>(),
        )?,
        dtype,
    )
}
