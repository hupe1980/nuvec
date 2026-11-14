//! Products of 1 to [`FEW_ROWS`] rows whose bits do not depend on the number of rows.
//!
//! Each output element is a reduction over K in one fixed order: term `p` goes to partial
//! sum `p mod 8`, each partial is updated in increasing `p` with a fused multiply-add from
//! +0, and the partials are combined by one fixed tree. The order depends only on K, so a
//! row of an M-row product has the bits of the same row alone, whatever the layout of the
//! rows or of the weights. A zero term leaves a partial unchanged (the partials start at +0
//! and never become -0), so a reduction padded with zero terms keeps its bits: attention
//! over a causal key prefix padded by masked keys equals attention over the prefix alone.
//! Each weight is read once for all rows.

use rayon::prelude::*;

use crate::matmul::MatView;

/// Rows per pass over the weights (nuvec's `ROW_INDEPENDENT_ROWS`).
pub const FEW_ROWS: usize = 8;

/// Products with at most this many rows run here, [`FEW_ROWS`] rows per pass, so a batch
/// of short products folded into one keeps each row's bits. Larger products take the
/// general GEMM.
pub const EXACT_ROWS: usize = 16;

/// Partial sums per output element.
const L: usize = 8;

/// Columns per parallel task.
const COLUMNS: usize = 64;

/// `s[l] = fma(x[l], w[l], s[l])` for every partial: NEON on aarch64, scalar elsewhere
/// (a fused multiply-add rounds once either way, so the bits are the same).
#[inline(always)]
fn fma(s: &mut [f32; L], x: &[f32; L], w: &[f32; L]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{float32x4_t, vfmaq_f32, vld1q_f32, vst1q_f32};
        let load = |v: &[f32]| -> float32x4_t {
            debug_assert!(v.len() >= 4);
            // SAFETY: NEON is part of the aarch64 baseline; `v` holds at least 4 `f32`s.
            unsafe { vld1q_f32(v.as_ptr()) }
        };
        for h in [0, 4] {
            let (a, b, c) = (load(&s[h..]), load(&x[h..]), load(&w[h..]));
            // SAFETY: NEON is part of the aarch64 baseline.
            let r = unsafe { vfmaq_f32(a, b, c) };
            let out = &mut s[h..h + 4];
            // SAFETY: as above; `out` holds 4 `f32`s.
            unsafe { vst1q_f32(out.as_mut_ptr(), r) };
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for l in 0..L {
        s[l] = x[l].mul_add(w[l], s[l]);
    }
}

/// The fixed combination of the partial sums.
#[inline(always)]
fn combine(s: &[f32; L]) -> f32 {
    ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]))
}

/// `dst (row-major m × n) = a · b` for `1 ≤ m ≤ EXACT_ROWS`. The views must be in bounds
/// (checked by [`crate::matmul`]) and `k > 0`.
pub(crate) fn rows(
    m: usize,
    n: usize,
    k: usize,
    a: MatView<'_, f32>,
    b: MatView<'_, f32>,
    dst: &mut [f32],
) {
    debug_assert!((1..=EXACT_ROWS).contains(&m) && k > 0 && dst.len() == m * n);
    let at = |v: &MatView<'_, f32>, i: usize, j: usize| {
        v.data[(v.offset + i as isize * v.row_stride + j as isize * v.col_stride) as usize]
    };
    // The rows, contiguous (at most 16 · K elements).
    let x: Vec<f32> = (0..m)
        .flat_map(|i| (0..k).map(move |p| (i, p)))
        .map(|(i, p)| at(&a, i, p))
        .collect();
    // Outputs column by column (`t[j · m + i]`), so parallel tasks own disjoint slices.
    let mut t = vec![0.0f32; n * m];
    let parallel = n * k >= 1 << 16;
    if b.row_stride == 1 || k == 1 {
        let columns = |j: usize| {
            let start = (b.offset + j as isize * b.col_stride) as usize;
            &b.data[start..start + k]
        };
        run(&mut t, m, parallel, |j0, out| {
            along_k(m, k, &x, j0, out, &columns)
        });
    } else if b.col_stride == 1 || n == 1 {
        run(&mut t, m, parallel, |j0, out| {
            along_n(m, k, &x, j0, out, &b)
        });
    } else {
        // Any other layout: the columns copied contiguous along K.
        let copy: Vec<f32> = (0..n)
            .flat_map(|j| (0..k).map(move |p| (p, j)))
            .map(|(p, j)| at(&b, p, j))
            .collect();
        let columns = |j: usize| &copy[j * k..(j + 1) * k];
        run(&mut t, m, parallel, |j0, out| {
            along_k(m, k, &x, j0, out, &columns)
        });
    }
    for (j, column) in t.chunks_exact(m).enumerate() {
        for (i, &v) in column.iter().enumerate() {
            dst[i * n + j] = v;
        }
    }
}

/// Fills `t` (column-major outputs) by blocks of [`COLUMNS`] columns: `f(j0, block)`.
fn run(t: &mut [f32], m: usize, parallel: bool, f: impl Fn(usize, &mut [f32]) + Sync) {
    if parallel {
        t.par_chunks_mut(COLUMNS * m)
            .enumerate()
            .for_each(|(c, block)| f(c * COLUMNS, block));
    } else {
        for (c, block) in t.chunks_mut(COLUMNS * m).enumerate() {
            f(c * COLUMNS, block);
        }
    }
}

/// Columns whose weights stay in L1 while every block of rows passes over them.
const GROUP: usize = 4;

/// Weights contiguous along K (`[N, K]` used transposed): [`GROUP`] columns at a time, each
/// block of [`FEW_ROWS`] rows over them in turn.
fn along_k<'c>(
    m: usize,
    k: usize,
    x: &[f32],
    j0: usize,
    out: &mut [f32],
    columns: &(impl Fn(usize) -> &'c [f32] + Sync),
) {
    let width = out.len() / m;
    let mut c0 = 0;
    while c0 < width {
        let group = GROUP.min(width - c0);
        for r0 in (0..m).step_by(FEW_ROWS) {
            let at = Tile {
                k,
                x: &x[r0 * k..],
                m,
                r0,
                j: j0 + c0,
                c0,
                group,
            };
            macro_rules! rows {
                ($($b:literal => $c:literal),*) => {
                    match FEW_ROWS.min(m - r0) {
                        $($b => tile::<$b, $c>(&at, out, columns),)*
                        _ => unreachable!("1 to FEW_ROWS rows"),
                    }
                };
            }
            // Columns per pass: as many as keep the partial sums in registers.
            rows!(1 => 4, 2 => 4, 3 => 2, 4 => 2, 5 => 1, 6 => 1, 7 => 1, 8 => 1);
        }
        c0 += group;
    }
}

/// Where a [`tile`] reads and writes.
struct Tile<'x> {
    k: usize,
    /// The block's first row (`M` rows of `k`).
    x: &'x [f32],
    /// Rows of the whole product.
    m: usize,
    /// The block's first row in the product.
    r0: usize,
    /// The first column of the group in the weights.
    j: usize,
    /// The first column of the group in `out`.
    c0: usize,
    /// Columns in the group.
    group: usize,
}

/// `M` rows over a group of columns, `C` columns per pass.
fn tile<'c, const M: usize, const C: usize>(
    at: &Tile<'_>,
    out: &mut [f32],
    columns: &(impl Fn(usize) -> &'c [f32] + Sync),
) {
    let (k, x) = (at.k, at.x);
    let full = k / L * L;
    let mut c = 0;
    while c < at.group {
        let cols = C.min(at.group - c);
        let w: [&[f32]; C] = std::array::from_fn(|q| columns(at.j + c + q.min(cols - 1)));
        let mut s = [[[0.0f32; L]; M]; C];
        for q in (0..full).step_by(L) {
            let wq: [&[f32; L]; C] =
                std::array::from_fn(|v| w[v][q..q + L].try_into().expect("L elements"));
            for i in 0..M {
                let xq: &[f32; L] = x[i * k + q..i * k + q + L].try_into().expect("L elements");
                for v in 0..C {
                    fma(&mut s[v][i], xq, wq[v]);
                }
            }
        }
        for p in full..k {
            for i in 0..M {
                for v in 0..C {
                    s[v][i][p - full] = x[i * k + p].mul_add(w[v][p], s[v][i][p - full]);
                }
            }
        }
        for v in 0..cols {
            for i in 0..M {
                out[(at.c0 + c + v) * at.m + at.r0 + i] = combine(&s[v][i]);
            }
        }
        c += cols;
    }
}

/// Positions of K per pass in [`along_n`]: a block of weights that stays in L1.
const POSITIONS: usize = 64;

/// Weights contiguous along N (`[K, N]`): 8 columns at a time with 8 partials each, over K
/// in blocks of [`POSITIONS`] (the partials kept between blocks).
fn along_n(m: usize, k: usize, x: &[f32], j0: usize, out: &mut [f32], b: &MatView<'_, f32>) {
    let width = out.len() / m;
    let blocks = width.div_ceil(L);
    // `s[block · m + i][l][jj]`: partial `l` of row `i`, column `block · 8 + jj`.
    let mut s = vec![[[0.0f32; L]; L]; blocks * m];
    let at = |p: usize, c: usize| {
        (b.offset + p as isize * b.row_stride + (j0 + c) as isize * b.col_stride) as usize
    };
    for p0 in (0..k).step_by(POSITIONS) {
        let p1 = (p0 + POSITIONS).min(k);
        for block in 0..blocks {
            let (c, cw) = (block * L, L.min(width - block * L));
            for i in 0..m {
                // A local copy, so the partials live in registers during the block.
                let mut acc = s[block * m + i];
                if cw == L && b.col_stride == 1 {
                    let row = |p: usize| -> &[f32; L] {
                        b.data[at(p, c)..at(p, c) + L]
                            .try_into()
                            .expect("L elements")
                    };
                    // `p0` is a multiple of 8: eight positions at a time, each lane fixed.
                    let full = p0 + (p1 - p0) / L * L;
                    let xi = &x[i * k..(i + 1) * k];
                    for q in (p0..full).step_by(L) {
                        let xq: &[f32; L] = xi[q..q + L].try_into().expect("L elements");
                        let mut base = at(q, c) as isize;
                        for (l, lane) in acc.iter_mut().enumerate() {
                            let start = base as usize;
                            let bp: &[f32; L] =
                                b.data[start..start + L].try_into().expect("L elements");
                            fma(lane, &[xq[l]; L], bp);
                            base += b.row_stride;
                        }
                    }
                    for p in full..p1 {
                        fma(&mut acc[p % L], &[x[i * k + p]; L], row(p));
                    }
                } else {
                    for p in p0..p1 {
                        for (jj, lane) in acc[p % L].iter_mut().enumerate().take(cw) {
                            *lane = x[i * k + p].mul_add(b.data[at(p, c + jj)], *lane);
                        }
                    }
                }
                s[block * m + i] = acc;
            }
        }
    }
    for block in 0..blocks {
        for i in 0..m {
            let acc = &s[block * m + i];
            for jj in 0..L.min(width - block * L) {
                out[(block * L + jj) * m + i] = combine(&std::array::from_fn(|l| acc[l][jj]));
            }
        }
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    fn values(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.618).sin()).collect()
    }

    /// `b` (`k × n`) in three layouts: contiguous along K, along N, and neither (every
    /// other element of a wider buffer, read backwards along N).
    fn layouts(k: usize, n: usize, data: &[f32]) -> Vec<(Vec<f32>, isize, isize, isize)> {
        let along_k: Vec<f32> = (0..n)
            .flat_map(|j| (0..k).map(move |p| (p, j)))
            .map(|(p, j)| data[p * n + j])
            .collect();
        let mut sparse = vec![9.0f32; 2 * k * n + 1];
        for p in 0..k {
            for j in 0..n {
                sparse[2 * (p * n + (n - 1 - j))] = data[p * n + j];
            }
        }
        vec![
            (along_k, 0, 1, k as isize),
            (data.to_vec(), 0, n as isize, 1),
            (sparse, 2 * (n as isize - 1), 2 * n as isize, -2),
        ]
    }

    fn run(m: usize, n: usize, k: usize, x: &[f32], b: (&[f32], isize, isize, isize)) -> Vec<f32> {
        let a = MatView {
            data: x,
            offset: 0,
            row_stride: k as isize,
            col_stride: 1,
        };
        let b = MatView {
            data: b.0,
            offset: b.1,
            row_stride: b.2,
            col_stride: b.3,
        };
        let mut out = vec![0.0; m * n];
        rows(m, n, k, a, b, &mut out);
        out
    }

    #[test]
    fn rows_layouts_and_padding_keep_the_bits() {
        for &(k, n) in &[(1, 1), (7, 9), (8, 8), (9, 70), (33, 129), (1000, 77)] {
            let data = values(k * n, 1.0);
            let x = values(EXACT_ROWS * k, 2.0);
            let all: Vec<Vec<f32>> = layouts(k, n, &data)
                .iter()
                .map(|l| run(FEW_ROWS, n, k, &x[..FEW_ROWS * k], (&l.0, l.1, l.2, l.3)))
                .collect();
            // The same bits in every layout, and close to the textbook sum.
            for out in &all[1..] {
                assert_eq!(out, &all[0], "k {k} n {n}: layouts differ");
            }
            for i in 0..FEW_ROWS {
                for j in 0..n {
                    let want: f64 = (0..k)
                        .map(|p| f64::from(x[i * k + p]) * f64::from(data[p * n + j]))
                        .sum();
                    let got = all[0][i * n + j];
                    assert!((f64::from(got) - want).abs() <= 1e-5 * (k as f64).sqrt());
                }
            }
            // Each row of every M-row product equals the row alone.
            for m in (1..=FEW_ROWS).chain([9, EXACT_ROWS]) {
                for l in layouts(k, n, &data) {
                    let many = run(m, n, k, &x[..m * k], (&l.0, l.1, l.2, l.3));
                    for i in 0..m {
                        let one = run(1, n, k, &x[i * k..(i + 1) * k], (&l.0, l.1, l.2, l.3));
                        assert_eq!(
                            &many[i * n..(i + 1) * n],
                            &one[..],
                            "k {k} n {n} m {m} row {i}"
                        );
                    }
                }
            }
            // Zero terms appended to K (masked keys) keep the bits.
            let pad = 13;
            let kp = k + pad;
            let mut xp = vec![0.0f32; FEW_ROWS * kp];
            for i in 0..FEW_ROWS {
                xp[i * kp..i * kp + k].copy_from_slice(&x[i * k..(i + 1) * k]);
            }
            let mut dp = values(kp * n, 3.0);
            dp[..k * n].copy_from_slice(&data);
            for (l, padded) in layouts(kp, n, &dp).iter().enumerate() {
                let got = run(
                    FEW_ROWS,
                    n,
                    kp,
                    &xp,
                    (&padded.0, padded.1, padded.2, padded.3),
                );
                assert_eq!(
                    got, all[0],
                    "k {k} n {n} layout {l}: padding changed the bits"
                );
            }
        }
    }
}
