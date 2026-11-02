//! The JACCL collectives on real Thunderbolt 5 hardware, one process per Mac, started by
//! `mlx.launch --backend jaccl` (or `jaccl-ring`):
//!
//! ```sh
//! mlx.launch --backend jaccl --hostfile hosts.json -- \
//!     cargo test --release -p nuvec-distributed --features jaccl --test jaccl_hardware
//! ```
//!
//! Every collective, dtype and size class against the reference; the ranks then all-gather a
//! checksum of their results, which must agree. Skipped without MLX's JACCL environment.

#![cfg(feature = "jaccl")]

use nuvec_core::distributed::{self, Group};
use nuvec_core::{Array, DType};
use nuvec_distributed::{Backend, init_backend};

/// Small integers per rank and element, exact in every dtype and in their sums.
fn values(rank: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + rank * 13) % 23) as f32 - 11.0)
        .collect()
}

fn as_f64(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

fn check(g: &Group) -> u64 {
    let (rank, size) = (g.rank(), g.size());
    let mut digest = 0u64;
    let mut mix = |bits: &[f64]| {
        for b in bits {
            digest = digest.rotate_left(5) ^ b.to_bits();
        }
    };
    for dtype in [DType::I32, DType::F16, DType::BF16, DType::F32, DType::F64] {
        // Around 32 KB, 512 KB and every frame size, and 64 MB.
        let mut counts = vec![0usize, 1, size - 1, 1001, (16 << 20) / dtype.size()];
        for bytes in [32 << 10, 512 << 10] {
            counts.extend([bytes / dtype.size() - 1, bytes / dtype.size() + 1]);
        }
        counts.extend((0..8).map(|k| (4096 << k) / dtype.size() + 3));
        for n in counts {
            let x = |r| Array::from_slice(&values(r, n), [n]).astype(dtype);
            let all: Vec<Vec<f64>> = (0..size).map(|r| as_f64(&x(r))).collect();
            let fold = |f: fn(f64, f64) -> f64| -> Vec<f64> {
                (0..n)
                    .map(|i| all.iter().map(|v| v[i]).reduce(f).unwrap())
                    .collect()
            };
            let name = format!("{} × {n}", dtype.name());
            let sum = as_f64(&distributed::all_sum(&x(rank), Some(g)));
            assert_eq!(sum, fold(|a, b| a + b), "all_sum {name}");
            assert_eq!(
                as_f64(&distributed::all_max(&x(rank), Some(g))),
                fold(f64::max),
                "all_max {name}"
            );
            assert_eq!(
                as_f64(&distributed::all_min(&x(rank), Some(g))),
                fold(f64::min),
                "all_min {name}"
            );
            let gathered = as_f64(&distributed::all_gather(&x(rank), Some(g)));
            assert_eq!(gathered, all.concat(), "all_gather {name}");
            let k = n.div_ceil(size);
            let y = |r| Array::from_slice(&values(r, k * size), [k * size]).astype(dtype);
            let ys: Vec<Vec<f64>> = (0..size).map(|r| as_f64(&y(r))).collect();
            let want: Vec<f64> = (rank * k..(rank + 1) * k)
                .map(|i| ys.iter().map(|v| v[i]).sum())
                .collect();
            assert_eq!(
                as_f64(&distributed::sum_scatter(&y(rank), Some(g))),
                want,
                "sum_scatter {name}"
            );
            mix(&sum);
            mix(&gathered);
        }
    }
    // Point to point between neighbours (both topologies connect them).
    let right = (rank + 1) % size;
    let left = (rank + size - 1) % size;
    for n in [1usize, 100_000] {
        let x = Array::from_slice(&values(rank, n), [n]);
        let (sent, got) = if rank % 2 == 0 {
            let s = distributed::send(&x, right, Some(g));
            s.eval().unwrap();
            (s, distributed::recv([n], DType::F32, left, Some(g)))
        } else {
            let r = distributed::recv([n], DType::F32, left, Some(g));
            r.eval().unwrap();
            (distributed::send(&x, right, Some(g)), r)
        };
        sent.eval().unwrap();
        assert_eq!(got.to_vec::<f32>().unwrap(), values(left, n));
    }
    digest
}

#[test]
fn collectives_over_thunderbolt() {
    if std::env::var("MLX_IBV_DEVICES").is_err() && std::env::var("JACCL_IBV_DEVICES").is_err() {
        eprintln!("skipped: no JACCL environment (run under mlx.launch --backend jaccl)");
        return;
    }
    let g = init_backend(true, Backend::Jaccl).expect("the JACCL group");
    let digest = check(&g);
    let bits = f64::from_bits(digest);
    let all = distributed::all_gather(&Array::from_slice(&[bits], [1]), Some(&g))
        .to_vec::<f64>()
        .unwrap();
    assert!(
        all.iter().all(|d| d.to_bits() == digest),
        "the ranks' results differ"
    );
    eprintln!(
        "rank {} of {}: every collective matches",
        g.rank(),
        g.size()
    );
}
