//! The ring backend across processes: each test re-runs this test binary as 2–4 ranks over
//! loopback (a generated hostfile, `MLX_RANK`), every rank checks its results, and the
//! parent checks that every rank exits cleanly and that the ranks' reported bits agree.

use std::net::TcpListener;
use std::time::{Duration, Instant};

use nuvec_core::distributed::{self, Group};
use nuvec_core::{Array, DType, Device, Stream, array, grad, set_default_stream, vmap};

mod common;

use common::{CHILD, agree, ranks};

/// Small integers per rank and element, exact in every dtype and in their sums.
fn values(rank: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + rank * 13) % 23) as f32 - 11.0)
        .collect()
}

fn input(rank: usize, shape: &[usize], dtype: DType) -> Array {
    let n = shape.iter().product();
    let v: Vec<f32> = match dtype {
        DType::Bool => (0..n)
            .map(|i| u8::from((i + rank).is_multiple_of(3)) as f32)
            .collect(),
        d if d.name().starts_with('u') => values(rank, n).iter().map(|x| x + 11.0).collect(),
        _ => values(rank, n),
    };
    let a = Array::from_slice(&v, shape);
    if dtype == DType::Bool {
        a.not_equal(0.0)
    } else {
        a.astype(dtype)
    }
}

fn as_f64(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

/// Every collective, dtype and shape class against the locally computed reference.
fn collectives(g: Group) {
    let (rank, size) = (g.rank(), g.size());
    let shapes: [&[usize]; 6] = [&[], &[0], &[1], &[3], &[7, 5], &[1001]];
    let dtypes = [
        DType::Bool,
        DType::U8,
        DType::U32,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for dtype in dtypes {
        for shape in shapes {
            let x = input(rank, shape, dtype);
            let all: Vec<Vec<f64>> = (0..size).map(|r| as_f64(&input(r, shape, dtype))).collect();
            let fold = |f: fn(f64, f64) -> f64| -> Vec<f64> {
                (0..all[0].len())
                    .map(|i| all.iter().map(|v| v[i]).reduce(f).unwrap())
                    .collect()
            };
            let name = format!("{} {shape:?}", dtype.name());
            let sum = distributed::all_sum(&x, None);
            let want_sum: Vec<f64> = match dtype {
                DType::Bool => fold(f64::max),
                _ => fold(|a, b| a + b),
            };
            assert_eq!(sum.shape(), x.shape(), "{name}");
            assert_eq!(sum.dtype(), dtype, "{name}");
            assert_eq!(as_f64(&sum), want_sum, "all_sum {name}");
            assert_eq!(
                as_f64(&distributed::all_max(&x, None)),
                fold(f64::max),
                "all_max {name}"
            );
            assert_eq!(
                as_f64(&distributed::all_min(&x, None)),
                fold(f64::min),
                "all_min {name}"
            );
            let gathered = distributed::all_gather(&x, None);
            let want: Vec<f64> = all.concat();
            assert_eq!(as_f64(&gathered), want, "all_gather {name}");
            let mut shape_g = shape.to_vec();
            match shape_g.first_mut() {
                Some(d) => *d *= size,
                None => shape_g.push(size),
            }
            let dims: Vec<i32> = shape_g.iter().map(|&d| d as i32).collect();
            assert_eq!(gathered.shape(), dims.as_slice(), "all_gather shape {name}");
        }
    }
}

#[test]
fn collectives_on_two_ranks() {
    ranks("collectives_on_two_ranks", 2, collectives);
}

#[test]
fn collectives_on_three_ranks() {
    ranks("collectives_on_three_ranks", 3, collectives);
}

#[test]
fn collectives_on_four_ranks() {
    ranks("collectives_on_four_ranks", 4, collectives);
}

/// Random float32 sums (not exact): every rank gets the same bits, and a rerun repeats them.
fn floats(g: Group) {
    let key = Array::from_slice(&[0u32, g.rank() as u32 + 7], [2]);
    let x = nuvec_core::random::normal([4099], DType::F32, 0.0, 1.0, &key);
    for _ in 0..2 {
        let s = distributed::all_sum(&x, None);
        let bits: u64 = s
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .fold(0u64, |h, v| h.rotate_left(5) ^ u64::from(v.to_bits()));
        println!("BITS {bits:016x}");
    }
}

#[test]
fn float_sums_agree_bit_for_bit() {
    for n in [2, 3, 4] {
        if let Some(outputs) = ranks("float_sums_agree_bit_for_bit", n, floats) {
            agree(&outputs);
        } else {
            return;
        }
    }
}

/// 64 MB per rank: split over the connection and the ring's segments.
fn large(g: Group) {
    let n = 16 << 20;
    let x = nuvec_core::ops::full([n], g.rank() as f32 + 0.5);
    let s = distributed::all_sum(&x, None);
    let want = (0..g.size()).map(|r| r as f32 + 0.5).sum::<f32>();
    let v = s.to_vec::<f32>().unwrap();
    assert!(v.iter().all(|&x| x == want), "{}", v[0]);
}

#[test]
fn large_arrays_on_two_and_four_ranks() {
    for n in [2, 4] {
        if ranks("large_arrays_on_two_and_four_ranks", n, large).is_none() {
            return;
        }
    }
}

/// Point to point between neighbours; a mismatched receive and a non-neighbour fail.
fn point_to_point(g: Group) {
    let (rank, size) = (g.rank(), g.size());
    let (right, left) = ((rank + 1) % size, (rank + size - 1) % size);
    let x = Array::from_slice(&values(rank, 6), [2, 3]);
    // Even ranks send first; odd ranks receive first.
    let sent = || distributed::send(&x, right, None).eval().unwrap();
    let got = if rank % 2 == 0 {
        sent();
        distributed::recv([2, 3], DType::F32, left, None)
    } else {
        let r = distributed::recv_like(&x, left, None);
        r.eval().unwrap();
        sent();
        r
    };
    assert_eq!(got.to_vec::<f32>().unwrap(), values(left, 6));
    // A receive of another shape is an error, not reinterpreted data.
    if rank == 0 {
        distributed::send(&x, 1, None).eval().unwrap();
    } else if rank == 1 {
        let wrong = distributed::recv([3, 2], DType::F32, 0, None);
        let err = wrong.eval().unwrap_err().to_string();
        assert!(err.contains("shape"), "{err}");
    }
    if size == 4 {
        let err = distributed::send(&x, (rank + 2) % 4, None)
            .eval()
            .unwrap_err()
            .to_string();
        assert!(err.contains("neighbours"), "{err}");
    }
    assert!(distributed::send(&x, rank, None).eval().is_err());
}

#[test]
fn send_and_recv_between_neighbours() {
    for n in [2, 4] {
        if ranks("send_and_recv_between_neighbours", n, point_to_point).is_none() {
            return;
        }
    }
}

/// Gradients follow MLX's rules, compile replays communicate again, `vmap` batches, and
/// GPU-produced inputs give the CPU's results.
fn transforms(g: Group) {
    let (rank, size) = (g.rank(), g.size());
    let x = Array::from_slice(&values(rank, 4), [4]);
    let total: Vec<f32> = (0..4)
        .map(|i| (0..size).map(|r| values(r, 4)[i]).sum())
        .collect();
    // loss = sum(all_sum(x)²): the cotangent passes through, so d loss / dx = 2 · all_sum(x).
    let loss = |x: &Array| {
        let s = distributed::all_sum(x, None);
        s.multiply(&s).sum(..)
    };
    let dx = grad(loss)(&x).unwrap();
    let want: Vec<f32> = total.iter().map(|t| 2.0 * t).collect();
    assert_eq!(dx.to_vec::<f32>().unwrap(), want);
    // Finite differences of rank 0's loss in its own input: only rank 0 moves its input,
    // the others hold theirs (every rank runs the same collectives).
    let h = 1e-2f32;
    let mut e = vec![0.0f32; 4];
    if rank == 0 {
        e[1] = h;
    }
    let plus = loss(&x.add(Array::from_slice(&e, [4])))
        .item::<f32>()
        .unwrap();
    let minus = loss(&x.subtract(Array::from_slice(&e, [4])))
        .item::<f32>()
        .unwrap();
    if rank == 0 {
        assert!(((plus - minus) / (2.0 * h) - want[1]).abs() < 1e-2);
    }
    // all_gather's vjp: this rank's rows of the cotangent.
    let w = nuvec_core::ops::arange(0, 4 * size as i32, 1).astype(DType::F32);
    let gathered_loss = |x: &Array| distributed::all_gather(x, None).multiply(&w).sum(..);
    let dg = grad(gathered_loss)(&x).unwrap().to_vec::<f32>().unwrap();
    let rows: Vec<f32> = (4 * rank..4 * rank + 4).map(|i| i as f32).collect();
    assert_eq!(dg, rows);
    // Compiled: equal to the uncompiled call, every call.
    let f = nuvec_compile::compile(|x: &Array| distributed::all_sum(x, None).multiply(2.0));
    for k in 0..3 {
        let y = x.add(k as f32);
        let want = distributed::all_sum(&y, None).multiply(2.0);
        assert_eq!(
            f(&y).unwrap().to_vec::<f32>().unwrap(),
            want.to_vec::<f32>().unwrap()
        );
    }
    // vmap over a batch of rows: each row reduced on its own.
    let batch = Array::from_slice(&values(rank, 8), [2, 4]);
    let batched = vmap(|r: &Array| distributed::all_sum(r, None))
        .call(&batch)
        .unwrap();
    assert_eq!(
        batched.to_vec::<f32>().unwrap(),
        distributed::all_sum(&batch, None).to_vec::<f32>().unwrap()
    );
    // Inputs produced on the GPU.
    if cfg!(target_os = "macos") {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        if (&Array::from(1.0f32) + 1.0).eval().is_ok() {
            let y = x.multiply(3.0);
            let s = distributed::all_sum(&y, None).add(1.0);
            let want: Vec<f32> = total.iter().map(|t| 3.0 * t + 1.0).collect();
            assert_eq!(s.to_vec::<f32>().unwrap(), want);
        }
        set_default_stream(Stream::default_for(Device::cpu()));
    }
}

#[test]
fn collectives_compose_with_transforms() {
    for n in [2, 3] {
        if ranks("collectives_compose_with_transforms", n, transforms).is_none() {
            return;
        }
    }
}

/// A peer that exits: the others fail with an error naming a rank, within the timeout, and
/// every later collective fails at once.
fn peer_exits(g: Group) {
    if g.rank() == g.size() - 1 {
        std::process::exit(0);
    }
    let x = array![1.0, 2.0];
    let start = Instant::now();
    let err = distributed::all_sum(&x, None)
        .eval()
        .unwrap_err()
        .to_string();
    assert!(err.contains("rank"), "{err}");
    assert!(start.elapsed() < Duration::from_secs(25));
    // The ring is broken from then on: the next collective fails at once (it would read
    // whatever the failed transfer left in the sockets).
    let start = Instant::now();
    let err = distributed::all_sum(&x, None)
        .eval()
        .unwrap_err()
        .to_string();
    assert!(err.contains("failed earlier"), "{err}");
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn a_peer_that_exits_fails_the_others() {
    ranks("a_peer_that_exits_fails_the_others", 3, peer_exits);
}

/// Without a hostfile the group is this process alone, and every collective the identity.
#[test]
fn a_single_process_runs_unchanged() {
    if std::env::var(CHILD).is_ok() {
        return;
    }
    let g = nuvec_distributed::init(false).unwrap();
    assert_eq!((g.rank(), g.size()), (0, 1));
    assert!(nuvec_distributed::init(true).is_err());
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    for y in [
        distributed::all_sum(&x, None),
        distributed::all_max(&x, None),
        distributed::all_gather(&x, None),
    ] {
        assert_eq!(y.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    }
    assert!(distributed::send(&x, 1, None).eval().is_err());
}

/// A rank that never starts: `init` fails naming it, within the timeout.
#[test]
fn a_rank_that_never_starts_fails_init() {
    if std::env::var(CHILD).is_ok() {
        return;
    }
    let ports: Vec<u16> = (0..2)
        .map(|_| {
            TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        })
        .collect();
    let nodes: Vec<Vec<String>> = ports
        .iter()
        .map(|p| vec![format!("127.0.0.1:{p}")])
        .collect();
    let options = nuvec_distributed::Options::default().timeout(Duration::from_secs(1));
    let start = Instant::now();
    let err = nuvec_distributed::init_with(&nodes, 0, &options)
        .unwrap_err()
        .to_string();
    assert!(err.contains("rank 1"), "{err}");
    assert!(start.elapsed() < Duration::from_secs(10));
}
