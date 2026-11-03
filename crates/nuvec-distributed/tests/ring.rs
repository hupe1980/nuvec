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

/// Max and min of signed zeros and NaNs, where the order of the operands decides the bits:
/// every rank still gets the same ones.
fn signed_zeros(g: Group) {
    let r = g.rank() as f32;
    let v = [
        0.0 * (1.0 - 2.0 * (r % 2.0)),
        f32::NAN * r,
        1.0 - r,
        -0.0 + r,
    ];
    let x = Array::from_slice(&v, [4]);
    for s in [
        distributed::all_max(&x, None),
        distributed::all_min(&x, None),
    ] {
        let bits: Vec<u32> = s
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        println!("BITS {bits:x?}");
    }
}

#[test]
fn max_and_min_agree_bit_for_bit() {
    for n in [2, 3] {
        if let Some(outputs) = ranks("max_and_min_agree_bit_for_bit", n, signed_zeros) {
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

/// Split all-reduces (both ring directions over every connection) of 256 KB to 64 MB with
/// a different value per element, exact against the reference: one part (256 KB), parts in
/// pairs, and on two ranks over two connections a pair and one part alone (1.5 MB).
fn split(g: Group) {
    let (rank, size) = (g.rank() as f32, g.size() as f32);
    for kb in [256usize, 1024, 1536, 4096, 65536] {
        let n = (kb << 10) / 4;
        let v: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 + rank).collect();
        let x = Array::from_slice(&v, [n]).multiply(1.0);
        let v = distributed::all_sum(&x, None).to_vec::<f32>().unwrap();
        let ranks: f32 = (0..g.size()).map(|r| r as f32).sum();
        for (i, got) in v.iter().enumerate() {
            let want = size * (i % 1000) as f32 + ranks;
            assert_eq!(*got, want, "{kb} KB, element {i}");
        }
    }
}

#[test]
fn split_all_reduces_over_one_and_two_connections() {
    for (n, connections) in [(2, 1), (2, 2), (4, 1), (4, 2)] {
        let file = std::env::temp_dir().join(format!(
            "nuvec-split-{}-{n}-{connections}.json",
            std::process::id()
        ));
        let entries: Vec<String> = (0..n)
            .map(|_| {
                let a: Vec<String> = common::free_addresses(connections)
                    .iter()
                    .map(|a| format!("\"{a}\""))
                    .collect();
                format!("[{}]", a.join(", "))
            })
            .collect();
        std::fs::write(&file, format!("[{}]", entries.join(", "))).unwrap();
        let hostfile = file.display().to_string();
        let env = [("MLX_HOSTFILE", hostfile.as_str())];
        if common::ranks_with(
            "split_all_reduces_over_one_and_two_connections",
            n,
            &env,
            split,
        )
        .is_none()
        {
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

/// `sum_scatter` for every dtype and shape class, on contiguous, strided and offset inputs,
/// with its gradient (each rank's loss is its own), jvp, vmap and compile.
fn scatter(g: Group) {
    let (rank, size) = (g.rank(), g.size());
    let dtypes = [
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
        for (k, rest) in [(0usize, &[][..]), (1, &[]), (3, &[]), (2, &[5, 3])] {
            let mut shape = vec![size * k];
            shape.extend_from_slice(rest);
            let x = input(rank, &shape, dtype);
            let all: Vec<Vec<f64>> = (0..size)
                .map(|r| as_f64(&input(r, &shape, dtype)))
                .collect();
            let per: usize = rest.iter().product::<usize>() * k;
            let want: Vec<f64> = (rank * per..(rank + 1) * per)
                .map(|i| all.iter().map(|v| v[i]).sum())
                .collect();
            let y = distributed::sum_scatter(&x, None);
            let name = format!("{} {shape:?}", dtype.name());
            let mut out_shape = vec![k as i32];
            out_shape.extend(rest.iter().map(|&d| d as i32));
            assert_eq!(y.shape(), out_shape.as_slice(), "sum_scatter shape {name}");
            assert_eq!(as_f64(&y), want, "sum_scatter {name}");
        }
    }
    // A transposed view and a slice at an offset scatter as their contiguous copies.
    let base = Array::from_slice(&values(rank, 6 * size * 4), [4, 6 * size]);
    for view in [
        base.transpose([1, 0]),
        base.transpose([1, 0]).i(size as i32..(3 * size) as i32),
    ] {
        let dense = view.contiguous();
        assert_eq!(
            distributed::sum_scatter(&view, None)
                .to_vec::<f32>()
                .unwrap(),
            distributed::sum_scatter(&dense, None)
                .to_vec::<f32>()
                .unwrap()
        );
    }
    // The shape errors (MLX's): a scalar, a first axis not divisible by the size.
    let e = distributed::sum_scatter(&Array::from(1.0f32), None)
        .check()
        .unwrap_err();
    assert!(e.to_string().contains("first axis"), "{e}");
    let odd = Array::from_slice(&vec![1.0f32; size + 1], [size + 1]);
    assert!(distributed::sum_scatter(&odd, None).check().is_err());
    // Gradient: d sum(sum_scatter(x)·w) / dx is w in this rank's rows, zero elsewhere.
    let n = 2 * size;
    let x = Array::from_slice(&values(rank, n), [n]);
    let w = Array::from_slice(&[3.0f32, -1.0], [2]);
    let loss = |x: &Array| distributed::sum_scatter(x, None).multiply(&w).sum(..);
    let dx = grad(loss)(&x).unwrap().to_vec::<f32>().unwrap();
    let mut want = vec![0.0f32; n];
    want[2 * rank] = 3.0;
    want[2 * rank + 1] = -1.0;
    assert_eq!(dx, want);
    // Finite differences of each rank's loss in its own input (every rank moves element 0
    // of rank 0's rows in turn; only the owner's loss changes).
    let h = 1e-2f32;
    for i in 0..n {
        let mut e = vec![0.0f32; n];
        e[i] = h;
        let e = Array::from_slice(&e, [n]);
        let fd = (loss(&x.add(&e)).item::<f32>().unwrap()
            - loss(&x.subtract(&e)).item::<f32>().unwrap())
            / (2.0 * h);
        // Every rank perturbed element i together, so the loss moved by the sum of the
        // ranks' own derivatives in row i: size·want only within this rank's rows.
        let expected = if i / 2 == rank {
            want[i] * size as f32
        } else {
            0.0
        };
        assert!(
            (fd - expected).abs() < 2e-2,
            "finite difference {i}: {fd} vs {expected}"
        );
    }
    // jvp: the op on the tangent.
    let t = Array::from_slice(&values(rank + 1, n), [n]);
    let (_, jt) = nuvec_core::jvp(|x: &Array| distributed::sum_scatter(x, None), &x, &t).unwrap();
    assert_eq!(
        jt.to_vec::<f32>().unwrap(),
        distributed::sum_scatter(&t, None).to_vec::<f32>().unwrap()
    );
    // vmap over axis 0 and 1: each example scattered on its own.
    let batch = Array::from_slice(&values(rank, 3 * n), [3, n]);
    let batched = vmap(|r: &Array| distributed::sum_scatter(r, None))
        .call(&batch)
        .unwrap();
    let looped: Vec<f32> = (0..3)
        .flat_map(|b| {
            distributed::sum_scatter(&batch.i(b), None)
                .to_vec::<f32>()
                .unwrap()
        })
        .collect();
    assert_eq!(batched.to_vec::<f32>().unwrap(), looped);
    let cols = batch.transpose([1, 0]).contiguous();
    let along = vmap(|r: &Array| distributed::sum_scatter(r, None))
        .in_axes(1)
        .call(&cols)
        .unwrap();
    assert_eq!(along.to_vec::<f32>().unwrap(), looped);
    // Compiled: equal to the uncompiled call.
    let f = nuvec_compile::compile(|x: &Array| distributed::sum_scatter(x, None).multiply(2.0));
    assert_eq!(
        f(&x).unwrap().to_vec::<f32>().unwrap(),
        distributed::sum_scatter(&x, None)
            .multiply(2.0)
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
fn sum_scatter_on_two_three_and_four_ranks() {
    for n in [2, 3, 4] {
        if ranks("sum_scatter_on_two_three_and_four_ranks", n, scatter).is_none() {
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

/// Whether the GPU evaluates here; sets the GPU stream if so.
fn on_gpu() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    (&Array::from(1.0f32) + 1.0).eval().is_ok()
}

fn bits_of(a: &Array) -> Vec<u32> {
    a.astype(DType::F32)
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Collectives on the GPU stream (non-blocking, behind GPU work) and on the CPU stream
/// (blocking) give the same bits, on GPU-produced, transposed and half-precision inputs.
fn streams(g: Group) {
    let key = Array::from_slice(&[1u32, g.rank() as u32], [2]);
    let x = nuvec_core::random::normal([24, 65], DType::F32, 0.0, 1.0, &key);
    x.eval().unwrap();
    let graph = |x: &Array| -> Vec<Array> {
        let t = x.multiply(1.5).transpose([1, 0]);
        vec![
            distributed::all_sum(&t, None),
            distributed::all_max(&x.astype(DType::BF16), None),
            distributed::all_min(&x.subtract(0.25), None),
            distributed::all_gather(&t.add(1.0), None),
            distributed::sum_scatter(&x.astype(DType::F16).multiply(2.0), None),
        ]
    };
    let mut gpu = Vec::new();
    if on_gpu() {
        gpu = graph(&x).iter().map(bits_of).collect();
    }
    set_default_stream(Stream::default_for(Device::cpu()));
    let cpu: Vec<Vec<u32>> = graph(&x).iter().map(bits_of).collect();
    if !gpu.is_empty() {
        assert_eq!(gpu, cpu);
    }
    // `sum_scatter` leaves each rank its own segment; the others are the same everywhere.
    let replicated = cpu[..4].concat();
    // On a line of its own: the harness's "test ... " line may still be open.
    println!();
    println!(
        "BITS {:x}",
        replicated
            .iter()
            .fold(0u64, |h, &b| h.rotate_left(5) ^ u64::from(b))
    );
}

#[test]
fn gpu_and_cpu_streams_agree_bit_for_bit() {
    for n in [2, 3, 4] {
        if let Some(outputs) = ranks("gpu_and_cpu_streams_agree_bit_for_bit", n, streams) {
            agree(&outputs);
        } else {
            return;
        }
    }
}

/// Decode's shape: 56 GPU ops each followed by an all_sum in one evaluation, three times,
/// equal to the CPU stream's; then two such graphs evaluated with `async_eval` before either
/// is read; then 600 collectives in one graph.
fn decode(g: Group) {
    let x = nuvec_core::ops::full([1, 1024], g.rank() as f32 + 0.5).astype(DType::BF16);
    let chain = |x: &Array, n: usize| {
        let mut y = x.clone();
        for _ in 0..n {
            y = distributed::all_sum(&y.multiply(0.5), None);
        }
        y
    };
    if !on_gpu() {
        return;
    }
    let gpu: Vec<Vec<u32>> = (0..3).map(|_| bits_of(&chain(&x, 56))).collect();
    let (a, b) = (chain(&x, 20), chain(&x.add(1.0), 20));
    nuvec_core::async_eval(&a).unwrap();
    nuvec_core::async_eval(&b).unwrap();
    let (a, b) = (bits_of(&a), bits_of(&b));
    let long = bits_of(&chain(&x, 600));
    set_default_stream(Stream::default_for(Device::cpu()));
    let cpu = bits_of(&chain(&x, 56));
    assert!(gpu.iter().all(|v| *v == cpu));
    assert_eq!(a, bits_of(&chain(&x, 20)));
    assert_eq!(b, bits_of(&chain(&x.add(1.0), 20)));
    assert_eq!(long, bits_of(&chain(&x, 600)));
}

#[test]
fn decode_shaped_graphs_match_the_cpu_stream() {
    for n in [2, 3] {
        if ranks("decode_shaped_graphs_match_the_cpu_stream", n, decode).is_none() {
            return;
        }
    }
}

/// The blocking path: every collective on the CPU stream.
fn on_cpu(g: Group) {
    set_default_stream(Stream::default_for(Device::cpu()));
    collectives(g);
}

#[test]
fn collectives_on_the_cpu_stream() {
    for n in [2, 3] {
        if ranks("collectives_on_the_cpu_stream", n, on_cpu).is_none() {
            return;
        }
    }
}
