//! The JACCL algorithms on the emulated fabric: 2–8 ranks as threads, every collective and
//! dtype, sizes around every threshold, both topologies and several cables, against the
//! reference reduction; every rank must agree bit for bit. Faults must fail every rank within
//! the timeout, naming the peer.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nuvec_core::distributed::{AllReduce, Communicator};
use nuvec_core::{DType, bf16, f16};

use super::config::Config;
use super::emulated::{Emulated, Fault};
use super::group::Jaccl;
use super::{Options, connect};

/// An in-process all-gather for the side channel: every rank's call blocks until all ranks
/// have contributed.
/// Contributions of the current round, the round number, the last round's result.
type Round = (Vec<Option<Vec<u8>>>, u64, Vec<u8>);

#[derive(Default)]
struct Gather {
    state: Mutex<Round>,
    ready: Condvar,
}

impl Gather {
    fn all_gather(&self, rank: usize, size: usize, data: &[u8]) -> Vec<u8> {
        let mut s = self.state.lock().unwrap();
        if s.0.len() != size {
            s.0 = vec![None; size];
        }
        s.0[rank] = Some(data.to_vec());
        let round = s.1;
        if s.0.iter().all(Option::is_some) {
            s.2 = s.0.iter_mut().flat_map(|c| c.take().unwrap()).collect();
            s.1 += 1;
            self.ready.notify_all();
        } else {
            while s.1 == round {
                s = self.ready.wait(s).unwrap();
            }
        }
        s.2.clone()
    }
}

fn mesh(n: usize, wires: usize) -> Vec<Vec<Vec<String>>> {
    (0..n)
        .map(|s| {
            (0..n)
                .map(|d| {
                    if s == d {
                        vec![]
                    } else {
                        (0..wires).map(|w| format!("rdma_en{}", 2 + w)).collect()
                    }
                })
                .collect()
        })
        .collect()
}

fn ring(n: usize, cables: usize) -> Vec<Vec<Vec<String>>> {
    (0..n)
        .map(|s| {
            (0..n)
                .map(|d| {
                    if d == (s + 1) % n || d == (s + n - 1) % n {
                        (0..cables).map(|w| format!("rdma_en{}", 2 + w)).collect()
                    } else {
                        vec![]
                    }
                })
                .collect()
        })
        .collect()
}

struct Setup {
    devices: Vec<Vec<Vec<String>>>,
    prefer_ring: bool,
    large: bool,
    faults: Vec<Fault>,
    timeout: Duration,
}

impl Setup {
    fn new(devices: Vec<Vec<Vec<String>>>) -> Self {
        Self {
            devices,
            prefer_ring: false,
            large: true,
            faults: Vec::new(),
            timeout: Duration::from_secs(20),
        }
    }

    /// Connects every rank and runs `body` on each, on its own thread; returns each rank's
    /// result.
    fn run<T: Send>(&self, body: impl Fn(usize, &Jaccl) -> T + Sync) -> Vec<T> {
        let n = self.devices.len();
        let fabric = Emulated::new(self.faults.clone());
        let gather = Arc::new(Gather::default());
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n)
                .map(|rank| {
                    let (fabric, gather, body) = (fabric.clone(), gather.clone(), &body);
                    s.spawn(move || {
                        let config = Config::new(rank, self.devices.clone())
                            .unwrap()
                            .prefer_ring(self.prefer_ring);
                        let options = Options::default()
                            .timeout(self.timeout)
                            .side_channel(move |d| Ok(gather.all_gather(rank, n, d)));
                        let comm =
                            connect(&config, &fabric.node(rank), &options, self.large).unwrap();
                        body(rank, &comm)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }
}

/// Small integers per rank and element: exact in every dtype and in their sums over 8 ranks.
fn value(dtype: DType, rank: usize, i: usize) -> f64 {
    let v = ((i * 7 + rank * 13) % 23) as f64 - 11.0;
    match dtype {
        DType::Bool => f64::from(u8::from((i + rank).is_multiple_of(3))),
        d if d.name().starts_with('u') => v + 11.0,
        _ => v,
    }
}

fn encode(dtype: DType, values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DType::Bool => vec![u8::from(v != 0.0)],
            DType::U8 => vec![v as u8],
            DType::U32 => (v as u32).to_le_bytes().to_vec(),
            DType::I32 => (v as i32).to_le_bytes().to_vec(),
            DType::I64 => (v as i64).to_le_bytes().to_vec(),
            DType::F16 => f16::from_f64(v).to_bits().to_le_bytes().to_vec(),
            DType::BF16 => bf16::from_f64(v).to_bits().to_le_bytes().to_vec(),
            DType::F32 => (v as f32).to_le_bytes().to_vec(),
            DType::F64 => v.to_le_bytes().to_vec(),
            // A complex value with real part v and imaginary part 2v.
            DType::Complex64 => [(v as f32).to_le_bytes(), (2.0 * v as f32).to_le_bytes()].concat(),
            d => panic!("{d:?}"),
        })
        .collect()
}

fn input(dtype: DType, rank: usize, count: usize) -> Vec<u8> {
    encode(
        dtype,
        &(0..count)
            .map(|i| value(dtype, rank, i))
            .collect::<Vec<_>>(),
    )
}

/// The reduction over `size` ranks of element `i`.
fn reference(op: AllReduce, dtype: DType, size: usize, i: usize) -> f64 {
    let vals = (0..size).map(|r| value(dtype, r, i));
    match (op, dtype) {
        (AllReduce::Sum, DType::Bool) => f64::from(u8::from(vals.sum::<f64>() > 0.0)),
        (AllReduce::Sum, _) => vals.sum(),
        (AllReduce::Max, _) => vals.fold(f64::MIN, f64::max),
        (_, _) => vals.fold(f64::MAX, f64::min),
    }
}

const DTYPES: [DType; 10] = [
    DType::Bool,
    DType::U8,
    DType::U32,
    DType::I32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::Complex64,
];

/// Element counts around every threshold of the algorithms for `dtype` with `size` ranks.
fn counts(dtype: DType, size: usize, full: bool) -> Vec<usize> {
    let e = dtype.size();
    let mut c = vec![0, 1, size - 1, size, 7, 1001];
    if full {
        for bytes in [32 << 10, 64 << 10, 512 << 10] {
            c.extend([bytes / e - 1, bytes / e, bytes / e + 1]);
        }
        for k in 0..8 {
            c.push((4096 << k) / e + 3);
        }
        // Several pieces per frame on every cable and direction (the ring's large path).
        c.push((32 << 10) * size * 2 / e + 5);
    }
    c
}

/// Runs every collective for `dtypes` on every rank and checks each against the reference
/// and the ranks against each other.
fn collectives(setup: &Setup, dtypes: &[DType], full: bool) {
    let n = setup.devices.len();
    let results = setup.run(|rank, comm| {
        let mut out = Vec::new();
        for &dtype in dtypes {
            for count in counts(
                dtype,
                n,
                full && dtype == DType::F32 || dtype == DType::BF16 && full,
            ) {
                let ops: &[AllReduce] = if dtype == DType::Complex64 {
                    &[AllReduce::Sum]
                } else {
                    &[AllReduce::Sum, AllReduce::Max, AllReduce::Min]
                };
                for &op in ops {
                    let mut data = input(dtype, rank, count);
                    comm.all_reduce(op, dtype, &mut data).unwrap();
                    let want: Vec<f64> = (0..count).map(|i| reference(op, dtype, n, i)).collect();
                    let want = encode(dtype, &want);
                    assert!(
                        data == want,
                        "all_reduce {op:?} {dtype:?} × {count} on rank {rank}"
                    );
                    out.push(data);
                }
                let x = input(dtype, rank, count);
                let mut gathered = vec![0u8; x.len() * n];
                comm.all_gather(&x, &mut gathered).unwrap();
                let want: Vec<u8> = (0..n).flat_map(|r| input(dtype, r, count)).collect();
                assert!(
                    gathered == want,
                    "all_gather {dtype:?} × {count} on rank {rank}"
                );
                if dtype != DType::Bool {
                    // Rank r's rows r·k..(r+1)·k of every rank's n·k elements, summed.
                    let k = count;
                    let x = input(dtype, rank, n * k);
                    let mut part = vec![0u8; x.len() / n];
                    comm.sum_scatter(dtype, &x, &mut part).unwrap();
                    let want: Vec<f64> = (rank * k..(rank + 1) * k)
                        .map(|i| reference(AllReduce::Sum, dtype, n, i))
                        .collect();
                    assert!(
                        part == encode(dtype, &want),
                        "sum_scatter {dtype:?} × {k} on rank {rank}"
                    );
                }
            }
        }
        out
    });
    assert!(
        results.iter().all(|r| r == &results[0]),
        "the ranks disagree"
    );
}

#[test]
fn mesh_collectives_match_the_reference() {
    for (n, wires) in [(2, 1), (3, 1), (4, 2)] {
        collectives(&Setup::new(mesh(n, wires)), &DTYPES, false);
    }
    collectives(&Setup::new(mesh(8, 1)), &[DType::F32, DType::I64], false);
}

#[test]
fn mesh_thresholds_and_large_messages() {
    for (n, wires) in [(2, 2), (3, 1), (4, 2)] {
        collectives(
            &Setup::new(mesh(n, wires)),
            &[DType::F32, DType::BF16],
            true,
        );
    }
    let mut setup = Setup::new(mesh(3, 2));
    setup.large = false;
    collectives(&setup, &[DType::F32], true);
}

#[test]
fn ring_collectives_match_the_reference() {
    for (n, cables) in [(2, 1), (3, 2), (4, 1)] {
        collectives(&Setup::new(ring(n, cables)), &DTYPES, false);
    }
}

#[test]
fn ring_thresholds_and_large_messages() {
    for (n, cables) in [(2, 2), (3, 4), (4, 2)] {
        collectives(
            &Setup::new(ring(n, cables)),
            &[DType::F32, DType::BF16],
            true,
        );
    }
    // A full mesh with the ring preferred runs the ring.
    let mut setup = Setup::new(mesh(4, 1));
    setup.prefer_ring = true;
    setup.large = false;
    collectives(&setup, &[DType::F32], true);
}

/// Floating-point sums of inexact values: every rank and every rerun get the same bits.
#[test]
fn reductions_are_deterministic() {
    for devices in [mesh(4, 2), ring(4, 2), mesh(3, 1)] {
        let setup = Setup::new(devices);
        let runs: Vec<Vec<Vec<u8>>> = (0..3)
            .map(|_| {
                setup.run(|rank, comm| {
                    let mut out = Vec::new();
                    for count in [5usize, 3000, 70_000, 300_000] {
                        let v: Vec<f64> = (0..count)
                            .map(|i| {
                                ((i * 31 + rank * 17) % 1000) as f64 / 7.0 + 1e-3 * rank as f64
                            })
                            .collect();
                        let mut data = encode(DType::F32, &v);
                        comm.all_reduce(AllReduce::Sum, DType::F32, &mut data)
                            .unwrap();
                        out.extend(data);
                        let mut data = encode(DType::BF16, &v);
                        comm.all_reduce(AllReduce::Sum, DType::BF16, &mut data)
                            .unwrap();
                        out.extend(data);
                    }
                    out
                })
            })
            .collect();
        for run in &runs {
            assert!(run.iter().all(|r| r == &run[0]), "the ranks disagree");
        }
        assert!(runs.iter().all(|r| r[0] == runs[0][0]), "reruns disagree");
    }
}

#[test]
fn point_to_point() {
    // Mesh: every pair, both directions, small and large.
    let setup = Setup::new(mesh(3, 2));
    setup.run(|rank, comm| {
        for len in [0usize, 5, 4096, 600_000] {
            for peer in (0..3).filter(|&p| p != rank) {
                let mine: Vec<u8> = (0..len).map(|i| (i + rank) as u8).collect();
                let mut got = vec![0u8; len];
                if rank < peer {
                    comm.send(peer, &mine).unwrap();
                    comm.recv(peer, &mut got).unwrap();
                } else {
                    comm.recv(peer, &mut got).unwrap();
                    comm.send(peer, &mine).unwrap();
                }
                assert!(got.iter().enumerate().all(|(i, &b)| b == (i + peer) as u8));
            }
        }
    });
    // Ring: neighbours on several cables; two ranks (left = right); not a neighbour.
    for (n, cables) in [(2, 2), (4, 2)] {
        Setup::new(ring(n, cables)).run(|rank, comm| {
            let right = (rank + 1) % n;
            let left = (rank + n - 1) % n;
            for len in [1usize, 100_000] {
                let mine = vec![rank as u8; len];
                let mut got = vec![0u8; len];
                if rank % 2 == 0 {
                    comm.send(right, &mine).unwrap();
                    comm.recv(left, &mut got).unwrap();
                } else {
                    comm.recv(left, &mut got).unwrap();
                    comm.send(right, &mine).unwrap();
                }
                assert!(got.iter().all(|&b| b == left as u8));
            }
            if n == 4 {
                let e = comm.send((rank + 2) % n, &[1]).unwrap_err().to_string();
                assert!(e.contains("only its neighbours"), "{e}");
            }
        });
    }
}

/// Every rank fails within the timeout, naming a peer, and the group then refuses work.
fn fails_everywhere(setup: &Setup, op: impl Fn(usize, &Jaccl) -> nuvec_core::Result<()> + Sync) {
    let start = Instant::now();
    let errors = setup.run(|rank, comm| {
        let first = op(rank, comm);
        let again = comm.all_reduce(AllReduce::Sum, DType::F32, &mut [0; 4]);
        (first, again)
    });
    assert!(
        start.elapsed() < setup.timeout * 2 + Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
    let mut failed = 0;
    for (first, again) in errors {
        if let Err(e) = first {
            failed += 1;
            let e = e.to_string();
            assert!(e.contains("rank "), "{e}");
            assert!(!e.contains("panicked"), "{e}");
            let again = again.unwrap_err().to_string();
            assert!(again.contains("failed earlier"), "{again}");
        }
    }
    assert!(failed >= 1, "some rank must fail");
}

#[test]
fn faults_fail_within_the_timeout() {
    let quick = |devices, faults| {
        let mut s = Setup::new(devices);
        s.faults = faults;
        s.timeout = Duration::from_millis(300);
        s
    };
    let lost = vec![Fault::Drop {
        from: 1,
        to: 0,
        nth: 0,
    }];
    let failing = vec![Fault::Fail {
        rank: 2,
        nth: 1,
        code: 12,
    }];
    for devices in [mesh(3, 1), ring(3, 1)] {
        for faults in [lost.clone(), failing.clone()] {
            let s = quick(devices.clone(), faults);
            fails_everywhere(&s, |_, c| {
                c.all_reduce(AllReduce::Sum, DType::F32, &mut vec![0; 4000])
            });
            fails_everywhere(&s, |_, c| {
                let mut out = vec![0; 3 * 4000];
                c.all_gather(&[1; 4000], &mut out)
            });
            fails_everywhere(&s, |_, c| {
                let mut out = vec![0; 4000];
                c.sum_scatter(DType::F32, &[0; 12_000], &mut out)
            });
        }
    }
    // A rank that never joins the operation: the others wait for it and fail naming it.
    let s = quick(mesh(3, 1), Vec::new());
    let errors = s.run(|rank, comm| {
        if rank == 2 {
            return None;
        }
        Some(
            comm.all_reduce(AllReduce::Sum, DType::F32, &mut [0; 64])
                .unwrap_err()
                .to_string(),
        )
    });
    for e in errors.into_iter().flatten() {
        assert!(e.contains("rank 2") && e.contains("no completion"), "{e}");
    }
}

#[test]
fn groups_run_array_collectives() {
    use nuvec_core::distributed::{self as dist, Group};
    let setup = Setup::new(mesh(3, 1));
    let comms: Vec<Arc<Jaccl>> = {
        let fabric = Emulated::new(Vec::new());
        let gather = Arc::new(Gather::default());
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..3)
                .map(|rank| {
                    let (fabric, gather, devices) =
                        (fabric.clone(), gather.clone(), setup.devices.clone());
                    s.spawn(move || {
                        let config = Config::new(rank, devices).unwrap();
                        let options = Options::default()
                            .side_channel(move |d| Ok(gather.all_gather(rank, 3, d)));
                        Arc::new(connect(&config, &fabric.node(rank), &options, true).unwrap())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    let groups: Vec<Group> = comms.into_iter().map(|c| Group::new(c)).collect();
    std::thread::scope(|s| {
        for g in &groups {
            s.spawn(move || {
                let x = nuvec_core::Array::from_slice(&[g.rank() as f32 + 1.0; 6], [6]);
                let sum = dist::all_sum(&x, Some(g)).to_vec::<f32>().unwrap();
                assert_eq!(sum, [6.0; 6]);
                let part = dist::sum_scatter(&x, Some(g)).to_vec::<f32>().unwrap();
                assert_eq!(part, [6.0; 2]);
                let all = dist::all_gather(&x, Some(g));
                assert_eq!(all.shape(), [18]);
            });
        }
    });
}
