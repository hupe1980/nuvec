//! Groups that survive a lost Mac: a lost peer is found while idle, the failed group
//! is closed and a new one formed in the same processes; keys authenticate every
//! connection, and a stray client cannot block or join a forming group.

use std::time::{Duration, Instant};

use nuvec_core::distributed::{self, Group};
use nuvec_core::{Array, DType, Device, Stream, set_default_stream};
use nuvec_distributed::{Options, init_with};

mod common;

use common::{agree, free_addresses, ranks_vanishing, ranks_with, stop_self};

fn nodes(var: &str) -> Vec<Vec<String>> {
    std::env::var(var)
        .unwrap()
        .split(',')
        .map(|a| vec![a.to_string()])
        .collect()
}

/// Every collective on `g` against the reference.
fn works(g: &Group) {
    let (rank, size) = (g.rank(), g.size());
    let x = Array::from_slice(&[rank as f32 + 1.0; 4], [4]);
    let total = (size * (size + 1) / 2) as f32;
    assert_eq!(
        distributed::all_sum(&x, Some(g)).to_vec::<f32>().unwrap(),
        [total; 4]
    );
    assert_eq!(
        distributed::all_gather(&x, Some(g)).shape(),
        [4 * size as i32]
    );
    let y = Array::from_slice(&vec![1.0f32; 2 * size], [2 * size]);
    assert_eq!(
        distributed::sum_scatter(&y, Some(g))
            .to_vec::<f32>()
            .unwrap(),
        [size as f32; 2]
    );
    if size > 1 {
        let right = (rank + 1) % size;
        let left = (rank + size - 1) % size;
        let sent = distributed::send(&x, right, Some(g));
        let got = distributed::recv([4], DType::F32, left, Some(g));
        if rank % 2 == 0 {
            sent.eval().unwrap();
            got.eval().unwrap();
        } else {
            got.eval().unwrap();
            sent.eval().unwrap();
        }
        assert_eq!(got.to_vec::<f32>().unwrap(), [left as f32 + 1.0; 4]);
    }
}

fn reform(g: Group) {
    // Every rank healthy, quickly, before rank 2 leaves.
    distributed::all_sum(&Array::from(1.0f32), Some(&g))
        .eval()
        .unwrap();
    // Under a millisecond (the fastest of five, robust to a loaded machine).
    let fastest = (0..5)
        .map(|_| {
            let t = Instant::now();
            g.health().unwrap();
            t.elapsed()
        })
        .min()
        .unwrap();
    assert!(fastest < Duration::from_millis(1), "{fastest:?}");
    distributed::all_sum(&Array::from(1.0f32), Some(&g))
        .eval()
        .unwrap();
    if g.rank() == 2 {
        println!("\nBITS ok");
        std::process::exit(0);
    }
    // Found while idle, within 2 s, naming a rank.
    let left = Instant::now();
    let e = loop {
        if let Err(e) = g.health() {
            break e;
        }
        assert!(
            left.elapsed() < Duration::from_secs(2),
            "the lost rank went unnoticed"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    // Rank 2, or a neighbour that already closed the failed group itself.
    let e = e.to_string();
    assert!(
        e.contains("rank 2") || e.contains("rank 1") || e.contains("rank 0"),
        "{e}"
    );
    assert!(e.contains("closed") || e.contains("failed"), "{e}");
    // Closed: operations fail; a new group of two forms in this process.
    g.close();
    let closed = distributed::all_sum(&Array::from(1.0f32), Some(&g)).eval();
    assert!(closed.unwrap_err().to_string().contains("closed"));
    let two = init_with(&nodes("NUVEC_TEST_TWO"), g.rank(), &Options::default()).unwrap();
    works(&two);
    // `init` drops the failed default group and forms the new one from the environment.
    // SAFETY: this rank's process runs one test thread, and nothing reads the environment
    // concurrently.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var(
            "MLX_HOSTFILE",
            std::env::var("NUVEC_TEST_TWO_FILE").unwrap(),
        );
    }
    let again = nuvec_distributed::init(true).unwrap();
    assert_eq!(again.size(), 2);
    works(&again);
    println!("\nBITS ok");
}

#[test]
fn a_group_reforms_after_a_lost_mac() {
    let two = free_addresses(2);
    let two_again = free_addresses(2);
    let file = std::env::temp_dir().join(format!("nuvec-two-{}.json", std::process::id()));
    let entries: Vec<String> = two_again.iter().map(|a| format!("[\"{a}\"]")).collect();
    std::fs::write(&file, format!("[{}]", entries.join(", "))).unwrap();
    let env = [
        ("NUVEC_TEST_TWO", two.join(",")),
        ("NUVEC_TEST_TWO_FILE", file.display().to_string()),
    ];
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    if let Some(out) = ranks_with("a_group_reforms_after_a_lost_mac", 3, &env, reform) {
        agree(&out);
    }
}

const KEY: &str = "office-secret-0123456789abcdef";

fn keys(g: Group) {
    let rank = g.rank();
    // Every case starts together (a refused case can end seconds apart on the two ranks).
    let barrier = || {
        distributed::all_sum(&Array::from(1.0f32), Some(&g))
            .eval()
            .unwrap()
    };
    let opts = |k: &str| Options::default().timeout(Duration::from_secs(3)).key(k);
    // The same key: a working group.
    let same = init_with(&nodes("NUVEC_TEST_SAME"), rank, &opts(KEY)).unwrap();
    works(&same);
    barrier();
    // Different keys: an authentication error that never shows the key.
    let other = format!("{KEY}-{rank}");
    let e = init_with(&nodes("NUVEC_TEST_DIFF"), rank, &opts(&other))
        .err()
        .unwrap()
        .to_string();
    assert!(
        e.contains("authentication failed") && !e.contains(KEY),
        "{e}"
    );
    barrier();
    // A key on rank 0 only.
    let e = if rank == 0 {
        init_with(&nodes("NUVEC_TEST_ONE"), rank, &opts(KEY))
    } else {
        init_with(
            &nodes("NUVEC_TEST_ONE"),
            rank,
            &Options::default().timeout(Duration::from_secs(3)),
        )
    }
    .err()
    .unwrap()
    .to_string();
    // Rank 0 sees the unkeyed peer; rank 1 is refused and can only time out.
    if rank == 0 {
        assert!(e.contains("has no key"), "{e}");
    }
    barrier();
    // A stray client connecting to rank 0 first is dropped, and the group forms.
    let stray_at = nodes("NUVEC_TEST_STRAY")[0][0].clone();
    let stray = (rank == 0).then(|| {
        std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(3) {
                if let Ok(mut s) = std::net::TcpStream::connect(&stray_at) {
                    use std::io::Write;
                    let _ = s.write_all(b"GET / HTTP/1.1\r\n\r\n");
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        })
    });
    if rank == 1 {
        std::thread::sleep(Duration::from_millis(300));
    }
    let g2 = init_with(&nodes("NUVEC_TEST_STRAY"), rank, &opts(KEY)).unwrap();
    works(&g2);
    if let Some(h) = stray {
        h.join().unwrap();
    }
    barrier();
    // A short secret is refused.
    let e = init_with(&nodes("NUVEC_TEST_SAME"), rank, &opts("short"))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("at least 16 bytes"), "{e}");
    println!("\nBITS ok");
}

#[test]
fn keys_authenticate_every_connection() {
    let lists: Vec<(String, String)> = ["SAME", "DIFF", "ONE", "STRAY"]
        .iter()
        .map(|n| (format!("NUVEC_TEST_{n}"), free_addresses(2).join(",")))
        .collect();
    let env: Vec<(&str, &str)> = lists
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    if let Some(out) = ranks_with("keys_authenticate_every_connection", 2, &env, keys) {
        agree(&out);
    }
}

/// `"a|b,c|d"`: two addresses (connections) per rank.
fn multi(var: &str) -> Vec<Vec<String>> {
    std::env::var(var)
        .unwrap()
        .split(',')
        .map(|r| r.split('|').map(str::to_string).collect())
        .collect()
}

/// The collectives of `g` on data of several megabytes (many frames and packets), as bits.
fn battery(g: &Group) -> Vec<Vec<u32>> {
    let (rank, size) = (g.rank(), g.size());
    let n = (2 << 20) + 37;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + rank * 13) % 1001) as f32 * 0.37 - 150.0)
        .collect();
    let x = Array::from_slice(&data, [n as i32]);
    let bits = |a: &Array| {
        a.to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<u32>>()
    };
    let half = x.astype(DType::BF16);
    let mut out = vec![
        bits(&distributed::all_sum(&x, Some(g))),
        bits(&distributed::all_sum(&half, Some(g)).astype(DType::F32)),
        bits(&distributed::all_max(&x, Some(g))),
        bits(&distributed::all_gather(&x, Some(g))),
    ];
    let m = n - n % size;
    let y = Array::from_slice(&data[..m], [m as i32]);
    out.push(bits(&distributed::sum_scatter(&y, Some(g))));
    let (right, left) = ((rank + 1) % size, (rank + size - 1) % size);
    let sent = distributed::send(&x, right, Some(g));
    let got = distributed::recv([n as i32], DType::F32, left, Some(g));
    if rank % 2 == 0 {
        sent.eval().unwrap();
        got.eval().unwrap();
    } else {
        got.eval().unwrap();
        sent.eval().unwrap();
    }
    out.push(bits(&got));
    out
}

fn encryption(g: Group) {
    let rank = g.rank();
    let barrier = || {
        distributed::all_sum(&Array::from(1.0f32), Some(&g))
            .eval()
            .unwrap()
    };
    let opts = || Options::default().timeout(Duration::from_secs(5));
    // The same number of connections (they split the reductions), with and without.
    let plain = init_with(&multi("NUVEC_TEST_TWOWAY"), rank, &opts()).unwrap();
    let sealed = init_with(&multi("NUVEC_TEST_SEALED"), rank, &opts().key(KEY)).unwrap();
    let signed = init_with(
        &multi("NUVEC_TEST_SIGNED"),
        rank,
        &opts().key(KEY).encrypt(false),
    )
    .unwrap();
    let (a, b) = (battery(&plain), battery(&sealed));
    assert!(a == b, "rank {rank}: encrypted results differ");
    assert!(
        a == battery(&signed),
        "rank {rank}: authenticated results differ"
    );
    // The collectives whose results every rank shares (not the scatter or the receive).
    let digest: u64 = a[..4]
        .iter()
        .flatten()
        .fold(0xcbf2_9ce4_8422_2325, |h, &v| {
            (h ^ u64::from(v)).wrapping_mul(0x100_0000_01b3)
        });
    println!("\nBITS {digest:x}");
    barrier();
    // A rank that does not encrypt is refused. The lower rank of a pair accepts first: with
    // two ranks rank 0 accepts rank 1 and names it; with three, rank 1 accepts rank 0 first
    // and names it (the others then time out or are refused).
    let e = init_with(
        &nodes("NUVEC_TEST_MODE"),
        rank,
        &opts().key(KEY).encrypt(rank != 1),
    )
    .err()
    .unwrap()
    .to_string();
    match (g.size(), rank) {
        (2, 0) => assert!(e.contains("rank 1 does not encrypt"), "{e}"),
        (3, 1) => assert!(e.contains("rank 0 encrypts"), "{e}"),
        _ => {}
    }
    barrier();
    // Encryption needs a key.
    let e = init_with(&nodes("NUVEC_TEST_PLAIN"), rank, &opts().encrypt(true))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("encryption needs a key"), "{e}");
    println!("\nBITS ok");
}

fn encrypted(test: &str, n: usize) {
    let two = || -> String {
        let r: Vec<String> = (0..n).map(|_| free_addresses(2).join("|")).collect();
        r.join(",")
    };
    let lists = [
        ("NUVEC_TEST_TWOWAY", two()),
        ("NUVEC_TEST_PLAIN", free_addresses(n).join(",")),
        ("NUVEC_TEST_SEALED", two()),
        ("NUVEC_TEST_SIGNED", two()),
        ("NUVEC_TEST_MODE", free_addresses(n).join(",")),
    ];
    let env: Vec<(&str, &str)> = lists.iter().map(|(k, v)| (*k, v.as_str())).collect();
    if let Some(out) = ranks_with(test, n, &env, encryption) {
        agree(&out);
    }
}

#[test]
fn encrypted_rings_of_two_give_the_same_bits() {
    encrypted("encrypted_rings_of_two_give_the_same_bits", 2);
}

#[test]
fn encrypted_rings_of_three_give_the_same_bits() {
    encrypted("encrypted_rings_of_three_give_the_same_bits", 3);
}

/// Rank 2 of three stops mid-transfer (a sleeping Mac: no FIN): the receive waiting on it,
/// on the GPU stream where there is one, fails within the timeout naming it, and the other
/// two form a new group.
fn vanished(g: Group, key: Option<&str>) {
    let rank = g.rank();
    let timeout = Duration::from_secs(3);
    let mut opts = Options::default().timeout(timeout);
    if let Some(k) = key {
        opts = opts.key(k);
    }
    let three = init_with(&nodes("NUVEC_TEST_THREE"), rank, &opts).unwrap();
    distributed::all_sum(&Array::from(1.0f32), Some(&three))
        .eval()
        .unwrap();
    if rank == 2 {
        println!("\nBITS ok");
        stop_self();
    }
    if cfg!(target_os = "macos") {
        set_default_stream(Stream::default_for(Device::gpu(0)));
    }
    let start = Instant::now();
    let e = distributed::recv([1 << 20], DType::F32, 2, Some(&three))
        .eval()
        .unwrap_err()
        .to_string();
    let took = start.elapsed();
    assert!(e.contains("rank 2"), "{e}");
    assert!(
        took >= timeout - Duration::from_millis(100) && took < timeout + Duration::from_secs(2),
        "{took:?}: {e}"
    );
    three.close();
    let two = init_with(&nodes("NUVEC_TEST_TWO"), rank, &opts).unwrap();
    works(&two);
    println!("\nBITS ok");
}

fn vanishing(test: &str, key: Option<&'static str>) {
    let lists = [
        ("NUVEC_TEST_THREE", free_addresses(3).join(",")),
        ("NUVEC_TEST_TWO", free_addresses(2).join(",")),
    ];
    let env: Vec<(&str, &str)> = lists.iter().map(|(k, v)| (*k, v.as_str())).collect();
    if let Some(out) = ranks_vanishing(test, 3, &env, Some(2), |g| vanished(g, key)) {
        agree(&out);
    }
}

#[test]
fn a_vanished_peer_fails_the_transfer_within_the_timeout() {
    vanishing(
        "a_vanished_peer_fails_the_transfer_within_the_timeout",
        None,
    );
}

#[test]
fn a_vanished_peer_fails_an_encrypted_transfer_within_the_timeout() {
    vanishing(
        "a_vanished_peer_fails_an_encrypted_transfer_within_the_timeout",
        Some(KEY),
    );
}
