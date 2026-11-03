//! Groups that survive a lost Mac: a lost peer is found while idle, the failed group
//! is closed and a new one formed in the same processes; keys authenticate every
//! connection, and a stray client cannot block or join a forming group.

use std::time::{Duration, Instant};

use nuvec_core::distributed::{self, Group};
use nuvec_core::{Array, DType};
use nuvec_distributed::{Options, init_with};

mod common;

use common::{agree, free_addresses, ranks_with};

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
