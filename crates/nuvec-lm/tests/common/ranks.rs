//! The multi-process harness: a test re-runs its own test binary as the ranks of a ring.

#![allow(dead_code)]

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use nuvec::distributed::Group;

pub(crate) const CHILD: &str = "NUVEC_TEST_CHILD";

/// Runs `body` as `n` ranks of one ring. In the parent, starts the ranks and returns their
/// standard outputs (in rank order) once all succeed; in a rank, runs `body` and returns
/// `None`.
pub(crate) fn ranks(test: &str, n: usize, body: impl FnOnce(Group)) -> Option<Vec<String>> {
    ranks_with(test, n, &[], body)
}

/// [`ranks`] with `env` set in every rank's environment.
pub(crate) fn ranks_with(
    test: &str,
    n: usize,
    env: &[(&str, &str)],
    body: impl FnOnce(Group),
) -> Option<Vec<String>> {
    if let Ok(name) = std::env::var(CHILD) {
        if name == test {
            body(nuvec::distributed::init(true).expect("init"));
        }
        return None;
    }
    // A free port can be taken by another process before its rank binds it: retry then.
    let mut report = String::new();
    for _ in 0..5 {
        match launch(test, n, env) {
            Ok(outputs) => return Some(outputs),
            Err(e) if e.contains("Address already in use") => report = e,
            Err(e) => panic!("{e}"),
        }
    }
    panic!("{report}")
}

/// Starts `n` ranks of `test` on free ports; their standard outputs once all succeed, or the
/// report of every rank (all are stopped) once one fails.
fn launch(test: &str, n: usize, env: &[(&str, &str)]) -> Result<Vec<String>, String> {
    let ports: Vec<u16> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect::<Vec<_>>()
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect();
    let hostfile =
        std::env::temp_dir().join(format!("nuvec-lm-ranks-{}-{test}.json", std::process::id()));
    let entries: Vec<String> = ports
        .iter()
        .map(|p| format!("[\"127.0.0.1:{p}\"]"))
        .collect();
    std::fs::write(&hostfile, format!("[{}]", entries.join(", "))).unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut children: Vec<_> = (0..n)
        .map(|rank| {
            Command::new(&exe)
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(CHILD, test)
                .env("MLX_HOSTFILE", &hostfile)
                .env("MLX_RANK", rank.to_string())
                .env("NUVEC_DISTRIBUTED_TIMEOUT", "20")
                .envs(env.iter().copied())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(120);
    // Wait for every rank, or the first failure.
    let mut statuses = vec![None; n];
    let failed = loop {
        for (rank, child) in children.iter_mut().enumerate() {
            if statuses[rank].is_none() {
                statuses[rank] = child.try_wait().unwrap();
            }
        }
        if statuses.iter().flatten().any(|s| !s.success()) || Instant::now() > deadline {
            break true;
        }
        if statuses.iter().all(Option::is_some) {
            break false;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if failed {
        for c in &mut children {
            let _ = c.kill();
        }
    }
    let mut outputs = Vec::new();
    let mut report = String::new();
    for (rank, child) in children.iter_mut().enumerate() {
        let status = child.wait().ok();
        let (mut out, mut err) = (String::new(), String::new());
        let _ = child.stdout.take().unwrap().read_to_string(&mut out);
        let _ = child.stderr.take().unwrap().read_to_string(&mut err);
        report.push_str(&format!("{test}: rank {rank} ({status:?})\n{out}\n{err}\n"));
        outputs.push(out);
    }
    let _ = std::fs::remove_file(&hostfile);
    match failed {
        false => Ok(outputs),
        true => Err(report),
    }
}

/// The lines of each rank's output starting with `BITS`, which must agree.
pub(crate) fn agree(outputs: &[String]) {
    fn bits(o: &str) -> Vec<&str> {
        o.lines().filter(|l| l.starts_with("BITS")).collect()
    }
    let first = bits(&outputs[0]);
    assert!(!first.is_empty(), "no BITS lines");
    for (rank, o) in outputs.iter().enumerate() {
        assert_eq!(bits(o), first, "rank {rank} differs from rank 0");
    }
}
