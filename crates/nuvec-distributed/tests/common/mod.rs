//! The multi-process harness: a test re-runs its own test binary as the ranks of a ring.

#![allow(dead_code)]

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use nuvec_core::distributed::Group;

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
    ranks_vanishing(test, n, env, None, body)
}

/// [`ranks_with`] where rank `vanishing` stops itself (a Mac that sleeps: [`stop_self`]):
/// it is not waited for, and is killed once the others have finished.
pub(crate) fn ranks_vanishing(
    test: &str,
    n: usize,
    env: &[(&str, &str)],
    vanishing: Option<usize>,
    body: impl FnOnce(Group),
) -> Option<Vec<String>> {
    if let Ok(name) = std::env::var(CHILD) {
        if name == test {
            body(nuvec_distributed::init(true).expect("init"));
        }
        return None;
    }
    let ports: Vec<u16> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect::<Vec<_>>()
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect();
    let hostfile =
        std::env::temp_dir().join(format!("nuvec-ring-{}-{test}.json", std::process::id()));
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
    let mut results = Vec::new();
    // The vanishing rank last, once the others are done.
    let mut order: Vec<usize> = (0..n).filter(|&r| Some(r) != vanishing).collect();
    order.extend(vanishing);
    let mut outputs: Vec<Option<_>> = (0..n).map(|_| None).collect();
    for rank in order {
        let child = &mut children[rank];
        if Some(rank) == vanishing {
            let _ = child.kill();
            let _ = child.wait();
        }
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if Instant::now() > deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let mut out = String::new();
        let mut err = String::new();
        child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut out)
            .unwrap();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut err)
            .unwrap();
        if status.is_none() {
            let _ = child.kill();
        }
        // A killed vanishing rank succeeded if it reached its stop.
        let status = match Some(rank) == vanishing {
            true => out.contains("BITS").then(success),
            false => status,
        };
        outputs[rank] = Some((status, out, err));
    }
    results.extend(outputs.into_iter().map(|r| r.expect("every rank")));
    let _ = std::fs::remove_file(&hostfile);
    // On a failure, every rank's output: the first rank to fail is often not rank 0.
    if results
        .iter()
        .any(|(s, ..)| !s.is_some_and(|s| s.success()))
    {
        let report: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(rank, (s, out, err))| format!("rank {rank} ({s:?}):\n{out}\n{err}"))
            .collect();
        panic!("{test}: a rank failed\n{}", report.join("\n"));
    }
    Some(results.into_iter().map(|(_, out, _)| out).collect())
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

/// `n` free loopback addresses.
pub(crate) fn free_addresses(n: usize) -> Vec<String> {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    listeners
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.local_addr().unwrap().port()))
        .collect()
}

/// A successful exit status.
fn success() -> std::process::ExitStatus {
    Command::new("true").status().unwrap()
}

/// Stops this process (SIGSTOP), as a sleeping Mac: its sockets stay open and send nothing,
/// with no FIN. The harness kills it ([`ranks_vanishing`]).
pub(crate) fn stop_self() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let pid = std::process::id().to_string();
    Command::new("kill").args(["-STOP", &pid]).status().unwrap();
    unreachable!("a stopped process is killed, never continued");
}
