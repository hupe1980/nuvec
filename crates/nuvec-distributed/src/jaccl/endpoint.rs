//! One connection to one peer over one cable, as the algorithms see it: post sends and
//! receives of whole frames, poll completions. The system RDMA library and the emulated
//! fabric implement it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use nuvec_core::{Error, Result};

use super::frame::Frame;

/// A work request's kind in its `wr_id` (MLX's encoding).
pub(crate) const SEND: u64 = 1;
pub(crate) const RECV: u64 = 2;

/// MLX's `wr_id`: `kind << 16 | slot << 8 | tag`, the tag being the peer's rank (mesh) or
/// the direction (ring).
pub(crate) fn wr_id(kind: u64, slot: usize, tag: usize) -> u64 {
    kind << 16 | (slot as u64) << 8 | tag as u64
}

/// The address of a connection's end, exchanged over the side channel: MLX's
/// `Destination` (`int` LID, queue pair number and packet sequence number, then the GID).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Destination {
    pub(crate) lid: i32,
    pub(crate) qpn: i32,
    pub(crate) psn: i32,
    pub(crate) gid: [u8; 16],
}

impl Destination {
    pub(crate) const BYTES: usize = 28;

    pub(crate) fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut b = [0u8; Self::BYTES];
        b[0..4].copy_from_slice(&self.lid.to_le_bytes());
        b[4..8].copy_from_slice(&self.qpn.to_le_bytes());
        b[8..12].copy_from_slice(&self.psn.to_le_bytes());
        b[12..].copy_from_slice(&self.gid);
        b
    }

    pub(crate) fn from_bytes(b: &[u8]) -> Self {
        let int = |i: usize| i32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        Self {
            lid: int(0),
            qpn: int(4),
            psn: int(8),
            gid: b[12..28].try_into().expect("16 bytes"),
        }
    }
}

/// A completion's status: success, or the library's status code and name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Success,
    Error { code: i32, name: String },
}

/// A completed work request; a receive carries the sender's sequence number.
#[derive(Clone, Debug)]
pub(crate) struct Completion {
    pub(crate) wr_id: u64,
    pub(crate) status: Status,
    pub(crate) sequence: Option<u32>,
}

/// Numbers a connection's messages: every send carries the next number (as immediate
/// data), every receive must carry the next expected one. UC queue pairs drop a lost frame
/// silently, and the sender's next frame would land in its receive; the numbers turn that
/// into an error naming the peer.
#[derive(Default)]
pub(crate) struct Sequence {
    sent: AtomicU32,
    received: AtomicU32,
}

impl Sequence {
    /// The number of the next send.
    pub(crate) fn next_send(&self) -> u32 {
        self.sent.fetch_add(1, Ordering::Relaxed)
    }

    /// Checks a receive's number, turning a gap into an error status.
    pub(crate) fn check(&self, c: &mut Completion) {
        if c.kind() != RECV || c.status != Status::Success {
            return;
        }
        let want = self.received.fetch_add(1, Ordering::Relaxed);
        if c.sequence != Some(want) {
            c.status = Status::Error {
                code: -1,
                name: match c.sequence {
                    Some(got) => format!("message {want} was lost (received message {got})"),
                    None => "a message without a sequence number".into(),
                },
            };
        }
    }
}

impl Completion {
    pub(crate) fn kind(&self) -> u64 {
        self.wr_id >> 16
    }

    pub(crate) fn slot(&self) -> usize {
        ((self.wr_id >> 8) & 0xff) as usize
    }

    pub(crate) fn tag(&self) -> usize {
        (self.wr_id & 0xff) as usize
    }

    /// An error naming `peer` unless the request succeeded.
    pub(crate) fn check(&self, peer: usize) -> Result<()> {
        match &self.status {
            Status::Success => Ok(()),
            Status::Error { code, name } => {
                let what = if self.kind() == SEND {
                    "send"
                } else {
                    "receive"
                };
                Err(Error::Backend(format!(
                    "jaccl: rank {peer}: {what} failed: {name} ({code})"
                )))
            }
        }
    }
}

/// One connection's end. Calls on one endpoint come from one thread at a time (the group's
/// order lock and one thread per cable).
pub(crate) trait Endpoint: Send + Sync {
    /// This end's address for the peer.
    fn destination(&self) -> Result<Destination>;
    /// Connects to the peer's end (MLX's INIT → RTR → RTS).
    fn connect(&self, peer: &Destination) -> Result<()>;
    /// Makes `frame` usable by this connection (once per frame).
    fn register(&self, frame: &Frame) -> Result<()>;
    /// Sends the whole of `frame`; its completion carries `wr_id`.
    fn post_send(&self, frame: &Frame, wr_id: u64) -> Result<()>;
    /// Receives the peer's next message into `frame`.
    fn post_recv(&self, frame: &Frame, wr_id: u64) -> Result<()>;
    /// Moves up to `out.len()` completions into `out` and returns their number.
    fn poll(&self, out: &mut Vec<Completion>, max: usize) -> Result<usize>;
}

/// Opens connection ends on this rank's RDMA devices.
pub(crate) trait Fabric: Send + Sync {
    /// A new connection end on `device` toward rank `peer`.
    fn open(&self, device: &str, peer: usize) -> Result<Arc<dyn Endpoint>>;
}

/// How long to spin on empty polls before yielding the thread.
const SPIN: Duration = Duration::from_micros(50);

/// Bounds a polling loop: spins (the latency path), then yields once no completion has
/// arrived for 50 µs, and fails past the timeout.
pub(crate) struct Waiter {
    timeout: Duration,
    since: Instant,
    empty: u32,
    yielding: bool,
}

impl Waiter {
    pub(crate) fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            since: Instant::now(),
            empty: 0,
            yielding: false,
        }
    }

    /// A poll returned `n` completions; `waiting` names the peers still expected.
    pub(crate) fn polled(&mut self, n: usize, waiting: impl FnOnce() -> String) -> Result<()> {
        if n > 0 {
            self.empty = 0;
            self.yielding = false;
            self.since = Instant::now();
            return Ok(());
        }
        self.empty += 1;
        if self.yielding {
            std::thread::yield_now();
        } else if !self.empty.is_multiple_of(1024) {
            return Ok(());
        }
        let idle = self.since.elapsed();
        if idle > self.timeout {
            return Err(Error::Backend(format!(
                "jaccl: {}: no completion within {:.1} s",
                waiting(),
                self.timeout.as_secs_f64()
            )));
        }
        self.yielding = idle > SPIN;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_work_requests_and_destinations() {
        let c = Completion {
            wr_id: wr_id(RECV, 1, 7),
            status: Status::Success,
            sequence: Some(0),
        };
        assert_eq!((c.kind(), c.slot(), c.tag()), (RECV, 1, 7));
        let d = Destination {
            lid: 3,
            qpn: -2,
            psn: 7,
            gid: [9; 16],
        };
        assert_eq!(Destination::from_bytes(&d.to_bytes()), d);
        let bad = Completion {
            wr_id: wr_id(SEND, 0, 2),
            status: Status::Error {
                code: 5,
                name: "transport retry counter exceeded".into(),
            },
            sequence: None,
        };
        let e = bad.check(2).unwrap_err().to_string();
        assert!(e.contains("rank 2: send failed"), "{e}");
        let seq = Sequence::default();
        let mut ok = c.clone();
        seq.check(&mut ok);
        assert_eq!(ok.status, Status::Success);
        let mut gap = Completion {
            sequence: Some(2),
            ..c
        };
        seq.check(&mut gap);
        let e = gap.check(7).unwrap_err().to_string();
        assert!(
            e.contains("rank 7: receive failed: message 1 was lost"),
            "{e}"
        );
    }

    #[test]
    fn waits_are_bounded() {
        let mut w = Waiter::new(Duration::from_millis(20));
        let start = Instant::now();
        let e = loop {
            if let Err(e) = w.polled(0, || "rank 1".into()) {
                break e;
            }
        };
        assert!(start.elapsed() >= Duration::from_millis(20));
        assert!(e.to_string().contains("rank 1: no completion"), "{e}");
    }
}
