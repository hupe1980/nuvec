//! An in-process RDMA fabric for tests: every connection delivers each send, in order, into
//! the peer's next posted receive, as the algorithms rely on Apple's driver to do. A send
//! posted before the peer's receive is held until the receive is posted; a receive frame
//! shorter than the message completes both ends with a length error. Faults (lost messages,
//! error completions) are injected by rule.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use nuvec_core::{Error, Result};

use super::endpoint::{Completion, Destination, Endpoint, Fabric, Sequence, Status};
use super::frame::Frame;

/// The status code and name of a too-short receive (`IBV_WC_LOC_LEN_ERR`).
const LOC_LEN_ERR: i32 = 1;

/// An injected fault.
#[derive(Clone, Debug)]
pub(crate) enum Fault {
    /// The `nth` message (0-based) from rank `from` to rank `to` is lost: the send completes,
    /// the receive never does (UC drops it silently).
    Drop { from: usize, to: usize, nth: usize },
    /// The `nth` completion (0-based) on rank `rank`'s connections fails with `code`.
    Fail { rank: usize, nth: usize, code: i32 },
}

struct End {
    rank: usize,
    peer: usize,
    remote: Option<usize>,
    /// Posted receives.
    receives: VecDeque<(Frame, u64)>,
    /// Sends from the remote end waiting for a receive here, with their sequence numbers.
    held: VecDeque<(Frame, u64, u32)>,
    completions: VecDeque<Completion>,
}

struct State {
    ends: Vec<End>,
    faults: Vec<Fault>,
    /// Messages sent per (from, to) rank pair, and completions per rank.
    sent: Vec<((usize, usize), usize)>,
    completed: Vec<usize>,
}

impl State {
    fn complete(&mut self, end: usize, wr_id: u64, status: Status, sequence: Option<u32>) {
        let rank = self.ends[end].rank;
        if self.completed.len() <= rank {
            self.completed.resize(rank + 1, 0);
        }
        let nth = self.completed[rank];
        self.completed[rank] += 1;
        let status = self
            .faults
            .iter()
            .find_map(|f| match *f {
                Fault::Fail {
                    rank: r,
                    nth: n,
                    code,
                } if r == rank && n == nth => Some(Status::Error {
                    code,
                    name: "injected failure".into(),
                }),
                _ => None,
            })
            .unwrap_or(status);
        self.ends[end].completions.push_back(Completion {
            wr_id,
            status,
            sequence,
        });
    }

    /// Delivers held sends into posted receives on `end`.
    fn deliver(&mut self, end: usize) {
        while !self.ends[end].held.is_empty() && !self.ends[end].receives.is_empty() {
            let (frame, send_id, seq) = self.ends[end].held.pop_front().expect("non-empty");
            let sender = self.ends[end].remote.expect("connected");
            let (from, to) = (self.ends[sender].rank, self.ends[end].rank);
            let nth = match self.sent.iter_mut().find(|(k, _)| *k == (from, to)) {
                Some((_, n)) => {
                    *n += 1;
                    *n - 1
                }
                None => {
                    self.sent.push(((from, to), 1));
                    0
                }
            };
            let lost = self.faults.iter().any(|f| {
                matches!(*f, Fault::Drop { from: a, to: b, nth: n } if a == from && b == to && n == nth)
            });
            if lost {
                self.complete(sender, send_id, Status::Success, None);
                continue;
            }
            let (into, recv_id) = self.ends[end].receives.pop_front().expect("non-empty");
            if into.len() < frame.len() {
                let err = || Status::Error {
                    code: LOC_LEN_ERR,
                    name: "local length error".into(),
                };
                self.complete(sender, send_id, err(), None);
                self.complete(end, recv_id, err(), Some(seq));
                continue;
            }
            into.copy_from(&frame, frame.len());
            self.complete(sender, send_id, Status::Success, None);
            self.complete(end, recv_id, Status::Success, Some(seq));
        }
    }
}

/// The fabric shared by every rank of an emulated group.
#[derive(Clone)]
pub(crate) struct Emulated(Arc<Mutex<State>>);

impl Emulated {
    pub(crate) fn new(faults: Vec<Fault>) -> Self {
        Self(Arc::new(Mutex::new(State {
            ends: Vec::new(),
            faults,
            sent: Vec::new(),
            completed: Vec::new(),
        })))
    }

    /// Rank `rank`'s view of the fabric.
    pub(crate) fn node(&self, rank: usize) -> Node {
        Node {
            fabric: self.clone(),
            rank,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One rank's devices on the emulated fabric (any device name opens a connection end).
pub(crate) struct Node {
    fabric: Emulated,
    rank: usize,
}

impl Fabric for Node {
    fn open(&self, device: &str, peer: usize) -> Result<Arc<dyn Endpoint>> {
        if device.is_empty() {
            return Err(Error::InvalidArgument("jaccl: empty device name".into()));
        }
        let mut s = self.fabric.lock();
        s.ends.push(End {
            rank: self.rank,
            peer,
            remote: None,
            receives: VecDeque::new(),
            held: VecDeque::new(),
            completions: VecDeque::new(),
        });
        Ok(Arc::new(EmulatedEnd {
            fabric: self.fabric.clone(),
            id: s.ends.len() - 1,
            sequence: Sequence::default(),
        }))
    }
}

struct EmulatedEnd {
    fabric: Emulated,
    id: usize,
    sequence: Sequence,
}

impl Endpoint for EmulatedEnd {
    fn destination(&self) -> Result<Destination> {
        let s = self.fabric.lock();
        Ok(Destination {
            lid: s.ends[self.id].rank as i32,
            qpn: self.id as i32,
            psn: 7,
            gid: [0; 16],
        })
    }

    fn connect(&self, peer: &Destination) -> Result<()> {
        let mut s = self.fabric.lock();
        let remote = usize::try_from(peer.qpn)
            .ok()
            .filter(|&r| r < s.ends.len() && s.ends[r].rank == s.ends[self.id].peer)
            .ok_or_else(|| {
                Error::Backend(format!(
                    "jaccl: rank {}: no such connection end",
                    s.ends[self.id].peer
                ))
            })?;
        s.ends[self.id].remote = Some(remote);
        Ok(())
    }

    fn register(&self, _frame: &Frame) -> Result<()> {
        Ok(())
    }

    fn post_send(&self, frame: &Frame, wr_id: u64) -> Result<()> {
        let mut s = self.fabric.lock();
        let remote = s.ends[self.id]
            .remote
            .ok_or_else(|| Error::Backend("jaccl: send on an unconnected end".into()))?;
        let seq = self.sequence.next_send();
        s.ends[remote].held.push_back((frame.clone(), wr_id, seq));
        s.deliver(remote);
        Ok(())
    }

    fn post_recv(&self, frame: &Frame, wr_id: u64) -> Result<()> {
        let mut s = self.fabric.lock();
        s.ends[self.id].receives.push_back((frame.clone(), wr_id));
        s.deliver(self.id);
        Ok(())
    }

    fn poll(&self, out: &mut Vec<Completion>, max: usize) -> Result<usize> {
        let mut s = self.fabric.lock();
        let c = &mut s.ends[self.id].completions;
        let n = c.len().min(max);
        for mut c in c.drain(..n) {
            self.sequence.check(&mut c);
            out.push(c);
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::super::endpoint::{RECV, SEND, wr_id};
    use super::*;

    fn pair(faults: Vec<Fault>) -> (Arc<dyn Endpoint>, Arc<dyn Endpoint>) {
        let f = Emulated::new(faults);
        let a = f.node(0).open("rdma_en1", 1).unwrap();
        let b = f.node(1).open("rdma_en1", 0).unwrap();
        a.connect(&b.destination().unwrap()).unwrap();
        b.connect(&a.destination().unwrap()).unwrap();
        (a, b)
    }

    fn drain(e: &dyn Endpoint) -> Vec<Completion> {
        let mut out = Vec::new();
        e.poll(&mut out, 16).unwrap();
        out
    }

    #[test]
    fn delivers_in_order_and_holds_early_sends() {
        let (a, b) = pair(Vec::new());
        let (s0, s1) = (Frame::new(4096).unwrap(), Frame::new(4096).unwrap());
        s0.write(&[1]);
        s1.write(&[2]);
        a.post_send(&s0, wr_id(SEND, 0, 1)).unwrap();
        a.post_send(&s1, wr_id(SEND, 1, 1)).unwrap();
        assert!(drain(&*a).is_empty(), "held until a receive is posted");
        let (r0, r1) = (Frame::new(4096).unwrap(), Frame::new(4096).unwrap());
        b.post_recv(&r0, wr_id(RECV, 0, 0)).unwrap();
        b.post_recv(&r1, wr_id(RECV, 1, 0)).unwrap();
        assert_eq!(r0.read(1, |b| b[0]), 1);
        assert_eq!(r1.read(1, |b| b[0]), 2);
        let slots: Vec<_> = drain(&*b).iter().map(Completion::slot).collect();
        assert_eq!(slots, [0, 1]);
        assert_eq!(drain(&*a).len(), 2);
    }

    #[test]
    fn short_receives_and_faults() {
        let (a, b) = pair(vec![
            Fault::Drop {
                from: 0,
                to: 1,
                nth: 1,
            },
            Fault::Fail {
                rank: 1,
                nth: 1,
                code: 12,
            },
        ]);
        let big = Frame::new(8192).unwrap();
        b.post_recv(&Frame::new(4096).unwrap(), 1).unwrap();
        a.post_send(&big, 2).unwrap();
        assert!(drain(&*b)[0].check(0).is_err(), "length error");
        assert!(drain(&*a)[0].check(1).is_err(), "on both ends");
        // The second message is lost: its send completes, no receive does.
        b.post_recv(&Frame::new(8192).unwrap(), 3).unwrap();
        a.post_send(&big, 4).unwrap();
        assert!(drain(&*a).iter().all(|c| c.check(1).is_ok()));
        assert!(drain(&*b).is_empty());
        // The third arrives in the posted receive, numbered after the lost one; rank 1's
        // second completion also fails by rule.
        a.post_send(&big, 5).unwrap();
        let e = drain(&*b)[0].check(0).unwrap_err().to_string();
        assert!(e.contains("injected failure"), "{e}");
    }
}
