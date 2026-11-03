//! MLX's ring backend over TCP: each rank connects to its right neighbour and accepts its
//! left one, with one connection per hostfile address. Reductions are a reduce-scatter
//! followed by an all-gather around the ring, so every segment is reduced once, in ring
//! order, and then copied: all ranks get the same bits, and reruns repeat them.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::thread;
use std::time::{Duration, Instant};

use nuvec_core::distributed::{AllReduce, Communicator};
use nuvec_core::{DType, Error, Result};

use crate::reduce::combine;

pub(crate) struct Ring {
    rank: usize,
    size: usize,
    /// Connections to the right neighbour (this rank connected), one per address.
    right: Vec<TcpStream>,
    /// Connections from the left neighbour (this rank accepted).
    left: Vec<TcpStream>,
    /// Receive buffers of a split all-reduce's parts, kept between calls.
    scratch: Vec<std::sync::Mutex<Vec<u8>>>,
    /// The first transport error: a failed transfer leaves unread bytes in the sockets, so
    /// every later operation fails with it instead of reading misaligned data.
    failed: std::sync::Mutex<Option<String>>,
    /// How long a transfer may make no progress before it fails.
    timeout: Duration,
}

fn peer_error(peer: usize, what: &str, e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("distributed: rank {peer}: {what}: {e}"))
}

fn resolve(address: &str) -> Result<SocketAddr> {
    address
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .ok_or_else(|| Error::InvalidArgument(format!("distributed: cannot resolve {address}")))
}

/// Applies the socket options of a ring connection.
fn configure(s: &TcpStream, timeout: Duration, peer: usize) -> Result<()> {
    let set = || -> std::io::Result<()> {
        s.set_nonblocking(false)?;
        s.set_nodelay(true)?;
        s.set_read_timeout(Some(timeout))?;
        s.set_write_timeout(Some(timeout))?;
        crate::keepalive::enable(s)
    };
    set().map_err(|e| peer_error(peer, "configuring the connection", e))
}

impl Ring {
    /// Connects rank `rank` of the ring described by `nodes` (the addresses per rank).
    pub(crate) fn connect(
        nodes: &[Vec<String>],
        rank: usize,
        timeout: Duration,
        key: Option<&crate::auth::Key>,
    ) -> Result<Self> {
        let size = nodes.len();
        if rank >= size {
            return Err(Error::InvalidArgument(format!(
                "distributed: rank {rank} of a hostfile with {size} ranks"
            )));
        }
        let k = nodes[rank].len();
        if nodes.iter().any(|n| n.len() != k) {
            return Err(Error::InvalidArgument(
                "distributed: every rank needs the same number of addresses (connections)".into(),
            ));
        }
        let (right, left) = ((rank + 1) % size, (rank + size - 1) % size);
        let deadline = Instant::now() + timeout;
        let listeners = nodes[rank]
            .iter()
            .map(|a| {
                let l = TcpListener::bind(resolve(a)?)
                    .map_err(|e| Error::Backend(format!("distributed: listening on {a}: {e}")))?;
                l.set_nonblocking(true)
                    .map_err(|e| Error::Backend(format!("distributed: {a}: {e}")))?;
                Ok(l)
            })
            .collect::<Result<Vec<_>>>()?;
        let accept = || -> Result<Vec<TcpStream>> {
            listeners
                .iter()
                .map(|l| {
                    // With a key, a connection that fails the handshake (a stray client) is
                    // dropped and accepting goes on until the deadline.
                    let mut refused = None;
                    loop {
                        let s = match l.accept() {
                            Ok((s, _)) => s,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                if Instant::now() > deadline {
                                    return Err(refused.unwrap_or_else(|| {
                                        peer_error(left, "never connected", "timed out")
                                    }));
                                }
                                thread::sleep(Duration::from_millis(5));
                                continue;
                            }
                            Err(e) => return Err(peer_error(left, "accepting", e)),
                        };
                        configure(&s, timeout, left)?;
                        let who = format!("rank {left}");
                        match crate::auth::accept(&s, key, rank as u64, Some(left as u64), &who) {
                            Ok(_) => return Ok(s),
                            Err(e) if key.is_some() => refused = Some(e),
                            Err(e) => return Err(e),
                        }
                    }
                })
                .collect()
        };
        let connect = || -> Result<Vec<TcpStream>> {
            nodes[right]
                .iter()
                .map(|a| {
                    let address = resolve(a)?;
                    let s = loop {
                        match TcpStream::connect_timeout(&address, Duration::from_secs(1)) {
                            Ok(s) => break s,
                            Err(e) if Instant::now() > deadline => {
                                return Err(peer_error(right, &format!("connecting to {a}"), e));
                            }
                            Err(_) => thread::sleep(Duration::from_millis(20)),
                        }
                    };
                    configure(&s, timeout, right)?;
                    let who = format!("rank {right}");
                    crate::auth::connect(&s, key, rank as u64, right as u64, &who)?;
                    Ok(s)
                })
                .collect()
        };
        // The lower rank of each pair accepts first, so the ring cannot deadlock.
        let (right_s, left_s) = if rank < right {
            let l = accept()?;
            (connect()?, l)
        } else {
            let r = connect()?;
            (r, accept()?)
        };
        // Non-blocking from here on: every transfer waits in poll(2) (see `duplex`).
        for s in right_s.iter().chain(&left_s) {
            s.set_nonblocking(true).map_err(|e| {
                Error::Backend(format!("distributed: configuring a connection: {e}"))
            })?;
        }
        let parts = right_s.len() + left_s.len();
        Ok(Self {
            rank,
            size,
            right: right_s,
            left: left_s,
            scratch: (0..parts).map(|_| std::sync::Mutex::default()).collect(),
            failed: std::sync::Mutex::new(None),
            timeout,
        })
    }

    fn right_rank(&self) -> usize {
        (self.rank + 1) % self.size
    }

    fn left_rank(&self) -> usize {
        (self.rank + self.size - 1) % self.size
    }

    /// The sockets to `dst` for sending, or `src` for receiving (direct neighbours only).
    fn sockets(&self, peer: usize, sending: bool) -> Result<&[TcpStream]> {
        let (first, second) = if sending {
            (
                (self.right_rank(), &self.right),
                (self.left_rank(), &self.left),
            )
        } else {
            (
                (self.left_rank(), &self.left),
                (self.right_rank(), &self.right),
            )
        };
        if peer == first.0 {
            Ok(first.1)
        } else if peer == second.0 {
            Ok(second.1)
        } else {
            Err(Error::InvalidArgument(format!(
                "distributed: the ring sends and receives between neighbours only (rank {} \
                 and rank {peer})",
                self.rank
            )))
        }
    }
}

/// The bytes of `read` (shared) and `write` (mutable) in `buf`: two disjoint ranges.
fn disjoint(
    buf: &mut [u8],
    read: std::ops::Range<usize>,
    write: std::ops::Range<usize>,
) -> (&[u8], &mut [u8]) {
    if read.end <= write.start {
        let (a, b) = buf.split_at_mut(write.start);
        (&a[read], &mut b[..write.len()])
    } else {
        let (a, b) = buf.split_at_mut(read.start);
        (&b[..read.len()], &mut a[write])
    }
}

/// `data` split into `k` contiguous parts of whole elements of `elem` bytes.
fn parts(len: usize, elem: usize, k: usize) -> Vec<std::ops::Range<usize>> {
    let n = len / elem;
    (0..k)
        .map(|i| (i * n / k) * elem..((i + 1) * n / k) * elem)
        .collect()
}

/// Up to this many bytes, two ranks exchange the whole buffer in one step instead of the
/// ring's two (decode's all_sums).
const SMALL: usize = 32 << 10;

/// The packets of a step's receive of `len` bytes (elements of `elem` bytes), each reduced
/// while the next arrives: half the step, from 32K elements up to 8 MB (MLX's packets).
fn packet(len: usize, elem: usize) -> usize {
    (len / 2).min(8 << 20).max(32_768 * elem)
}

/// Whether a failed transfer is transient: nothing to do for the moment (`EAGAIN`), the
/// kernel's buffers are full (`ENOBUFS` on macOS loopback under load) or the call was
/// interrupted. MLX's ring retries these too ("made no progress").
fn transient(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
    ) || e.raw_os_error() == Some(libc::ENOBUFS)
}

/// The receiving side of [`duplex`]: `len` bytes from `from` (rank `rank`), received into
/// the buffers `slot` hands out (`None` while none is free; with `true`, it waits for one)
/// and passed to `full` (their offset in the received bytes and the buffer) once filled.
struct Receive<'a, S, F> {
    from: &'a TcpStream,
    rank: usize,
    len: usize,
    slot: S,
    full: F,
}

/// Sends `data` to `to` (rank `to_rank`) and receives as `recv` says, both on this thread:
/// it waits in poll(2) for whichever socket is ready and moves what the kernel takes (on
/// macOS loopback, two threads sending and receiving on one socket at once halve its
/// throughput). Fails once neither socket has made progress for `timeout`.
fn duplex<'b, S, F>(
    to: &TcpStream,
    data: &[u8],
    to_rank: usize,
    mut recv: Option<Receive<'_, S, F>>,
    timeout: Duration,
) -> Result<()>
where
    S: FnMut(bool) -> Option<&'b mut [u8]>,
    F: FnMut(usize, &'b mut [u8]) -> Result<()>,
{
    let mut sent = 0;
    let mut got = 0;
    // The buffer being filled and its bytes so far.
    let mut current: Option<(&'b mut [u8], usize)> = None;
    let mut progress = Instant::now();
    loop {
        let receiving = recv.as_ref().is_some_and(|r| got < r.len);
        let sending = sent < data.len();
        if !receiving && !sending {
            return Ok(());
        }
        if let Some(r) = recv.as_mut().filter(|_| receiving && current.is_none()) {
            current = (r.slot)(!sending).map(|b| (b, 0));
            if current.is_none() && !sending {
                return Err(Error::Backend("distributed: the reduction stopped".into()));
            }
        }
        // Without a free buffer, look again shortly: the consumer frees one without any
        // socket becoming ready.
        let starved = receiving && current.is_none();
        let wait = if starved {
            Duration::from_micros(100)
        } else {
            timeout.saturating_sub(progress.elapsed())
        };
        let read = recv.as_ref().filter(|_| current.is_some()).map(|r| r.from);
        let ready = crate::socket::wait(sending.then_some(to), read, wait)
            .map_err(|e| peer_error(to_rank, "waiting for the connection", e))?;
        if !ready && progress.elapsed() >= timeout {
            let peer = match &recv {
                Some(r) if !sending => r.rank,
                _ => to_rank,
            };
            return Err(peer_error(peer, "transferring", "timed out"));
        }
        if let (Some(r), Some((buf, filled))) = (recv.as_mut(), current.as_mut()) {
            match (&*r.from).read(&mut buf[*filled..]) {
                Ok(0) => return Err(peer_error(r.rank, "receiving", "connection closed")),
                Ok(n) => {
                    progress = Instant::now();
                    *filled += n;
                    if *filled == buf.len() {
                        let (buf, filled) = current.take().expect("a buffer being filled");
                        (r.full)(got, buf)?;
                        got += filled;
                    }
                }
                Err(e) if transient(&e) => {}
                Err(e) => return Err(peer_error(r.rank, "receiving", e)),
            }
        }
        if sending {
            match (&*to).write(&data[sent..]) {
                Ok(0) => return Err(peer_error(to_rank, "sending", "connection closed")),
                Ok(n) => {
                    progress = Instant::now();
                    sent += n;
                }
                // ENOBUFS: poll reports the socket writable while the kernel has no
                // buffers; wait a little instead of spinning.
                Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {
                    thread::sleep(Duration::from_micros(50));
                }
                Err(e) if transient(&e) => {}
                Err(e) => return Err(peer_error(to_rank, "sending", e)),
            }
        }
    }
}

/// No receive, for [`duplex`].
type NoSlot<'b> = fn(bool) -> Option<&'b mut [u8]>;
type NoFull<'b> = fn(usize, &'b mut [u8]) -> Result<()>;

impl Ring {
    fn write_all(&self, s: &TcpStream, data: &[u8], peer: usize) -> Result<()> {
        duplex::<NoSlot, NoFull>(s, data, peer, None, self.timeout)
    }

    fn read_exact(&self, s: &TcpStream, data: &mut [u8], peer: usize) -> Result<()> {
        self.exchange(s, &[], peer, s, data, peer, usize::MAX, |_, _| Ok(()))
    }

    /// Sends `data` to `to` while receiving `into.len()` bytes from `from` in packets of
    /// `packet` bytes; `each` consumes every received packet (its offset in the received
    /// bytes and the bytes) as soon as it is complete. One packet is received in place;
    /// several pass through the first two packets of `into`, which then hold no particular
    /// contents.
    #[allow(clippy::too_many_arguments)]
    fn exchange(
        &self,
        to: &TcpStream,
        data: &[u8],
        to_rank: usize,
        from: &TcpStream,
        into: &mut [u8],
        from_rank: usize,
        packet: usize,
        mut each: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let len = into.len();
        if len <= packet {
            // In place, consumed once the send is done too.
            let mut whole = Some(&mut *into);
            duplex(
                to,
                data,
                to_rank,
                Some(Receive {
                    from,
                    rank: from_rank,
                    len,
                    slot: |_| whole.take(),
                    full: |_, _| Ok(()),
                }),
                self.timeout,
            )?;
            return each(0, into);
        }
        // Several packets, through two alternating slots of `into` (MLX's `recv_buffers`):
        // the transfer thread fills one while this one consumes the other.
        let (a, b) = into.split_at_mut(packet);
        let b = &mut b[..packet.min(len - packet)];
        let (tx, rx) = std::sync::mpsc::channel::<(usize, &mut [u8])>();
        let (free_tx, free_rx) = std::sync::mpsc::channel::<&mut [u8]>();
        let _ = free_tx.send(a);
        let _ = free_tx.send(b);
        let (moved, consumed) = crate::pool::join(
            move || {
                let mut at = 0;
                duplex(
                    to,
                    data,
                    to_rank,
                    Some(Receive {
                        from,
                        rank: from_rank,
                        len,
                        slot: |wait: bool| {
                            let slot = if wait {
                                free_rx.recv().ok()
                            } else {
                                free_rx.try_recv().ok()
                            }?;
                            let n = packet.min(len - at);
                            at += n;
                            Some(&mut slot[..n])
                        },
                        full: |at, slot| {
                            tx.send((at, slot)).map_err(|_| {
                                Error::Backend("distributed: the reduction stopped".into())
                            })
                        },
                    }),
                    self.timeout,
                )
            },
            move || {
                for (at, slot) in rx {
                    each(at, slot)?;
                    let _ = free_tx.send(slot);
                }
                Ok(())
            },
        );
        consumed.and(moved)
    }
}

impl Ring {
    /// The ring reductions of `forward` (sent right, received from the left) and `back` (the
    /// opposite direction) over one connection pair, step by step together: each socket's
    /// thread sends one part's segment and receives the other's on that socket, so no socket
    /// is used by two threads at once (one thread per socket, as MLX's socket threads).
    #[allow(clippy::too_many_arguments)]
    fn reduce_pair(
        &self,
        op: AllReduce,
        dtype: DType,
        forward: &mut [u8],
        back: &mut [u8],
        right: &TcpStream,
        left: &TcpStream,
        incoming: (&mut Vec<u8>, &mut Vec<u8>),
    ) -> Result<()> {
        let (size, elem) = (self.size, dtype.size());
        let (right_rank, left_rank) = (self.right_rank(), self.left_rank());
        // This rank's position in each ring (the reversed one counts the other way round).
        let (rf, rb) = (self.rank, (size - self.rank) % size);
        let (nf, nb) = (forward.len() / elem, back.len() / elem);
        let seg = |n: usize, s: usize| (s * n / size) * elem..((s + 1) * n / size) * elem;
        let (inf, inb) = incoming;
        inf.resize(inf.len().max(nf.div_ceil(size) * elem), 0);
        inb.resize(inb.len().max(nb.div_ceil(size) * elem), 0);
        // Reduce-scatter: after it, this rank holds segment `rank + 1` of each ring reduced.
        for step in 0..size - 1 {
            let send = |r: usize| (r + size - step) % size;
            let recv = |r: usize| (r + 2 * size - step - 1) % size;
            let (sent_f, acc_f) = disjoint(forward, seg(nf, send(rf)), seg(nf, recv(rf)));
            let (sent_b, acc_b) = disjoint(back, seg(nb, send(rb)), seg(nb, recv(rb)));
            let into_f = &mut inf[..acc_f.len()];
            let into_b = &mut inb[..acc_b.len()];
            let (packet_f, packet_b) = (packet(acc_f.len(), elem), packet(acc_b.len(), elem));
            let (on_right, on_left) = crate::pool::join(
                // Right socket: the forward segment out, the backward one in.
                || {
                    self.exchange(
                        right,
                        sent_f,
                        right_rank,
                        right,
                        into_b,
                        right_rank,
                        packet_b,
                        |at, p| combine(op, dtype, &mut acc_b[at..at + p.len()], p),
                    )
                },
                || {
                    self.exchange(
                        left,
                        sent_b,
                        left_rank,
                        left,
                        into_f,
                        left_rank,
                        packet_f,
                        |at, p| combine(op, dtype, &mut acc_f[at..at + p.len()], p),
                    )
                },
            );
            on_right.and(on_left)?;
        }
        // All-gather: pass the reduced segments around, received in place.
        for step in 0..size - 1 {
            let send = |r: usize| (r + 1 + size - step) % size;
            let recv = |r: usize| (r + size - step) % size;
            let (sent_f, out_f) = disjoint(forward, seg(nf, send(rf)), seg(nf, recv(rf)));
            let (sent_b, out_b) = disjoint(back, seg(nb, send(rb)), seg(nb, recv(rb)));
            let (whole_f, whole_b) = (out_f.len(), out_b.len());
            let (on_right, on_left) = crate::pool::join(
                || {
                    self.exchange(
                        right,
                        sent_f,
                        right_rank,
                        right,
                        out_b,
                        right_rank,
                        whole_b,
                        |_, _| Ok(()),
                    )
                },
                || {
                    self.exchange(
                        left,
                        sent_b,
                        left_rank,
                        left,
                        out_f,
                        left_rank,
                        whole_f,
                        |_, _| Ok(()),
                    )
                },
            );
            on_right.and(on_left)?;
        }
        Ok(())
    }

    /// The ring reduction of `buf` (elements of `elem` bytes) over one socket pair: sending
    /// right and receiving from the left, or with `reverse` the other way round (the ring
    /// in the opposite direction, as MLX's second half of a split reduction).
    #[allow(clippy::too_many_arguments)]
    fn reduce_part(
        &self,
        op: AllReduce,
        dtype: DType,
        buf: &mut [u8],
        to: &TcpStream,
        from: &TcpStream,
        reverse: bool,
        incoming: &mut Vec<u8>,
    ) -> Result<()> {
        let (size, elem) = (self.size, dtype.size());
        // In the reversed ring, this rank's position counts the other way round.
        let rank = if reverse {
            (size - self.rank) % size
        } else {
            self.rank
        };
        let n = buf.len() / elem;
        let segment = |s: usize| (s * n / size) * elem..((s + 1) * n / size) * elem;
        let (right, left) = match reverse {
            false => (self.right_rank(), self.left_rank()),
            true => (self.left_rank(), self.right_rank()),
        };
        incoming.resize(incoming.len().max(n.div_ceil(size) * elem), 0);
        // Reduce-scatter: after it, this rank holds segment `rank + 1` reduced.
        for step in 0..size - 1 {
            let send = segment((rank + size - step) % size);
            let recv = segment((rank + 2 * size - step - 1) % size);
            // The segments are disjoint: send one while reducing into the other.
            let (sent, acc) = disjoint(buf, send, recv.clone());
            let into = &mut incoming[..recv.len()];
            let packet = packet(recv.len(), elem);
            self.exchange(to, sent, right, from, into, left, packet, |at, packet| {
                combine(op, dtype, &mut acc[at..at + packet.len()], packet)
            })?;
        }
        // All-gather: pass the reduced segments around, received in place.
        for step in 0..size - 1 {
            let send = segment((rank + 1 + size - step) % size);
            let recv = segment((rank + size - step) % size);
            let (sent, out) = disjoint(buf, send, recv);
            let whole = out.len();
            self.exchange(to, sent, right, from, out, left, whole, |_, _| Ok(()))?;
        }
        Ok(())
    }

    /// The reduce-scatter of `buf` (`size` equal segments) over one socket pair, shifted so
    /// that this rank ends holding segment `rank` (the all-reduce's first half ends on
    /// `rank + 1`).
    fn scatter_part(
        &self,
        dtype: DType,
        buf: &mut [u8],
        to: &TcpStream,
        from: &TcpStream,
    ) -> Result<()> {
        let (size, rank) = (self.size, self.rank);
        let len = buf.len() / size;
        let segment = |s: usize| s * len..(s + 1) * len;
        let (right, left) = (self.right_rank(), self.left_rank());
        let mut incoming = vec![0u8; len];
        for step in 0..size - 1 {
            let send = segment((rank + 2 * size - step - 1) % size);
            let recv = segment((rank + 2 * size - step - 2) % size);
            let (sent, acc) = disjoint(buf, send, recv);
            let packet = packet(len, dtype.size());
            self.exchange(
                to,
                sent,
                right,
                from,
                &mut incoming,
                left,
                packet,
                |at, packet| {
                    combine(
                        AllReduce::Sum,
                        dtype,
                        &mut acc[at..at + packet.len()],
                        packet,
                    )
                },
            )?;
        }
        Ok(())
    }
}

impl Ring {
    /// Runs one operation unless an earlier one failed; a transport error marks the ring
    /// failed (argument errors do not).
    fn guarded<T>(&self, op: impl FnOnce() -> Result<T>) -> Result<T> {
        let mut failed = self.failed.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(first) = &*failed {
            return Err(Error::Backend(format!(
                "distributed: the ring failed earlier ({first}); no further operation runs"
            )));
        }
        let result = op();
        if let Err(Error::Backend(e)) = &result {
            *failed = Some(e.clone());
        }
        result
    }
}

impl Communicator for Ring {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        self.guarded(|| self.all_reduce_now(op, dtype, data))
    }

    fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        self.guarded(|| self.all_gather_now(input, output))
    }

    fn send(&self, dst: usize, data: &[u8]) -> Result<()> {
        self.guarded(|| self.send_now(dst, data))
    }

    fn recv(&self, src: usize, data: &mut [u8]) -> Result<()> {
        self.guarded(|| self.recv_now(src, data))
    }

    fn sum_scatter(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        self.guarded(|| self.sum_scatter_now(dtype, input, output))
    }

    fn health(&self) -> Result<()> {
        self.guarded(|| {
            let sockets = self
                .right
                .iter()
                .map(|s| (s, self.right_rank()))
                .chain(self.left.iter().map(|s| (s, self.left_rank())));
            for (s, peer) in sockets {
                probe(s, peer)?;
            }
            Ok(())
        })
    }
}

/// Whether a connection is still open: a non-blocking peek sees the peer's close (or an
/// error) at once; pending bytes (a peer already in the next collective) mean it is alive.
pub(crate) fn probe(s: &TcpStream, peer: usize) -> Result<()> {
    match crate::socket::peek(s) {
        Ok(0) => Err(peer_error(
            peer,
            "the connection was closed",
            "the peer left",
        )),
        Ok(_) => Ok(()),
        Err(e) if transient(&e) => Ok(()),
        Err(e) => Err(peer_error(peer, "the connection failed", e)),
    }
}

impl Ring {
    fn all_reduce_now(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        if self.size == 2 && data.len() <= SMALL {
            // Two ranks, a small buffer (decode's): one exchange of the whole buffer instead
            // of the ring's two steps. Both combine in rank order, so they get the same bits.
            let mut other = vec![0u8; data.len()];
            let peer = self.right_rank();
            self.exchange(
                &self.right[0],
                data,
                peer,
                &self.left[0],
                &mut other,
                peer,
                SMALL,
                |_, _| Ok(()),
            )?;
            if self.rank == 0 {
                return combine(op, dtype, data, &other);
            }
            combine(op, dtype, &mut other, data)?;
            data.copy_from_slice(&other);
            return Ok(());
        }
        // Large buffers are split over every connection in both directions (at least 256 KB
        // per rank's segment each, as MLX): part `i` goes around the ring on connection
        // pair `i / 2`, the odd parts in the opposite direction, all in parallel.
        let k = (2 * self.right.len().min(self.left.len()))
            .min(data.len() / (self.size * 262_144))
            .max(1);
        let ranges = parts(data.len(), dtype.size(), k);
        let mut chunks = Vec::with_capacity(k);
        let mut rest = data;
        for r in &ranges {
            let (head, tail) = rest.split_at_mut(r.len());
            chunks.push(head);
            rest = tail;
        }
        // On two ranks, parts `2c` and `2c + 1` share connection pair `c` and run together
        // (see `reduce_pair`); an odd last part runs alone. With more ranks every part runs on
        // its own: the two rings' steps exchange with different neighbours, and keeping them
        // in step would make each step wait for the slower one.
        let scratch = |i: usize| {
            self.scratch[i]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        // A unit: its first part's index and its parts (two on two ranks, else one).
        let unit = |i: usize, mut parts: Vec<&mut [u8]>| -> Result<()> {
            let (right, left) = (&self.right[i / 2], &self.left[i / 2]);
            match parts.as_mut_slice() {
                [forward, back] => {
                    let (mut sf, mut sb) = (scratch(i), scratch(i + 1));
                    self.reduce_pair(op, dtype, forward, back, right, left, (&mut sf, &mut sb))
                }
                [only] => {
                    let (to, from, reverse) = match i % 2 {
                        0 => (right, left, false),
                        _ => (left, right, true),
                    };
                    self.reduce_part(op, dtype, only, to, from, reverse, &mut scratch(i))
                }
                _ => Ok(()),
            }
        };
        let mut units: Vec<(usize, Vec<&mut [u8]>)> = Vec::new();
        for (i, chunk) in chunks.into_iter().enumerate() {
            match units.last_mut() {
                Some((first, parts)) if self.size == 2 && i % 2 == 1 && *first == i - 1 => {
                    parts.push(chunk);
                }
                _ => units.push((i, vec![chunk])),
            }
        }
        // Every unit but the first on a helper thread, the first on this one.
        fn all(
            unit: &(impl Fn(usize, Vec<&mut [u8]>) -> Result<()> + Sync),
            mut units: Vec<(usize, Vec<&mut [u8]>)>,
        ) -> Result<()> {
            let Some((i, parts)) = units.pop() else {
                return Ok(());
            };
            if units.is_empty() {
                return unit(i, parts);
            }
            let (last, rest) = crate::pool::join(|| unit(i, parts), || all(unit, units));
            rest.and(last)
        }
        all(&unit, units)
    }

    fn sum_scatter_now(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        // Every chunk is split over the connections (at least 256 KB per rank's segment each, as
        // the all-reduce); each connection reduces its part of every chunk around its own ring.
        let chunk = output.len();
        let k = self
            .right
            .len()
            .min(chunk * self.size / (self.size * 262_144))
            .max(1);
        let ranges = parts(chunk, dtype.size(), k);
        let mut bufs: Vec<Vec<u8>> = ranges
            .iter()
            .map(|r| {
                input
                    .chunks(chunk.max(1))
                    .flat_map(|c| &c[r.clone()])
                    .copied()
                    .collect()
            })
            .collect();
        let rank = self.rank;
        let result = thread::scope(|scope| {
            let mut bufs = bufs.iter_mut();
            let first = bufs.next();
            let handles: Vec<_> = bufs
                .enumerate()
                .map(|(i, buf)| {
                    let (to, from) = (&self.right[i + 1], &self.left[i + 1]);
                    scope.spawn(move || self.scatter_part(dtype, buf, to, from))
                })
                .collect();
            let first = match first {
                Some(buf) => self.scatter_part(dtype, buf, &self.right[0], &self.left[0]),
                None => Ok(()),
            };
            handles.into_iter().fold(first, |acc, h| {
                acc.and(h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("distributed: a thread panicked".into()))
                }))
            })
        });
        result?;
        for (r, buf) in ranges.iter().zip(&bufs) {
            let len = r.len();
            output[r.clone()].copy_from_slice(&buf[rank * len..(rank + 1) * len]);
        }
        Ok(())
    }

    fn all_gather_now(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let (size, rank, n) = (self.size, self.rank, input.len());
        output[rank * n..(rank + 1) * n].copy_from_slice(input);
        let (right, left) = (self.right_rank(), self.left_rank());
        for step in 0..size - 1 {
            let send = (rank + size - step) % size;
            let recv = (rank + 2 * size - step - 1) % size;
            // Received in place, as MLX's.
            let (block, out) = disjoint(output, send * n..(send + 1) * n, recv * n..(recv + 1) * n);
            self.exchange(
                &self.right[0],
                block,
                right,
                &self.left[0],
                out,
                left,
                n,
                |_, _| Ok(()),
            )?;
        }
        Ok(())
    }

    fn send_now(&self, dst: usize, data: &[u8]) -> Result<()> {
        let sockets = self.sockets(dst, true)?;
        let ranges = parts(data.len(), 1, sockets.len());
        thread::scope(|scope| {
            let handles: Vec<_> = sockets
                .iter()
                .zip(ranges)
                .map(|(s, r)| scope.spawn(move || self.write_all(s, &data[r], dst)))
                .collect();
            handles.into_iter().try_for_each(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("distributed: a thread panicked".into()))
                })
            })
        })
    }

    fn recv_now(&self, src: usize, data: &mut [u8]) -> Result<()> {
        let sockets = self.sockets(src, false)?;
        let ranges = parts(data.len(), 1, sockets.len());
        let mut chunks = Vec::with_capacity(ranges.len());
        let mut rest = data;
        for r in &ranges {
            let (head, tail) = rest.split_at_mut(r.len());
            chunks.push(head);
            rest = tail;
        }
        thread::scope(|scope| {
            let handles: Vec<_> = sockets
                .iter()
                .zip(chunks)
                .map(|(s, chunk)| scope.spawn(move || self.read_exact(s, chunk, src)))
                .collect();
            handles.into_iter().try_for_each(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("distributed: a thread panicked".into()))
                })
            })
        })
    }
}
