//! MLX's ring backend over TCP: each rank connects to its right neighbour and accepts its
//! left one, with one connection per hostfile address. Reductions are a reduce-scatter
//! followed by an all-gather around the ring, so every segment is reduced once, in ring
//! order, and then copied: all ranks get the same bits, and reruns repeat them.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::thread;
use std::time::{Duration, Instant};

use nuvec_core::distributed::{AllReduce, Communicator};
use nuvec_core::{DType, Error, Result, bf16, f16};

pub(crate) struct Ring {
    rank: usize,
    size: usize,
    /// Connections to the right neighbour (this rank connected), one per address.
    right: Vec<TcpStream>,
    /// Connections from the left neighbour (this rank accepted).
    left: Vec<TcpStream>,
    /// The first transport error: a failed transfer leaves unread bytes in the sockets, so
    /// every later operation fails with it instead of reading misaligned data.
    failed: std::sync::Mutex<Option<String>>,
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
        s.set_write_timeout(Some(timeout))
    };
    set().map_err(|e| peer_error(peer, "configuring the connection", e))
}

impl Ring {
    /// Connects rank `rank` of the ring described by `nodes` (the addresses per rank).
    pub(crate) fn connect(nodes: &[Vec<String>], rank: usize, timeout: Duration) -> Result<Self> {
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
                    let s = loop {
                        match l.accept() {
                            Ok((s, _)) => break s,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                if Instant::now() > deadline {
                                    return Err(peer_error(left, "never connected", "timed out"));
                                }
                                thread::sleep(Duration::from_millis(5));
                            }
                            Err(e) => return Err(peer_error(left, "accepting", e)),
                        }
                    };
                    configure(&s, timeout, left)?;
                    let mut who = [0u8; 8];
                    (&s).read_exact(&mut who)
                        .map_err(|e| peer_error(left, "handshake", e))?;
                    if u64::from_le_bytes(who) != left as u64 {
                        return Err(peer_error(
                            left,
                            "handshake",
                            format!("rank {} connected instead", u64::from_le_bytes(who)),
                        ));
                    }
                    Ok(s)
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
                    (&s).write_all(&(rank as u64).to_le_bytes())
                        .map_err(|e| peer_error(right, "handshake", e))?;
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
        Ok(Self {
            rank,
            size,
            right: right_s,
            left: left_s,
            failed: std::sync::Mutex::new(None),
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

fn write_all(s: &TcpStream, data: &[u8], peer: usize) -> Result<()> {
    let mut s = s;
    s.write_all(data)
        .map_err(|e| peer_error(peer, "sending", e))
}

fn read_exact(s: &TcpStream, data: &mut [u8], peer: usize) -> Result<()> {
    let mut s = s;
    s.read_exact(data).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => peer_error(peer, "receiving", "connection closed"),
        _ => peer_error(peer, "receiving", e),
    })
}

/// Below this many bytes a step writes, then reads, on one thread: both directions fit in
/// the sockets' buffers, and a thread would cost more than the transfer.
const SMALL: usize = 32 << 10;

/// Packets of a step's receive: each is reduced while the next arrives (MLX's 32 KB–8 MB
/// packets, double-buffered).
const PACKET: usize = 1 << 20;

/// Sends `data` to `to` while receiving `into` from `from`; `each` consumes every received
/// packet (its offset in `into` and its bytes) as soon as it is complete.
fn exchange(
    to: &TcpStream,
    data: &[u8],
    to_rank: usize,
    from: &TcpStream,
    into: &mut [u8],
    from_rank: usize,
    mut each: impl FnMut(usize, &[u8]) -> Result<()>,
) -> Result<()> {
    if data.len() <= SMALL && into.len() <= SMALL {
        write_all(to, data, to_rank)?;
        read_exact(from, into, from_rank)?;
        return each(0, into);
    }
    thread::scope(|scope| {
        let sender = scope.spawn(|| write_all(to, data, to_rank));
        let mut received = Ok(());
        let mut at = 0;
        while at < into.len() && received.is_ok() {
            let end = (at + PACKET).min(into.len());
            received = read_exact(from, &mut into[at..end], from_rank)
                .and_then(|()| each(at, &into[at..end]));
            at = end;
        }
        let sent = sender.join().unwrap_or_else(|_| {
            Err(Error::Backend(
                "distributed: the sending thread panicked".into(),
            ))
        });
        received.and(sent)
    })
}

macro_rules! fold {
    ($acc:ident, $x:ident, $t:ty, $n:expr, $f:expr) => {{
        for (a, b) in $acc.chunks_exact_mut($n).zip($x.chunks_exact($n)) {
            let (p, q) = (
                <$t>::from_le_bytes(a.try_into().unwrap()),
                <$t>::from_le_bytes(b.try_into().unwrap()),
            );
            let f: fn($t, $t) -> $t = $f;
            a.copy_from_slice(&f(p, q).to_le_bytes());
        }
    }};
}

macro_rules! numeric {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        let n = size_of::<$t>();
        match $op {
            AllReduce::Sum => fold!($acc, $x, $t, n, |a, b| a + b),
            AllReduce::Max => fold!($acc, $x, $t, n, |a, b| if a < b { b } else { a }),
            _ => fold!($acc, $x, $t, n, |a, b| if b < a { b } else { a }),
        }
    }};
}

macro_rules! wrapping {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        let n = size_of::<$t>();
        match $op {
            AllReduce::Sum => fold!($acc, $x, $t, n, |a, b| a.wrapping_add(b)),
            AllReduce::Max => fold!($acc, $x, $t, n, |a, b| a.max(b)),
            _ => fold!($acc, $x, $t, n, |a, b| a.min(b)),
        }
    }};
}

macro_rules! half_float {
    ($op:expr, $acc:ident, $x:ident, $t:ty) => {{
        // Exact in float32, so rounded once (float32 has more than twice the bits).
        for (a, b) in $acc.chunks_exact_mut(2).zip($x.chunks_exact(2)) {
            let p = <$t>::from_bits(u16::from_le_bytes([a[0], a[1]]));
            let q = <$t>::from_bits(u16::from_le_bytes([b[0], b[1]]));
            let r = match $op {
                AllReduce::Sum => <$t>::from_f32(p.to_f32() + q.to_f32()),
                AllReduce::Max => {
                    if p < q {
                        q
                    } else {
                        p
                    }
                }
                _ => {
                    if q < p {
                        q
                    } else {
                        p
                    }
                }
            };
            a.copy_from_slice(&r.to_bits().to_le_bytes());
        }
    }};
}

/// `acc = op(acc, x)` element-wise for elements of `dtype`.
fn combine(op: AllReduce, dtype: DType, acc: &mut [u8], x: &[u8]) -> Result<()> {
    match dtype {
        DType::Bool => {
            for (a, b) in acc.iter_mut().zip(x) {
                *a = match op {
                    AllReduce::Min => u8::from(*a != 0 && *b != 0),
                    _ => u8::from(*a != 0 || *b != 0),
                };
            }
        }
        DType::U8 => wrapping!(op, acc, x, u8),
        DType::U16 => wrapping!(op, acc, x, u16),
        DType::U32 => wrapping!(op, acc, x, u32),
        DType::U64 => wrapping!(op, acc, x, u64),
        DType::I8 => wrapping!(op, acc, x, i8),
        DType::I16 => wrapping!(op, acc, x, i16),
        DType::I32 => wrapping!(op, acc, x, i32),
        DType::I64 => wrapping!(op, acc, x, i64),
        DType::F32 => numeric!(op, acc, x, f32),
        DType::F64 => numeric!(op, acc, x, f64),
        DType::F16 => half_float!(op, acc, x, f16),
        DType::BF16 => half_float!(op, acc, x, bf16),
        DType::Complex64 if op == AllReduce::Sum => fold!(acc, x, f32, 4, |a, b| a + b),
        d => {
            return Err(Error::InvalidArgument(format!(
                "distributed: no {op:?} reduction for {}",
                d.name()
            )));
        }
    }
    Ok(())
}

impl Ring {
    /// The ring reduction of `buf` (elements of `elem` bytes) over one socket pair.
    fn reduce_part(
        &self,
        op: AllReduce,
        dtype: DType,
        buf: &mut [u8],
        to: &TcpStream,
        from: &TcpStream,
    ) -> Result<()> {
        let (size, rank, elem) = (self.size, self.rank, dtype.size());
        let n = buf.len() / elem;
        let segment = |s: usize| (s * n / size) * elem..((s + 1) * n / size) * elem;
        let (right, left) = (self.right_rank(), self.left_rank());
        let mut incoming = vec![0u8; (n.div_ceil(size)) * elem];
        // Reduce-scatter: after it, this rank holds segment `rank + 1` reduced.
        for step in 0..size - 1 {
            let send = segment((rank + size - step) % size);
            let recv = segment((rank + 2 * size - step - 1) % size);
            // The segments are disjoint: send one while reducing into the other.
            let (sent, acc) = disjoint(buf, send, recv.clone());
            let into = &mut incoming[..recv.len()];
            exchange(to, sent, right, from, into, left, |at, packet| {
                combine(op, dtype, &mut acc[at..at + packet.len()], packet)
            })?;
        }
        // All-gather: pass the reduced segments around.
        for step in 0..size - 1 {
            let send = segment((rank + 1 + size - step) % size);
            let recv = segment((rank + size - step) % size);
            let (sent, out) = disjoint(buf, send, recv.clone());
            let into = &mut incoming[..recv.len()];
            exchange(to, sent, right, from, into, left, |at, packet| {
                out[at..at + packet.len()].copy_from_slice(packet);
                Ok(())
            })?;
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
}

impl Ring {
    fn all_reduce_now(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        // Large buffers are split over the connections (at least 256 KB per rank's segment
        // each, as MLX), each part reduced around the ring in parallel.
        let k = self
            .right
            .len()
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
        if k == 1 {
            return self.reduce_part(op, dtype, chunks.remove(0), &self.right[0], &self.left[0]);
        }
        thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let (to, from) = (&self.right[i], &self.left[i]);
                    scope.spawn(move || self.reduce_part(op, dtype, chunk, to, from))
                })
                .collect();
            handles.into_iter().try_for_each(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("distributed: a thread panicked".into()))
                })
            })
        })
    }

    fn all_gather_now(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let (size, rank, n) = (self.size, self.rank, input.len());
        output[rank * n..(rank + 1) * n].copy_from_slice(input);
        let (right, left) = (self.right_rank(), self.left_rank());
        let mut incoming = vec![0u8; n];
        for step in 0..size - 1 {
            let send = (rank + size - step) % size;
            let recv = (rank + 2 * size - step - 1) % size;
            let (block, out) = disjoint(output, send * n..(send + 1) * n, recv * n..(recv + 1) * n);
            exchange(
                &self.right[0],
                block,
                right,
                &self.left[0],
                &mut incoming,
                left,
                |at, p| {
                    out[at..at + p.len()].copy_from_slice(p);
                    Ok(())
                },
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
                .map(|(s, r)| scope.spawn(move || write_all(s, &data[r], dst)))
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
                .map(|(s, chunk)| scope.spawn(move || read_exact(s, chunk, src)))
                .collect();
            handles.into_iter().try_for_each(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("distributed: a thread panicked".into()))
                })
            })
        })
    }
}
