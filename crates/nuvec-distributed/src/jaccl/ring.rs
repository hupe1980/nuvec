//! The ring over Thunderbolt cables to both neighbours, ported from MLX's JACCL
//! (`mlx/distributed/jaccl/lib/jaccl/ring_impl.h`, MIT, see THIRD_PARTY_NOTICES).
//!
//! Data moves in two directions at once (direction 0 sends right and receives from the
//! left, direction 1 the reverse), each split across the cables. Every element of a
//! reduction is reduced once, by its chunk's owner, and then copied around the ring, so every
//! rank gets the same bits and reruns repeat them.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use nuvec_core::distributed::AllReduce;
use nuvec_core::{DType, Error, Result};

use super::endpoint::{Completion, Endpoint, RECV, SEND, Waiter, wr_id};
use super::frame::{self, CLASSES, Frame};
use super::on_wires;
use crate::reduce::{combine, combine_into};

/// MLX's limit of cables per direction.
pub(crate) const MAX_CABLES: usize = 4;

const PIPELINE: usize = 2;
const WC: usize = PIPELINE * 2 * 2;

/// One frame of each kind per slot and direction.
struct Frames {
    send: Vec<Frame>,
    recv: Vec<Frame>,
}

/// One cable to each neighbour.
pub(crate) struct RingWire {
    rank: usize,
    size: usize,
    left: Arc<dyn Endpoint>,
    right: Arc<dyn Endpoint>,
    classes: Vec<Option<Frames>>,
    large: bool,
    timeout: Duration,
    completions: Vec<Completion>,
    /// Posted requests not yet completed on the left and the right connection.
    outstanding: [usize; 2],
}

/// The bytes `[i·n, (i+1)·n)` of a piece, clamped to it.
fn frame_range(len: usize, i: usize, n: usize) -> std::ops::Range<usize> {
    (i * n).min(len)..((i + 1) * n).min(len)
}

impl RingWire {
    pub(crate) fn new(
        rank: usize,
        size: usize,
        left: Arc<dyn Endpoint>,
        right: Arc<dyn Endpoint>,
        large: bool,
        timeout: Duration,
    ) -> Self {
        Self {
            rank,
            size,
            left,
            right,
            classes: (0..CLASSES).map(|_| None).collect(),
            large,
            timeout,
            completions: Vec::with_capacity(WC),
            outstanding: [0; 2],
        }
    }

    fn left_rank(&self) -> usize {
        (self.rank + self.size - 1) % self.size
    }

    fn right_rank(&self) -> usize {
        (self.rank + 1) % self.size
    }

    /// Direction 0 sends right and receives from the left; direction 1 the reverse
    /// (MLX's `send_to`/`recv_from`). Returns the endpoint and its side (0 left, 1 right).
    fn sending(&self, dir: usize) -> (&Arc<dyn Endpoint>, usize) {
        if dir == 1 {
            (&self.left, 0)
        } else {
            (&self.right, 1)
        }
    }

    fn receiving(&self, dir: usize) -> (&Arc<dyn Endpoint>, usize) {
        if dir == 1 {
            (&self.right, 1)
        } else {
            (&self.left, 0)
        }
    }

    fn frames(&mut self, k: usize) -> Result<()> {
        if self.classes[k].is_none() {
            let len = frame::FRAME << k;
            let (mut send, mut recv) = (Vec::new(), Vec::new());
            for _slot in 0..PIPELINE {
                for dir in 0..2 {
                    let f = Frame::new(len)?;
                    self.sending(dir).0.register(&f)?;
                    send.push(f);
                    let f = Frame::new(len)?;
                    self.receiving(dir).0.register(&f)?;
                    recv.push(f);
                }
            }
            self.classes[k] = Some(Frames { send, recv });
        }
        Ok(())
    }

    fn send_frame(&self, k: usize, slot: usize, dir: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").send[slot * 2 + dir].clone()
    }

    fn recv_frame(&self, k: usize, slot: usize, dir: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").recv[slot * 2 + dir].clone()
    }

    fn post_send(&mut self, k: usize, slot: usize, dir: usize) -> Result<()> {
        let f = self.send_frame(k, slot, dir);
        let (e, side) = self.sending(dir);
        let e = e.clone();
        self.outstanding[side] += 1;
        e.post_send(&f, wr_id(SEND, slot, dir))
    }

    fn post_recv(&mut self, k: usize, slot: usize, dir: usize) -> Result<()> {
        let f = self.recv_frame(k, slot, dir);
        let (e, side) = self.receiving(dir);
        let e = e.clone();
        self.outstanding[side] += 1;
        e.post_recv(&f, wr_id(RECV, slot, dir))
    }

    /// Waits for completions on both connections, every one successful.
    fn poll(&mut self, waiter: &mut Waiter) -> Result<Vec<Completion>> {
        let mut out = std::mem::take(&mut self.completions);
        out.clear();
        loop {
            self.left.poll(&mut out, WC)?;
            let split = out.len();
            let room = WC - out.len();
            self.right.poll(&mut out, room)?;
            let (left, right) = (self.left_rank(), self.right_rank());
            let outstanding = self.outstanding;
            waiter.polled(out.len(), || match outstanding {
                [0, _] => format!("rank {right}"),
                [_, 0] => format!("rank {left}"),
                _ if left == right => format!("rank {left}"),
                _ => format!("rank {left}, rank {right}"),
            })?;
            if !out.is_empty() {
                for (i, c) in out.iter().enumerate() {
                    let side = usize::from(i >= split);
                    self.outstanding[side] -= 1;
                    c.check(if side == 0 { left } else { right })?;
                }
                return Ok(out);
            }
        }
    }

    /// `size − 1` pipelined steps in both directions (MLX's `ring_pass`): each step sends the
    /// piece of chunk `send[d]` and receives into chunk `recv[d]`, reducing with `op` or
    /// copying, then the windows rotate (direction 0 backward, 1 forward).
    #[allow(clippy::too_many_arguments)]
    fn ring_pass(
        &mut self,
        k: usize,
        n: usize,
        steps: usize,
        dirs: usize,
        views: &mut [Vec<&mut [u8]>; 2],
        send: &mut [usize; 2],
        recv: &mut [usize; 2],
        op: Option<(AllReduce, DType)>,
    ) -> Result<()> {
        let mut waiter = Waiter::new(self.timeout);
        for _ in 0..self.size - 1 {
            let (mut sent, mut received, mut in_flight, mut slot) = ([0; 2], [0; 2], 0, 0);
            while slot < steps && slot < PIPELINE {
                for d in 0..dirs {
                    self.post_recv(k, slot, d)?;
                }
                for d in 0..dirs {
                    let piece = &views[d][send[d]];
                    self.send_frame(k, slot, d)
                        .write(&piece[frame_range(piece.len(), sent[d], n)]);
                    sent[d] += 1;
                    self.post_send(k, slot, d)?;
                }
                slot += 1;
                in_flight += 2 * dirs;
            }
            while in_flight > 0 {
                let done = self.poll(&mut waiter)?;
                for c in &done {
                    in_flight -= 1;
                    let (d, b) = (c.tag(), c.slot());
                    if c.kind() == SEND && sent[d] < steps {
                        let piece = &views[d][send[d]];
                        self.send_frame(k, b, d)
                            .write(&piece[frame_range(piece.len(), sent[d], n)]);
                        self.post_send(k, b, d)?;
                        in_flight += 1;
                        sent[d] += 1;
                    } else if c.kind() == RECV {
                        let piece = &mut views[d][recv[d]];
                        let r = frame_range(piece.len(), received[d], n);
                        let dst = &mut piece[r];
                        self.recv_frame(k, b, d).read(dst.len(), |src| match op {
                            Some((op, dtype)) => combine(op, dtype, dst, src),
                            None => {
                                dst.copy_from_slice(src);
                                Ok(())
                            }
                        })?;
                        received[d] += 1;
                        if received[d] + (PIPELINE - 1) < steps {
                            self.post_recv(k, b, d)?;
                            in_flight += 1;
                        }
                    }
                }
                self.completions = done;
            }
            let s = self.size;
            send[0] = (send[0] + s - 1) % s;
            recv[0] = (recv[0] + s - 1) % s;
            send[1] = (send[1] + 1) % s;
            recv[1] = (recv[1] + 1) % s;
        }
        Ok(())
    }

    /// MLX's standalone dual-direction reduce-scatter on one cable: both directions
    /// converge on chunk `rank`; `out[d]` is this cable's piece of the output for direction
    /// `d` and `input[d][c]` the same piece of input chunk `c`. A received slice is reduced
    /// only once the send of the partial it overwrites is staged.
    #[allow(clippy::needless_range_loop)] // `d` indexes several per-direction arrays
    fn reduce_scatter(
        &mut self,
        dtype: DType,
        k: usize,
        n: usize,
        steps: usize,
        input: &[Vec<&[u8]>; 2],
        out: &mut [&mut [u8]; 2],
    ) -> Result<()> {
        let s = self.size;
        let r = self.rank;
        let mut in_send = [(r + s - 1) % s, (r + 1) % s];
        let mut in_recv = [(r + 2 * s - 2) % s, (r + 2) % s];
        let mut waiter = Waiter::new(self.timeout);
        for step in 0..s - 1 {
            let (mut sent, mut received, mut deferred) = ([0; 2], [0; 2], [0; 2]);
            let (mut in_flight, mut slot) = (0, 0);
            let stage = |this: &Self, out: &[&mut [u8]; 2], b: usize, d: usize, i: usize| {
                let piece: &[u8] = if step == 0 {
                    input[d][in_send[d]]
                } else {
                    out[d]
                };
                this.send_frame(k, b, d)
                    .write(&piece[frame_range(piece.len(), i, n)]);
            };
            while slot < steps && slot < PIPELINE {
                for d in 0..2 {
                    self.post_recv(k, slot, d)?;
                }
                for d in 0..2 {
                    stage(self, out, slot, d, sent[d]);
                    sent[d] += 1;
                    self.post_send(k, slot, d)?;
                }
                slot += 1;
                in_flight += 4;
            }
            while in_flight > 0 {
                let done = self.poll(&mut waiter)?;
                for c in &done {
                    in_flight -= 1;
                    let d = c.tag();
                    if c.kind() == SEND {
                        if sent[d] < steps {
                            stage(self, out, c.slot(), d, sent[d]);
                            sent[d] += 1;
                            self.post_send(k, c.slot(), d)?;
                            in_flight += 1;
                        }
                    } else if c.kind() == RECV {
                        deferred[d] += 1;
                    }
                    // Apply the receives whose partials are staged, in arrival order.
                    while deferred[d] > 0 && received[d] < sent[d] {
                        let b = received[d] % PIPELINE;
                        let rg = frame_range(out[d].len(), received[d], n);
                        let base = &input[d][in_recv[d]][rg.clone()];
                        let dst = &mut out[d][rg];
                        self.recv_frame(k, b, d).read(dst.len(), |src| {
                            combine_into(AllReduce::Sum, dtype, dst, base, src)
                        })?;
                        received[d] += 1;
                        deferred[d] -= 1;
                        if received[d] + (PIPELINE - 1) < steps {
                            self.post_recv(k, b, d)?;
                            in_flight += 1;
                        }
                    }
                }
                self.completions = done;
            }
            in_send[0] = (in_send[0] + s - 1) % s;
            in_recv[0] = (in_recv[0] + s - 1) % s;
            in_send[1] = (in_send[1] + 1) % s;
            in_recv[1] = (in_recv[1] + 1) % s;
        }
        Ok(())
    }

    /// MLX's point-to-point send of one cable's piece in direction `dir`; `frame_bytes` is
    /// the per-cable size both ends size their frames by.
    fn send(&mut self, data: &[u8], dir: usize, frame_bytes: usize) -> Result<()> {
        let (k, n) = frame::class(frame_bytes, self.large);
        self.frames(k)?;
        let mut waiter = Waiter::new(self.timeout);
        let (mut read, mut in_flight, mut slot) = (0, 0, 0);
        while read < data.len() && slot < PIPELINE {
            self.send_frame(k, slot, dir)
                .write(&data[read..(read + n).min(data.len())]);
            self.post_send(k, slot, dir)?;
            slot += 1;
            read += n;
            in_flight += 1;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter)?;
            for c in &done {
                in_flight -= 1;
                if read < data.len() {
                    self.send_frame(k, c.slot(), dir)
                        .write(&data[read..(read + n).min(data.len())]);
                    self.post_send(k, c.slot(), dir)?;
                    read += n;
                    in_flight += 1;
                }
            }
            self.completions = done;
        }
        Ok(())
    }

    fn recv(&mut self, data: &mut [u8], dir: usize, frame_bytes: usize) -> Result<()> {
        let (k, n) = frame::class(frame_bytes, self.large);
        self.frames(k)?;
        let mut waiter = Waiter::new(self.timeout);
        let (mut written, mut in_flight, mut slot) = (0, 0, 0);
        while n * slot < data.len() && slot < PIPELINE {
            self.post_recv(k, slot, dir)?;
            slot += 1;
            in_flight += 1;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter)?;
            for c in &done {
                in_flight -= 1;
                let len = n.min(data.len() - written);
                let dst = &mut data[written..written + len];
                self.recv_frame(k, c.slot(), dir)
                    .read(len, |src| dst.copy_from_slice(src));
                written += n;
                if written + (PIPELINE - 1) * n < data.len() {
                    self.post_recv(k, c.slot(), dir)?;
                    in_flight += 1;
                }
            }
            self.completions = done;
        }
        Ok(())
    }
}

/// Splits `data` (regions of `region` bytes, the last possibly shorter) into per-cable,
/// per-direction pieces: piece `(d, w)` of every region is `[(d·wires + w)·piece, +piece)`,
/// clamped to the region (MLX's `wire_offset`/`wire_end`). Returns `[cable][direction][region]`.
#[allow(clippy::needless_range_loop)] // `d` and `w` index the piece layout
fn split(
    mut data: &mut [u8],
    region: usize,
    regions: usize,
    piece: usize,
    dirs: usize,
    wires: usize,
) -> Vec<[Vec<&mut [u8]>; 2]> {
    let mut out: Vec<[Vec<&mut [u8]>; 2]> = (0..wires).map(|_| [Vec::new(), Vec::new()]).collect();
    for _ in 0..regions {
        let len = region.min(data.len());
        let (mut r, rest) = std::mem::take(&mut data).split_at_mut(len);
        data = rest;
        for d in 0..dirs {
            for w in 0..wires {
                let end = (piece * (d * wires + w + 1)).min(len);
                let start = (piece * (d * wires + w)).min(len);
                let (p, tail) = std::mem::take(&mut r).split_at_mut(end - start);
                r = tail;
                out[w][d].push(p);
            }
        }
    }
    out
}

/// [`split`] of a shared buffer.
fn split_ref(
    data: &[u8],
    region: usize,
    regions: usize,
    piece: usize,
    wires: usize,
) -> Vec<[Vec<&[u8]>; 2]> {
    let mut out: Vec<[Vec<&[u8]>; 2]> = (0..wires).map(|_| [Vec::new(), Vec::new()]).collect();
    for c in 0..regions {
        let r = &data[(c * region).min(data.len())..((c + 1) * region).min(data.len())];
        for d in 0..2 {
            for (w, o) in out.iter_mut().enumerate() {
                let start = (piece * (d * wires + w)).min(r.len());
                let end = (piece * (d * wires + w + 1)).min(r.len());
                o[d].push(&r[start..end]);
            }
        }
    }
    out
}

/// The ring topology: one [`RingWire`] per cable.
pub(crate) struct Ring {
    pub(crate) rank: usize,
    pub(crate) size: usize,
    pub(crate) wires: Vec<Mutex<RingWire>>,
}

impl Ring {
    fn lock(&self, w: usize) -> std::sync::MutexGuard<'_, RingWire> {
        self.wires[w].lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// MLX's ring all-reduce in place: one direction on one cable up to 32 KB (or when the
    /// elements do not fill both directions of every cable), else both on every cable.
    pub(crate) fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        let elem = dtype.size();
        let count = data.len() / elem;
        let cables = self.wires.len();
        let (dirs, wires) = if data.len() <= 32768 || count < self.size * 2 * cables {
            (1, 1)
        } else {
            (2, cables)
        };
        let chunk = count.div_ceil(self.size);
        let per_wire = chunk.div_ceil(dirs * wires);
        let large = self.lock(0).large;
        let (k, n) = frame::class(per_wire * elem, large);
        let steps = (per_wire * elem).div_ceil(n);
        let views = split(data, chunk * elem, self.size, per_wire * elem, dirs, wires);
        let (rank, size) = (self.rank, self.size);
        on_wires(views, |w, mut views| {
            let mut wire = self.lock(w);
            wire.frames(k)?;
            let mut send = [rank; 2];
            let mut recv = [(rank + size - 1) % size, (rank + 1) % size];
            wire.ring_pass(
                k,
                n,
                steps,
                dirs,
                &mut views,
                &mut send,
                &mut recv,
                Some((op, dtype)),
            )?;
            wire.ring_pass(k, n, steps, dirs, &mut views, &mut send, &mut recv, None)
        })
    }

    /// MLX's ring all-gather: every rank's block passes around in both directions on
    /// every cable.
    pub(crate) fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let n_bytes = input.len();
        output[self.rank * n_bytes..(self.rank + 1) * n_bytes].copy_from_slice(input);
        let wires = self.wires.len();
        let per_wire = n_bytes.div_ceil(2 * wires);
        let large = self.lock(0).large;
        let (k, n) = frame::class(per_wire, large);
        let steps = per_wire.div_ceil(n);
        let views = split(output, n_bytes, self.size, per_wire, 2, wires);
        let (rank, size) = (self.rank, self.size);
        on_wires(views, |w, mut views| {
            let mut wire = self.lock(w);
            wire.frames(k)?;
            let mut send = [rank; 2];
            let mut recv = [(rank + size - 1) % size, (rank + 1) % size];
            wire.ring_pass(k, n, steps, 2, &mut views, &mut send, &mut recv, None)
        })
    }

    /// MLX's ring reduce-scatter: one cable up to 64 KB of output (or when the elements do
    /// not fill both directions of every cable), else every cable.
    pub(crate) fn sum_scatter(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        let elem = dtype.size();
        let count = output.len() / elem;
        let cables = self.wires.len();
        let wires = if self.size * count < self.size * 2 * cables || output.len() <= 65536 {
            1
        } else {
            cables
        };
        let per_wire = count.div_ceil(2 * wires);
        let large = self.lock(0).large;
        let (k, n) = frame::class(per_wire * elem, large);
        let steps = (per_wire * elem).div_ceil(n);
        let inputs = split_ref(input, count * elem, self.size, per_wire * elem, wires);
        let outs = split(output, count * elem, 1, per_wire * elem, 2, wires);
        let work: Vec<_> = inputs.into_iter().zip(outs).collect();
        on_wires(work, |w, (input, out)| {
            let mut wire = self.lock(w);
            wire.frames(k)?;
            let [mut a, mut b] = out;
            let mut out = [a.remove(0), b.remove(0)];
            wire.reduce_scatter(dtype, k, n, steps, &input, &mut out)
        })
    }

    fn neighbour(&self, op: &str, peer: usize) -> Result<()> {
        let (left, right) = (
            (self.rank + self.size - 1) % self.size,
            (self.rank + 1) % self.size,
        );
        if peer != left && peer != right {
            return Err(Error::InvalidArgument(format!(
                "jaccl: rank {} can {op} only its neighbours {left} and {right} in the ring \
                 topology, not {peer}",
                self.rank
            )));
        }
        Ok(())
    }

    /// MLX's ring send to a neighbour, split across the cables (with two ranks the left
    /// connections send and the right ones receive).
    pub(crate) fn send(&self, data: &[u8], dst: usize) -> Result<()> {
        self.neighbour("send to", dst)?;
        let dir = usize::from(dst == (self.rank + self.size - 1) % self.size);
        let per_wire = data.len().div_ceil(self.wires.len());
        let parts: Vec<_> = data.chunks(per_wire.max(1)).collect();
        on_wires(parts, |w, part| self.lock(w).send(part, dir, per_wire))
    }

    pub(crate) fn recv(&self, data: &mut [u8], src: usize) -> Result<()> {
        self.neighbour("receive from", src)?;
        let dir = usize::from(src == (self.rank + 1) % self.size);
        let per_wire = data.len().div_ceil(self.wires.len());
        let parts: Vec<_> = data.chunks_mut(per_wire.max(1)).collect();
        on_wires(parts, |w, part| self.lock(w).recv(part, dir, per_wire))
    }
}
