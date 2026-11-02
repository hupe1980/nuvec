//! The fully connected mesh over one cable per peer, ported from MLX's JACCL
//! (`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h`, MIT, see THIRD_PARTY_NOTICES).
//!
//! Every peer pair has a connection; transfers are double-buffered in frames of the
//! message's size class. Reductions add the ranks' values in rank order, so every rank gets
//! the same bits, on every path and every run.

use std::sync::Arc;
use std::time::Duration;

use nuvec_core::distributed::AllReduce;
use nuvec_core::{DType, Result};

use super::endpoint::{Completion, Endpoint, RECV, SEND, Waiter, wr_id};
use super::frame::{self, CLASSES, Frame};
use crate::reduce::combine;

/// MLX's mesh limit.
pub(crate) const MAX_PEERS: usize = 8;

/// Frames in flight per peer.
const PIPELINE: usize = 2;

/// Completions read per poll (MLX's `WC_NUM`).
const WC: usize = PIPELINE * MAX_PEERS * 2;

/// The frames of one size class: per slot, one mesh frame per rank (this rank's is the send
/// frame, registered with every peer; peer `j`'s receives from `j`), and per slot `2·size`
/// scatter frames (send to peer `p`, then receive from peer `p`).
struct Frames {
    mesh: Vec<Frame>,
    scatter: Vec<Frame>,
}

/// The mesh over one cable: this rank's connection to every peer.
pub(crate) struct MeshWire {
    rank: usize,
    size: usize,
    ends: Vec<Option<Arc<dyn Endpoint>>>,
    classes: Vec<Option<Frames>>,
    large: bool,
    timeout: Duration,
    /// This rank's chunks while the output is overwritten (MLX's staging memory).
    staging: [Vec<u8>; PIPELINE],
    completions: Vec<Completion>,
    /// Posted requests per peer not yet completed, to name the peers a wait is for.
    outstanding: Vec<usize>,
}

impl MeshWire {
    pub(crate) fn new(
        rank: usize,
        ends: Vec<Option<Arc<dyn Endpoint>>>,
        large: bool,
        timeout: Duration,
    ) -> Self {
        let size = ends.len();
        Self {
            rank,
            size,
            ends,
            classes: (0..CLASSES).map(|_| None).collect(),
            large,
            timeout,
            staging: [Vec::new(), Vec::new()],
            completions: Vec::with_capacity(WC),
            outstanding: vec![0; size],
        }
    }

    /// The frames of class `k`, allocated and registered on first use.
    fn frames(&mut self, k: usize) -> Result<&Frames> {
        if self.classes[k].is_none() {
            let len = frame::FRAME << k;
            let mut mesh = Vec::with_capacity(PIPELINE * self.size);
            let mut scatter = Vec::with_capacity(PIPELINE * 2 * self.size);
            for _ in 0..PIPELINE {
                for j in 0..self.size {
                    let f = Frame::new(len)?;
                    if j == self.rank {
                        for e in self.ends.iter().flatten() {
                            e.register(&f)?;
                        }
                    } else {
                        self.end(j).register(&f)?;
                    }
                    mesh.push(f);
                }
                for j in 0..2 * self.size {
                    let f = Frame::new(len)?;
                    let p = j % self.size;
                    if p != self.rank {
                        self.end(p).register(&f)?;
                    }
                    scatter.push(f);
                }
            }
            self.classes[k] = Some(Frames { mesh, scatter });
        }
        Ok(self.classes[k].as_ref().expect("allocated"))
    }

    fn end(&self, peer: usize) -> &Arc<dyn Endpoint> {
        self.ends[peer]
            .as_ref()
            .expect("a connection to every peer")
    }

    fn send_frame(&self, k: usize, slot: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").mesh[slot * self.size + self.rank].clone()
    }

    fn recv_frame(&self, k: usize, slot: usize, peer: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").mesh[slot * self.size + peer].clone()
    }

    fn scatter_send(&self, k: usize, slot: usize, peer: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").scatter[slot * 2 * self.size + peer].clone()
    }

    fn scatter_recv(&self, k: usize, slot: usize, peer: usize) -> Frame {
        self.classes[k].as_ref().expect("allocated").scatter
            [slot * 2 * self.size + self.size + peer]
            .clone()
    }

    fn peers(&self) -> impl Iterator<Item = usize> + use<> {
        let rank = self.rank;
        (0..self.size).filter(move |&p| p != rank)
    }

    fn post_send(&mut self, frame: &Frame, slot: usize, peer: usize) -> Result<()> {
        self.outstanding[peer] += 1;
        self.end(peer).post_send(frame, wr_id(SEND, slot, peer))
    }

    fn post_recv(&mut self, frame: &Frame, slot: usize, peer: usize) -> Result<()> {
        self.outstanding[peer] += 1;
        self.end(peer).post_recv(frame, wr_id(RECV, slot, peer))
    }

    fn post_send_all(&mut self, k: usize, slot: usize) -> Result<()> {
        let f = self.send_frame(k, slot);
        for p in self.peers() {
            self.post_send(&f, slot, p)?;
        }
        Ok(())
    }

    fn post_recv_all(&mut self, k: usize, slot: usize) -> Result<()> {
        for p in self.peers() {
            let f = self.recv_frame(k, slot, p);
            self.post_recv(&f, slot, p)?;
        }
        Ok(())
    }

    /// Waits for completions on `peers` (all when `None`) and returns them, every one
    /// successful.
    fn poll(&mut self, waiter: &mut Waiter, peers: Option<usize>) -> Result<Vec<Completion>> {
        let mut out = std::mem::take(&mut self.completions);
        out.clear();
        loop {
            match peers {
                Some(p) => {
                    self.end(p).poll(&mut out, WC)?;
                }
                None => {
                    for e in self.ends.iter().flatten() {
                        let room = WC - out.len();
                        if room == 0 {
                            break;
                        }
                        e.poll(&mut out, room)?;
                    }
                }
            }
            let outstanding = &self.outstanding;
            waiter.polled(out.len(), || {
                let ranks: Vec<_> = outstanding
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| **n > 0)
                    .map(|(r, _)| r.to_string())
                    .collect();
                format!("rank {}", ranks.join(", rank "))
            })?;
            if !out.is_empty() {
                for c in &out {
                    self.outstanding[c.tag()] -= 1;
                    c.check(c.tag())?;
                }
                return Ok(out);
            }
        }
    }

    fn done(&mut self, completions: Vec<Completion>) {
        self.completions = completions;
    }

    /// MLX's fully connected all-reduce of `data` in place: rank 0's chunk seeds the output
    /// and every later rank's chunk is reduced into it in rank order.
    pub(crate) fn all_reduce(
        &mut self,
        op: AllReduce,
        dtype: DType,
        data: &mut [u8],
    ) -> Result<()> {
        let total = data.len();
        if total == 0 {
            return Ok(());
        }
        let (k, n) = frame::class(total, self.large);
        self.frames(k)?;
        for s in &mut self.staging {
            s.resize(n, 0);
        }
        let peers = self.size - 1;
        let chunks = total.div_ceil(n);
        let mut waiter = Waiter::new(self.timeout);
        let (mut in_flight, mut read) = (0usize, 0usize);
        let mut sent = [0usize; PIPELINE];
        let mut recv_end = vec![0usize; self.size];
        let (mut chunk, mut from) = (0usize, 0usize);
        let mut slot = 0;
        while read < total && slot < PIPELINE {
            self.post_recv_all(k, slot)?;
            let len = n.min(total - read);
            self.staging[slot][..len].copy_from_slice(&data[read..read + len]);
            self.send_frame(k, slot).write(&data[read..read + len]);
            recv_end[self.rank] += 1;
            self.post_send_all(k, slot)?;
            slot += 1;
            in_flight += 2 * peers;
            read += n;
        }
        while chunk < chunks {
            let done = self.poll(&mut waiter, None)?;
            for c in &done {
                in_flight -= 1;
                if c.kind() == SEND && read < total {
                    let b = c.slot();
                    sent[b] += 1;
                    if sent[b] == peers {
                        let len = n.min(total - read);
                        self.send_frame(k, b).write(&data[read..read + len]);
                        self.post_send_all(k, b)?;
                        sent[b] = 0;
                        in_flight += peers;
                        read += n;
                    }
                } else if c.kind() == RECV {
                    recv_end[c.tag()] += 1;
                }
            }
            self.done(done);
            // Reduce the chunks that have arrived, in order.
            while chunk < chunks {
                let at = chunk * n;
                if at >= read || recv_end[from] <= chunk {
                    break;
                }
                let b = chunk % PIPELINE;
                let len = n.min(total - at);
                let out = &mut data[at..at + len];
                if from == self.rank {
                    if from == 0 {
                        out.copy_from_slice(&self.staging[b][..len]);
                    } else {
                        combine(op, dtype, out, &self.staging[b][..len])?;
                    }
                    let next = chunk + PIPELINE;
                    if next < chunks {
                        let off = next * n;
                        let len = n.min(total - off);
                        self.staging[b][..len].copy_from_slice(&data[off..off + len]);
                        recv_end[self.rank] += 1;
                    }
                } else {
                    let f = self.recv_frame(k, b, from);
                    f.read(len, |src| {
                        if from == 0 {
                            out.copy_from_slice(src);
                            Ok(())
                        } else {
                            combine(op, dtype, out, src)
                        }
                    })?;
                    if chunk + PIPELINE < chunks {
                        self.post_recv(&f, b, from)?;
                        in_flight += 1;
                    }
                }
                from += 1;
                if from == self.size {
                    from = 0;
                    chunk += 1;
                }
            }
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter, None)?;
            in_flight -= done.len();
            self.done(done);
        }
        Ok(())
    }

    /// MLX's fully connected all-gather: `regions[r]` receives rank `r`'s bytes; this rank's
    /// region holds its own.
    pub(crate) fn all_gather(&mut self, regions: &mut [&mut [u8]]) -> Result<()> {
        let total = regions[self.rank].len();
        if total == 0 {
            return Ok(());
        }
        let (k, n) = frame::class(total, self.large);
        self.frames(k)?;
        let peers = self.size - 1;
        let mut waiter = Waiter::new(self.timeout);
        let (mut in_flight, mut read) = (0usize, 0usize);
        let mut sent = [0usize; PIPELINE];
        let mut written = vec![0usize; self.size];
        let mut slot = 0;
        while read < total && slot < PIPELINE {
            self.post_recv_all(k, slot)?;
            let end = (read + n).min(total);
            self.send_frame(k, slot)
                .write(&regions[self.rank][read..end]);
            self.post_send_all(k, slot)?;
            slot += 1;
            in_flight += 2 * peers;
            read += n;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter, None)?;
            for c in &done {
                in_flight -= 1;
                let b = c.slot();
                if c.kind() == SEND && read < total {
                    sent[b] += 1;
                    if sent[b] == peers {
                        let end = (read + n).min(total);
                        self.send_frame(k, b).write(&regions[self.rank][read..end]);
                        self.post_send_all(k, b)?;
                        sent[b] = 0;
                        in_flight += peers;
                        read += n;
                    }
                } else if c.kind() == RECV {
                    let p = c.tag();
                    let at = written[p];
                    let len = n.min(total - at);
                    let f = self.recv_frame(k, b, p);
                    f.read(len, |src| regions[p][at..at + len].copy_from_slice(src));
                    written[p] += n;
                    if written[p] + n * (PIPELINE - 1) < total {
                        self.post_recv(&f, b, p)?;
                        in_flight += 1;
                    }
                }
            }
            self.done(done);
        }
        Ok(())
    }

    /// The fully connected reduce-scatter: every rank sends its chunk `p` to rank `p` and
    /// reduces every rank's chunk `rank` into `out` with `op`, slice by slice in rank order
    /// (MLX reduces in arrival order). `parts[p]` is
    /// this rank's chunk `p` for every peer; this rank's own chunk is `own`, or `out` itself
    /// when `None` (in place).
    pub(crate) fn reduce_scatter(
        &mut self,
        op: AllReduce,
        dtype: DType,
        parts: &[&[u8]],
        own: Option<&[u8]>,
        out: &mut [u8],
    ) -> Result<()> {
        let total = out.len();
        if total == 0 {
            return Ok(());
        }
        let (k, n) = frame::class(total, self.large);
        self.frames(k)?;
        self.staging[0].resize(n, 0);
        let peers = self.size - 1;
        let slices = total.div_ceil(n);
        let mut waiter = Waiter::new(self.timeout);
        let (mut in_flight, mut read) = (0usize, 0usize);
        let mut sent = [0usize; PIPELINE];
        // Received slices per peer; slice `s` is reduced once every peer's has arrived.
        let mut arrived = vec![0usize; self.size];
        let mut reduced = 0usize;
        let mut slot = 0;
        while read < total && slot < PIPELINE {
            let len = n.min(total - read);
            for p in self.peers() {
                let f = self.scatter_recv(k, slot, p);
                self.post_recv(&f, slot, p)?;
            }
            for p in self.peers() {
                let f = self.scatter_send(k, slot, p);
                f.write(&parts[p][read..read + len]);
                self.post_send(&f, slot, p)?;
            }
            slot += 1;
            in_flight += 2 * peers;
            read += n;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter, None)?;
            for c in &done {
                in_flight -= 1;
                let b = c.slot();
                if c.kind() == SEND && read < total {
                    sent[b] += 1;
                    if sent[b] == peers {
                        let len = n.min(total - read);
                        for p in self.peers() {
                            let f = self.scatter_send(k, b, p);
                            f.write(&parts[p][read..read + len]);
                            self.post_send(&f, b, p)?;
                        }
                        sent[b] = 0;
                        in_flight += peers;
                        read += n;
                    }
                } else if c.kind() == RECV {
                    arrived[c.tag()] += 1;
                }
            }
            self.done(done);
            while reduced < slices && self.peers().all(|p| arrived[p] > reduced) {
                let (b, at) = (reduced % PIPELINE, reduced * n);
                let len = n.min(total - at);
                let mut mine = std::mem::take(&mut self.staging[0]);
                mine[..len].copy_from_slice(match own {
                    Some(own) => &own[at..at + len],
                    None => &out[at..at + len],
                });
                let dst = &mut out[at..at + len];
                for p in 0..self.size {
                    if p == self.rank {
                        if p == 0 {
                            dst.copy_from_slice(&mine[..len]);
                        } else {
                            combine(op, dtype, dst, &mine[..len])?;
                        }
                    } else {
                        self.scatter_recv(k, b, p).read(len, |src| {
                            if p == 0 {
                                dst.copy_from_slice(src);
                                Ok(())
                            } else {
                                combine(op, dtype, dst, src)
                            }
                        })?;
                    }
                }
                self.staging[0] = mine;
                if (reduced + PIPELINE) * n < total {
                    for p in self.peers() {
                        let f = self.scatter_recv(k, b, p);
                        self.post_recv(&f, b, p)?;
                        in_flight += 1;
                    }
                }
                reduced += 1;
            }
        }
        Ok(())
    }

    /// MLX's bandwidth path for large messages on more than two ranks: a reduce-scatter
    /// into `data`'s own chunk, then an all-gather of the reduced chunks, in place; a tail
    /// of fewer than `size` elements takes the fully connected path.
    pub(crate) fn all_reduce_scatter_gather(
        &mut self,
        op: AllReduce,
        dtype: DType,
        data: &mut [u8],
    ) -> Result<()> {
        let elem = dtype.size();
        let chunk = data.len() / elem / self.size * elem;
        let (body, tail) = data.split_at_mut(chunk * self.size);
        if chunk > 0 {
            let mut chunks: Vec<&mut [u8]> = body.chunks_mut(chunk).collect();
            let own = std::mem::take(&mut chunks[self.rank]);
            let parts: Vec<&[u8]> = chunks.iter().map(|c| &**c).collect();
            self.reduce_scatter(op, dtype, &parts, None, own)?;
            let mut chunks: Vec<&mut [u8]> = body.chunks_mut(chunk).collect();
            self.all_gather(&mut chunks)?;
        }
        if !tail.is_empty() {
            self.all_reduce(op, dtype, tail)?;
        }
        Ok(())
    }

    /// MLX's point-to-point send to `dst`, double-buffered.
    pub(crate) fn send(&mut self, data: &[u8], dst: usize) -> Result<()> {
        let total = data.len();
        if total == 0 {
            return Ok(());
        }
        let (k, n) = frame::class(total, self.large);
        self.frames(k)?;
        let mut waiter = Waiter::new(self.timeout);
        let (mut in_flight, mut read, mut slot) = (0usize, 0usize, 0usize);
        while read < total && slot < PIPELINE {
            let f = self.send_frame(k, slot);
            f.write(&data[read..(read + n).min(total)]);
            self.post_send(&f, slot, dst)?;
            slot += 1;
            read += n;
            in_flight += 1;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter, Some(dst))?;
            for c in &done {
                in_flight -= 1;
                if read < total {
                    let f = self.send_frame(k, c.slot());
                    f.write(&data[read..(read + n).min(total)]);
                    self.post_send(&f, c.slot(), dst)?;
                    read += n;
                    in_flight += 1;
                }
            }
            self.done(done);
        }
        Ok(())
    }

    /// MLX's point-to-point receive from `src`.
    pub(crate) fn recv(&mut self, data: &mut [u8], src: usize) -> Result<()> {
        let total = data.len();
        if total == 0 {
            return Ok(());
        }
        let (k, n) = frame::class(total, self.large);
        self.frames(k)?;
        let mut waiter = Waiter::new(self.timeout);
        let (mut in_flight, mut written, mut slot) = (0usize, 0usize, 0usize);
        while n * slot < total && slot < PIPELINE {
            let f = self.recv_frame(k, slot, src);
            self.post_recv(&f, slot, src)?;
            in_flight += 1;
            slot += 1;
        }
        while in_flight > 0 {
            let done = self.poll(&mut waiter, Some(src))?;
            for c in &done {
                in_flight -= 1;
                let f = self.recv_frame(k, c.slot(), src);
                let len = n.min(total - written);
                f.read(len, |b| data[written..written + len].copy_from_slice(b));
                written += n;
                if written + (PIPELINE - 1) * n < total {
                    self.post_recv(&f, c.slot(), src)?;
                    in_flight += 1;
                }
            }
            self.done(done);
        }
        Ok(())
    }
}

/// From this many bytes a mesh transfer is split across the cables (MLX's
/// `MESH_MULTI_WIRE_MIN_BYTES`).
const MULTI_WIRE: usize = 512 * 1024;

/// The mesh topology: one [`MeshWire`] per cable.
pub(crate) struct Mesh {
    pub(crate) rank: usize,
    pub(crate) size: usize,
    pub(crate) wires: Vec<std::sync::Mutex<MeshWire>>,
}

/// `[w·per, (w+1)·per)` of `total`, clamped (MLX's `split_wires`).
fn wire_range(total: usize, per: usize, w: usize) -> std::ops::Range<usize> {
    (w * per).min(total)..((w + 1) * per).min(total)
}

impl Mesh {
    fn lock(&self, w: usize) -> std::sync::MutexGuard<'_, MeshWire> {
        self.wires[w]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn cables(&self, bytes: usize) -> usize {
        if bytes >= MULTI_WIRE {
            self.wires.len()
        } else {
            1
        }
    }

    /// MLX's mesh all-reduce in place: per cable, the scatter-gather path above 32 KB with
    /// more than two ranks, else the fully connected one.
    pub(crate) fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        let elem = dtype.size();
        let wires = self.cables(data.len());
        let per = (data.len() / elem).div_ceil(wires) * elem;
        let parts: Vec<_> = data.chunks_mut(per.max(elem)).collect();
        let size = self.size;
        super::on_wires(parts, |w, part| {
            let mut wire = self.lock(w);
            if size > 2 && part.len() > 32 * 1024 {
                wire.all_reduce_scatter_gather(op, dtype, part)
            } else {
                wire.all_reduce(op, dtype, part)
            }
        })
    }

    pub(crate) fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let n = input.len();
        output[self.rank * n..(self.rank + 1) * n].copy_from_slice(input);
        if n == 0 {
            return Ok(());
        }
        let wires = self.cables(n);
        let per = n.div_ceil(wires);
        let mut work: Vec<Vec<&mut [u8]>> = (0..wires).map(|_| Vec::new()).collect();
        for region in output.chunks_mut(n) {
            let mut rest = region;
            for (w, regions) in work.iter_mut().enumerate() {
                let len = wire_range(n, per, w).len();
                let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
                regions.push(head);
                rest = tail;
            }
        }
        super::on_wires(work, |w, mut regions| self.lock(w).all_gather(&mut regions))
    }

    pub(crate) fn sum_scatter(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        let elem = dtype.size();
        let n = output.len();
        if n == 0 {
            return Ok(());
        }
        let wires = self.cables(n);
        let per = (n / elem).div_ceil(wires) * elem;
        let chunks: Vec<&[u8]> = input.chunks(n).collect();
        let outs: Vec<_> = output.chunks_mut(per).collect();
        let rank = self.rank;
        super::on_wires(outs, |w, out| {
            let r = wire_range(n, per, w);
            let parts: Vec<&[u8]> = chunks.iter().map(|c| &c[r.clone()]).collect();
            self.lock(w)
                .reduce_scatter(AllReduce::Sum, dtype, &parts, Some(parts[rank]), out)
        })
    }

    pub(crate) fn send(&self, data: &[u8], dst: usize) -> Result<()> {
        let per = data.len().div_ceil(self.cables(data.len()));
        let parts: Vec<_> = data.chunks(per.max(1)).collect();
        super::on_wires(parts, |w, part| self.lock(w).send(part, dst))
    }

    pub(crate) fn recv(&self, data: &mut [u8], src: usize) -> Result<()> {
        let per = data.len().div_ceil(self.cables(data.len()));
        let parts: Vec<_> = data.chunks_mut(per.max(1)).collect();
        super::on_wires(parts, |w, part| self.lock(w).recv(part, src))
    }
}
