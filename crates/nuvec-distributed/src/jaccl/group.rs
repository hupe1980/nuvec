//! A JACCL group: the connections set up in MLX's order over the side channel, and the
//! [`Communicator`] over the mesh or the ring.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use nuvec_core::distributed::{AllReduce, Communicator};
use nuvec_core::{DType, Error, Result};

use super::config::{Config, Topology};
use super::endpoint::{Destination, Endpoint, Fabric};
use super::mesh::{Mesh, MeshWire};
use super::ring::{Ring, RingWire};
use super::side_channel::SideChannel;

enum Topo {
    Mesh(Mesh),
    Ring(Ring),
}

/// The JACCL communicator.
pub(crate) struct Jaccl {
    rank: usize,
    size: usize,
    topo: Topo,
    failed: Mutex<Option<String>>,
    /// Kept open after setup, so `health` sees a lost rank between collectives.
    side: SideChannel,
}

fn destinations(ends: &[Arc<dyn Endpoint>]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(ends.len() * Destination::BYTES);
    for e in ends {
        out.extend_from_slice(&e.destination()?.to_bytes());
    }
    Ok(out)
}

/// Every rank's connection addresses; every rank must list as many as this one.
fn gather(side: &SideChannel, info: &[u8], size: usize) -> Result<Vec<Vec<u8>>> {
    let all = side.all_gather_vec(info, size)?;
    if let Some(r) = all.iter().position(|a| a.len() != info.len()) {
        return Err(Error::InvalidArgument(format!(
            "jaccl: rank {r} has other connections than this rank: the ranks read different \
             device files"
        )));
    }
    Ok(all)
}

fn destination(all: &[Vec<u8>], rank: usize, index: usize) -> Destination {
    let at = index * Destination::BYTES;
    Destination::from_bytes(&all[rank][at..at + Destination::BYTES])
}

/// Connects this rank of the group `config` describes: opens every connection end on
/// `fabric`, exchanges their addresses over `side`, connects them, then waits for every rank.
pub(crate) fn connect(
    config: &Config,
    fabric: &dyn Fabric,
    side: SideChannel,
    large: bool,
    timeout: Duration,
) -> Result<Jaccl> {
    let (rank, size) = (config.rank(), config.size());
    let topo = match config.topology()? {
        Topology::Mesh { wires } => {
            let devices = config.mesh_devices(wires);
            let mut ends: Vec<Vec<Option<Arc<dyn Endpoint>>>> = Vec::with_capacity(wires);
            let mut info = Vec::new();
            for names in &devices {
                let mut wire = Vec::with_capacity(size);
                for (p, name) in names.iter().enumerate() {
                    if p == rank {
                        info.extend_from_slice(&Destination::default().to_bytes());
                        wire.push(None);
                    } else {
                        let e = fabric.open(name, p)?;
                        info.extend_from_slice(&e.destination()?.to_bytes());
                        wire.push(Some(e));
                    }
                }
                ends.push(wire);
            }
            let all = gather(&side, &info, size)?;
            for (w, wire) in ends.iter().enumerate() {
                for (p, e) in wire.iter().enumerate() {
                    if let Some(e) = e {
                        e.connect(&destination(&all, p, w * size + rank))?;
                    }
                }
            }
            side.barrier(size)?;
            Topo::Mesh(Mesh {
                rank,
                size,
                wires: ends
                    .into_iter()
                    .map(|e| Mutex::new(MeshWire::new(rank, e, large, timeout)))
                    .collect(),
            })
        }
        Topology::Ring { .. } => {
            let (left_rank, right_rank) = ((rank + size - 1) % size, (rank + 1) % size);
            let (ld, rd) = config.ring_devices();
            let left = ld
                .iter()
                .map(|d| fabric.open(d, left_rank))
                .collect::<Result<Vec<_>>>()?;
            let right = rd
                .iter()
                .map(|d| fabric.open(d, right_rank))
                .collect::<Result<Vec<_>>>()?;
            let (li, ri) = (destinations(&left)?, destinations(&right)?);
            let all_left = gather(&side, &li, size)?;
            let all_right = gather(&side, &ri, size)?;
            for (i, e) in left.iter().enumerate() {
                e.connect(&destination(&all_right, left_rank, i))?;
            }
            for (i, e) in right.iter().enumerate() {
                e.connect(&destination(&all_left, right_rank, i))?;
            }
            side.barrier(size)?;
            Topo::Ring(Ring {
                rank,
                size,
                wires: left
                    .into_iter()
                    .zip(right)
                    .map(|(l, r)| Mutex::new(RingWire::new(rank, size, l, r, large, timeout)))
                    .collect(),
            })
        }
    };
    Ok(Jaccl {
        rank,
        size,
        topo,
        failed: Mutex::new(None),
        side,
    })
}

impl Jaccl {
    /// Runs one operation unless an earlier one failed; a transport error marks the group
    /// failed (argument errors do not).
    fn guarded(&self, op: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut failed = self.failed.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(first) = &*failed {
            return Err(Error::Backend(format!(
                "jaccl: the group failed earlier ({first}); create a new group"
            )));
        }
        let result = op();
        if let Err(Error::Backend(e)) = &result {
            *failed = Some(e.clone());
        }
        result
    }
}

impl Communicator for Jaccl {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
        self.guarded(|| match &self.topo {
            Topo::Mesh(m) => m.all_reduce(op, dtype, data),
            Topo::Ring(r) => r.all_reduce(op, dtype, data),
        })
    }

    fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        self.guarded(|| match &self.topo {
            Topo::Mesh(m) => m.all_gather(input, output),
            Topo::Ring(r) => r.all_gather(input, output),
        })
    }

    fn send(&self, dst: usize, data: &[u8]) -> Result<()> {
        self.guarded(|| match &self.topo {
            Topo::Mesh(m) => m.send(data, dst),
            Topo::Ring(r) => r.send(data, dst),
        })
    }

    fn recv(&self, src: usize, data: &mut [u8]) -> Result<()> {
        self.guarded(|| match &self.topo {
            Topo::Mesh(m) => m.recv(data, src),
            Topo::Ring(r) => r.recv(data, src),
        })
    }

    fn health(&self) -> Result<()> {
        self.guarded(|| self.side.health())
    }

    fn sum_scatter(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        self.guarded(|| match &self.topo {
            Topo::Mesh(m) => m.sum_scatter(dtype, input, output),
            Topo::Ring(r) => r.sum_scatter(dtype, input, output),
        })
    }
}
