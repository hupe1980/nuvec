//! Distributed backends for nuvec: MLX's TCP ring, and MLX's JACCL (RDMA over
//! Thunderbolt 5; experimental, feature `jaccl`).
//!
//! [`init`] reads MLX's launch environment, connects a backend and makes it the default
//! group of `nuvec::distributed`'s collectives, as `mx.distributed.init()`. The ring reads
//! `MLX_HOSTFILE` (a JSON list with one entry per rank of `"host:port"` addresses) and
//! `MLX_RANK`; JACCL reads the variables `mlx.launch --backend jaccl` writes
//! (`jaccl::Config::from_env`). [`init_backend`] names the backend. Every rank runs the
//! same program; the processes may share one Mac (over loopback) or run on several (over
//! Ethernet or Thunderbolt IP for the ring, Thunderbolt 5 cables for JACCL).
//!
//! ```no_run
//! use nuvec_core::{array, distributed};
//!
//! # fn main() -> nuvec_core::Result<()> {
//! let group = nuvec_distributed::init(false)?; // size 1 without a hostfile
//! let x = array![1.0, 2.0];
//! let total = distributed::all_sum(&x, None); // the sum over every rank
//! println!("rank {} of {}: {:?}", group.rank(), group.size(), total.to_vec::<f32>()?);
//! # Ok(())
//! # }
//! ```
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.

#![deny(unsafe_code)]

mod auth;
mod backend;
mod hostfile;
#[cfg(feature = "jaccl")]
pub mod jaccl;
mod json;
mod keepalive;
mod pool;
mod reduce;
mod ring;
mod socket;

use std::sync::Arc;
use std::time::Duration;

use nuvec_core::Result;
use nuvec_core::distributed::Group;

pub use backend::{Backend, init_backend, is_available};

/// Options of [`init_with`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// How long connecting and every transfer may wait for a peer before failing with an
    /// error naming it. Default 30 s (`NUVEC_DISTRIBUTED_TIMEOUT`, in seconds, for
    /// [`init`]).
    pub timeout: Duration,
    key: auth::Secret,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            key: auth::Secret::default(),
        }
    }
}

impl Options {
    /// Sets the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Authenticates every connection with this pre-shared secret (at least 16 bytes, the
    /// same on every rank; `NUVEC_DISTRIBUTED_KEY` for [`init`]): a host without it cannot
    /// join the group. The traffic itself is not encrypted; on an untrusted network, run the
    /// group over a VPN.
    pub fn key(mut self, secret: impl AsRef<[u8]>) -> Self {
        self.key = auth::Secret(Some(secret.as_ref().to_vec()));
        self
    }
}

/// Initialises a backend from MLX's environment and makes it the default group, as
/// `mx.distributed.init()`: [`init_backend`] with [`Backend::Any`].
pub fn init(strict: bool) -> Result<Group> {
    init_backend(strict, Backend::Any)
}

/// Connects rank `rank` of the ring over `nodes` (per rank its addresses, one connection
/// per address, as MLX's hostfile) and returns its group (not made the default). A single
/// rank is the group of this process alone.
pub fn init_with(nodes: &[Vec<String>], rank: usize, options: &Options) -> Result<Group> {
    if nodes.len() <= 1 {
        return Ok(Group::single());
    }
    let key = options.key.key()?;
    let ring = ring::Ring::connect(nodes, rank, options.timeout, key.as_ref())?;
    Ok(Group::new(Arc::new(ring)))
}
