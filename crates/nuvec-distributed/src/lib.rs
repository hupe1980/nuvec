//! Distributed backends for nuvec: MLX's TCP ring.
//!
//! [`init`] reads MLX's launch environment (`MLX_HOSTFILE`, a JSON list with one entry per
//! rank of `"host:port"` addresses, and `MLX_RANK`), connects the ring and makes it the
//! default group of `nuvec::distributed`'s collectives. Every rank runs the same program;
//! the processes may share one Mac (over loopback) or run on several (over Ethernet or
//! Thunderbolt IP).
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

#![forbid(unsafe_code)]

mod hostfile;
mod ring;

use std::sync::Arc;
use std::time::Duration;

use nuvec_core::distributed::{Group, set_default_group};
use nuvec_core::{Error, Result};

/// Options of [`init_with`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// How long connecting and every transfer may wait for a peer before failing with an
    /// error naming it. Default 30 s (`NUVEC_DISTRIBUTED_TIMEOUT`, in seconds, for
    /// [`init`]).
    pub timeout: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }
}

impl Options {
    /// Sets the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Initialises the ring from MLX's environment (`MLX_HOSTFILE`, `MLX_RANK`) and makes it
/// the default group, as `mx.distributed.init()`. Without the environment: an error if
/// `strict`, else the group of this process alone.
pub fn init(strict: bool) -> Result<Group> {
    let (hostfile, rank) = (std::env::var("MLX_HOSTFILE"), std::env::var("MLX_RANK"));
    let (Ok(hostfile), Ok(rank)) = (hostfile, rank) else {
        if strict {
            return Err(Error::InvalidArgument(
                "distributed: MLX_HOSTFILE and MLX_RANK are not set".into(),
            ));
        }
        return Ok(Group::single());
    };
    let text = std::fs::read_to_string(&hostfile)
        .map_err(|e| Error::Io(format!("distributed: reading {hostfile}: {e}")))?;
    let rank = rank
        .trim()
        .parse()
        .map_err(|_| Error::InvalidArgument(format!("distributed: MLX_RANK {rank:?}")))?;
    let mut options = Options::default();
    if let Ok(secs) = std::env::var("NUVEC_DISTRIBUTED_TIMEOUT") {
        let invalid = || {
            Error::InvalidArgument(format!(
                "distributed: NUVEC_DISTRIBUTED_TIMEOUT {secs:?} (seconds, a positive number)"
            ))
        };
        let parsed: f64 = secs.trim().parse().map_err(|_| invalid())?;
        options.timeout = Duration::try_from_secs_f64(parsed)
            .ok()
            .filter(|d| !d.is_zero())
            .ok_or_else(invalid)?;
    }
    let group = init_with(&hostfile::parse(&text)?, rank, &options)?;
    set_default_group(group.clone());
    Ok(group)
}

/// Connects rank `rank` of the ring over `nodes` (per rank its addresses, one connection
/// per address, as MLX's hostfile) and returns its group (not made the default). A single
/// rank is the group of this process alone.
pub fn init_with(nodes: &[Vec<String>], rank: usize, options: &Options) -> Result<Group> {
    if nodes.len() <= 1 {
        return Ok(Group::single());
    }
    let ring = ring::Ring::connect(nodes, rank, options.timeout)?;
    Ok(Group::new(Arc::new(ring)))
}
