//! **Experimental:** MLX's JACCL backend, RDMA over Thunderbolt 5 (macOS 26.2+), behind the
//! `jaccl` feature. It is unverified on Thunderbolt 5 hardware; its algorithms are
//! tested on an emulated fabric.
//!
//! RDMA must be enabled once per Mac in recoveryOS (`rdma_ctl enable`), and every pair of
//! ranks cabled (the mesh) or every rank cabled to its neighbours (the ring). The group is
//! described by MLX's device file and started as MLX's (`mlx.launch --backend jaccl` or
//! `--backend jaccl-ring`); see [`Config::from_env`]. [`devices`] lists this Mac's RDMA
//! devices to check a setup.

mod config;
#[cfg(test)]
mod emulated;
mod endpoint;
mod frame;
mod group;
mod mesh;
mod ring;
mod side_channel;
#[cfg(test)]
mod tests;
#[cfg(target_os = "macos")]
mod verbs;

use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nuvec_core::distributed::Group;
use nuvec_core::{Error, Result};

pub use config::{Config, Topology, parse_devices};
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub use verbs::layout as verbs_layout;

use side_channel::{AllGather, SideChannel, Star};

/// Options of [`init_with`].
#[derive(Clone)]
#[non_exhaustive]
pub struct Options {
    /// How long connecting and every transfer may wait for a peer before failing with an
    /// error naming it. Default 30 s (`NUVEC_DISTRIBUTED_TIMEOUT`, in seconds, for `init`).
    pub timeout: Duration,
    side_channel: Option<AllGather>,
    key: crate::auth::Secret,
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("timeout", &self.timeout)
            .field(
                "side_channel",
                &self.side_channel.as_ref().map(|_| "custom"),
            )
            .field("key", &self.key)
            .finish()
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            side_channel: None,
            key: crate::auth::Secret::default(),
        }
    }
}

impl Options {
    /// Sets the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Authenticates the side channel's connections with this pre-shared secret (at least 16
    /// bytes, the same on every rank; `NUVEC_DISTRIBUTED_KEY` for `init`).
    pub fn key(mut self, secret: impl AsRef<[u8]>) -> Self {
        self.key = crate::auth::Secret(Some(secret.as_ref().to_vec()));
        self
    }

    /// Exchanges the connection details through `all_gather` instead of the coordinator's
    /// TCP star, as MLX's `all_gather_factory`: it takes this rank's bytes and returns every
    /// rank's (equal lengths), concatenated in rank order.
    pub fn side_channel(
        mut self,
        all_gather: impl Fn(&[u8]) -> Result<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        self.side_channel = Some(Arc::new(all_gather));
        self
    }
}

/// Connects this rank's JACCL group (not made the default group).
pub fn init_with(config: &Config, options: &Options) -> Result<Group> {
    #[cfg(target_os = "macos")]
    {
        let fabric = verbs::Verbs::load()?;
        let comm = connect(config, &fabric, options, frame::large_frames())?;
        Ok(Group::new(Arc::new(comm)))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (config, options);
        Err(unavailable())
    }
}

/// Connects over any fabric (the RDMA library, or the emulated one in tests).
fn connect(
    config: &Config,
    fabric: &dyn endpoint::Fabric,
    options: &Options,
    large: bool,
) -> Result<group::Jaccl> {
    let side = match &options.side_channel {
        Some(f) => SideChannel::Custom(f.clone()),
        None => {
            let coordinator = config.coordinator.as_deref().ok_or_else(|| {
                Error::InvalidArgument(
                    "jaccl: no coordinator (JACCL_COORDINATOR or MLX_JACCL_COORDINATOR) and no \
                     side channel"
                        .into(),
                )
            })?;
            SideChannel::Tcp(Mutex::new(Star::connect(
                config.rank(),
                config.size(),
                coordinator,
                options.timeout,
                options.key.key()?.as_ref(),
            )?))
        }
    };
    group::connect(config, fabric, side, large, options.timeout)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn unavailable() -> Error {
    Error::Backend(
        "jaccl: RDMA is not available (librdma not found, or no RDMA device: enable RDMA over \
         Thunderbolt with `rdma_ctl enable` in recoveryOS)"
            .into(),
    )
}

/// Whether RDMA can run here: the system library loads and lists at least one device.
pub fn is_available() -> bool {
    devices().is_ok_and(|d| !d.is_empty())
}

/// A port's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PortState {
    /// No link.
    Down,
    /// Initialising.
    Init,
    /// Armed.
    Armed,
    /// Ready.
    Active,
    /// Ready, deferred.
    ActiveDefer,
    /// A state the library reports beyond these.
    Unknown(i32),
}

/// One of this Mac's RDMA devices (a Thunderbolt port).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RdmaDevice {
    /// The device name, as in the device file (e.g. `rdma_en3`).
    pub name: String,
    /// Port 1's state.
    pub port_state: PortState,
    /// The IPv4 address of the port's IPv4-mapped GID; `None` means the Thunderbolt
    /// interface has no IPv4 address yet, which connecting needs.
    pub ipv4: Option<Ipv4Addr>,
}

/// This Mac's RDMA devices (empty when RDMA is not enabled); an error when the system
/// library cannot be loaded.
pub fn devices() -> Result<Vec<RdmaDevice>> {
    #[cfg(target_os = "macos")]
    {
        verbs::devices()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(unavailable())
    }
}

/// Runs `f` for every item, the first on this thread and the others on one thread each
/// (one per cable), and returns the first error.
pub(crate) fn on_wires<T: Send>(
    items: Vec<T>,
    f: impl Fn(usize, T) -> Result<()> + Sync,
) -> Result<()> {
    let mut items = items.into_iter().enumerate();
    let Some((_, first)) = items.next() else {
        return Ok(());
    };
    if items.len() == 0 {
        return f(0, first);
    }
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = items.map(|(w, item)| s.spawn(move || f(w, item))).collect();
        let first = f(0, first);
        let rest: Vec<Result<()>> = handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::Backend("jaccl: a cable's thread panicked".into()))
                })
            })
            .collect();
        first.and(rest.into_iter().collect())
    })
}
