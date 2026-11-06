//! MLX's backend selection: `init(strict, backend)` with a group cached per backend, and
//! `is_available(backend)`.

use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use nuvec_core::distributed::{Group, set_default_group};
use nuvec_core::{Error, Result};

use crate::{Options, hostfile, init_with};

/// A distributed backend (MLX's `backend` argument).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The first backend whose environment is set: the ring, then JACCL.
    Any,
    /// MLX's TCP ring.
    Ring,
    /// MLX's JACCL, RDMA over Thunderbolt 5 (experimental, feature `jaccl`).
    Jaccl,
}

impl std::str::FromStr for Backend {
    type Err = Error;

    /// MLX's names: `any`, `ring`, `jaccl`.
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "any" => Ok(Self::Any),
            "ring" => Ok(Self::Ring),
            "jaccl" => Ok(Self::Jaccl),
            _ => Err(Error::InvalidArgument(format!(
                "distributed: no backend {s:?} (any, ring or jaccl)"
            ))),
        }
    }
}

/// Initialised groups per backend; `Any` maps to the first one (MLX's `get_backends`).
static GROUPS: Mutex<Vec<(Backend, Group)>> = Mutex::new(Vec::new());

/// `NUVEC_DISTRIBUTED_TIMEOUT` in seconds, or 30 s.
fn timeout() -> Result<Duration> {
    let Ok(secs) = std::env::var("NUVEC_DISTRIBUTED_TIMEOUT") else {
        return Ok(Duration::from_secs(30));
    };
    let invalid = || {
        Error::InvalidArgument(format!(
            "distributed: NUVEC_DISTRIBUTED_TIMEOUT {secs:?} (seconds, a positive number)"
        ))
    };
    let parsed: f64 = secs.trim().parse().map_err(|_| invalid())?;
    Duration::try_from_secs_f64(parsed)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(invalid)
}

/// `NUVEC_DISTRIBUTED_KEY`, the pre-shared secret of every connection.
fn secret() -> Option<String> {
    std::env::var("NUVEC_DISTRIBUTED_KEY").ok()
}

/// `NUVEC_DISTRIBUTED_ENCRYPT`: `0` keeps a keyed ring authenticated only, `1` encrypts
/// (the default with a key).
fn encrypt() -> Result<Option<bool>> {
    match std::env::var("NUVEC_DISTRIBUTED_ENCRYPT").as_deref() {
        Err(_) => Ok(None),
        Ok("0") => Ok(Some(false)),
        Ok("1") => Ok(Some(true)),
        Ok(v) => Err(Error::InvalidArgument(format!(
            "distributed: NUVEC_DISTRIBUTED_ENCRYPT {v:?} (0 or 1)"
        ))),
    }
}

/// The ring from `MLX_HOSTFILE` and `MLX_RANK`; `None` when they are not set.
fn ring() -> Result<Option<Group>> {
    let (Ok(hostfile), Ok(rank)) = (std::env::var("MLX_HOSTFILE"), std::env::var("MLX_RANK"))
    else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&hostfile)
        .map_err(|e| Error::Io(format!("distributed: reading {hostfile}: {e}")))?;
    let rank = rank
        .trim()
        .parse()
        .map_err(|_| Error::InvalidArgument(format!("distributed: MLX_RANK {rank:?}")))?;
    let mut options = Options::default().timeout(timeout()?);
    if let Some(secret) = secret() {
        options = options.key(secret);
    }
    if let Some(on) = encrypt()? {
        options = options.encrypt(on);
    }
    init_with(&hostfile::parse(&text)?, rank, &options).map(Some)
}

/// JACCL from its environment; `None` when it is not set.
#[cfg(feature = "jaccl")]
fn jaccl() -> Result<Option<Group>> {
    let Some(config) = crate::jaccl::Config::from_env()? else {
        return Ok(None);
    };
    let mut options = crate::jaccl::Options::default().timeout(timeout()?);
    if let Some(secret) = secret() {
        options = options.key(secret);
    }
    crate::jaccl::init_with(&config, &options).map(Some)
}

#[cfg(not(feature = "jaccl"))]
fn jaccl() -> Result<Option<Group>> {
    Ok(None)
}

const RING_VARS: &str = "the ring needs MLX_HOSTFILE and MLX_RANK";
const JACCL_VARS: &str = "JACCL needs MLX_RANK (or JACCL_RANK), MLX_IBV_DEVICES (or \
                          JACCL_IBV_DEVICES) and MLX_JACCL_COORDINATOR (or JACCL_COORDINATOR)";

/// Initialises `backend` from MLX's environment and makes it the default group, as MLX's
/// `init(strict, backend)`. A backend initialised before returns its group (`Any`: the first
/// group initialised). Without the environment: an error naming the missing variables if
/// `strict`, else the group of this process alone.
pub fn init_backend(strict: bool, backend: Backend) -> Result<Group> {
    if let Some(g) = cached(backend) {
        return Ok(g);
    }
    let (group, missing) = match backend {
        Backend::Ring => (ring()?.map(|g| (Backend::Ring, g)), RING_VARS.to_string()),
        Backend::Jaccl => {
            if cfg!(not(feature = "jaccl")) {
                return Err(Error::InvalidArgument(
                    "distributed: the JACCL backend is experimental and not built in: enable \
                     the `jaccl` feature"
                        .into(),
                ));
            }
            (
                jaccl()?.map(|g| (Backend::Jaccl, g)),
                JACCL_VARS.to_string(),
            )
        }
        Backend::Any => {
            let group = match ring()? {
                Some(g) => Some((Backend::Ring, g)),
                None => jaccl()?.map(|g| (Backend::Jaccl, g)),
            };
            let tried = if cfg!(feature = "jaccl") {
                format!("{RING_VARS}; {JACCL_VARS}")
            } else {
                RING_VARS.to_string()
            };
            (group, tried)
        }
    };
    match group {
        Some((b, g)) => {
            let mut groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
            for key in [b, Backend::Any] {
                if !groups.iter().any(|(k, _)| *k == key) {
                    groups.push((key, g.clone()));
                }
            }
            set_default_group(g.clone());
            Ok(g)
        }
        None if strict => Err(Error::InvalidArgument(format!(
            "distributed: no backend could be initialised ({missing})"
        ))),
        None => Ok(Group::single()),
    }
}

/// The group initialised for `backend`, unless it failed or was closed: then every cached
/// entry of it is dropped (and the default group reset), so the next `init` forms a new one.
fn cached(backend: Backend) -> Option<Group> {
    let mut groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
    let g = groups
        .iter()
        .find(|(k, _)| *k == backend)
        .map(|(_, g)| g.clone())?;
    if g.health().is_ok() {
        return Some(g);
    }
    groups.retain(|(_, c)| c.health().is_ok());
    nuvec_core::distributed::set_default_group(Group::single());
    None
}

/// Whether `backend` can run here, as MLX's `is_available(backend)`: the ring always;
/// JACCL when built in and the system's RDMA library lists a device.
pub fn is_available(backend: Backend) -> bool {
    match backend {
        Backend::Any | Backend::Ring => true,
        #[cfg(feature = "jaccl")]
        Backend::Jaccl => crate::jaccl::is_available(),
        #[cfg(not(feature = "jaccl"))]
        Backend::Jaccl => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests run without the distributed environment (the multi-process tests set it in
    /// their child processes only).
    fn clean() -> bool {
        let set = ["MLX_HOSTFILE", "MLX_IBV_DEVICES", "JACCL_IBV_DEVICES"]
            .iter()
            .any(|v| std::env::var(v).is_ok());
        if set {
            eprintln!("skipped: a distributed environment is set");
        }
        !set
    }

    #[test]
    fn selects_backends_as_mlx() {
        assert_eq!("jaccl".parse::<Backend>().unwrap(), Backend::Jaccl);
        assert!("mpi".parse::<Backend>().is_err());
        assert!(is_available(Backend::Ring) && is_available(Backend::Any));
        if !clean() {
            return;
        }
        assert_eq!(init_backend(false, Backend::Any).unwrap().size(), 1);
        assert_eq!(init_backend(false, Backend::Ring).unwrap().size(), 1);
        let e = init_backend(true, Backend::Ring).unwrap_err().to_string();
        assert!(e.contains("MLX_HOSTFILE and MLX_RANK"), "{e}");
        let e = init_backend(true, Backend::Jaccl).unwrap_err().to_string();
        if cfg!(feature = "jaccl") {
            assert!(e.contains("MLX_IBV_DEVICES"), "{e}");
            assert_eq!(init_backend(false, Backend::Jaccl).unwrap().size(), 1);
        } else {
            assert!(e.contains("enable the `jaccl` feature"), "{e}");
            assert!(!is_available(Backend::Jaccl));
        }
        let e = init_backend(true, Backend::Any).unwrap_err().to_string();
        assert!(e.contains("MLX_HOSTFILE"), "{e}");
    }
}
