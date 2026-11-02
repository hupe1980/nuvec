//! The side channel that exchanges connection details once, at initialisation: MLX's TCP
//! star (every rank sends to rank 0, which sends the concatenation back), or a caller's
//! all-gather. Every wait is bounded by the group's timeout.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use nuvec_core::{Error, Result};

/// A caller's all-gather: one rank's bytes in, every rank's bytes (equal lengths)
/// concatenated in rank order out, as MLX's `all_gather_factory`'s functions.
pub(crate) type AllGather = Arc<dyn Fn(&[u8]) -> Result<Vec<u8>> + Send + Sync>;

pub(crate) enum SideChannel {
    Tcp(Mutex<Star>),
    Custom(AllGather),
}

pub(crate) struct Star {
    rank: usize,
    size: usize,
    /// Rank 0: the sockets of ranks 1..size in rank order; others: the one to rank 0.
    sockets: Vec<TcpStream>,
}

fn side_error(peer: usize, what: &str, e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("jaccl: side channel: rank {peer}: {what}: {e}"))
}

fn resolve(address: &str) -> Result<SocketAddr> {
    address
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .ok_or_else(|| {
            Error::InvalidArgument(format!("jaccl: coordinator {address:?} does not resolve"))
        })
}

impl Star {
    /// Connects the star: rank 0 listens on `coordinator` and orders the peers by the
    /// 4-byte rank each sends; the others connect, retrying until `timeout`.
    pub(crate) fn connect(
        rank: usize,
        size: usize,
        coordinator: &str,
        timeout: Duration,
        key: Option<&crate::auth::Key>,
    ) -> Result<Self> {
        let address = resolve(coordinator)?;
        let deadline = Instant::now() + timeout;
        let configure = |s: &TcpStream, peer: usize| {
            s.set_nodelay(true)
                .and_then(|()| s.set_read_timeout(Some(timeout)))
                .and_then(|()| s.set_write_timeout(Some(timeout)))
                .and_then(|()| crate::keepalive::enable(s))
                .map_err(|e| side_error(peer, "configuring", e))
        };
        if rank == 0 {
            let listener = TcpListener::bind(address).map_err(|e| {
                Error::Backend(format!("jaccl: side channel: listening on {address}: {e}"))
            })?;
            listener
                .set_nonblocking(true)
                .map_err(|e| Error::Backend(format!("jaccl: side channel: {e}")))?;
            let mut slots: Vec<Option<TcpStream>> = (1..size).map(|_| None).collect();
            let mut joined = 0;
            let mut refused: Option<Error> = None;
            while joined < size - 1 {
                match listener.accept() {
                    Ok((s, _)) => {
                        s.set_nonblocking(false)
                            .map_err(|e| Error::Backend(format!("jaccl: side channel: {e}")))?;
                        configure(&s, 0)?;
                        // With a key, a connection that fails the handshake is dropped.
                        let peer = match crate::auth::accept(&s, key, 0, None, "a peer") {
                            Ok(p) => p,
                            Err(e) if key.is_some() => {
                                refused = Some(e);
                                continue;
                            }
                            Err(e) => return Err(e),
                        };
                        let slot = usize::try_from(peer)
                            .ok()
                            .filter(|&p| (1..size).contains(&p))
                            .ok_or_else(|| {
                                Error::Backend(format!(
                                    "jaccl: side channel: a peer sent rank {peer}"
                                ))
                            })?;
                        if slots[slot - 1].replace(s).is_some() {
                            return Err(Error::Backend(format!(
                                "jaccl: side channel: rank {slot} connected twice"
                            )));
                        }
                        joined += 1;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() > deadline {
                            if let Some(e) = refused {
                                return Err(e);
                            }
                            let missing: Vec<_> = slots
                                .iter()
                                .enumerate()
                                .filter(|(_, s)| s.is_none())
                                .map(|(i, _)| (i + 1).to_string())
                                .collect();
                            return Err(Error::Backend(format!(
                                "jaccl: side channel: rank {} did not connect within {:.1} s",
                                missing.join(", rank "),
                                timeout.as_secs_f64()
                            )));
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(e) => return Err(Error::Backend(format!("jaccl: side channel: {e}"))),
                }
            }
            Ok(Self {
                rank,
                size,
                sockets: slots
                    .into_iter()
                    .map(|s| s.expect("every rank joined"))
                    .collect(),
            })
        } else {
            let s = loop {
                match TcpStream::connect_timeout(&address, Duration::from_millis(500)) {
                    Ok(s) => break s,
                    Err(e) if Instant::now() > deadline => {
                        return Err(side_error(0, &format!("connecting to {address}"), e));
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(20)),
                }
            };
            configure(&s, 0)?;
            crate::auth::connect(&s, key, rank as u64, 0, "rank 0")?;
            Ok(Self {
                rank,
                size,
                sockets: vec![s],
            })
        }
    }

    fn all_gather(&self, data: &[u8]) -> Result<Vec<u8>> {
        let n = data.len();
        if self.rank == 0 {
            let mut out = vec![0u8; self.size * n];
            out[..n].copy_from_slice(data);
            for (i, s) in self.sockets.iter().enumerate() {
                let mut s = s;
                s.read_exact(&mut out[(i + 1) * n..(i + 2) * n])
                    .map_err(|e| side_error(i + 1, "receiving", e))?;
            }
            for (i, s) in self.sockets.iter().enumerate() {
                let mut s = s;
                s.write_all(&out)
                    .map_err(|e| side_error(i + 1, "sending", e))?;
            }
            Ok(out)
        } else {
            let mut s = &self.sockets[0];
            s.write_all(data).map_err(|e| side_error(0, "sending", e))?;
            let mut out = vec![0u8; self.size * n];
            s.read_exact(&mut out)
                .map_err(|e| side_error(0, "receiving", e))?;
            Ok(out)
        }
    }
}

impl Star {
    /// Whether the star's connections are still open (rank 0: to every rank; the others:
    /// to rank 0).
    fn health(&self) -> Result<()> {
        for (i, s) in self.sockets.iter().enumerate() {
            let peer = if self.rank == 0 { i + 1 } else { 0 };
            crate::ring::probe(s, peer)
                .map_err(|e| Error::Backend(format!("jaccl: side channel: {e}")))?;
        }
        Ok(())
    }
}

impl SideChannel {
    /// Whether the side channel's connections are open (a caller's side channel: `Ok`).
    pub(crate) fn health(&self) -> Result<()> {
        match self {
            Self::Tcp(star) => star.lock().unwrap_or_else(PoisonError::into_inner).health(),
            Self::Custom(_) => Ok(()),
        }
    }

    /// Every rank's `data` (equal lengths on every rank), concatenated in rank order.
    pub(crate) fn all_gather(&self, data: &[u8], size: usize) -> Result<Vec<u8>> {
        let out = match self {
            Self::Tcp(star) => star
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .all_gather(data)?,
            Self::Custom(f) => f(data)?,
        };
        if out.len() != size * data.len() {
            return Err(Error::Backend(format!(
                "jaccl: side channel: an all-gather of {} bytes from {size} ranks returned {}",
                data.len(),
                out.len()
            )));
        }
        Ok(out)
    }

    /// Every rank's `data` of any length (lengths first, then the data padded to the
    /// longest, as MLX's container all-gather).
    pub(crate) fn all_gather_vec(&self, data: &[u8], size: usize) -> Result<Vec<Vec<u8>>> {
        let lens = self.all_gather(&(data.len() as u64).to_le_bytes(), size)?;
        let lens: Vec<usize> = lens
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c) as usize)
            .collect();
        let max = lens.iter().copied().max().unwrap_or(0);
        let mut padded = data.to_vec();
        padded.resize(max, 0);
        let all = self.all_gather(&padded, size)?;
        Ok(lens
            .iter()
            .enumerate()
            .map(|(r, &len)| all[r * max..r * max + len].to_vec())
            .collect())
    }

    /// Every rank has reached this point (two all-gathers, as MLX's barrier).
    pub(crate) fn barrier(&self, size: usize) -> Result<()> {
        self.all_gather(&[0; 4], size)?;
        self.all_gather(&[0; 4], size)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn the_star_gathers_in_rank_order() {
        let coordinator = format!("127.0.0.1:{}", free_port());
        let results: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..3)
                .map(|rank| {
                    let c = coordinator.clone();
                    s.spawn(move || {
                        let star = Star::connect(rank, 3, &c, Duration::from_secs(10), None)?;
                        let ch = SideChannel::Tcp(Mutex::new(star));
                        ch.barrier(3)?;
                        let fixed = ch.all_gather(&[rank as u8; 2], 3)?;
                        let var = ch.all_gather_vec(&vec![7u8; rank], 3)?;
                        Ok::<_, Error>((fixed, var))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap().unwrap())
                .collect()
        });
        for (fixed, var) in results {
            assert_eq!(fixed, [0, 0, 1, 1, 2, 2]);
            assert_eq!(var, [vec![], vec![7], vec![7, 7]]);
        }
    }

    #[test]
    fn a_missing_rank_fails_within_the_timeout() {
        let coordinator = format!("127.0.0.1:{}", free_port());
        let start = Instant::now();
        let e = Star::connect(0, 3, &coordinator, Duration::from_millis(200), None)
            .err()
            .unwrap()
            .to_string();
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(e.contains("rank 1, rank 2 did not connect"), "{e}");
        let e = Star::connect(1, 2, &coordinator, Duration::from_millis(200), None)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("rank 0: connecting"), "{e}");
    }
}
