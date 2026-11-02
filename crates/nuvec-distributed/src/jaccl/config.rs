//! MLX's JACCL configuration: the rank, the device file (per rank and peer, the RDMA
//! devices of the cables between them), the coordinator and the ring preference.

use nuvec_core::{Error, Result};

use super::mesh::MAX_PEERS;
use super::ring::MAX_CABLES;
use crate::json::{self, Value};

/// A JACCL group's configuration, as MLX's `jaccl::Config`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub(crate) rank: usize,
    pub(crate) devices: Vec<Vec<Vec<String>>>,
    pub(crate) coordinator: Option<String>,
    pub(crate) prefer_ring: bool,
}

/// The topology a configuration selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Topology {
    /// Every pair of ranks is cabled; `wires` cables per pair.
    Mesh {
        /// Cables per pair (the smallest count over all pairs).
        wires: usize,
    },
    /// Every rank is cabled to its two neighbours.
    Ring {
        /// Cables per direction.
        cables: usize,
    },
}

/// MLX's device file: a JSON array with one entry per rank, each an array with one entry
/// per peer: `null` (no cable; the rank itself), a device name, or a list of names (one per
/// cable; the `k`-th cable of a pair is the `k`-th name on both sides).
pub fn parse_devices(text: &str) -> Result<Vec<Vec<Vec<String>>>> {
    let invalid = |what: String| Error::InvalidArgument(format!("jaccl: device file: {what}"));
    let Value::Array(ranks) = json::parse(text, "jaccl: device file")? else {
        return Err(invalid("expected an array of arrays".into()));
    };
    let size = ranks.len();
    ranks
        .into_iter()
        .enumerate()
        .map(|(rank, row)| {
            let Value::Array(peers) = row else {
                return Err(invalid(format!("rank {rank}: expected an array")));
            };
            if peers.len() != size {
                return Err(invalid(format!(
                    "rank {rank} lists {} peers, the file has {size} ranks",
                    peers.len()
                )));
            }
            peers
                .into_iter()
                .enumerate()
                .map(|(peer, entry)| match entry {
                    Value::Null => Ok(Vec::new()),
                    Value::String(name) => Ok(vec![name]),
                    Value::Array(names) => names
                        .into_iter()
                        .map(|n| match n {
                            Value::String(n) => Ok(n),
                            _ => Err(invalid(format!(
                                "rank {rank} → {peer}: device names are strings"
                            ))),
                        })
                        .collect(),
                    _ => Err(invalid(format!(
                        "rank {rank} → {peer}: expected null, a device name or a list of names"
                    ))),
                })
                .collect()
        })
        .collect()
}

impl Config {
    /// Rank `rank` of the group the device matrix `devices` describes (`size` rows of `size`
    /// entries).
    pub fn new(rank: usize, devices: Vec<Vec<Vec<String>>>) -> Result<Self> {
        let size = devices.len();
        for (r, row) in devices.iter().enumerate() {
            if row.len() != size {
                return Err(Error::InvalidArgument(format!(
                    "jaccl: the device matrix has {size} rows but row {r} has {} entries",
                    row.len()
                )));
            }
        }
        if rank >= size {
            return Err(Error::InvalidArgument(format!(
                "jaccl: rank {rank} is not a rank of a group of {size}"
            )));
        }
        Ok(Self {
            rank,
            devices,
            coordinator: None,
            prefer_ring: false,
        })
    }

    /// The coordinator's `host:port` (rank 0 listens on it during initialisation).
    pub fn coordinator(mut self, address: impl Into<String>) -> Self {
        self.coordinator = Some(address.into());
        self
    }

    /// Prefers the ring topology when the cabling allows both.
    pub fn prefer_ring(mut self, prefer: bool) -> Self {
        self.prefer_ring = prefer;
        self
    }

    /// This process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of ranks.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// The configuration from MLX's environment, `None` when it is not set: `JACCL_RANK` or
    /// `MLX_RANK`, `JACCL_IBV_DEVICES` or `MLX_IBV_DEVICES` (the device file),
    /// `JACCL_COORDINATOR` or `MLX_JACCL_COORDINATOR`, and `JACCL_RING` or `MLX_JACCL_RING`
    /// (set: prefer the ring), as `mlx.launch --backend jaccl` writes them.
    pub fn from_env() -> Result<Option<Self>> {
        Self::from_vars(|name| std::env::var(name).ok())
    }

    fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let env = |names: &[&str]| names.iter().find_map(|n| var(n));
        let (Some(file), Some(rank)) = (
            env(&["JACCL_IBV_DEVICES", "MLX_IBV_DEVICES"]),
            env(&["JACCL_RANK", "MLX_RANK"]),
        ) else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(&file)
            .map_err(|e| Error::Io(format!("jaccl: reading the device file {file}: {e}")))?;
        let rank = rank
            .trim()
            .parse()
            .map_err(|_| Error::InvalidArgument(format!("jaccl: rank {rank:?}")))?;
        let mut config = Self::new(rank, parse_devices(&text)?)?
            .prefer_ring(env(&["JACCL_RING", "MLX_JACCL_RING"]).is_some());
        config.coordinator = env(&["JACCL_COORDINATOR", "MLX_JACCL_COORDINATOR"]);
        Ok(Some(config))
    }

    /// The first rule a mesh breaks, or `None` for a valid mesh (MLX's `is_valid_mesh`).
    fn mesh_error(&self) -> Option<String> {
        let size = self.size();
        if size < 2 {
            return Some("a group needs at least 2 ranks".into());
        }
        for (src, row) in self.devices.iter().enumerate() {
            for (dst, cables) in row.iter().enumerate() {
                if src == dst && !cables.is_empty() {
                    return Some(format!("rank {src} lists a cable to itself"));
                }
                if src != dst && cables.is_empty() {
                    return Some(format!("rank {src} → {dst}: no cable in the device file"));
                }
            }
        }
        None
    }

    /// The first rule a ring breaks, or `None` for a valid ring (MLX's `is_valid_ring`).
    fn ring_error(&self) -> Option<String> {
        let size = self.size();
        if size < 2 {
            return Some("a group needs at least 2 ranks".into());
        }
        let cables = self.devices[0][1].len();
        if cables == 0 {
            return Some("rank 0 → 1: no cable in the device file".into());
        }
        if cables > MAX_CABLES {
            return Some(format!(
                "the ring supports up to {MAX_CABLES} cables per direction, rank 0 → 1 has {cables}"
            ));
        }
        for src in 0..size {
            for dst in [(src + size - 1) % size, (src + 1) % size] {
                let n = self.devices[src][dst].len();
                if n != cables {
                    return Some(format!(
                        "rank {src} → {dst}: {n} cables, rank 0 → 1 has {cables}"
                    ));
                }
            }
        }
        None
    }

    /// The topology MLX's `init` selects: the ring when preferred and valid, else the mesh
    /// when valid, else the ring when valid.
    pub fn topology(&self) -> Result<Topology> {
        let (mesh, ring) = (self.mesh_error(), self.ring_error());
        let as_ring = || Topology::Ring {
            cables: self.devices[0][1].len(),
        };
        match (mesh, ring) {
            (_, None) if self.prefer_ring => Ok(as_ring()),
            // MLX selects a valid mesh of any size and then rejects more than 8 ranks.
            (None, _) if self.size() > MAX_PEERS => Err(Error::InvalidArgument(format!(
                "jaccl: the mesh supports up to {MAX_PEERS} ranks, the file has {}",
                self.size()
            ))),
            (None, _) => Ok(Topology::Mesh {
                wires: (0..self.size())
                    .flat_map(|s| {
                        (0..self.size())
                            .filter(move |&d| d != s)
                            .map(move |d| (s, d))
                    })
                    .map(|(s, d)| self.devices[s][d].len())
                    .min()
                    .unwrap_or(1),
            }),
            (Some(_), None) => Ok(as_ring()),
            (Some(mesh), Some(ring)) => Err(Error::InvalidArgument(format!(
                "jaccl: the device file defines neither a mesh ({mesh}) nor a ring ({ring})"
            ))),
        }
    }

    /// The mesh's devices of this rank: per cable, per peer (empty for itself).
    pub(crate) fn mesh_devices(&self, wires: usize) -> Vec<Vec<String>> {
        (0..wires)
            .map(|w| {
                (0..self.size())
                    .map(|p| {
                        if p == self.rank {
                            String::new()
                        } else {
                            self.devices[self.rank][p][w].clone()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// The ring's devices of this rank: to the left and to the right neighbour.
    pub(crate) fn ring_devices(&self) -> (Vec<String>, Vec<String>) {
        let size = self.size();
        let (left, right) = ((self.rank + size - 1) % size, (self.rank + 1) % size);
        (
            self.devices[self.rank][left].clone(),
            self.devices[self.rank][right].clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MESH4: &str = r#"[
        [null, "rdma_en5", "rdma_en4", "rdma_en3"],
        ["rdma_en5", null, "rdma_en3", "rdma_en4"],
        ["rdma_en4", "rdma_en3", null, "rdma_en5"],
        ["rdma_en3", "rdma_en4", "rdma_en5", null]
    ]"#;

    #[test]
    fn reads_mlx_device_files() {
        let d = parse_devices(MESH4).unwrap();
        assert_eq!(d[1][2], ["rdma_en3"]);
        assert!(d[2][2].is_empty());
        let c = Config::new(2, d).unwrap();
        assert_eq!(c.topology().unwrap(), Topology::Mesh { wires: 1 });
        assert_eq!(
            c.clone().prefer_ring(true).topology().unwrap(),
            Topology::Ring { cables: 1 }
        );
        assert_eq!(
            c.mesh_devices(1),
            [vec!["rdma_en4", "rdma_en3", "", "rdma_en5"]]
        );
        assert_eq!(
            c.ring_devices(),
            (vec!["rdma_en3".into()], vec!["rdma_en5".into()])
        );
        // Two cables between two Macs.
        let two = r#"[[null, ["rdma_en2", "rdma_en3"]], [["rdma_en2", "rdma_en3"], null]]"#;
        let c = Config::new(0, parse_devices(two).unwrap()).unwrap();
        assert_eq!(c.topology().unwrap(), Topology::Mesh { wires: 2 });
        // A ring of four: neighbours only.
        let ring = r#"[[null, "a", null, "b"], ["a", null, "c", null],
                       [null, "c", null, "d"], ["b", null, "d", null]]"#;
        let c = Config::new(1, parse_devices(ring).unwrap()).unwrap();
        assert_eq!(c.topology().unwrap(), Topology::Ring { cables: 1 });
    }

    #[test]
    fn reads_mlx_and_jaccl_variables() {
        let file = std::env::temp_dir().join(format!("nuvec-jaccl-{}.json", std::process::id()));
        std::fs::write(&file, MESH4).unwrap();
        let path = file.to_str().unwrap().to_string();
        let vars = |pairs: &[(&str, &str)]| {
            let pairs: Vec<(String, String)> = pairs
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            move |n: &str| pairs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone())
        };
        assert_eq!(Config::from_vars(vars(&[])).unwrap(), None);
        let c = Config::from_vars(vars(&[
            ("MLX_IBV_DEVICES", &path),
            ("MLX_RANK", "1"),
            ("JACCL_RANK", "3"),
            ("MLX_JACCL_COORDINATOR", "10.0.0.1:5000"),
            ("MLX_JACCL_RING", "1"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!((c.rank(), c.size(), c.prefer_ring), (3, 4, true));
        assert_eq!(c.coordinator.as_deref(), Some("10.0.0.1:5000"));
        assert!(Config::from_vars(vars(&[("MLX_IBV_DEVICES", &path), ("MLX_RANK", "x")])).is_err());
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn invalid_files_name_the_entry() {
        for (text, want) in [
            ("[[null, \"a\"]]", "rank 0 lists 2 peers"),
            ("[[null, 3], [\"a\", null]]", "rank 0 → 1"),
            ("{}", "array of arrays"),
        ] {
            let e = parse_devices(text).unwrap_err().to_string();
            assert!(e.contains(want), "{text}: {e}");
        }
        let self_cable = parse_devices(r#"[["x", "a"], ["a", null]]"#).unwrap();
        // Not a mesh; MLX's ring rule reads only the neighbours' entries.
        let c = Config::new(0, self_cable).unwrap();
        assert!(
            c.mesh_error()
                .unwrap()
                .contains("rank 0 lists a cable to itself")
        );
        assert_eq!(c.topology().unwrap(), Topology::Ring { cables: 1 });
        let missing =
            parse_devices(r#"[[null, "a", null], ["a", null, null], [null, null, null]]"#).unwrap();
        let e = Config::new(0, missing)
            .unwrap()
            .topology()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("rank 0 → 2: no cable") && e.contains("rank 0 → 2: 0 cables"),
            "{e}"
        );
        let nine: Vec<Vec<Vec<String>>> = (0..9)
            .map(|s| {
                (0..9)
                    .map(|d| if s == d { vec![] } else { vec!["x".into()] })
                    .collect()
            })
            .collect();
        let e = Config::new(0, nine)
            .unwrap()
            .topology()
            .unwrap_err()
            .to_string();
        assert!(e.contains("up to 8 ranks"), "{e}");
        let five = r#"[[null, ["a","b","c","d","e"]], [["a","b","c","d","e"], null]]"#;
        let c = Config::new(0, parse_devices(five).unwrap()).unwrap();
        assert_eq!(c.topology().unwrap(), Topology::Mesh { wires: 5 });
        let e = c.prefer_ring(true).ring_error().unwrap();
        assert!(e.contains("up to 4 cables"), "{e}");
        assert!(Config::new(3, parse_devices(MESH4).unwrap()).is_ok());
        assert!(Config::new(4, parse_devices(MESH4).unwrap()).is_err());
    }
}
