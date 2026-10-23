//! Loading and saving weights by name ([`crate::Module::load_weights`],
//! [`crate::Module::save_weights`]).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use nuvec::tree::{LeafKind, Leaves, LoadMode, Tree, Visit, Visitor};
use nuvec::{Array, Error, Result};

/// Where [`crate::Module::load_weights`] reads named weights from: a safetensors file (a
/// path), or named arrays (`Vec<(String, Array)>`, a `HashMap` or `BTreeMap`).
pub trait WeightSource {
    /// The named weights.
    fn weights(self) -> Result<Vec<(String, Array)>>;
}

fn from_file(path: &Path) -> Result<Vec<(String, Array)>> {
    Ok(nuvec::io::load_safetensors(path)?
        .tensors
        .into_iter()
        .collect())
}

impl WeightSource for &Path {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        from_file(self)
    }
}

impl WeightSource for &PathBuf {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        from_file(self)
    }
}

impl WeightSource for PathBuf {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        from_file(&self)
    }
}

impl WeightSource for &str {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        from_file(Path::new(self))
    }
}

impl WeightSource for String {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        from_file(Path::new(&self))
    }
}

impl WeightSource for Vec<(String, Array)> {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        Ok(self)
    }
}

impl<const N: usize> WeightSource for [(String, Array); N] {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        Ok(self.into())
    }
}

impl<S> WeightSource for HashMap<String, Array, S> {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        Ok(self.into_iter().collect())
    }
}

impl WeightSource for BTreeMap<String, Array> {
    fn weights(self) -> Result<Vec<(String, Array)>> {
        Ok(self.into_iter().collect())
    }
}

/// `load_flat`'s error, named for `op`.
fn renamed(e: Error, op: &str) -> Error {
    match e {
        Error::InvalidArgument(m) => Error::InvalidArgument(m.replacen("load_flat", op, 1)),
        e => e,
    }
}

pub(crate) fn load<T: Tree>(tree: &mut T, src: impl WeightSource, mode: LoadMode) -> Result<()> {
    let op = match mode {
        LoadMode::Strict => "load_weights",
        _ => "load_weights_partial",
    };
    tree.load_flat(src.weights()?, mode)
        .map_err(|e| renamed(e, op))
}

/// Writes the leaves of `tree` (all, or only the trainable ones) as safetensors, named by
/// path. Leaves that cannot be evaluated (unloaded placeholders) are reported by name.
pub(crate) fn save<T: Leaves + ?Sized>(tree: &T, path: &Path, trainable_only: bool) -> Result<()> {
    struct Named(Vec<(String, Array)>, bool);
    impl Visitor for Named {
        fn leaf(&mut self, path: &str, a: &Array, _: LeafKind, trainable: bool) {
            if trainable || !self.1 {
                self.0.push((path.to_owned(), a.clone()));
            }
        }
    }
    let op = if trainable_only {
        "save_trainable"
    } else {
        "save_weights"
    };
    let mut named = Named(Vec::new(), trainable_only);
    tree.visit(&mut Visit::new(&mut named));
    let named = named.0;
    let mut seen = std::collections::HashSet::new();
    for (p, _) in &named {
        if !seen.insert(p.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "{op}: the path {p:?} appears twice"
            )));
        }
    }
    let arrays: Vec<Array> = named.iter().map(|(_, a)| a.clone()).collect();
    if nuvec::eval(&arrays).is_err() {
        let failed: Vec<&str> = named
            .iter()
            .filter(|(_, a)| a.eval().is_err())
            .map(|(p, _)| p.as_str())
            .collect();
        return Err(Error::InvalidArgument(format!(
            "{op}: cannot evaluate {} (unloaded placeholders?): {}",
            if failed.len() == 1 {
                "a leaf"
            } else {
                "leaves"
            },
            failed.join(", ")
        )));
    }
    let tensors: Vec<(&str, &Array)> = named.iter().map(|(p, a)| (p.as_str(), a)).collect();
    nuvec::io::save_safetensors(path, &tensors, &[("format", "mlx")])
}
