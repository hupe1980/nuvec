//! Any tree of arrays as a safetensors file, by path: [`save`] and [`load_into`].

use std::path::Path;

use nuvec_core::tree::{LeafKind, Leaves, LoadMode, Tree, Visit, Visitor};
use nuvec_core::{Array, DType, Error, Result};

use crate::{load_safetensors, save_safetensors};

/// Writes the leaves of `tree` as a safetensors file, each named by its path (MLX's
/// `tree_flatten` names), evaluating them.
///
/// The file's metadata records `format = "nuvec-tree"`, the root type name and, when the
/// tree holds 0-D leaves named `step` and `learning_rate` (an optimizer's state), their
/// values (informational: the tensors are authoritative). Checkpointing a training run is
/// `nuvec::save(path, &(model, opt))?`, resuming it
/// `nuvec::load_into(path, &mut (model, opt))?` (after `opt.init(&model)`).
///
/// Duplicate paths are an error.
pub fn save<T: Leaves + ?Sized>(path: impl AsRef<Path>, tree: &T) -> Result<()> {
    struct Named(Vec<(String, Array)>);
    impl Visitor for Named {
        fn leaf(&mut self, path: &str, a: &Array, _: LeafKind, _: bool) {
            self.0.push((path.to_owned(), a.clone()));
        }
    }
    let mut named = Named(Vec::new());
    tree.visit(&mut Visit::new(&mut named));
    let named = named.0;
    let mut seen = std::collections::HashSet::new();
    for (p, _) in &named {
        if !seen.insert(p.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "save: the tree has the path {p:?} twice"
            )));
        }
    }
    let mut metadata = vec![
        ("format".to_owned(), "nuvec-tree".to_owned()),
        ("root".to_owned(), std::any::type_name::<T>().to_owned()),
    ];
    for key in ["step", "learning_rate"] {
        let found = named
            .iter()
            .find(|(p, a)| a.ndim() == 0 && (p == key || p.ends_with(&format!(".{key}"))));
        if let Some((_, a)) = found {
            let value = match a.dtype() {
                DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                    a.astype(DType::F64).item::<f64>()?.to_string()
                }
                _ => a.astype(DType::I64).item::<i64>()?.to_string(),
            };
            metadata.push((key.to_owned(), value));
        }
    }
    let tensors: Vec<(&str, &Array)> = named.iter().map(|(p, a)| (p.as_str(), a)).collect();
    let metadata: Vec<(&str, &str)> = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    save_safetensors(path, &tensors, &metadata)
}

/// Reads a file written by [`save`] into `tree` by path, strictly: every leaf must be in
/// the file with its shape, and nothing else (loaded dtypes are kept). On any problem the
/// error lists every one and `tree` is unchanged.
pub fn load_into<T: Tree>(path: impl AsRef<Path>, tree: &mut T) -> Result<()> {
    let file = load_safetensors(path)?;
    tree.load_flat(file.tensors, LoadMode::Strict)
        .map_err(|e| match e {
            Error::InvalidArgument(m) => {
                Error::InvalidArgument(m.replacen("load_flat", "load_into", 1))
            }
            e => e,
        })
}
