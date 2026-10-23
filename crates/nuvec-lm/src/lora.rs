//! LoRA fine-tuning support, as mlx-lm's `tuner`: replacing a loaded model's projections by
//! LoRA layers by path ([`linear_to_lora_layers`]), and adapter directories in mlx-lm's
//! layout ([`save_adapters`], [`load_adapters`]): `adapters.safetensors` with the adapter
//! weights and `adapter_config.json`.
//!
//! ```no_run
//! use nuvec_lm::lora::{self, LoraParams};
//! use nuvec_nn::Module;
//!
//! # fn main() -> nuvec::Result<()> {
//! let mut model = nuvec_lm::load_model("models/Qwen3-0.6B")?;
//! model.freeze(|_| true)?;
//! let keys = ["self_attn.q_proj", "self_attn.v_proj"];
//! lora::linear_to_lora_layers(&mut model, 8, Some(&keys), LoraParams::default())?;
//! assert!(model.trainable_parameters().iter().all(|(p, _)| p.contains(".lora_")));
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;
use std::path::Path;

use nuvec::io::json::{self, Json};
use nuvec::{Array, Error, Param, Result};
use nuvec_nn::{Linear, LoraBase, LoraLinear, Module};

use crate::model::{Model, Projection, SwitchProjection};

/// A LoRA layer's rank, scale and dropout (mlx-lm's `lora_parameters`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoraParams {
    /// The rank of the update (mlx-lm's default 8).
    pub rank: i32,
    /// The update's scale (mlx-lm's default 20).
    pub scale: f32,
    /// Dropout on the update's input (mlx-lm's default 0).
    pub dropout: f32,
}

impl Default for LoraParams {
    fn default() -> Self {
        Self {
            rank: 8,
            scale: 20.0,
            dropout: 0.0,
        }
    }
}

/// Where LoRA layers go, and their parameters (mlx-lm's `adapter_config.json` for
/// `fine_tune_type: "lora"`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdapterConfig {
    /// The number of blocks adapted, from the last one.
    pub num_layers: usize,
    /// The adapted projections, by path within a block (`self_attn.q_proj`); `None` for
    /// every linear layer (mlx-lm's default).
    pub keys: Option<Vec<String>>,
    /// Rank, scale and dropout.
    pub params: LoraParams,
}

impl AdapterConfig {
    /// LoRA on every linear layer of the last `num_layers` blocks.
    pub fn new(num_layers: usize, params: LoraParams) -> Self {
        Self {
            num_layers,
            keys: None,
            params,
        }
    }

    /// Only the projections at these paths within a block.
    pub fn keys(mut self, keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.keys = Some(keys.into_iter().map(Into::into).collect());
        self
    }
}

/// The block index and the path within the block of a model path
/// (`model.layers.3.self_attn.q_proj`).
fn in_block(path: &str) -> Option<(usize, &str)> {
    let rest = path.strip_prefix("model.layers.")?;
    let (index, rel) = rest.split_once('.')?;
    Some((index.parse().ok()?, rel))
}

/// Replaces the projections at `keys` (paths within a block, such as
/// `self_attn.q_proj`; `None`: every linear layer of the block, as mlx-lm's default) in the
/// last `num_layers` blocks by LoRA layers, as mlx-lm's `linear_to_lora_layers`: names
/// become `….q_proj.linear.weight`, `….q_proj.lora_a`, `….q_proj.lora_b`; full-precision and
/// quantized bases work, and each base is frozen. A key equal to a top-level path
/// (`lm_head`) adapts that projection too. Returns how many projections were adapted.
///
/// The new layers take the model's train/eval mode. Mixture-of-experts expert layers are
/// not supported: selecting one is an error naming the layer type (nothing changes).
pub fn linear_to_lora_layers(
    model: &mut Model,
    num_layers: usize,
    keys: Option<&[&str]>,
    params: LoraParams,
) -> Result<usize> {
    let layers = model.config.num_hidden_layers;
    let first = layers.saturating_sub(num_layers);
    let keys: Option<HashSet<&str>> = keys.map(|k| k.iter().copied().collect());
    let selected = |path: &str| match in_block(path) {
        Some((i, rel)) => i >= first && keys.as_ref().is_none_or(|k| k.contains(rel)),
        None => keys.as_ref().is_some_and(|k| k.contains(path)),
    };
    // Mixture-of-experts experts have no LoRA layer here: reject before changing anything.
    let mut experts = Vec::new();
    model.map_modules(selected, |path, _: &mut SwitchProjection| {
        experts.push(path.to_owned());
        Ok(())
    })?;
    if !experts.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "linear_to_lora_layers: LoRA on mixture-of-experts layers (SwitchProjection) is not \
             supported: {}",
            experts.join(", ")
        )));
    }
    let training = model.training();
    model.map_modules(selected, |path, proj: &mut Projection| {
        let placeholder = Projection::Dense(Linear {
            weight: Param::default(),
            bias: None,
        });
        let base: LoraBase = match std::mem::replace(proj, placeholder) {
            Projection::Dense(l) => l.into(),
            Projection::Quantized(q) => q.into(),
            Projection::Lora(l) => {
                *proj = Projection::Lora(l);
                return Err(Error::InvalidArgument(format!(
                    "linear_to_lora_layers: {path} is a LoRA layer already"
                )));
            }
        };
        let mut lora = LoraLinear::from_base(base, params.rank)
            .scale(params.scale)
            .dropout(params.dropout);
        lora.train(training);
        *proj = Projection::Lora(Box::new(lora));
        Ok(())
    })
}

/// Whether `path` names an adapter weight (mlx-lm's adapter leaves).
fn is_adapter(path: &str) -> bool {
    matches!(path.rsplit('.').next(), Some("lora_a" | "lora_b"))
}

/// Writes `model`'s adapter weights (`lora_a`, `lora_b`) to `dir/adapters.safetensors` and
/// `config` to `dir/adapter_config.json`, in mlx-lm's layout, creating `dir`.
pub fn save_adapters(dir: impl AsRef<Path>, model: &Model, config: &AdapterConfig) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    let adapters: Vec<(String, Array)> = model
        .parameters()
        .into_iter()
        .filter(|(p, _)| is_adapter(p))
        .collect();
    let tensors: Vec<(&str, &Array)> = adapters.iter().map(|(p, a)| (p.as_str(), a)).collect();
    nuvec::io::save_safetensors(dir.join("adapters.safetensors"), &tensors, &[])?;
    let p = &config.params;
    let keys = match &config.keys {
        Some(keys) => {
            let quoted: Vec<String> = keys.iter().map(|k| quote(k)).collect();
            format!(",\n    \"keys\": [{}]", quoted.join(", "))
        }
        None => String::new(),
    };
    let text = format!(
        "{{\n  \"fine_tune_type\": \"lora\",\n  \"num_layers\": {},\n  \"lora_parameters\": {{\n    \"rank\": {},\n    \"scale\": {:?},\n    \"dropout\": {:?}{keys}\n  }}\n}}\n",
        config.num_layers, p.rank, p.scale, p.dropout
    );
    let path = dir.join("adapter_config.json");
    std::fs::write(&path, text).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

/// A JSON string literal.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Loads an adapter directory in mlx-lm's layout onto `model` (mlx-lm's `load_adapters`):
/// reads `adapter_config.json` (`fine_tune_type` must be `"lora"`), applies
/// [`linear_to_lora_layers`] and loads `adapters.safetensors`, whose tensors must be adapter
/// weights of the model with their shapes (every problem is reported). Returns the
/// configuration. For inference, switch the model to evaluation mode
/// (`model.train(false)`), as `mlx_lm.load` does.
pub fn load_adapters(dir: impl AsRef<Path>, model: &mut Model) -> Result<AdapterConfig> {
    let dir = dir.as_ref();
    let path = dir.join("adapter_config.json");
    let bytes = std::fs::read(&path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let json = json::parse(&bytes)?;
    let invalid = |what: &str| Error::InvalidArgument(format!("{}: {what}", path.display()));
    let kind = json
        .get("fine_tune_type")
        .and_then(Json::as_str)
        .unwrap_or("lora");
    if kind != "lora" {
        return Err(invalid(&format!(
            "fine_tune_type {kind:?} is not supported (only \"lora\")"
        )));
    }
    let num_layers = json
        .get("num_layers")
        .and_then(Json::as_i64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("num_layers is missing or invalid"))?;
    let lp = json
        .get("lora_parameters")
        .ok_or_else(|| invalid("lora_parameters is missing"))?;
    let number = |key: &str| {
        lp.get(key)
            .and_then(Json::as_f64)
            .ok_or_else(|| invalid(&format!("lora_parameters.{key} is missing or invalid")))
    };
    let rank = number("rank")?;
    if rank < 1.0 || rank.fract() != 0.0 || rank > f64::from(i32::MAX) {
        return Err(invalid("lora_parameters.rank must be a positive integer"));
    }
    let params = LoraParams {
        rank: rank as i32,
        scale: number("scale")? as f32,
        dropout: lp.get("dropout").and_then(Json::as_f64).unwrap_or(0.0) as f32,
    };
    let keys: Option<Vec<String>> = match lp.get("keys") {
        None | Some(Json::Null) => None,
        Some(k) => Some(
            k.as_array()
                .ok_or_else(|| invalid("lora_parameters.keys must be a list"))?
                .iter()
                .map(|k| {
                    k.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("lora_parameters.keys must be strings"))
                })
                .collect::<Result<_>>()?,
        ),
    };
    let weights = nuvec::io::load_safetensors(dir.join("adapters.safetensors"))?.tensors;
    let mut adapted = model.clone();
    let key_refs: Option<Vec<&str>> = keys
        .as_ref()
        .map(|k| k.iter().map(String::as_str).collect());
    linear_to_lora_layers(&mut adapted, num_layers, key_refs.as_deref(), params)?;
    let allowed: std::collections::HashMap<String, Vec<i32>> = adapted
        .parameters()
        .into_iter()
        .filter(|(p, _)| is_adapter(p))
        .map(|(p, a)| (p, a.shape().to_vec()))
        .collect();
    let mut problems = Vec::new();
    for (name, w) in &weights {
        match allowed.get(name) {
            None => problems.push(format!("  {name}: not an adapter parameter of the model")),
            Some(shape) if shape.as_slice() != w.shape() => problems.push(format!(
                "  {name}: shape {:?}, the model's is {shape:?}",
                w.shape()
            )),
            Some(_) => {}
        }
    }
    if !problems.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "load_adapters: {} has incompatible tensors:\n{}",
            dir.display(),
            problems.join("\n")
        )));
    }
    adapted.load_weights_partial(weights)?;
    *model = adapted;
    Ok(AdapterConfig {
        num_layers,
        keys,
        params,
    })
}
