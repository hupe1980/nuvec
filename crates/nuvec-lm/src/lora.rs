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

/// What is trained (mlx-lm's `fine_tune_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FineTuneType {
    /// LoRA adapters ([`LoraLinear`]).
    #[default]
    Lora,
    /// Weight-decomposed adapters ([`LoraLinear::dora`], mlx-lm's `DoRALinear`).
    Dora,
    /// Every parameter of the last `num_layers` blocks (a float model).
    Full,
}

impl FineTuneType {
    /// mlx-lm's name: `lora`, `dora` or `full`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Dora => "dora",
            Self::Full => "full",
        }
    }

    /// The type named `name` (mlx-lm's names).
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "lora" => Ok(Self::Lora),
            "dora" => Ok(Self::Dora),
            "full" => Ok(Self::Full),
            other => Err(Error::InvalidArgument(format!(
                "unknown fine-tune type {other:?} (lora, dora or full)"
            ))),
        }
    }
}

/// What is trained and where (mlx-lm's `adapter_config.json`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdapterConfig {
    /// LoRA, DoRA or full fine-tuning.
    pub fine_tune_type: FineTuneType,
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
            fine_tune_type: FineTuneType::Lora,
            num_layers,
            keys: None,
            params,
        }
    }

    /// Trains `kind` instead of LoRA adapters.
    pub fn fine_tune_type(mut self, kind: FineTuneType) -> Self {
        self.fine_tune_type = kind;
        self
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
/// The new layers take the model's train/eval mode. Mixture-of-experts expert layers have
/// no LoRA layer: without `keys` they are left as they are, and naming one in `keys` is an
/// error naming it. A layer that is LoRA already is an error too. On an error nothing
/// changes.
pub fn linear_to_lora_layers(
    model: &mut Model,
    num_layers: usize,
    keys: Option<&[&str]>,
    params: LoraParams,
) -> Result<usize> {
    to_adapters(model, num_layers, keys, params, false)
}

/// [`linear_to_lora_layers`] with DoRA layers ([`LoraLinear::dora`], mlx-lm's
/// `use_dora=True`): each also learns a magnitude `m` per output feature.
pub fn linear_to_dora_layers(
    model: &mut Model,
    num_layers: usize,
    keys: Option<&[&str]>,
    params: LoraParams,
) -> Result<usize> {
    to_adapters(model, num_layers, keys, params, true)
}

/// Prepares `model` for training as `config` says (mlx-lm's `train_model`): freezes it, then
/// adds LoRA or DoRA layers, or (full fine-tuning) unfreezes every parameter of the last
/// `num_layers` blocks, which must be full precision. Returns how many projections were
/// adapted (full: parameters unfrozen).
pub fn apply(model: &mut Model, config: &AdapterConfig) -> Result<usize> {
    let keys: Option<Vec<&str>> = config
        .keys
        .as_ref()
        .map(|k| k.iter().map(String::as_str).collect());
    let mut m = model.clone();
    m.freeze(|_| true)?;
    let n = match config.fine_tune_type {
        FineTuneType::Lora => {
            linear_to_lora_layers(&mut m, config.num_layers, keys.as_deref(), config.params)?
        }
        FineTuneType::Dora => {
            linear_to_dora_layers(&mut m, config.num_layers, keys.as_deref(), config.params)?
        }
        FineTuneType::Full => {
            let first = m.config.num_hidden_layers.saturating_sub(config.num_layers);
            let quantized: Vec<String> = m
                .parameters()
                .into_iter()
                .filter(|(p, a)| {
                    a.dtype() == nuvec::DType::U32 && in_block(p).is_some_and(|(i, _)| i >= first)
                })
                .map(|(p, _)| p)
                .collect();
            if !quantized.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "full fine-tuning needs a full-precision model; quantized: {}",
                    quantized.join(", ")
                )));
            }
            m.unfreeze(|p| in_block(p).is_some_and(|(i, _)| i >= first))?
        }
    };
    *model = m;
    Ok(n)
}

fn to_adapters(
    model: &mut Model,
    num_layers: usize,
    keys: Option<&[&str]>,
    params: LoraParams,
    dora: bool,
) -> Result<usize> {
    let layers = model.config.num_hidden_layers;
    let first = layers.saturating_sub(num_layers);
    let keys: Option<HashSet<&str>> = keys.map(|k| k.iter().copied().collect());
    let selected = |path: &str| match in_block(path) {
        Some((i, rel)) => i >= first && keys.as_ref().is_none_or(|k| k.contains(rel)),
        None => keys.as_ref().is_some_and(|k| k.contains(path)),
    };
    // Checked before changing anything: experts named explicitly, and layers that are LoRA
    // already.
    let mut experts = Vec::new();
    if keys.is_some() {
        model.map_modules(selected, |path, _: &mut SwitchProjection| {
            experts.push(path.to_owned());
            Ok(())
        })?;
    }
    if !experts.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "linear_to_lora_layers: LoRA on mixture-of-experts layers (SwitchProjection) is not \
             supported: {}",
            experts.join(", ")
        )));
    }
    let mut lora = Vec::new();
    model.map_modules(selected, |path, proj: &mut Projection| {
        if matches!(proj, Projection::Lora(_)) {
            lora.push(path.to_owned());
        }
        Ok(())
    })?;
    if !lora.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "linear_to_lora_layers: LoRA layers already: {}",
            lora.join(", ")
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
            Projection::Lora(_) => unreachable!("checked above: {path}"),
        };
        let mut lora = LoraLinear::from_base(base, params.rank)
            .scale(params.scale)
            .dropout(params.dropout);
        if dora {
            lora = lora.dora();
        }
        lora.train(training);
        *proj = Projection::Lora(Box::new(lora));
        Ok(())
    })
}

/// Writes `model`'s trainable parameters (the adapters; for full fine-tuning, the trained
/// blocks' weights) to `dir/adapters.safetensors` and `config` to
/// `dir/adapter_config.json`, in mlx-lm's layout, creating `dir`.
pub fn save_adapters(dir: impl AsRef<Path>, model: &Model, config: &AdapterConfig) -> Result<()> {
    let dir = dir.as_ref();
    save_weights(dir.join("adapters.safetensors"), model)?;
    let p = &config.params;
    let keys = match &config.keys {
        Some(keys) => {
            let quoted: Vec<String> = keys.iter().map(|k| quote(k)).collect();
            format!(",\n    \"keys\": [{}]", quoted.join(", "))
        }
        None => String::new(),
    };
    let lora = match config.fine_tune_type {
        FineTuneType::Full => "null".to_owned(),
        _ => format!(
            "{{\n    \"rank\": {},\n    \"scale\": {:?},\n    \"dropout\": {:?}{keys}\n  }}",
            p.rank, p.scale, p.dropout
        ),
    };
    let text = format!(
        "{{\n  \"fine_tune_type\": \"{}\",\n  \"num_layers\": {},\n  \"lora_parameters\": {lora}\n}}\n",
        config.fine_tune_type.name(),
        config.num_layers,
    );
    let path = dir.join("adapter_config.json");
    std::fs::write(&path, text).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

/// Writes `model`'s trainable parameters to the safetensors file `path` (mlx-lm's
/// `adapters.safetensors` and its numbered checkpoints), creating its directory.
pub fn save_weights(path: impl AsRef<Path>, model: &Model) -> Result<()> {
    let path = path.as_ref();
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    }
    let trained = model.trainable_parameters();
    let tensors: Vec<(&str, &Array)> = trained.iter().map(|(p, a)| (p.as_str(), a)).collect();
    nuvec::io::save_safetensors(path, &tensors, &[])
}

/// Loads a safetensors file of trained weights onto `model` (mlx-lm's
/// `resume_adapter_file`, `load_weights(..., strict=False)`): every tensor must name a
/// parameter of the model with its shape.
pub fn load_weights(path: impl AsRef<Path>, model: &mut Model) -> Result<()> {
    let path = path.as_ref();
    let weights = nuvec::io::load_safetensors(path)?.tensors;
    let shapes: std::collections::HashMap<String, Vec<i32>> = model
        .parameters()
        .into_iter()
        .map(|(p, a)| (p, a.shape().to_vec()))
        .collect();
    check_tensors(path, &weights, |name| shapes.get(name).cloned())?;
    model.load_weights_partial(weights)
}

/// Every tensor of `weights` names an allowed parameter with its shape; else one error
/// listing the problems.
fn check_tensors(
    path: &Path,
    weights: &std::collections::BTreeMap<String, Array>,
    allowed: impl Fn(&str) -> Option<Vec<i32>>,
) -> Result<()> {
    let mut problems = Vec::new();
    for (name, w) in weights {
        match allowed(name) {
            None => problems.push(format!("  {name}: not a trainable parameter of the model")),
            Some(shape) if shape.as_slice() != w.shape() => problems.push(format!(
                "  {name}: shape {:?}, the model's is {shape:?}",
                w.shape()
            )),
            Some(_) => {}
        }
    }
    match problems.is_empty() {
        true => Ok(()),
        false => Err(Error::InvalidArgument(format!(
            "{} has incompatible tensors:\n{}",
            path.display(),
            problems.join("\n")
        ))),
    }
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
    let kind = FineTuneType::from_name(
        json.get("fine_tune_type")
            .and_then(Json::as_str)
            .unwrap_or("lora"),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    // mlx-lm writes -1 for every layer.
    let num_layers = json
        .get("num_layers")
        .and_then(Json::as_i64)
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
        .ok_or_else(|| invalid("num_layers is missing or invalid"))?;
    // Full fine-tuning has no LoRA parameters (mlx-lm writes null).
    let empty = Json::Object(Default::default());
    let lp = match (json.get("lora_parameters"), kind) {
        (Some(Json::Null) | None, FineTuneType::Full) => &empty,
        (Some(lp), _) => lp,
        (None, _) => return Err(invalid("lora_parameters is missing")),
    };
    let number = |key: &str| {
        lp.get(key)
            .and_then(Json::as_f64)
            .ok_or_else(|| invalid(&format!("lora_parameters.{key} is missing or invalid")))
    };
    let rank = match kind {
        FineTuneType::Full => f64::from(LoraParams::default().rank),
        _ => number("rank")?,
    };
    if rank < 1.0 || rank.fract() != 0.0 || rank > f64::from(i32::MAX) {
        return Err(invalid("lora_parameters.rank must be a positive integer"));
    }
    let params = LoraParams {
        rank: rank as i32,
        scale: match kind {
            FineTuneType::Full => LoraParams::default().scale,
            _ => number("scale")? as f32,
        },
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
    let config = AdapterConfig {
        fine_tune_type: kind,
        num_layers,
        keys,
        params,
    };
    let file = dir.join("adapters.safetensors");
    let weights = nuvec::io::load_safetensors(&file)?.tensors;
    let mut adapted = model.clone();
    apply(&mut adapted, &config)?;
    let allowed: std::collections::HashMap<String, Vec<i32>> = adapted
        .trainable_parameters()
        .into_iter()
        .map(|(p, a)| (p, a.shape().to_vec()))
        .collect();
    check_tensors(&file, &weights, |name| allowed.get(name).cloned())
        .map_err(|e| Error::InvalidArgument(format!("load_adapters: {e}")))?;
    adapted.load_weights_partial(weights)?;
    *model = adapted;
    Ok(config)
}
