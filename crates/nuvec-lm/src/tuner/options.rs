//! `mlx_lm.lora`'s optimizer and schedule options by name ([`build_optimizer`],
//! [`build_schedule`]) and its YAML configuration files ([`read_config`]).

use std::path::Path;

use nuvec::io::json::Json;
use nuvec::tree::{Visit, VisitMut};
use nuvec::{Array, Error, Result, Tree};
use nuvec_nn::optim::{Adafactor, Adam, AdamW, LearningRate, Muon, Optimizer, Sgd, schedules};

/// One of `mlx_lm.lora`'s optimizers (`--optimizer`), built by [`build_optimizer`].
#[derive(Debug, Clone, Tree)]
pub enum AnyOptimizer {
    /// `adam`.
    Adam(Adam),
    /// `adamw`.
    AdamW(AdamW),
    /// `sgd`.
    Sgd(Sgd),
    /// `muon`.
    Muon(Muon),
    /// `adafactor`.
    Adafactor(Adafactor),
}

macro_rules! each {
    ($self:expr, $o:ident => $e:expr) => {
        match $self {
            AnyOptimizer::Adam($o) => $e,
            AnyOptimizer::AdamW($o) => $e,
            AnyOptimizer::Sgd($o) => $e,
            AnyOptimizer::Muon($o) => $e,
            AnyOptimizer::Adafactor($o) => $e,
        }
    };
}

impl Optimizer for AnyOptimizer {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        each!(self, o => o.apply(params, grads))
    }
    fn init_params(&mut self, params: &[(&str, &Array)]) {
        each!(self, o => o.init_params(params))
    }
    fn visit_state(&self, v: &mut Visit<'_>) {
        each!(self, o => o.visit_state(v))
    }
    fn visit_state_mut(&mut self, v: &mut VisitMut<'_>) {
        each!(self, o => o.visit_state_mut(v))
    }
    fn learning_rate(&self) -> Array {
        each!(self, o => o.learning_rate())
    }
    fn set_learning_rate(&mut self, lr: f32) {
        each!(self, o => o.set_learning_rate(lr))
    }
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// mlx-lm's optimizer `name` (`adam`, `adamw`, `muon`, `sgd`, `adafactor`) with its
/// `optimizer_config` entry `config` (an object of MLX's keyword arguments, such as
/// `betas`, `eps`, `weight_decay`, `momentum`; `Json::Null` for the defaults) at
/// `learning_rate`. Unknown names and keys are errors.
pub fn build_optimizer(
    name: &str,
    config: &Json,
    learning_rate: LearningRate,
) -> Result<AnyOptimizer> {
    let entries: &[(String, Json)] = match config {
        Json::Object(e) => e,
        Json::Null => &[],
        _ => {
            return Err(invalid(format!(
                "optimizer_config.{name} must be a mapping"
            )));
        }
    };
    let lr = learning_rate;
    let mut o = match name.to_ascii_lowercase().as_str() {
        "adam" => AnyOptimizer::Adam(Adam::new(lr)),
        "adamw" => AnyOptimizer::AdamW(AdamW::new(lr)),
        "sgd" => AnyOptimizer::Sgd(Sgd::new(lr)),
        "muon" => AnyOptimizer::Muon(Muon::new(lr)),
        "adafactor" => AnyOptimizer::Adafactor(Adafactor::new(lr)),
        other => {
            return Err(invalid(format!(
                "unsupported optimizer {other:?} (adam, adamw, muon, sgd or adafactor)"
            )));
        }
    };
    for (key, value) in entries {
        let bad = || {
            invalid(format!(
                "optimizer_config.{name}.{key}: invalid value {value:?}"
            ))
        };
        let num = || value.as_f64().ok_or_else(bad);
        let pair = || -> Result<(f64, f64)> {
            match value.as_array() {
                Some([a, b]) => Ok((a.as_f64().ok_or_else(bad)?, b.as_f64().ok_or_else(bad)?)),
                _ => Err(bad()),
            }
        };
        let flag = || match value {
            Json::Bool(b) => Ok(*b),
            _ => Err(bad()),
        };
        o = match (o, key.as_str()) {
            (AnyOptimizer::Adam(a), "betas") => AnyOptimizer::Adam(a.betas(pair()?.0, pair()?.1)),
            (AnyOptimizer::Adam(a), "eps") => AnyOptimizer::Adam(a.eps(num()?)),
            (AnyOptimizer::Adam(a), "bias_correction") => {
                AnyOptimizer::Adam(if flag()? { a.bias_correction() } else { a })
            }
            (AnyOptimizer::AdamW(a), "betas") => AnyOptimizer::AdamW(a.betas(pair()?.0, pair()?.1)),
            (AnyOptimizer::AdamW(a), "eps") => AnyOptimizer::AdamW(a.eps(num()?)),
            (AnyOptimizer::AdamW(a), "weight_decay") => AnyOptimizer::AdamW(a.weight_decay(num()?)),
            (AnyOptimizer::AdamW(a), "bias_correction") => {
                AnyOptimizer::AdamW(if flag()? { a.bias_correction() } else { a })
            }
            (AnyOptimizer::Sgd(s), "momentum") => AnyOptimizer::Sgd(s.momentum(num()?)),
            (AnyOptimizer::Sgd(s), "weight_decay") => AnyOptimizer::Sgd(s.weight_decay(num()?)),
            (AnyOptimizer::Sgd(s), "dampening") => AnyOptimizer::Sgd(s.dampening(num()?)),
            (AnyOptimizer::Sgd(s), "nesterov") => {
                AnyOptimizer::Sgd(if flag()? { s.nesterov() } else { s })
            }
            (AnyOptimizer::Muon(m), "momentum") => AnyOptimizer::Muon(m.momentum(num()?)),
            (AnyOptimizer::Muon(m), "weight_decay") => AnyOptimizer::Muon(m.weight_decay(num()?)),
            (AnyOptimizer::Muon(m), "nesterov") => {
                AnyOptimizer::Muon(if flag()? { m } else { m.without_nesterov() })
            }
            (AnyOptimizer::Muon(m), "ns_steps") => AnyOptimizer::Muon(m.ns_steps(num()? as usize)),
            (AnyOptimizer::Adafactor(a), "eps") => {
                AnyOptimizer::Adafactor(a.eps(pair()?.0, pair()?.1))
            }
            (AnyOptimizer::Adafactor(a), "clip_threshold") => {
                AnyOptimizer::Adafactor(a.clip_threshold(num()?))
            }
            (AnyOptimizer::Adafactor(a), "decay_rate") => {
                AnyOptimizer::Adafactor(a.decay_rate(num()?))
            }
            (AnyOptimizer::Adafactor(a), "beta_1") => match value {
                Json::Null => AnyOptimizer::Adafactor(a),
                _ => AnyOptimizer::Adafactor(a.beta_1(num()?)),
            },
            (AnyOptimizer::Adafactor(a), "weight_decay") => {
                AnyOptimizer::Adafactor(a.weight_decay(num()?))
            }
            (AnyOptimizer::Adafactor(a), "scale_parameter") => {
                AnyOptimizer::Adafactor(a.scale_parameter(flag()?))
            }
            (AnyOptimizer::Adafactor(a), "relative_step") => {
                AnyOptimizer::Adafactor(a.relative_step(flag()?))
            }
            (AnyOptimizer::Adafactor(a), "warmup_init") => {
                AnyOptimizer::Adafactor(a.warmup_init(flag()?))
            }
            (_, key) => {
                return Err(invalid(format!(
                    "optimizer_config.{name}: unknown key {key:?}"
                )));
            }
        };
    }
    Ok(o)
}

/// mlx-lm's `build_schedule`: the schedule `config.name` (`cosine_decay`,
/// `exponential_decay`, `step_decay`, `linear_schedule`) called with `config.arguments`;
/// with `warmup` steps, a linear warm-up from `warmup_init` (0) to the first argument
/// joined before it at `warmup + 1`.
pub fn build_schedule(config: &Json) -> Result<LearningRate> {
    let name = config
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| invalid("lr_schedule.name is missing".into()))?;
    let args: Vec<f64> = config
        .get("arguments")
        .and_then(Json::as_array)
        .ok_or_else(|| invalid("lr_schedule.arguments is missing".into()))?
        .iter()
        .map(|a| {
            a.as_f64()
                .ok_or_else(|| invalid(format!("lr_schedule.arguments: {a:?}")))
        })
        .collect::<Result<_>>()?;
    let arg = |i: usize, default: Option<f64>| {
        args.get(i)
            .copied()
            .or(default)
            .ok_or_else(|| invalid(format!("lr_schedule {name}: argument {} is missing", i + 1)))
    };
    let schedule = match name {
        "cosine_decay" => {
            schedules::cosine_decay(arg(0, None)?, arg(1, None)? as u64, arg(2, Some(0.0))?)
        }
        "exponential_decay" => schedules::exponential_decay(arg(0, None)?, arg(1, None)?),
        "step_decay" => schedules::step_decay(arg(0, None)?, arg(1, None)?, arg(2, None)? as u64),
        "linear_schedule" => {
            schedules::linear_schedule(arg(0, None)?, arg(1, None)?, arg(2, None)? as u64)
        }
        other => {
            return Err(invalid(format!(
                "unknown lr_schedule {other:?} (cosine_decay, exponential_decay, step_decay or linear_schedule)"
            )));
        }
    };
    let warmup = config.get("warmup").and_then(Json::as_f64).unwrap_or(0.0) as u64;
    if warmup == 0 {
        return Ok(schedule);
    }
    let init = config
        .get("warmup_init")
        .and_then(Json::as_f64)
        .unwrap_or(0.0);
    let ramp = schedules::linear_schedule(init, arg(0, None)?, warmup);
    schedules::join_schedules(vec![ramp, schedule], vec![warmup + 1])
}

fn invalid(m: String) -> Error {
    Error::InvalidArgument(m)
}

/// Reads an `mlx_lm.lora` YAML configuration file (`--config`) into a JSON value.
///
/// mlx-lm's files use a subset of YAML, which is what this reads: block mappings by
/// indentation, block sequences (`- item`), flow sequences (`[a, b]`), quoted and plain
/// scalars (`null`, booleans, integers and floats such as `1e-5`, as mlx-lm's loader reads
/// them), and `#` comments. Anything else is an error naming the line.
pub fn read_config(path: impl AsRef<Path>) -> Result<Json> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    parse_yaml(&text).map_err(|e| Error::InvalidArgument(format!("{}: {e}", path.display())))
}

/// The YAML subset of [`read_config`].
pub(crate) fn parse_yaml(text: &str) -> std::result::Result<Json, String> {
    let lines: Vec<(usize, usize, &str)> = text
        .lines()
        .enumerate()
        .filter_map(|(n, raw)| {
            let line = strip_comment(raw).trim_end();
            let content = line.trim_start();
            (!content.is_empty() && content != "---")
                .then(|| (n + 1, line.len() - content.len(), content))
        })
        .collect();
    let mut at = 0;
    let value = block(&lines, &mut at, 0)?;
    match lines.get(at) {
        Some((n, ..)) => Err(format!("line {n}: unexpected indentation")),
        None => Ok(value.unwrap_or(Json::Null)),
    }
}

/// A `#` outside quotes starts a comment (when it begins the line or follows a space).
fn strip_comment(line: &str) -> &str {
    let mut quote = None;
    let mut prev = ' ';
    for (i, c) in line.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '#') if prev.is_whitespace() => return &line[..i],
            _ => {}
        }
        prev = c;
    }
    line
}

/// The block (mapping or sequence) at indentation `indent` starting at `lines[*at]`.
fn block(
    lines: &[(usize, usize, &str)],
    at: &mut usize,
    indent: usize,
) -> std::result::Result<Option<Json>, String> {
    let Some(&(_, first, content)) = lines.get(*at) else {
        return Ok(None);
    };
    if first < indent {
        return Ok(None);
    }
    let indent = first;
    if content.starts_with("- ") || content == "-" {
        let mut items = Vec::new();
        while let Some(&(n, i, c)) = lines.get(*at) {
            if i != indent || !(c.starts_with("- ") || c == "-") {
                if i > indent {
                    return Err(format!("line {n}: unexpected indentation"));
                }
                break;
            }
            *at += 1;
            let rest = c[1..].trim_start();
            let column = indent + (c.len() - rest.len());
            items.push(match (rest.is_empty(), split_key(rest)) {
                (true, _) => block(lines, at, indent + 1)?.unwrap_or(Json::Null),
                // `- key: value` starts a mapping whose keys line up with `key`.
                (false, Some((key, value))) if !rest.starts_with(['[', '"', '\'']) => {
                    let first = match value.is_empty() {
                        true => block(lines, at, column + 1)?.unwrap_or(Json::Null),
                        false => scalar(value).map_err(|e| format!("line {n}: {e}"))?,
                    };
                    let mut entries = vec![(key, first)];
                    if let Some(Json::Object(more)) = block(lines, at, column)? {
                        entries.extend(more);
                    }
                    Json::Object(entries)
                }
                (false, _) => scalar(rest).map_err(|e| format!("line {n}: {e}"))?,
            });
        }
        return Ok(Some(Json::Array(items)));
    }
    let mut entries = Vec::new();
    while let Some(&(n, i, c)) = lines.get(*at) {
        if i < indent {
            break;
        }
        if i > indent {
            return Err(format!("line {n}: unexpected indentation"));
        }
        let (key, rest) = split_key(c).ok_or_else(|| format!("line {n}: expected `key: value`"))?;
        *at += 1;
        let value = match rest.is_empty() {
            true => block(lines, at, indent + 1)?.unwrap_or(Json::Null),
            false => scalar(rest).map_err(|e| format!("line {n}: {e}"))?,
        };
        entries.push((key, value));
    }
    Ok(Some(Json::Object(entries)))
}

/// `key: rest` (the key unquoted).
fn split_key(line: &str) -> Option<(String, &str)> {
    let (key, rest) = match line.strip_prefix('"') {
        Some(q) => {
            let end = q.find('"')?;
            (q[..end].to_owned(), q[end + 1..].strip_prefix(':')?)
        }
        None => {
            let colon = line
                .find(": ")
                .or_else(|| line.ends_with(':').then(|| line.len() - 1))?;
            (line[..colon].trim().to_owned(), &line[colon + 1..])
        }
    };
    Some((key, rest.trim()))
}

/// A scalar or a flow sequence.
fn scalar(s: &str) -> std::result::Result<Json, String> {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('[') {
        let inner = inner.strip_suffix(']').ok_or("an unclosed `[`")?;
        let items = split_flow(inner);
        return items
            .into_iter()
            .filter(|i| !i.trim().is_empty())
            .map(scalar)
            .collect::<std::result::Result<_, _>>()
            .map(Json::Array);
    }
    if s.starts_with('{') {
        return Err("flow mappings are not supported".into());
    }
    for q in ['"', '\''] {
        if let Some(inner) = s.strip_prefix(q) {
            return inner
                .strip_suffix(q)
                .map(|v| Json::Str(v.to_owned()))
                .ok_or_else(|| "an unclosed quote".into());
        }
    }
    Ok(match s {
        "null" | "Null" | "NULL" | "~" => Json::Null,
        "true" | "True" | "TRUE" => Json::Bool(true),
        "false" | "False" | "FALSE" => Json::Bool(false),
        _ => match s.parse::<i64>() {
            Ok(i) => Json::Int(i),
            Err(_) => match s.parse::<f64>() {
                Ok(f) if s.chars().any(|c| c.is_ascii_digit()) => Json::Float(f),
                _ => Json::Str(s.to_owned()),
            },
        },
    })
}

/// Splits a flow sequence's inside at top-level commas.
fn split_flow(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut quote, mut start) = (Vec::new(), 0, None, 0);
    for (i, c) in s.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '[') => depth += 1,
            (None, ']') => depth -= 1,
            (None, ',') if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mlx-lm's `examples/lora_config.yaml` (comments and all).
    const EXAMPLE: &str = r#"
# The path to the local model directory or Hugging Face repo.
model: "mlx_model"
train: true
fine_tune_type: lora
optimizer: adamw
optimizer_config:
  adamw:
    betas: [0.9, 0.98]
    eps: 1e-6
    weight_decay: 0.05
    bias_correction: true
seed: 0
num_layers: 16
learning_rate: 1e-5
resume_adapter_file: null
grad_checkpoint: false
lora_parameters:
  # The layer keys to apply LoRA to.
  keys: ["self_attn.q_proj", "self_attn.v_proj"]
  rank: 8
  scale: 20.0
  dropout: 0.0
lr_schedule:
  name: cosine_decay
  warmup: 100 # 0 for no warmup
  warmup_init: 1e-7
  arguments: [1e-5, 1000, 1e-7]
hf_dataset:
  - name: a
    split: train
"#;

    #[test]
    fn reads_mlx_lm_config_files() {
        let c = parse_yaml(EXAMPLE).unwrap();
        assert_eq!(c.get("model").and_then(Json::as_str), Some("mlx_model"));
        assert_eq!(c.get("train"), Some(&Json::Bool(true)));
        assert_eq!(c.get("learning_rate").and_then(Json::as_f64), Some(1e-5));
        assert_eq!(c.get("resume_adapter_file"), Some(&Json::Null));
        let adamw = c
            .get("optimizer_config")
            .and_then(|o| o.get("adamw"))
            .unwrap();
        assert_eq!(adamw.get("eps").and_then(Json::as_f64), Some(1e-6));
        let keys = c
            .get("lora_parameters")
            .and_then(|l| l.get("keys"))
            .and_then(Json::as_array)
            .unwrap();
        assert_eq!(keys[1].as_str(), Some("self_attn.v_proj"));
        let datasets = c.get("hf_dataset").and_then(Json::as_array).unwrap();
        assert_eq!(
            datasets[0].get("split").and_then(Json::as_str),
            Some("train")
        );
        assert!(parse_yaml("a: 1\n   b: 2\n").is_err());
        assert!(parse_yaml("a: {b: 1}\n").is_err());
        let opt = build_optimizer("adamw", adamw, LearningRate::Constant(1e-5)).unwrap();
        assert!(matches!(opt, AnyOptimizer::AdamW(_)));
        assert!(build_optimizer("lamb", &Json::Null, 1e-5f32.into()).is_err());
        assert!(build_optimizer("adam", &parse_yaml("x: 1").unwrap(), 1e-5f32.into()).is_err());
    }

    /// The rates of `{cosine_decay, warmup 3, warmup_init 0.1, arguments [1, 4, 0.2]}`
    /// equal mlx-lm's `build_schedule` (recorded from mlx-lm 0.32.2 with MLX 0.32.4).
    #[test]
    fn schedules_equal_mlx_lm() {
        let c = parse_yaml(
            "name: cosine_decay\nwarmup: 3\nwarmup_init: 0.1\narguments: [1.0, 4, 0.2]\n",
        )
        .unwrap();
        let s = build_schedule(&c).unwrap();
        let want = MLX_RATES;
        for (step, w) in want.iter().enumerate() {
            let got = s.at(&Array::from(step as u64)).item::<f32>().unwrap();
            assert!(
                (f64::from(got) - w).abs() < 1e-6,
                "step {step}: {got} vs {w}"
            );
        }
        assert!(build_schedule(&parse_yaml("name: warmup\narguments: [1]\n").unwrap()).is_err());
    }

    const MLX_RATES: [f64; 10] = [
        0.100000001,
        0.400000006,
        0.700000048,
        1.0,
        1.0,
        0.88284272,
        0.599999964,
        0.317157298,
        0.200000003,
        0.200000003,
    ];
}
