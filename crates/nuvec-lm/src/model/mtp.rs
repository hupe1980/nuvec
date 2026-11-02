//! Qwen 3.5–3.8's multi-token-prediction (MTP) head, as mlx-vlm's `qwen3_5_mtp` drafter
//! (MIT): the next token's embedding and the target's final hidden state, each normalised,
//! projected to one hidden state and run through full-attention decoder layer(s) with their
//! own KV cache; the shared output head gives the logits of the token after.

use std::collections::BTreeMap;
use std::path::Path;

use nuvec::fast::AttentionMask;
use nuvec::{Array, Error, Result, ops};
use nuvec_nn::{Module, RmsNorm};

use super::{
    Attention, Block, BlockMlp, Builder, KvCache, Mixer, Mlp, Model, MoeMlp, Projection, add_norm,
};
use crate::config::{Config, LayerQuant};

/// The MTP head: `mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`,
/// `mtp.layers.{i}` and `mtp.norm`, as the original checkpoints name them.
#[derive(Debug, Clone, Module)]
pub(crate) struct MtpHead {
    fc: Projection,
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    layers: Vec<Block>,
    norm: RmsNorm,
}

impl MtpHead {
    pub(super) fn new(b: &Builder<'_>, c: &Config) -> Result<Self> {
        let d = c.hidden_size;
        let layers = (0..c.mtp_layers)
            .map(|i| {
                let path = |s: &str| format!("mtp.layers.{i}.{s}");
                Ok(Block {
                    mixer: Mixer::Attention(Attention::new(b, &path("self_attn"), c, None)?),
                    mlp: match &c.moe {
                        Some(moe) => BlockMlp::Moe(Box::new(MoeMlp::new(b, &path("mlp"), d, moe)?)),
                        None => BlockMlp::Dense(Box::new(Mlp::new(
                            b,
                            &path("mlp"),
                            d,
                            c.intermediate_size,
                            c.mlp_bias,
                        )?)),
                    },
                    input_layernorm: Builder::norm(d, c.rms_norm_eps),
                    post_attention_layernorm: Builder::norm(d, c.rms_norm_eps),
                    group: None,
                    gemma: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            fc: b.projection("mtp.fc", 2 * d, d, false)?,
            pre_fc_norm_embedding: Builder::norm(d, c.rms_norm_eps),
            pre_fc_norm_hidden: Builder::norm(d, c.rms_norm_eps),
            layers,
            norm: Builder::norm(d, c.rms_norm_eps),
        })
    }
}

/// The quantization of an MTP layer in a quantized model: `mtp.layers.{i}.<suffix>` takes
/// the setting the body's layers have for `<suffix>` (per-layer overrides such as an 8-bit
/// `mlp.gate`), any other projection the default; `None` in a full-precision model.
pub(super) fn mtp_quant(config: &Config, path: &str) -> Option<LayerQuant> {
    let q = config.quantization.as_ref()?;
    let entry = path
        .strip_prefix("mtp.layers.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(_, suffix)| {
            q.layers.iter().find(|l| {
                l.path
                    .strip_prefix("model.layers.")
                    .and_then(|rest| rest.split_once('.'))
                    .is_some_and(|(_, s)| s == suffix)
            })
        });
    Some(match entry {
        Some(l) => LayerQuant {
            path: path.to_owned(),
            ..l.clone()
        },
        None => LayerQuant {
            path: path.to_owned(),
            quant: Some(q.default),
            global_scale: false,
        },
    })
}

/// MTP tensors in the original checkpoints' layout (`mtp.*`, Hugging Face): zero-centred
/// norms plus one, fused or per-expert MoE projections as the body's `switch_mlp`.
fn sanitize_mtp(weights: BTreeMap<String, Array>) -> Result<BTreeMap<String, Array>> {
    const NORMS: [&str; 7] = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];
    let mut out = BTreeMap::new();
    for (k, v) in weights {
        if !k.starts_with("mtp.") {
            continue;
        }
        let v = match v.ndim() == 1 && NORMS.iter().any(|n| k.ends_with(n)) {
            true => v.add(1.0),
            false => v,
        };
        if let Some(prefix) = k.strip_suffix("experts.gate_up_proj") {
            let [gate, up] = v.split_n::<2>(-2);
            out.insert(format!("{prefix}switch_mlp.gate_proj.weight"), gate);
            out.insert(format!("{prefix}switch_mlp.up_proj.weight"), up);
        } else if let Some(prefix) = k.strip_suffix("experts.down_proj") {
            out.insert(format!("{prefix}switch_mlp.down_proj.weight"), v);
        } else {
            out.insert(k, v);
        }
    }
    // Per-expert projections (`experts.{e}.{name}.weight`), stacked.
    let first: Vec<String> = out
        .keys()
        .filter(|k| k.ends_with(".mlp.experts.0.up_proj.weight"))
        .cloned()
        .collect();
    for key in first {
        let prefix = key.trim_end_matches("experts.0.up_proj.weight").to_owned();
        for name in ["up_proj", "down_proj", "gate_proj"] {
            let mut experts = Vec::new();
            while let Some(w) =
                out.remove(&format!("{prefix}experts.{}.{name}.weight", experts.len()))
            {
                experts.push(w);
            }
            out.insert(
                format!("{prefix}switch_mlp.{name}.weight"),
                ops::stack(&experts, 0),
            );
        }
    }
    Ok(out)
}

impl Model {
    /// Whether the model has its multi-token-prediction head (loaded with the checkpoint, or
    /// [`Model::attach_mtp`]).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Builds the MTP head (placeholder weights) from the configuration and the
    /// quantization of each of its layers.
    pub(crate) fn build_mtp(
        &mut self,
        quantized: &dyn Fn(&str) -> Option<LayerQuant>,
    ) -> Result<()> {
        if self.config.mtp_layers == 0 {
            return Err(Error::InvalidArgument(format!(
                "{}: the configuration has no multi-token-prediction layers",
                self.config.model_type
            )));
        }
        self.mtp = Some(MtpHead::new(&Builder { quantized }, &self.config)?);
        Ok(())
    }

    /// Attaches the MTP head from a safetensors file of the original checkpoint's `mtp.*`
    /// tensors (mlx-community conversions drop them; `scripts/extract_mtp.py` writes the
    /// file). In a quantized model each MTP projection is quantized on load with the
    /// setting the body's layers have for it (an 8-bit `mlp.gate` stays 8-bit); a
    /// full-precision model keeps them in full precision. A missing tensor or a shape
    /// mismatch is an error naming it.
    pub fn attach_mtp(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let raw = nuvec::io::load_safetensors(path)?.tensors;
        let weights = sanitize_mtp(raw.into_iter().collect())?;
        if weights.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "{}: no mtp.* tensors",
                path.display()
            )));
        }
        let config = self.config.clone();
        let quant = |p: &str| mtp_quant(&config, p);
        self.build_mtp(&quant)?;
        let mut loaded = BTreeMap::new();
        for (name, w) in weights {
            let layer = name.strip_suffix(".weight").unwrap_or(&name);
            match quant(layer).and_then(|l| l.quant).filter(|_| w.ndim() >= 2) {
                Some(q) if !name.contains("norm") => {
                    let packed = w.quantize(q);
                    loaded.insert(format!("{layer}.weight"), packed.weight);
                    loaded.insert(format!("{layer}.scales"), packed.scales);
                    if let Some(b) = packed.biases {
                        loaded.insert(format!("{layer}.biases"), b);
                    }
                }
                _ => {
                    loaded.insert(name, w);
                }
            }
        }
        let names: Vec<(String, Array)> = loaded
            .into_iter()
            .map(|(k, v)| (k.strip_prefix("mtp.").unwrap_or(&k).to_owned(), v))
            .collect();
        // Loaded aside, so a failed load leaves the model as it was (no half-built head).
        let mut head = self.mtp.take().expect("built");
        let loaded = head.load_weights(names);
        if loaded.is_ok() {
            head.train(false);
            self.mtp = Some(head);
        }
        loaded
    }

    /// An empty cache per MTP layer.
    pub fn new_mtp_cache(&self) -> Vec<KvCache> {
        let n = self.mtp.as_ref().map_or(0, |m| m.layers.len());
        (0..n).map(|_| KvCache::default()).collect()
    }

    /// The MTP head on `tokens` `[B, L]` (each the token after the position of the matching
    /// row of `hidden`) and `hidden` `[B, L, hidden_size]` (the target's final hidden
    /// states, [`Model::forward_hidden`], or the head's own for further drafts): the head's
    /// final hidden states, whose [`Model::logits`] predict the token after each. `cache`
    /// holds one entry per MTP layer ([`Model::new_mtp_cache`]) and is extended by `L`
    /// positions.
    #[track_caller]
    pub fn mtp_forward(&self, tokens: &Array, hidden: &Array, cache: &mut [KvCache]) -> Array {
        let Some(m) = &self.mtp else {
            return Array::failure(
                "mtp_forward",
                nuvec::OpErrorKind::InvalidArgument("the model has no MTP head".into()),
            );
        };
        if cache.len() != m.layers.len() {
            return super::mismatched_layers(cache.len(), m.layers.len());
        }
        let e = self.model.embed_tokens.forward(tokens);
        let x = ops::concatenate(
            &[
                m.pre_fc_norm_embedding.forward(&e),
                m.pre_fc_norm_hidden.forward(hidden),
            ],
            -1,
        );
        let mut h = m.fc.forward(&x);
        let mask = match tokens.shape().get(1) {
            Some(&l) if l > 1 => AttentionMask::Causal,
            _ => AttentionMask::None,
        };
        let mut pending = None;
        for (layer, c) in m.layers.iter().zip(cache) {
            let (next, mlp) = layer.forward(&h, pending.as_ref(), mask, c);
            (h, pending) = (next, Some(mlp));
        }
        add_norm(&h, pending.as_ref(), &m.norm).1
    }
}
