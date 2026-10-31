//! Gemma 4's text model, as mlx-lm's `gemma4_text`: sliding and full layers with their own
//! head sizes and RoPE, q/k norms and an unweighted v norm, attention scale 1, sandwich
//! norms around a GeGLU MLP, per-layer inputs (E2B/E4B), KV-shared last layers, a
//! mixture-of-experts branch beside the MLP (26B-A4B), a per-layer scalar and the logits'
//! tanh soft cap.

use super::*;
use crate::config::Gemma4;

/// A decoder layer's Gemma-only parts (named as mlx-lm's `DecoderLayer`).
#[derive(Debug, Clone, Module)]
pub(super) struct Parts {
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Option<Projection>,
    per_layer_projection: Option<Projection>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_scalar: Param,
    router: Option<Router>,
    experts: Option<Experts>,
    post_feedforward_layernorm_1: Option<RmsNorm>,
    post_feedforward_layernorm_2: Option<RmsNorm>,
    pre_feedforward_layernorm_2: Option<RmsNorm>,
    /// The earlier layer whose keys and values a KV-shared layer reads.
    #[tree(skip)]
    kv_source: Option<usize>,
}

/// The experts' router: normalised (its weight times `hidden^-½`), projected, the top-k
/// softmax scaled per expert.
#[derive(Debug, Clone, Module)]
pub(super) struct Router {
    proj: Projection,
    scale: Param,
    per_expert_scale: Param,
    #[tree(skip)]
    eps: f32,
    #[tree(skip)]
    root: f32,
    #[tree(skip)]
    k: i32,
}

impl Router {
    /// Each token's experts (`[B, L, k]`) and their weights.
    #[track_caller]
    fn forward(&self, h: &Array) -> (Array, Array) {
        let weight = self.scale.value().multiply(self.root);
        let x = fast::rms_norm(h, Some(&weight), self.eps);
        let scores = self.proj.forward(&x);
        let k = self.k;
        let top = scores.argpartition(-k, -1).i((.., .., -k..));
        let weights = scores.take_along_axis(&top, -1).softmax(-1);
        let weights = weights.multiply(self.per_expert_scale.value().take(&top, 0));
        (top, weights)
    }
}

/// The GeGLU experts (`experts.switch_glu`).
#[derive(Debug, Clone, Module)]
pub(super) struct Experts {
    switch_glu: SwitchGlu,
}

/// The per-layer inputs (E2B/E4B): a token embedding per layer and a projection of the
/// scaled embedding, normalised, summed and scaled by `2^-½`.
#[derive(Debug, Clone, Module)]
pub(super) struct PerLayer {
    embed_tokens_per_layer: TokenEmbedding,
    per_layer_model_projection: Projection,
    per_layer_projection_norm: RmsNorm,
    #[tree(skip)]
    layers: i32,
    #[tree(skip)]
    width: i32,
}

impl PerLayer {
    /// `[B, L, layers, width]` for `tokens` and the scaled embedding `h`.
    #[track_caller]
    fn inputs(&self, tokens: &Array, h: &Array) -> Array {
        let [b, l, d] = h.shape_n::<3>().unwrap_or_default();
        let shape = [b, l, self.layers, self.width];
        let embedded = self
            .embed_tokens_per_layer
            .forward(tokens)
            .multiply((self.width as f32).sqrt())
            .reshape(shape);
        let projected = self
            .per_layer_model_projection
            .forward(h)
            .multiply((d as f32).powf(-0.5))
            .reshape(shape);
        let projected = self.per_layer_projection_norm.forward(&projected);
        projected.add(&embedded).multiply(0.5f32.sqrt())
    }
}

/// The model's structure with placeholder parameters.
pub(super) fn model(b: &Builder<'_>, c: &Config, g: &Gemma4) -> Result<Model> {
    let (d, eps, layers) = (c.hidden_size, c.rms_norm_eps, c.num_hidden_layers);
    let first_shared = g.first_shared(layers);
    let sliding = |i: usize| c.layer_windows.get(i).copied().flatten().is_some();
    let norm = |dims| Builder::norm(dims, eps);
    let blocks = (0..layers)
        .map(|i| {
            let path = |s: &str| format!("model.layers.{i}.{s}");
            let a = |s: &str| path(&format!("self_attn.{s}"));
            let shared = i >= first_shared && first_shared > 0;
            // The last earlier layer of the same kind (mlx-lm's `previous_kvs`).
            let kv_source = match shared {
                true => Some(
                    (0..first_shared)
                        .rev()
                        .find(|&j| sliding(j) == sliding(i))
                        .ok_or_else(|| {
                            nuvec::Error::InvalidArgument(format!(
                                "config.json: KV-shared layer {i} has no earlier layer of its kind"
                            ))
                        })?,
                ),
                false => None,
            };
            let full = !sliding(i);
            let hd = if full { g.global_head_dim } else { c.head_dim };
            let k_eq_v = full && g.k_eq_v;
            let kv_heads = if k_eq_v {
                g.global_kv_heads
            } else {
                c.num_key_value_heads
            };
            let (q, kv) = (c.num_attention_heads * hd, kv_heads * hd);
            let rope = match full {
                true => (g.full_rope.0, &g.full_rope.1),
                false => (g.sliding_rope.0, &g.sliding_rope.1),
            };
            let attention = Attention {
                q_proj: b.projection(&a("q_proj"), d, q, false)?,
                k_proj: (!shared)
                    .then(|| b.projection(&a("k_proj"), d, kv, false))
                    .transpose()?,
                v_proj: (!shared && !k_eq_v)
                    .then(|| b.projection(&a("v_proj"), d, kv, false))
                    .transpose()?,
                o_proj: b.projection(&a("o_proj"), q, d, false)?,
                q_norm: Some(norm(hd)),
                k_norm: (!shared).then(|| norm(hd)),
                heads: int(&a(""), c.num_attention_heads)?,
                kv_heads: int(&a(""), kv_heads)?,
                scale: 1.0,
                rope: Rope::with(hd, rope, c.rope_traditional)?,
                gate: false,
                sinks: None,
                window: c.layer_windows[i].map(|w| int(&a(""), w)).transpose()?,
                mscale: 1.0,
                v_norm: Some(eps),
            };
            let wide = if g.double_wide_mlp && shared { 2 } else { 1 };
            let mut mlp = Mlp::new(b, &path("mlp"), d, c.intermediate_size * wide, false)?;
            mlp.geglu = true;
            let ple = g.per_layer_input;
            let per_layer = |name: &str, input, output| {
                (ple > 0)
                    .then(|| b.projection(&path(name), input, output, false))
                    .transpose()
            };
            let moe = g.moe.as_ref();
            let moe_norm = || moe.map(|_| norm(d));
            let parts = Parts {
                pre_feedforward_layernorm: norm(d),
                post_feedforward_layernorm: norm(d),
                per_layer_input_gate: per_layer("per_layer_input_gate", d, ple)?,
                per_layer_projection: per_layer("per_layer_projection", ple, d)?,
                post_per_layer_input_norm: (ple > 0).then(|| norm(d)),
                layer_scalar: param(&[1], DType::F32),
                router: moe
                    .map(|m| -> Result<Router> {
                        Ok(Router {
                            proj: b.projection(&path("router.proj"), d, m.experts, false)?,
                            scale: param(&[d], DType::F32),
                            per_expert_scale: param(&[m.experts], DType::F32),
                            eps,
                            root: (d as f32).powf(-0.5),
                            k: int(&path("router"), m.experts_per_token)?,
                        })
                    })
                    .transpose()?,
                experts: moe
                    .map(|m| -> Result<Experts> {
                        Ok(Experts {
                            switch_glu: SwitchGlu::new(b, &path("experts.switch_glu"), d, m)?,
                        })
                    })
                    .transpose()?,
                post_feedforward_layernorm_1: moe_norm(),
                post_feedforward_layernorm_2: moe_norm(),
                pre_feedforward_layernorm_2: moe_norm(),
                kv_source,
            };
            Ok(Block {
                mixer: Mixer::Attention(attention),
                mlp: BlockMlp::Dense(Box::new(mlp)),
                input_layernorm: norm(d),
                post_attention_layernorm: norm(d),
                gemma: Some(Box::new(parts)),
            })
        })
        .collect::<Result<_>>()?;
    let ple = g.per_layer_input;
    let per_layer = match ple {
        0 => None,
        _ => Some(Box::new(PerLayer {
            embed_tokens_per_layer: b.embedding(
                "model.embed_tokens_per_layer",
                g.per_layer_vocab,
                layers * ple,
            )?,
            per_layer_model_projection: b.projection(
                "model.per_layer_model_projection",
                d,
                layers * ple,
                false,
            )?,
            per_layer_projection_norm: norm(ple),
            layers: int("model.embed_tokens_per_layer", layers)?,
            width: int("model.embed_tokens_per_layer", ple)?,
        })),
    };
    Ok(Model {
        training: true,
        config: c.clone(),
        model: Body {
            embed_tokens: b.embedding("model.embed_tokens", c.vocab_size, d)?,
            layers: blocks,
            norm: norm(d),
            per_layer,
        },
        lm_head: (!c.tie_word_embeddings)
            .then(|| b.projection("lm_head", d, c.vocab_size, false))
            .transpose()?,
        mtp: None,
    })
}

/// A layer's work after its attention output `o`: the sandwich norms around the MLP (and
/// the experts), the per-layer input and the layer scalar.
#[track_caller]
fn after_attention(
    layer: &Block,
    p: &Parts,
    h: &Array,
    o: &Array,
    per_layer: Option<&Array>,
    i: usize,
) -> Array {
    let o = layer.post_attention_layernorm.forward(o);
    // `h + o` and the MLP's input norm in one kernel.
    let (h1, n) = add_norm(h, Some(&o), &p.pre_feedforward_layernorm);
    let mut f = layer.mlp.forward(&n);
    if let (Some(router), Some(experts), Some(post1), Some(post2), Some(pre2)) = (
        &p.router,
        &p.experts,
        &p.post_feedforward_layernorm_1,
        &p.post_feedforward_layernorm_2,
        &p.pre_feedforward_layernorm_2,
    ) {
        let (top, weights) = router.forward(&h1);
        let y = experts.switch_glu.forward(&pre2.forward(&h1), &top);
        let y = y.multiply(weights.expand_dims(-1)).sum(-2);
        f = post1.forward(&f).add(post2.forward(&y));
    }
    let mut h = h1.add(p.post_feedforward_layernorm.forward(&f));
    if let (Some(inputs), Some(gate), Some(proj), Some(norm)) = (
        per_layer,
        &p.per_layer_input_gate,
        &p.per_layer_projection,
        &p.post_per_layer_input_norm,
    ) {
        let input = inputs.i((.., .., i as i32));
        let g = nuvec_nn::geglu(&gate.forward(&h), &input);
        h = h.add(norm.forward(&proj.forward(&g)));
    }
    h.multiply(p.layer_scalar.value())
}

/// The scaled embedding and the per-layer inputs of `tokens`.
#[track_caller]
fn embed(m: &Model, tokens: &Array) -> (Array, Option<Array>) {
    let body = &m.model;
    let d = m.config.hidden_size;
    let h = body
        .embed_tokens
        .forward(tokens)
        .multiply((d as f32).sqrt());
    let per_layer = body.per_layer.as_ref().map(|p| p.inputs(tokens, &h));
    (h, per_layer)
}

/// The final norm, head and soft cap.
#[track_caller]
fn logits(m: &Model, h: &Array) -> Array {
    let logits = m.head(&m.model.norm.forward(h));
    match m.config.gemma4.as_ref().and_then(|g| g.final_logit_softcap) {
        Some(cap) => nuvec_nn::logit_softcap(&logits, cap),
        None => logits,
    }
}

/// The attention layer and Gemma parts of a Gemma 4 block.
fn parts(layer: &Block) -> (&Attention, &Parts) {
    match (&layer.mixer, &layer.gemma) {
        (Mixer::Attention(a), Some(p)) => (a, p),
        _ => unreachable!("Gemma 4 layers are attention layers with Gemma parts"),
    }
}

/// [`Model::forward`] of a Gemma 4 model.
#[track_caller]
pub(super) fn forward(m: &Model, tokens: &Array, cache: &mut [KvCache]) -> Array {
    let (mut h, per_layer) = embed(m, tokens);
    let mask = match tokens.shape().get(1) {
        Some(&l) if l > 1 => AttentionMask::Causal,
        _ => AttentionMask::None,
    };
    // Each layer's keys and values and its first query's position, for KV-shared layers.
    let mut kvs: Vec<Option<((Array, Array), i32)>> = vec![None; m.model.layers.len()];
    for (i, (layer, c)) in m.model.layers.iter().zip(cache).enumerate() {
        let (attention, p) = parts(layer);
        let at = c.offset;
        let shared = p
            .kv_source
            .and_then(|s| kvs[s].as_ref())
            .map(|(kv, at)| (kv, *at));
        let n = layer.input_layernorm.forward(&h);
        let (o, kv) = attention.forward_kv(&n, mask, c, shared);
        kvs[i] = Some((kv, shared.map_or(at, |(_, at)| at)));
        h = after_attention(layer, p, &h, &o, per_layer.as_ref(), i);
    }
    logits(m, &h)
}

/// [`Model::forward_paged`] of a Gemma 4 model: KV-shared layers attend to their source
/// layer's pools.
#[track_caller]
pub(super) fn forward_paged(
    m: &Model,
    tokens: &Array,
    layers: &mut [PagedLayer],
    step: &PagedStep,
) -> Array {
    let (mut h, per_layer) = embed(m, tokens);
    let mut pools: Vec<Option<(Array, Array, usize)>> = vec![None; m.model.layers.len()];
    for (i, (layer, pages)) in m.model.layers.iter().zip(layers).enumerate() {
        let (attention, p) = parts(layer);
        let shared = p.kv_source.and_then(|s| pools[s].as_ref());
        let n = layer.input_layernorm.forward(&h);
        let (o, kv) = attention.forward_paged_kv(&n, pages, step, shared);
        pools[i] = Some(kv);
        h = after_attention(layer, p, &h, &o, per_layer.as_ref(), i);
    }
    logits(m, &h)
}
