//! Prompt caches in mlx-lm's file format (`save_prompt_cache`, `load_prompt_cache`): a
//! safetensors file whose tensors are each layer's cache state and whose metadata names the
//! caches' classes, the caller's metadata and the scalar entries.

use std::collections::BTreeMap;
use std::path::Path;

use nuvec::{Array, DType, Result};

use super::{KvCache, KvQuant, Model};

fn invalid(msg: String) -> nuvec::Error {
    nuvec::Error::InvalidArgument(format!("prompt cache: {msg}"))
}

/// Saves `model`'s `caches` (one per layer) and `metadata` to `path`, as mlx-lm's
/// `save_prompt_cache` (keys as mlx-lm caches them: with YaRN's mscale where `model` folds
/// it into the attention scale): attention layers as `KVCache` (keys, values, offset) or,
/// quantized, `QuantizedKVCache`; sliding-window layers as `RotatingKVCache` (their last
/// `window` positions in order); Gated DeltaNet layers as `ArraysCache` (the convolution's
/// inputs and the recurrent state). mlx-lm loads the file with `load_prompt_cache`, and
/// [`load_prompt_cache`] loads mlx-lm's. Quantized sliding-window caches, and quantized caches
/// of layers whose keys carry a folded mscale, have no mlx-lm form and are refused.
pub fn save_prompt_cache(
    path: impl AsRef<Path>,
    model: &Model,
    caches: &[KvCache],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let keys = |i: usize, k: Array| match model.cached_key_factor(i, k.dtype()) {
        1.0 => k,
        f => k.multiply(f64::from(f)),
    };
    let mut tensors: Vec<(String, Array)> = Vec::new();
    let mut meta: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| (format!("0.{k}"), v.clone()))
        .collect();
    // mlx-lm's scalar list starts with "" (so it is never empty).
    let mut scalars: Vec<(String, &str)> = Vec::new();
    let none = || Array::from_slice::<f32>(&[], [0]);
    for (i, c) in caches.iter().enumerate() {
        let mut put = |key: String, a: Array| tensors.push((key, a));
        let class = if c.conv.is_some() || c.recurrent.is_some() {
            put(format!("{i}.0.0"), c.conv.clone().unwrap_or_else(none));
            // LFM2's short convolutions keep one state (mlx-lm's `ArraysCache(size=1)`).
            if c.recurrent.is_some() || !model.short_conv(i) {
                put(format!("{i}.0.1"), c.recurrent.clone().unwrap_or_else(none));
            }
            for j in [1, 2] {
                put(format!("{i}.{j}"), none());
                scalars.push((format!("{i}.{j}"), "none"));
            }
            "ArraysCache"
        } else if let Some((k, v, q)) = c.quantized_keys_values() {
            if c.window.is_some() || c.start != 0 {
                return Err(invalid(format!(
                    "layer {i}: a quantized sliding-window cache (mlx-lm has none)"
                )));
            }
            if model.cached_key_factor(i, k[1].dtype()) != 1.0 {
                return Err(invalid(format!(
                    "layer {i}: quantized keys with a folded mscale cannot be rescaled"
                )));
            }
            for (side, t) in [k, v].into_iter().enumerate() {
                for (j, a) in t.into_iter().enumerate() {
                    put(format!("{i}.{side}.{j}"), a);
                }
            }
            for (j, value) in [(2, c.offset), (3, q.group_size), (4, q.bits)] {
                put(format!("{i}.{j}"), Array::from(value));
                scalars.push((format!("{i}.{j}"), "scalar"));
            }
            "QuantizedKVCache"
        } else if let Some(w) = c.window {
            let (k, v) = c
                .keys_values()
                .ok_or_else(|| invalid(format!("layer {i}: an empty sliding-window cache")))?;
            let n = k.shape()[2];
            let kept = n.min(w);
            let last = |a: &Array| a.i((.., .., n - kept..));
            put(format!("{i}.0"), keys(i, last(&k)));
            put(format!("{i}.1"), last(&v));
            for (j, value) in [(2, c.offset), (3, 0), (4, w), (5, kept)] {
                put(format!("{i}.{j}"), Array::from(value));
                scalars.push((format!("{i}.{j}"), "scalar"));
            }
            "RotatingKVCache"
        } else {
            match c.keys_values() {
                Some((k, v)) if c.start == 0 => {
                    put(format!("{i}.0"), keys(i, k));
                    put(format!("{i}.1"), v);
                }
                Some(_) => {
                    return Err(invalid(format!(
                        "layer {i}: a cache that dropped positions"
                    )));
                }
                None => {
                    for j in [0, 1] {
                        put(format!("{i}.{j}"), none());
                        scalars.push((format!("{i}.{j}"), "none"));
                    }
                }
            }
            put(format!("{i}.2"), Array::from(c.offset));
            scalars.push((format!("{i}.2"), "scalar"));
            "KVCache"
        };
        meta.push((format!("1.{i}"), class.to_string()));
    }
    meta.push(("2.0".into(), String::new()));
    for (j, (pos, kind)) in scalars.iter().enumerate() {
        meta.push((format!("2.{}.0", j + 1), pos.clone()));
        meta.push((format!("2.{}.1", j + 1), (*kind).to_string()));
    }
    let tensors: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
    let meta: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    nuvec::io::save_safetensors(path, &tensors, &meta)
}

/// Loads a prompt cache for `model` saved by [`save_prompt_cache`] or mlx-lm's
/// `save_prompt_cache`: one cache per layer and the caller's metadata. Keys of layers whose
/// YaRN mscale `model` folds into the attention scale are divided by it (one rounding).
/// mlx-lm's `KVCache`, `QuantizedKVCache`, `RotatingKVCache` (without kept leading
/// positions, as its language models use it) and `ArraysCache` load.
pub fn load_prompt_cache(
    path: impl AsRef<Path>,
    model: &Model,
) -> Result<(Vec<KvCache>, BTreeMap<String, String>)> {
    let file = nuvec::io::load_safetensors(path)?;
    let (t, m) = (&file.tensors, &file.metadata);
    let metadata = m
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("0.").map(|k| (k.to_string(), v.clone())))
        .collect();
    let mut classes: Vec<(usize, &str)> = m
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("1.")?.parse().ok().map(|i| (i, v.as_str())))
        .collect();
    classes.sort_unstable();
    let tensor = |key: String| {
        t.get(&key)
            .cloned()
            .ok_or_else(|| invalid(format!("no tensor `{key}`")))
    };
    let scalar = |key: String| -> Result<i32> {
        let v = tensor(key.clone())?.astype(DType::I64).to_vec::<i64>()?;
        v.first()
            .and_then(|&v| i32::try_from(v).ok())
            .ok_or_else(|| invalid(format!("`{key}` is not a scalar")))
    };
    let present = |a: Array| (a.size() > 0).then_some(a);
    let mut caches = Vec::with_capacity(classes.len());
    for (n, &(i, class)) in classes.iter().enumerate() {
        if i != n {
            return Err(invalid(format!("no class for layer {n}")));
        }
        let cache = match class {
            "KVCache" => KvCache {
                keys: present(tensor(format!("{i}.0"))?),
                values: present(tensor(format!("{i}.1"))?),
                offset: scalar(format!("{i}.2"))?,
                ..KvCache::default()
            },
            "RotatingKVCache" => {
                let (offset, keep) = (scalar(format!("{i}.2"))?, scalar(format!("{i}.3"))?);
                let (max_size, idx) = (scalar(format!("{i}.4"))?, scalar(format!("{i}.5"))?);
                if keep != 0 {
                    return Err(invalid(format!(
                        "layer {i}: a rotating cache keeping {keep} positions"
                    )));
                }
                // mlx-lm's `_temporal_order` (no kept positions).
                let order = |a: Array| {
                    let len = a.shape()[2];
                    if idx == len {
                        a
                    } else if idx < offset {
                        nuvec::ops::concatenate(&[a.i((.., .., idx..)), a.i((.., .., ..idx))], 2)
                    } else {
                        a.i((.., .., ..idx))
                    }
                };
                let (k, v) = (
                    order(tensor(format!("{i}.0"))?),
                    order(tensor(format!("{i}.1"))?),
                );
                let n = k.shape()[2];
                KvCache {
                    keys: Some(k),
                    values: Some(v),
                    window: Some(max_size),
                    start: offset - n,
                    offset,
                    ..KvCache::default()
                }
            }
            "QuantizedKVCache" => {
                let triple = |side: usize| -> Result<[Array; 3]> {
                    Ok([
                        tensor(format!("{i}.{side}.0"))?,
                        tensor(format!("{i}.{side}.1"))?,
                        tensor(format!("{i}.{side}.2"))?,
                    ])
                };
                let quant = KvQuant {
                    bits: scalar(format!("{i}.4"))?,
                    group_size: scalar(format!("{i}.3"))?,
                    start: 0,
                };
                quant.validate()?;
                let (qk, qv) = (triple(0)?, triple(1)?);
                if model.cached_key_factor(i, qk[1].dtype()) != 1.0 {
                    return Err(invalid(format!(
                        "layer {i}: quantized keys with a folded mscale cannot be rescaled"
                    )));
                }
                KvCache {
                    qkeys: Some(qk),
                    qvalues: Some(qv),
                    quant: Some(quant),
                    offset: scalar(format!("{i}.2"))?,
                    ..KvCache::default()
                }
            }
            // Size 2 (Gated DeltaNet, Mamba-2: convolution and recurrent state) or 1 (LFM2's
            // short convolution).
            "ArraysCache" => KvCache {
                conv: present(tensor(format!("{i}.0.0"))?),
                recurrent: t.get(&format!("{i}.0.1")).cloned().and_then(present),
                ..KvCache::default()
            },
            other => {
                return Err(invalid(format!(
                    "layer {i}: unsupported cache class `{other}`"
                )));
            }
        };
        caches.push(cache);
    }
    // mlx-lm's `ArraysCache` keeps no offset: the attention layers' position count.
    let offset = caches
        .iter()
        .filter(|c| c.conv.is_none() && c.recurrent.is_none())
        .map(|c| c.offset)
        .max()
        .unwrap_or_default();
    for c in caches
        .iter_mut()
        .filter(|c| c.conv.is_some() || c.recurrent.is_some())
    {
        c.offset = offset;
    }
    for (i, c) in caches.iter_mut().enumerate() {
        if let Some(k) = c.keys.take() {
            let f = model.cached_key_factor(i, k.dtype());
            c.keys = Some(if f == 1.0 { k } else { k.divide(f64::from(f)) });
        }
    }
    Ok((caches, metadata))
}
