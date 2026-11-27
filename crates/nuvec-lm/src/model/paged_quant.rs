//! Shared eight-bit KV pages. Every component uses the same physical page and slot.

use super::{
    Array, DType, GroupStep, KvCache, KvQuant, Mixer, Model, Result, ops, quantized_attention,
};
use nuvec::fast::{self, AttentionMask};

pub(crate) const FORMAT: KvQuant = KvQuant {
    bits: 8,
    group_size: 64,
    start: 0,
    rotate: false,
};

/// Shape and dtype are checked before a reservation can allocate blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    heads: i32,
    dims: i32,
    dtype: DType,
    pub(crate) bytes_per_position: usize,
}

impl Layout {
    pub(crate) fn for_model(model: &Model) -> Result<Self> {
        let c = &model.config;
        if c.model_type != "qwen3"
            || model.stage().is_some()
            || model.sharding().is_some()
            || !matches!(c.head_dim, 64 | 128)
            || c.num_key_value_heads == 0
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || !matches!(c.num_attention_heads / c.num_key_value_heads, 1 | 2 | 4 | 8)
            || model.model.layers.is_empty()
            || !model.model.layers.iter().all(|block| {
                let Mixer::Attention(attn) = &block.mixer else {
                    return false;
                };
                let width = c.num_key_value_heads.checked_mul(c.head_dim);
                attn.window.is_none()
                    && attn.heads as usize == c.num_attention_heads
                    && attn.kv_heads as usize == c.num_key_value_heads
                    && width.is_some()
                    && attn.k_proj.as_ref().map(super::Projection::outputs) == width
                    && attn.v_proj.as_ref().map(super::Projection::outputs) == width
            })
        {
            return Err(nuvec::Error::InvalidArgument("eight-bit pages require single-device full-attention Qwen3, head dimension 64 or 128 and GQA 1, 2, 4 or 8".into()));
        }
        let dtype = model.embed(&Array::from_slice(&[0u32], [1, 1])).dtype();
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(nuvec::Error::InvalidArgument(
                "eight-bit pages require floating-point activations".into(),
            ));
        }
        let bytes_per_head = 2 * (c.head_dim + 2 * (c.head_dim / 64) * dtype.size());
        let bytes_per_position = bytes_per_head
            .checked_mul(c.num_key_value_heads)
            .and_then(|b| b.checked_mul(model.model.layers.len()))
            .ok_or_else(|| {
                nuvec::Error::InvalidArgument("eight-bit page capacity overflow".into())
            })?;
        Ok(Self {
            heads: i32::try_from(c.num_key_value_heads)
                .map_err(|_| nuvec::Error::InvalidArgument("too many KV heads".into()))?,
            dims: c.head_dim as i32,
            dtype,
            bytes_per_position,
        })
    }

    pub(crate) fn import_rows(self, cache: &KvCache, len: usize) -> Result<Option<i32>> {
        if cache
            .quant
            .is_some_and(|q| q.bits != FORMAT.bits || q.group_size != FORMAT.group_size || q.rotate)
        {
            return Err(nuvec::Error::InvalidArgument(
                "paged cache: incompatible KV quantization".into(),
            ));
        }
        let Some((k, v, _)) = cache.quantized_keys_values() else {
            return if len == 0 {
                Ok(None)
            } else {
                Err(nuvec::Error::InvalidArgument(
                    "paged cache: expected an eight-bit prefill cache".into(),
                ))
            };
        };
        let batch = k[0].shape().first().copied().unwrap_or(0);
        if batch == 0
            || k.iter().chain(&v).enumerate().any(|(i, a)| {
                let codes = i % 3 == 0;
                a.dtype() != if codes { DType::U32 } else { self.dtype }
                    || a.shape()
                        != [
                            batch,
                            self.heads,
                            len as i32,
                            self.dims / if codes { 4 } else { 64 },
                        ]
            })
        {
            return Err(nuvec::Error::InvalidArgument(
                "paged cache: incompatible eight-bit prefill cache".into(),
            ));
        }
        Ok(Some(batch))
    }
}

/// Keys then values, each represented by codes, scales and biases. Allocation is lazy.
#[derive(Debug, Default, Clone)]
pub(crate) struct Pages {
    keys: [Option<Array>; 3],
    values: [Option<Array>; 3],
}

impl Pages {
    pub(crate) fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(&self.values).flatten()
    }

    /// Scatter matching prefill or projected step components [B, H, T, D] to slots [B, T].
    pub(crate) fn write(
        &mut self,
        size: (i32, i32),
        keys: [Array; 3],
        values: [Array; 3],
        slots: &Array,
    ) {
        for (dest, src) in self
            .keys
            .iter_mut()
            .chain(&mut self.values)
            .zip(keys.into_iter().chain(values))
        {
            let pool = super::take_pool(dest, size, &src);
            *dest = Some(fast::paged_write(
                &pool,
                &src.transpose([0, 2, 1, 3]),
                slots,
            ));
        }
    }

    pub(crate) fn attention(
        &self,
        q: &Array,
        group: &GroupStep,
        (lengths, device_lengths): (&[usize], &Array),
        hkv: i32,
        scale: f32,
    ) -> Array {
        if let Some(failed) = self.arrays().find(|a| a.error().is_some()) {
            return failed.clone();
        }
        if !nuvec::row_exact() && nuvec::default_stream().device == nuvec::Device::gpu(0) {
            fn refs(side: &[Option<Array>; 3]) -> [&Array; 3] {
                side.each_ref()
                    .map(|p| p.as_ref().expect("six components written before attention"))
            }
            return fast::paged_quantized_attention(
                q,
                refs(&self.keys),
                refs(&self.values),
                &group.tables,
                device_lengths,
                scale,
            );
        }
        let rows: Vec<_> = lengths
            .iter()
            .enumerate()
            .map(|(r, &n)| {
                let blocks = Array::from(group.host_tables[r].clone());
                let gather = |side: &[Option<Array>; 3]| {
                    side.each_ref().map(|p| {
                        let p = p.as_ref().expect("all components written before attention");
                        p.take(&blocks, 0)
                            .transpose([1, 0, 2, 3])
                            .reshape([1, hkv, -1, p.shape()[3]])
                            .i((.., .., ..n as i32, ..))
                    })
                };
                quantized_attention(
                    &q.i(r as i32..r as i32 + 1),
                    (&gather(&self.keys), &gather(&self.values), FORMAT),
                    scale,
                    AttentionMask::None,
                    (hkv, None, None),
                )
            })
            .collect();
        ops::concatenate(&rows, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nuvec::{Device, Stream};

    #[test]
    fn fragmented_packed_pages_match_quantized_rows() -> Result<()> {
        for device in [Device::cpu(), Device::gpu(0)] {
            if device == Device::gpu(0) && !nuvec::metal::is_available() {
                continue;
            }
            nuvec::set_default_stream(Stream::default_for(device));
            for (d, g) in [(64, 1), (128, 2), (64, 4), (128, 8)] {
                for (dtype, tolerance) in
                    [(DType::F32, 2e-5), (DType::F16, 5e-3), (DType::BF16, 3e-2)]
                {
                    let sample = |n, phase: f32| {
                        (ops::arange(0, n, 1).astype(DType::F32) * 0.05 + phase)
                            .sin()
                            .astype(dtype)
                    };
                    let quant = |x: Array| FORMAT.encode_values(&x);
                    let k = quant(sample(2 * 7 * d, 0.1).reshape([2, 1, 7, d]));
                    let v = quant(sample(2 * 7 * d, 0.8).reshape([2, 1, 7, d]));
                    let q = sample(2 * g * d, 0.3).reshape([2, g, 1, d]);
                    let group = GroupStep {
                        slots: Array::from_slice(
                            &[8, 9, 10, 11, 0, 1, 2, 4, 5, 6, 7, 12, 13, 14],
                            [2, 7],
                        ),
                        tables: Array::from_slice(&[2, 0, 1, 3], [2, 2]),
                        host_tables: vec![vec![2, 0], vec![1, 3]],
                    };
                    let mut pages = Pages::default();
                    pages.write((4, 4), k.clone(), v.clone(), &group.slots);
                    let scale = (d as f32).sqrt().recip();
                    let got = pages.attention(
                        &q,
                        &group,
                        (&[7, 5], &Array::from([7, 5].to_vec())),
                        1,
                        scale,
                    );
                    let exact = nuvec::with_row_exact(|| {
                        pages.attention(
                            &q,
                            &group,
                            (&[7, 5], &Array::from([7, 5].to_vec())),
                            1,
                            scale,
                        )
                    });
                    for (r, n) in [(0, 7), (1, 5)] {
                        let row =
                            |side: &[Array; 3]| side.each_ref().map(|a| a.i((r..r + 1, .., ..n)));
                        let (kr, vr) = (row(&k), row(&v));
                        let qr = q.i(r..r + 1);
                        let dense = |[codes, scales, biases]: &[Array; 3]| {
                            nuvec::Quantized::affine(
                                codes.clone(),
                                scales.clone(),
                                biases.clone(),
                                64,
                                8,
                            )
                            .dequantize(dtype)
                        };
                        let reference = fast::scaled_dot_product_attention(
                            &qr,
                            &dense(&kr),
                            &dense(&vr),
                            scale,
                            &fast::AttentionOptions::default(),
                        );
                        let err = (&got.i(r..r + 1).astype(DType::F32)
                            - reference.astype(DType::F32))
                        .abs()
                        .max(..)
                        .item::<f32>()?;
                        assert!(
                            err <= tolerance,
                            "{device:?} {dtype:?} G{g} D{d} error {err}"
                        );
                        let reference = nuvec::with_row_exact(|| {
                            quantized_attention(
                                &qr,
                                (&kr, &vr, FORMAT),
                                scale,
                                AttentionMask::None,
                                (1, None, None),
                            )
                        });
                        assert_eq!(
                            exact.i(r..r + 1).astype(DType::F32).to_vec::<f32>()?,
                            reference.astype(DType::F32).to_vec::<f32>()?
                        );
                        for (pool, want) in pages.arrays().zip(kr.iter().chain(&vr)) {
                            let blocks = Array::from(group.host_tables[r as usize].clone());
                            let actual = pool
                                .take(&blocks, 0)
                                .reshape([1, 1, 8, pool.shape()[3]])
                                .i((.., .., ..n));
                            assert_eq!(
                                ops::view(&actual.contiguous(), DType::U8).to_vec::<u8>()?,
                                ops::view(&want.contiguous(), DType::U8).to_vec::<u8>()?
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
