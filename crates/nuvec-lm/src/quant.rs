//! Quantized checkpoint tensors in Hugging Face layouts, as mlx-lm 0.32.2 adapts them.

use std::collections::BTreeMap;

use nuvec::{Array, DType, Error, Quant, Result};

use crate::config::PackedFormat;

/// gpt-oss's MXFP4 experts (`*_blocks`, uint8 `[..., K / 32, 16]`: two E2M1 codes per
/// byte, sixteen bytes per group of 32) as MLX's packed codes (uint32 `[..., K / 8]`), as
/// mlx-lm's `gpt_oss.sanitize` (`v.view(mx.uint32).flatten(-2)`); the matching `*_scales`
/// are MLX's E8M0 `scales` unchanged. The bytes are reinterpreted, not copied, when the
/// blocks are row-contiguous.
///
/// ```
/// use nuvec::array;
///
/// let blocks = array![[[[1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]]]]; // [1, 1, 1, 16]
/// let codes = nuvec_lm::quant::mxfp4_from_hf_blocks(&blocks);
/// assert_eq!(codes.shape(), [1, 1, 4]);
/// assert_eq!(codes.to_vec::<u32>().unwrap(), [1, 2, 3, 4]);
/// ```
#[track_caller]
pub fn mxfp4_from_hf_blocks(blocks: &Array) -> Array {
    let codes = blocks.view(DType::U32);
    let n = codes.ndim();
    if n < 2 {
        return codes;
    }
    let mut shape: Vec<i32> = codes.shape()[..n - 2].to_vec();
    shape.push(codes.shape()[n - 2] * codes.shape()[n - 1]);
    codes.reshape(shape)
}

/// AWQ's order of the eight 4-bit values in a word: value `c` of a group of eight outputs
/// sits in nibble `AWQ_ORDER[c]` (mlx-lm's `_unpack_awq_weights` shifts).
const AWQ_ORDER: [u32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// The layers of an AWQ or GPTQ checkpoint (`{prefix}.qweight`, `.qzeros`, `.scales`, and
/// GPTQ's `.g_idx`) as MLX's affine layout (`{prefix}.weight` uint32 `[out, in · bits /
/// 32]`, `.scales` and `.biases` `[out, groups]` with `bias = -zero · scale`), and every other
/// float tensor in the scales' dtype, as mlx-lm's `_transform_awq_weights` leaves the
/// model. Without `qzeros` the zero is `2^(bits - 1)` (symmetric). Other tensors pass
/// through.
///
/// ```
/// use std::collections::BTreeMap;
/// use nuvec::{Array, DType, Quant};
/// use nuvec_lm::config::PackedFormat;
///
/// // GPTQ v2, 4-bit, 32 inputs (one group), one output: the codes 0..16, 0..16 down the
/// // inputs, zero 8, scale 0.5.
/// let words: Vec<i32> = (0..4).map(|r| (0..8).fold(0u32, |w, k| w | (((r * 8 + k) % 16) << (4 * k))) as i32).collect();
/// let mut w = BTreeMap::new();
/// w.insert("l.qweight".to_owned(), Array::from_slice(&words, [4, 1]));
/// w.insert("l.qzeros".to_owned(), Array::from_slice(&[8i32], [1, 1]));
/// w.insert("l.scales".to_owned(), Array::from_slice(&[0.5f32], [1, 1]).astype(DType::F16));
/// let w = nuvec_lm::quant::convert_packed(w, PackedFormat::Gptq { v1: false }, Quant::affine(32, 4)).unwrap();
/// let q = nuvec::Quantized::new(w["l.weight"].clone(), w["l.scales"].clone(), w.get("l.biases").cloned(), Quant::affine(32, 4));
/// let dense = q.dequantize(DType::F32).to_vec::<f32>().unwrap();
/// let want: Vec<f32> = (0..32).map(|i| ((i % 16) as f32 - 8.0) * 0.5).collect();
/// assert_eq!(dense, want);
/// ```
pub fn convert_packed(
    mut weights: BTreeMap<String, Array>,
    format: PackedFormat,
    quant: Quant,
) -> Result<BTreeMap<String, Array>> {
    let bits = quant.bits as u32;
    let group = quant.group_size as usize;
    let per_word = (32 / bits) as usize;
    let mask = (1u32 << bits) - 1;
    let prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".qweight").map(str::to_owned))
        .collect();
    let mut dtype = None;
    for prefix in prefixes {
        let take =
            |w: &mut BTreeMap<String, Array>, name: &str| w.remove(&format!("{prefix}.{name}"));
        let qweight = take(&mut weights, "qweight").expect("listed");
        let scales = take(&mut weights, "scales")
            .ok_or_else(|| Error::InvalidArgument(format!("{prefix}: qweight without scales")))?;
        let qzeros = take(&mut weights, "qzeros");
        let g_idx = take(&mut weights, "g_idx");
        let [rows, cols] = dims2(&prefix, &qweight)?;
        let (inputs, outputs) = match format {
            PackedFormat::Awq => (rows, cols * per_word),
            PackedFormat::Gptq { .. } => (rows * per_word, cols),
        };
        let groups = inputs.div_ceil(group);
        if inputs % group != 0 || scales.shape() != [groups as i32, outputs as i32] {
            return Err(Error::InvalidArgument(format!(
                "{prefix}: {inputs} inputs and scales {:?} do not match groups of {group}",
                scales.shape()
            )));
        }
        if let Some(g_idx) = &g_idx {
            let ids = g_idx.astype(DType::I64).to_vec::<i64>()?;
            if ids.len() != inputs
                || ids
                    .iter()
                    .enumerate()
                    .any(|(i, &g)| g != (i / group) as i64)
            {
                return Err(Error::InvalidArgument(format!(
                    "{prefix}: g_idx reorders the inputs (activation reordering), which MLX's \
                     grouping cannot express"
                )));
            }
        }
        let words: Vec<u32> = qweight
            .astype(DType::I32)
            .to_vec::<i32>()?
            .into_iter()
            .map(|w| w as u32)
            .collect();
        // The code of input `i`, output `o`.
        let code = |i: usize, o: usize| -> u32 {
            match format {
                PackedFormat::Awq => {
                    let w = words[i * cols + o / per_word];
                    (w >> (4 * AWQ_ORDER[o % per_word])) & mask
                }
                PackedFormat::Gptq { .. } => {
                    let w = words[(i / per_word) * cols + o];
                    (w >> (bits * (i % per_word) as u32)) & mask
                }
            }
        };
        let mut packed = vec![0u32; outputs * inputs / per_word];
        for o in 0..outputs {
            let row = &mut packed[o * inputs / per_word..(o + 1) * inputs / per_word];
            for (i, value) in (0..inputs).map(|i| (i, code(i, o))) {
                row[i / per_word] |= value << (bits * (i % per_word) as u32);
            }
        }
        // Zeros `[groups, outputs]`, as stored (AWQ, GPTQ v2) or stored plus one (GPTQ v1).
        let zeros: Vec<f32> = match &qzeros {
            None => vec![(1u32 << (bits - 1)) as f32; groups * outputs],
            Some(z) => {
                let [zr, zc] = dims2(&prefix, z)?;
                let zw: Vec<u32> = z
                    .astype(DType::I32)
                    .to_vec::<i32>()?
                    .into_iter()
                    .map(|w| w as u32)
                    .collect();
                if zr != groups || zc != outputs.div_ceil(per_word) {
                    return Err(Error::InvalidArgument(format!(
                        "{prefix}: qzeros {:?} for {groups} groups of {outputs} outputs",
                        z.shape()
                    )));
                }
                let offset = u32::from(matches!(format, PackedFormat::Gptq { v1: true }));
                (0..groups * outputs)
                    .map(|at| {
                        let (g, o) = (at / outputs, at % outputs);
                        let w = zw[g * zc + o / per_word];
                        let shift = match format {
                            PackedFormat::Awq => 4 * AWQ_ORDER[o % per_word],
                            PackedFormat::Gptq { .. } => bits * (o % per_word) as u32,
                        };
                        (((w >> shift) & mask) + offset) as f32
                    })
                    .collect()
            }
        };
        let scale_dtype = scales.dtype();
        dtype.get_or_insert(scale_dtype);
        let zeros = Array::from_slice(&zeros, [groups as i32, outputs as i32]);
        // As mlx-lm: `-zero` in float32 times the scales, in the scales' dtype; both
        // `[out, groups]`.
        let biases = zeros
            .negative()
            .multiply(scales.astype(DType::F32))
            .t()
            .astype(scale_dtype);
        let scales = scales.t().contiguous();
        weights.insert(
            format!("{prefix}.weight"),
            Array::from_slice(&packed, [outputs as i32, (inputs / per_word) as i32]),
        );
        weights.insert(format!("{prefix}.scales"), scales);
        weights.insert(format!("{prefix}.biases"), biases.contiguous());
    }
    if let Some(dtype) = dtype {
        for (name, w) in weights.iter_mut() {
            let quantized = name.ends_with(".scales") || name.ends_with(".biases");
            if !quantized && w.dtype().category() == nuvec::Category::Float && w.dtype() != dtype {
                *w = w.astype(dtype);
            }
        }
    }
    Ok(weights)
}

fn dims2(prefix: &str, a: &Array) -> Result<[usize; 2]> {
    match a.shape() {
        &[r, c] => Ok([r as usize, c as usize]),
        s => Err(Error::InvalidArgument(format!(
            "{prefix}: a 2-D packed tensor, got {s:?}"
        ))),
    }
}
