/// Kernels compiled with contraction (MLX's default), so their bits are MLX's: products
/// and attention (fused multiply-adds also speed up attention decode by about 15%, 4-bit
/// prefill products by about 3%), and the normalisation and rotation kernels, where MLX's
/// contracted `x1 · cos − x2 · sin` and sums of squares differ from separate roundings. Exact kernels in these files (affine
/// `quantize`/`dequantize`) turn it off with `#pragma clang fp contract(off)`; the
/// floating-point modes' quantize and dequantize have no multiply-add to contract, and are
/// compiled as MLX compiles them.
const CONTRACTED: &[&str] = &[
    "sdpa.metal",
    "paged_attention.metal",
    "sdpa_quantized.metal",
    "quantized.metal",
    "fp_quantized.metal",
    "gemm.metal",
    "gemv.metal",
    "gather_gemm.metal",
    "steel_conv.metal",
    "steel_segmented.metal",
    "steel_masked.metal",
    "rope.metal",
    "rms_norm.metal",
    "layer_norm.metal",
    "conv.metal",
];
