/// Kernels compiled with contraction (MLX's default): they have no bit-exact CPU
/// counterpart, and fused multiply-adds speed up their dot products (attention decode by
/// about 15%, 4-bit prefill products by about 3%). Exact kernels in these files (affine
/// `quantize`/`dequantize`) turn it off with `#pragma clang fp contract(off)`.
const CONTRACTED: &[&str] = &[
    "sdpa.metal",
    "paged_attention.metal",
    "quantized.metal",
    "gemm.metal",
    "gemv.metal",
    "gather_gemm.metal",
];
