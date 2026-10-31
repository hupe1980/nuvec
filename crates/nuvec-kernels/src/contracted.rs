/// Kernels compiled with contraction (MLX's default): they have no bit-exact CPU
/// counterpart, and fused multiply-adds speed up their dot products (attention decode by
/// about 15%, 4-bit prefill products by about 3%). Exact kernels in these files (affine
/// `quantize`/`dequantize`) turn it off with `#pragma clang fp contract(off)`; the
/// floating-point modes' quantize and dequantize have no multiply-add to contract, and are
/// compiled as MLX compiles them.
const CONTRACTED: &[&str] = &[
    "sdpa.metal",
    "paged_attention.metal",
    "quantized.metal",
    "fp_quantized.metal",
    "gemm.metal",
    "gemv.metal",
    "gather_gemm.metal",
];
