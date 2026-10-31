// GEMM on the GPU's matrix units through Metal 4 TensorOps (Metal Performance Primitives
// `matmul2d`): the Neural Accelerators on M5-class GPUs. Built for Metal 4.0 and macOS
// 26.2 into a second library (`nuvec_nax.metallib`), so the baseline keeps macOS 15.
//
// Unlike MLX's NAX kernels, which fill `matmul2d`'s cooperative tensors in registers with a
// hard-coded element layout (MLX's `BaseNAXFrag`, which is not the layout of earlier GPUs
// that also run TensorOps), the operands are tensor views of device memory that
// `matmul2d` loads itself, and the epilogue reads the result through each element's
// reported coordinates. So the kernel is correct on every GPU with TensorOps, whatever
// layout the implementation chooses.
//
// The arguments are those of the steel GEMM (`steel_gemm.h`): `D = A·B`, or
// `D = alpha·A·B + beta·C` (`addmm`), with batching. Function constants: 10 batched
// (shape and strides), 100 a C source, 110 alpha/beta; split-K products take neither.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct NaxGemmParams {
  int M, N, K;
  int lda, ldb, ldd;
  int tiles_n, tiles_m;
  int64_t batch_stride_a, batch_stride_b, batch_stride_d;
  int swizzle_log, gemm_k_iterations_aligned;
  int batch_ndim;
};

struct NaxAddMmParams {
  int ldc, fdc;
  int64_t batch_stride_c;
  float alpha, beta;
};

constant bool has_batch [[function_constant(10)]];
constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

// The offset of batch element `elem` for one set of strides (row-major batch shape).
static int64_t batch_offset(uint elem, constant const int* shape, constant const int64_t* strides, int ndim) {
  int64_t offset = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; i--) {
    offset += int64_t(elem % uint(shape[i])) * strides[i];
    elem /= uint(shape[i]);
  }
  return offset;
}

// One BM x BN output tile per threadgroup of SG SIMD groups. Full products write `D` (with
// the `addmm` epilogue and batching); split-K products (`SPLIT`) multiply the K range of
// partition `tid.z` into float32 partials at `D + tid.z * partition_stride`, which the
// baseline library's `nuvec_gemm_splitk_accum[_axpby]` kernels sum.
template <typename T, bool TA, bool TB, int BM, int BN, int SG, bool SPLIT>
[[kernel, max_total_threads_per_threadgroup(SG * 32)]] void nax_gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device void* out [[buffer(3)]],
    const constant NaxGemmParams& p [[buffer(4)]],
    const constant NaxAddMmParams& addmm [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    const constant int2& split [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  const int m0 = int(tid.y) * BM, n0 = int(tid.x) * BN;
  if (m0 >= p.M || n0 >= p.N) {
    return;
  }
  int64_t off_c = 0;
  int k0 = 0, k = p.K;
  if (SPLIT) {
    // split = (partition size, partition stride)
    k0 = int(tid.z) * split.x;
    k = min(split.x, p.K - k0);
  } else if (has_batch) {
    A += batch_offset(tid.z, batch_shape, batch_strides, p.batch_ndim);
    B += batch_offset(tid.z, batch_shape, batch_strides + p.batch_ndim, p.batch_ndim);
    if (use_out_source) {
      off_c = batch_offset(tid.z, batch_shape, batch_strides + 2 * p.batch_ndim, p.batch_ndim);
    }
  } else {
    A += p.batch_stride_a * tid.z;
    B += p.batch_stride_b * tid.z;
    if (use_out_source) {
      off_c = addmm.batch_stride_c * tid.z;
    }
  }
  if (k <= 0) {
    return;
  }
  // The K range starts at k0: along the innermost dimension of a row-major A (and of a
  // transposed B), else one leading dimension per step.
  A += TA ? int64_t(k0) * p.lda : int64_t(k0);
  B += TB ? int64_t(k0) : int64_t(k0) * p.ldb;

  // Tensor views (innermost dimension first): A is M x K and B is K x N, each stored
  // row-major or transposed (TA, TB), as the descriptor's transpose flags say.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent), TA, TB, true);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<SG>> op;
  using View = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
  View tA = TA ? View((device T*)A, dextents<int32_t, 2>(p.M, k), array<int32_t, 2>{1, p.lda})
               : View((device T*)A, dextents<int32_t, 2>(k, p.M), array<int32_t, 2>{1, p.lda});
  View tB = TB ? View((device T*)B, dextents<int32_t, 2>(k, p.N), array<int32_t, 2>{1, p.ldb})
               : View((device T*)B, dextents<int32_t, 2>(p.N, k), array<int32_t, 2>{1, p.ldb});
  auto sA = TA ? tA.slice(m0, 0) : tA.slice(0, m0);
  auto sB = TB ? tB.slice(0, n0) : tB.slice(n0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();
  _Pragma("clang loop unroll(full)")
  for (ushort i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      acc[i] = 0;
    }
  }
  op.run(sA, sB, acc);

  device T* D = (device T*)out + (SPLIT ? 0 : p.batch_stride_d * tid.z);
  device float* P = (device float*)out + (SPLIT ? int64_t(tid.z) * split.y : 0);
  _Pragma("clang loop unroll(full)")
  for (ushort i = 0; i < acc.get_capacity(); i++) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    auto ix = acc.get_multidimensional_index(i);
    const int n = n0 + ix[0], m = m0 + ix[1];
    if (m >= p.M || n >= p.N) {
      continue;
    }
    float v = acc[i];
    if (SPLIT) {
      P[int64_t(m) * p.ldd + n] = v;
      continue;
    }
    if (use_out_source) {
      const float c = float(C[off_c + int64_t(m) * addmm.ldc + int64_t(n) * addmm.fdc]);
      v = do_axpby ? addmm.alpha * v + addmm.beta * c : v + c;
    }
    D[int64_t(m) * p.ldd + n] = T(v);
  }
}

// Names: `nuvec_nax_gemm_{ab}_{type}_bm{BM}_bn{BN}` (full products: 64 x 64 tiles, and
// 16- or 32-row (or column) tiles for thin products) and `nuvec_nax_gemm_splitk_{ab}_{type}
// _bm{BM}_bn{BN}` (float32 partials).
#define NAX_GEMM(tname, T, ta, tb, sa, sb, bm, bn, sg)                                     \
  template [[host_name("nuvec_nax_gemm_" #sa #sb "_" #tname "_bm" #bm "_bn" #bn)]]         \
  [[kernel]] decltype(nax_gemm<T, ta, tb, bm, bn, sg, false>)                               \
      nax_gemm<T, ta, tb, bm, bn, sg, false>;                                               \
  template [[host_name("nuvec_nax_gemm_splitk_" #sa #sb "_" #tname "_bm" #bm "_bn" #bn)]]  \
  [[kernel]] decltype(nax_gemm<T, ta, tb, bm, bn, sg, true>)                                \
      nax_gemm<T, ta, tb, bm, bn, sg, true>;

#define NAX_GEMM_TILES(tname, T, ta, tb, sa, sb)    \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 64, 64, 4)     \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 32, 64, 2)     \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 16, 64, 1)     \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 64, 32, 2)     \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 64, 16, 1)     \
  NAX_GEMM(tname, T, ta, tb, sa, sb, 32, 32, 1)

#define NAX_GEMM_ALL(tname, T)                         \
  NAX_GEMM_TILES(tname, T, false, false, n, n)         \
  NAX_GEMM_TILES(tname, T, false, true, n, t)          \
  NAX_GEMM_TILES(tname, T, true, false, t, n)          \
  NAX_GEMM_TILES(tname, T, true, true, t, t)

NAX_GEMM_ALL(float16, half)
NAX_GEMM_ALL(bfloat16, bfloat)
NAX_GEMM_ALL(float32, float)
