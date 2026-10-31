// Affine quantized products `y = x · dequantize(w)ᵀ` (MLX's `qmm_t`) on the GPU's matrix
// units through Metal 4 TensorOps (`matmul2d`): per 64-wide step along K, the threadgroup
// dequantizes a 64 x 64 weight tile into threadgroup memory, and `matmul2d` multiplies the
// activations (device memory) by it, accumulating in float32. As in `gemm.metal`, the
// operands are tensor views and the epilogue reads each element's reported coordinates,
// so the kernel is correct whatever cooperative-tensor layout the GPU uses.
//
// Bits 2, 4 and 8 (32 values are whole 32-bit words); K a multiple of 64; group sizes 32,
// 64 and 128. Arguments as the SIMD `qmm_t`: w, scales, biases, x, y, K, N, M, with
// optional batch strides (function constant 10); the sorted gather's as the SIMD
// `gather_qmm_rhs`.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

constant bool batched [[function_constant(10)]];

constexpr constant int BM = 64, BN = 64, BK = 64;

// One 64 x 64 output tile: rows `m0..m0 + rows` of `x` (row-major, K columns) times the
// dequantized rows `n0..` of `w`, into `y` (row-major, N columns).
template <typename T, int GS, int BITS>
METAL_FUNC void qmm_t_tile(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    int K,
    int N,
    int rows,
    int n0,
    threadgroup T* ws,
    ushort lid) {
  constexpr int WORDS = 32 * BITS / 32; // 32-bit words per 32 values
  constexpr uint32_t MASK = (1u << BITS) - 1;
  const int row_words = K * BITS / 32;
  const int groups = K / GS;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM, BN, BK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
  using Device = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
  using Local = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
  Device tx((device T*)x, dextents<int32_t, 2>(K, rows), array<int32_t, 2>{1, K});
  Local tw(ws, dextents<int32_t, 2>(BK, BN), array<int32_t, 2>{1, BK});
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (ushort i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      acc[i] = 0;
    }
  }

  // Each thread dequantizes 32 values: row `lid / 2`, half `lid % 2` of the tile's K.
  const int r = lid / 2, half_k = (lid % 2) * 32;
  const int n = n0 + r;
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup T* dst = ws + r * BK + half_k;
    if (n < N) {
      const int k = k0 + half_k;
      const float scale = float(scales[int64_t(n) * groups + k / GS]);
      const float bias = float(biases[int64_t(n) * groups + k / GS]);
      const device uint32_t* src = w + int64_t(n) * row_words + k * BITS / 32;
      for (int word = 0; word < WORDS; word++) {
        uint32_t bits = src[word];
        for (int j = 0; j < 32 / BITS; j++) {
          dst[word * (32 / BITS) + j] = T(fma(scale, float(bits & MASK), bias));
          bits >>= BITS;
        }
      }
    } else {
      for (int j = 0; j < 32; j++) {
        dst[j] = T(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sx = tx.slice(k0, 0);
    op.run(sx, tw, acc);
  }

  for (ushort i = 0; i < acc.get_capacity(); i++) {
    if (!acc.is_valid_element(i)) {
      continue;
    }
    auto ix = acc.get_multidimensional_index(i);
    const int col = n0 + ix[0], row = ix[1];
    if (row < rows && col < N) {
      y[int64_t(row) * N + col] = T(acc[i]);
    }
  }
}

template <typename T, int GS, int BITS>
[[kernel, max_total_threads_per_threadgroup(128)]] void nax_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int64_t* strides [[buffer(8), function_constant(batched)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort lid [[thread_index_in_threadgroup]]) {
  const int m0 = int(tid.y) * BM, n0 = int(tid.x) * BN;
  if (m0 >= M || n0 >= N) {
    return;
  }
  if (batched) {
    // Per batch item: x, w, scales, biases and y strides.
    x += strides[0] * tid.z;
    w += strides[1] * tid.z;
    scales += strides[2] * tid.z;
    biases += strides[3] * tid.z;
    y += strides[4] * tid.z;
  }
  threadgroup T ws[BN * BK];
  qmm_t_tile<T, GS, BITS>(
      w, scales, biases, x + int64_t(m0) * K, y + int64_t(m0) * N, K, N, min(BM, M - m0),
      n0, ws, lid);
}

// MLX's `schedule_row_tile` (quantized.h): the row tile `tile` of the experts' runs of
// sorted rows (`offsets[g]` the first row of expert `g`): its first row, expert and rows.
METAL_FUNC bool schedule_row_tile(
    const device int32_t* offsets,
    const int num_groups,
    const int M,
    const int tile,
    const uint simd_lane_id,
    thread int& row,
    thread int& group,
    thread int& rows) {
  int tiles_before = 0;
  for (int e = 0; e < num_groups; e += 32) {
    const int g = e + simd_lane_id;
    int start = M;
    int end = M;
    if (g < num_groups) {
      start = offsets[g];
      end = g + 1 < num_groups ? offsets[g + 1] : M;
    }
    const int n = (end - start + BM - 1) / BM;
    const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n);
    const ushort owner_lane = ushort(simd_sum(int(tiles_through <= tile)));
    if (owner_lane < 32) {
      group = e + owner_lane;
      row = simd_shuffle(start, owner_lane) +
          (tile - simd_shuffle(tiles_through - n, owner_lane)) * BM;
      rows = min(BM, simd_shuffle(end, owner_lane) - row);
      return true;
    }
    tiles_before = simd_shuffle(tiles_through, 31);
  }
  return false;
}

// The sorted mixture-of-experts gather (MLX's `gather_qmm_rhs`, transposed weights): rows
// of `x` (`[M, K]`) sorted by expert, `offsets` the first row of each; each threadgroup a
// 64 x 64 tile within one expert's rows, times that expert's weights.
template <typename T, int GS, int BITS>
[[kernel, max_total_threads_per_threadgroup(128)]] void nax_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& num_groups [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    ushort lid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  int row, group, rows;
  if (!schedule_row_tile(offsets, num_groups, M, int(tid.y), simd_lane_id, row, group, rows)) {
    return;
  }
  const int n0 = int(tid.x) * BN;
  if (n0 >= N || rows <= 0) {
    return;
  }
  const int64_t words = int64_t(N) * (K * BITS / 32), scale_count = int64_t(N) * (K / GS);
  threadgroup T ws[BN * BK];
  qmm_t_tile<T, GS, BITS>(
      w + group * words, scales + group * scale_count, biases + group * scale_count,
      x + int64_t(row) * K, y + int64_t(row) * N, K, N, rows, n0, ws, lid);
}

#define NAX_QMM(tname, T, gs, b)                                                        \
  template [[host_name("nuvec_nax_qmm_t_" #tname "_gs" #gs "_b" #b)]] [[kernel]]       \
  decltype(nax_qmm_t<T, gs, b>) nax_qmm_t<T, gs, b>;                                    \
  template [[host_name("nuvec_nax_gather_qmm_rhs_" #tname "_gs" #gs "_b" #b)]]         \
  [[kernel]] decltype(nax_gather_qmm_rhs<T, gs, b>) nax_gather_qmm_rhs<T, gs, b>;

#define NAX_QMM_BITS(tname, T, gs) \
  NAX_QMM(tname, T, gs, 2)         \
  NAX_QMM(tname, T, gs, 4)         \
  NAX_QMM(tname, T, gs, 8)

#define NAX_QMM_ALL(tname, T)   \
  NAX_QMM_BITS(tname, T, 32)    \
  NAX_QMM_BITS(tname, T, 64)    \
  NAX_QMM_BITS(tname, T, 128)

NAX_QMM_ALL(float16, half)
NAX_QMM_ALL(bfloat16, bfloat)
NAX_QMM_ALL(float32, float)
