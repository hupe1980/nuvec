// SPDX-License-Identifier: MIT
//
// MLX's segmented GEMM (`segmented_mm`: one product per segment of the inner axis),
// ported from MLX v0.32.4 (mlx/backend/metal/kernels/steel/gemm/kernels/
// steel_gemm_segmented.h) on top of steel_gemm.h. One change: the segments are clamped to
// [0, K], since they are array values. Compiled with contraction, as MLX's GEMMs.
//
// Copyright © 2025 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "steel_gemm.h"

typedef bfloat bfloat16_t;

constant bool segments_contiguous [[function_constant(199)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void nuvec_segmented_mm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* segments [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant mlx::steel::GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using namespace mlx::steel;
  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Find the block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  // Prepare threadgroup bounds
  const short tgp_bm = mlx::steel::align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = mlx::steel::align_N ? BN : short(min(BN, params->N - c_col));

  // Move the pointers to the output tile
  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  // Move the pointers to the start of the segment
  uint32_t k_start, k_end;
  if (segments_contiguous) {
    k_start = segments[2 * tid.z];
    k_end = segments[2 * tid.z + 1];
  } else {
    // We accept either contiguous (above) or weird strides where the beginning
    // of the next one is the previous one. Basically the last two strides are
    // both 1!
    k_start = segments[tid.z];
    k_end = segments[tid.z + 1];
  }
  // Segments outside [0, K] are clamped (nuvec): the bounds are array values.
  k_end = min(k_end, uint32_t(params->K));
  k_start = min(k_start, k_end);
  A += transpose_a ? int64_t(k_start) * params->lda : k_start;
  B += transpose_b ? k_start : int64_t(k_start) * params->ldb;
  C += tid.z * params->batch_stride_d;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

  // Matrix level alignment so only check K
  if (mlx::steel::align_M && mlx::steel::align_N) {
    uint32_t k = k_start + BK;
    for (; k <= k_end; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }
    short k_remain = BK - short(k - k_end);
    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
    if (k_remain > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(tile_dims_A);
      loader_b.load_safe(tile_dims_B);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
    }
    mma_op.store_result(C, params->ldd);
  } else {
    // Tile aligned do the same as above
    if ((mlx::steel::align_M || tgp_bm == BM) && (mlx::steel::align_N || tgp_bn == BN)) {
      uint32_t k = k_start + BK;
      for (; k <= k_end; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }
      short k_remain = BK - short(k - k_end);
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
      if (k_remain > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
      }
      mma_op.store_result(C, params->ldd);
    }

    // Tile partially aligned check rows
    else if (mlx::steel::align_N || tgp_bn == BN) {
      uint32_t k = k_start + BK;
      for (; k <= k_end; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load elements into threadgroup
        loader_a.load_safe(
            transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm));
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }
      short k_remain = BK - short(k - k_end);
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
      if (k_remain > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
      }
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Tile partially aligned check cols
    else if (mlx::steel::align_M || tgp_bm == BM) {
      uint32_t k = k_start + BK;
      for (; k <= k_end; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_safe(
            transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }
      short k_remain = BK - short(k - k_end);
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
      if (k_remain > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
      }
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Nothing aligned so check both rows and cols
    else {
      uint32_t k = k_start + BK;
      for (; k <= k_end; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load elements into threadgroup
        loader_a.load_safe(
            transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm));
        loader_b.load_safe(
            transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }
      short k_remain = BK - short(k - k_end);
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
      if (k_remain > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
      }
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}

#define NUVEC_SEGMENTED(tname, ta, tb, iname, T, bm, bn, bk, wm, wn)                               \
  template [[host_name("nuvec_segmented_mm_" #tname "_" #iname "_bm" #bm "_bn" #bn "_bk" #bk    \
                       "_wm" #wm "_wn" #wn)]] [[kernel]] decltype(nuvec_segmented_mm<T, bm, bn, bk, \
                       wm, wn, ta, tb, float>) nuvec_segmented_mm<T, bm, bn, bk, wm, wn, ta, tb, float>;

#define NUVEC_SEGMENTED_T(iname, T, bm, bn, bk, wm, wn)            \
  NUVEC_SEGMENTED(nn, false, false, iname, T, bm, bn, bk, wm, wn)  \
  NUVEC_SEGMENTED(nt, false, true, iname, T, bm, bn, bk, wm, wn)   \
  NUVEC_SEGMENTED(tn, true, false, iname, T, bm, bn, bk, wm, wn)   \
  NUVEC_SEGMENTED(tt, true, true, iname, T, bm, bn, bk, wm, wn)

#define NUVEC_SEGMENTED_SHAPES(iname, T)              \
  NUVEC_SEGMENTED_T(iname, T, 64, 64, 16, 2, 2)       \
  NUVEC_SEGMENTED_T(iname, T, 64, 64, 16, 1, 2)       \
  NUVEC_SEGMENTED_T(iname, T, 64, 32, 32, 2, 2)       \
  NUVEC_SEGMENTED_T(iname, T, 32, 64, 16, 1, 2)       \
  NUVEC_SEGMENTED_T(iname, T, 32, 32, 16, 2, 2)

NUVEC_SEGMENTED_SHAPES(float16, half)
NUVEC_SEGMENTED_SHAPES(bfloat16, bfloat16_t)
NUVEC_SEGMENTED_SHAPES(float32, float)
