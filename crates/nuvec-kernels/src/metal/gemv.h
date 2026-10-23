// SPDX-License-Identifier: MIT
//
// MLX's matrix-vector kernels (`gemv`: rows of a row-major matrix times a vector; `gemv_t`:
// a vector times a row-major matrix), with the batch and axpby (`addmm`) variants, ported
// from MLX v0.32.4 (mlx/backend/metal/kernels/gemv.h) without the complex, gather and wide
// kernels. Placed in `mlx::matvec`; needs steel_gemm.h first (elem_to_loc, bfloat16_t).
//
// Copyright © 2023-2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#pragma once

// From MLX's kernels/defines.h.
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// Copyright © 2023-2024 Apple Inc.


#include <metal_simdgroup>
#include <metal_stdlib>



using namespace metal;

namespace mlx {
namespace matvec {

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

template <typename U>
struct DefaultAccT {
  using type = float;
};

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  static_assert(
      SN == 4 || SN == 8 || SN == 16 || SN == 32,
      "gemv block must have a width of 4, 8, 16, or 32");

  // - The matrix of size (M = out_vec_size, K = in_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated blockM outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void
  load_unsafe(const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else { // Edgecase
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& matrix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread AccT result[TM] = {0};
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position
    int out_row = tid.x * blockM + bm;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += size_t(out_row) * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Loop over in_vec in blocks of blockN
    for (int i = 0; i < n_iter; ++i) {
      load_unsafe<AccT>(in_vec, v_coeff, bn);

      // Per thread work loop
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_unsafe(mat, inter, mat_offset + bn);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }

        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    if (leftover > 0) {
      load_safe<AccT>(in_vec, v_coeff, bn, in_size);

      // Per thread work loop
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // Write outputs
    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        if (kDoAxpby) {
          out_vec[out_row + tm] =
              static_cast<T>(alpha) * static_cast<T>(result[tm]) +
              static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
        } else {
          out_vec[out_row + tm] = static_cast<T>(result[tm]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVTKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //    and the corresponding scalar from the vector
  // 2. The thread then accumulates its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM*(blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& marix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];
    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    constexpr const uniform<int> loop_stride = make_uniform(blockM);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Edgecase handling
    if (out_col < out_vec_size) {
      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      for (int i = 0; i < n_iter; ++i) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          auto vc = static_cast<AccT>(v_coeff[tm]);
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[size_t(bm + tm) * marix_ld + out_col + tn];
          }
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc * inter[tn];
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[size_t(bm + tm) * marix_ld + out_col + tn];
          }

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += v_coeff[tm] * inter[tn];
          }
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    // Threadgroup accumulation and writing out results
    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] =
              static_cast<T>(alpha) * static_cast<T>(result[j]) +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = static_cast<T>(result[j]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}


template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}


///////////////////////////////////////////////////////////////////////////////
/// Multi-row matrix vector multiplication (nuvec)
///////////////////////////////////////////////////////////////////////////////

// `gemv` and `gemv_t` for up to R input vectors (rows of a matrix X) at once: each matrix
// element is loaded once and used for every row, and each row's arithmetic (block shape,
// per-thread accumulation order, SIMD-group and threadgroup reductions) is exactly the
// one-row kernel's, so row r of the result has the bits of the one-row product of row r
// (nuvec's row-independence contract; MLX's `gemv_wide` reads the matrix once in the same
// way). Rows past `n_rows` are skipped.

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const int R, /* Rows (input vectors) at most */
    const bool kDoAxpby,
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVRowsKernel {
  using one = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby, AccT>;
  MLX_MTL_CONST int blockM = one::blockM;
  MLX_MTL_CONST int blockN = one::blockN;
  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? R * BN * (blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const constant int& in_vec_size,
      const constant int& out_vec_size,
      const constant int& matrix_ld,
      const constant float& alpha,
      const constant float& beta,
      const constant int& bias_stride,
      const int n_rows,
      const int64_t vec_row_stride,
      const int bias_row_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint simd_gid,
      uint simd_lid) {
    thread AccT result[R][TM];
    thread T inter[TN];
    thread AccT v_coeff[R][TN];
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < R; r++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        result[r][tm] = 0;
      }
    }

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;
    int out_row = tid.x * blockM + bm;
    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    if (out_row >= out_vec_size)
      return;
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
    mat += size_t(out_row) * matrix_ld;

    for (int i = 0; i < n_iter; ++i) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < R; r++) {
        if (r < n_rows) {
          one::template load_unsafe<AccT>(in_vec + r * vec_row_stride, v_coeff[r], bn);
        }
      }
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        one::load_unsafe(mat, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < R; r++) {
          if (r < n_rows) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[r][tm] += inter[tn] * v_coeff[r][tn];
            }
          }
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }

    if (leftover > 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < R; r++) {
        if (r < n_rows) {
          one::template load_safe<AccT>(
              in_vec + r * vec_row_stride, v_coeff[r], bn, in_size);
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        one::load_safe(&mat[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < R; r++) {
          if (r < n_rows) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[r][tm] += inter[tn] * v_coeff[r][tn];
            }
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < R; r++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
          result[r][tm] += simd_shuffle_down(result[r][tm], sn);
        }
      }
    }

    if (needs_tgp_reduction) {
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < R; r++) {
          threadgroup AccT* tgp_results =
              tgp_memory + r * BN * (blockM + TM) + sgN * (blockM + TM) + bm;
          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            tgp_results[tm] = result[r][tm];
          }
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int r = 0; r < R; r++) {
            threadgroup AccT* tgp_results = tgp_memory + r * BN * (blockM + TM) + bm;
            MLX_MTL_PRAGMA_UNROLL
            for (int sgn = 1; sgn < BN; sgn++) {
              MLX_MTL_PRAGMA_UNROLL
              for (int tm = 0; tm < TM; tm++) {
                result[r][tm] += tgp_results[sgn * (blockM + TM) + tm];
              }
            }
          }
        }
      }
    }

    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < R; r++) {
        if (r < n_rows) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            if (kDoAxpby) {
              out_vec[r * out_vec_size + out_row + tm] =
                  static_cast<T>(alpha) * static_cast<T>(result[r][tm]) +
                  static_cast<T>(beta) *
                      bias[r * bias_row_stride + (out_row + tm) * bias_stride];
            } else {
              out_vec[r * out_vec_size + out_row + tm] = static_cast<T>(result[r][tm]);
            }
          }
        }
      }
    }
  }
};

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const int R,
    const bool kDoAxpby,
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVTRowsKernel {
  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;
  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;
  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? R * BM * (blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const constant int& in_vec_size,
      const constant int& out_vec_size,
      const constant int& marix_ld,
      const constant float& alpha,
      const constant float& beta,
      const constant int& bias_stride,
      const int n_rows,
      const int64_t vec_row_stride,
      const int bias_row_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint simd_gid,
      uint simd_lid) {
    AccT result[R][TN];
    T inter[TN];
    AccT v_coeff[R][TM];
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < R; r++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        result[r][tn] = 0;
      }
    }
    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = SM * sgM;
    const int simdN = SN * sgN;
    int cm = (simdM + thrM);
    int cn = (simdN + thrN);
    int bm = cm * TM;
    int bn = cn * TN;
    int out_col = tid.x * blockN + bn;

    constexpr const uniform<int> loop_stride = make_uniform(blockM);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    if (out_col < out_vec_size) {
      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;
      for (int i = 0; i < n_iter; ++i) {
        threadgroup_barrier(mem_flags::mem_none);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < R; r++) {
          if (r < n_rows) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              v_coeff[r][tm] = static_cast<AccT>(in_vec[r * vec_row_stride + bm + tm]);
            }
          }
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[size_t(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int r = 0; r < R; r++) {
            if (r < n_rows) {
              auto vc = static_cast<AccT>(v_coeff[r][tm]);
              for (int tn = 0; tn < TN; tn++) {
                result[r][tn] += vc * inter[tn];
              }
            }
          }
        }
        bm += blockM;
      }
      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[size_t(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int r = 0; r < R; r++) {
            if (r < n_rows) {
              v_coeff[r][tm] = static_cast<AccT>(in_vec[r * vec_row_stride + bm + tm]);
              MLX_MTL_PRAGMA_UNROLL
              for (int tn = 0; tn < TN; tn++) {
                result[r][tn] += v_coeff[r][tm] * inter[tn];
              }
            }
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < R; r++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
          result[r][tn] += simd_shuffle_down(result[r][tn], SN * sm);
        }
      }
    }

    if (needs_tgp_reduction) {
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < R; r++) {
          threadgroup AccT* tgp_results =
              tgp_memory + r * BM * (blockN + TN) + sgM * (blockN + TN) + bn;
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            tgp_results[tn] = result[r][tn];
          }
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int r = 0; r < R; r++) {
            threadgroup AccT* tgp_results = tgp_memory + r * BM * (blockN + TN) + bn;
            MLX_MTL_PRAGMA_UNROLL
            for (int sgm = 1; sgm < BM; sgm++) {
              MLX_MTL_PRAGMA_UNROLL
              for (int tn = 0; tn < TN; tn++) {
                result[r][tn] += tgp_results[sgm * (blockN + TN) + tn];
              }
            }
          }
        }
      }
    }

    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < R; r++) {
        if (r < n_rows) {
          MLX_MTL_PRAGMA_UNROLL
          for (int j = 0; j < TN; j++) {
            if (kDoAxpby) {
              out_vec[r * out_vec_size + out_col + j] =
                  static_cast<T>(alpha) * static_cast<T>(result[r][j]) +
                  static_cast<T>(beta) *
                      bias[r * bias_row_stride + (out_col + j) * bias_stride];
            } else {
              out_vec[r * out_vec_size + out_col + j] = static_cast<T>(result[r][j]);
            }
          }
        }
      }
    }
  }
};

#define NUVEC_GEMV_ROWS_KERNEL(fname, Kernel)                                          \
  template <                                                                          \
      typename T,                                                                     \
      const int BM,                                                                   \
      const int BN,                                                                   \
      const int SM,                                                                   \
      const int SN,                                                                   \
      const int TM,                                                                   \
      const int TN,                                                                   \
      const int R,                                                                    \
      const bool kDoNCBatch,                                                          \
      const bool kDoAxpby>                                                            \
  [[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void fname(             \
      const device T* mat [[buffer(0)]],                                              \
      const device T* in_vec [[buffer(1)]],                                           \
      const device T* bias [[buffer(2)]],                                             \
      device T* out_vec [[buffer(3)]],                                                \
      const constant int& in_vec_size [[buffer(4)]],                                  \
      const constant int& out_vec_size [[buffer(5)]],                                 \
      const constant int& marix_ld [[buffer(6)]],                                     \
      const constant float& alpha [[buffer(7)]],                                      \
      const constant float& beta [[buffer(8)]],                                       \
      const constant int& batch_ndim [[buffer(9)]],                                   \
      const constant int* batch_shape [[buffer(10)]],                                 \
      const constant int64_t* vector_batch_stride [[buffer(11)]],                     \
      const constant int64_t* matrix_batch_stride [[buffer(12)]],                     \
      const constant int64_t* bias_batch_stride [[buffer(13)]],                       \
      const constant int& bias_stride [[buffer(14)]],                                 \
      const constant int& n_rows [[buffer(15)]],                                      \
      const constant int64_t& vec_row_stride [[buffer(16)]],                          \
      const constant int& bias_row_stride [[buffer(17)]],                             \
      const constant int& row_groups [[buffer(18)]],                                  \
      uint3 group_id [[threadgroup_position_in_grid]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                                  \
    using rows_k = Kernel<T, BM, BN, SM, SN, TM, TN, R, kDoAxpby>;                      \
    /* Groups of R rows of one output tile are neighbours in the grid, so they read */  \
    /* the same matrix block at about the same time (from the cache after the first). */ \
    const int row_group = group_id.x % row_groups;                                    \
    const uint3 tid = uint3(group_id.x / row_groups, group_id.y, group_id.z);         \
    const int rows_here = min(R, n_rows - row_group * R);                             \
    threadgroup float tgp_memory[rows_k::tgp_mem_size == 0 ? 1 : rows_k::tgp_mem_size]; \
    if (kDoNCBatch) {                                                                 \
      in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);     \
      mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);        \
      if (kDoAxpby) {                                                                 \
        bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);       \
      }                                                                               \
    } else {                                                                          \
      in_vec += tid.z * vector_batch_stride[0];                                       \
      mat += tid.z * matrix_batch_stride[0];                                          \
      if (kDoAxpby) {                                                                 \
        bias += tid.z * bias_batch_stride[0];                                         \
      }                                                                               \
    }                                                                                 \
    out_vec += tid.z * n_rows * out_vec_size + row_group * R * out_vec_size;           \
    in_vec += row_group * R * vec_row_stride;                                         \
    if (kDoAxpby) {                                                                   \
      bias += row_group * R * bias_row_stride;                                        \
    }                                                                                 \
    rows_k::run(                                                                      \
        mat,                                                                          \
        in_vec,                                                                       \
        bias,                                                                         \
        out_vec,                                                                      \
        in_vec_size,                                                                  \
        out_vec_size,                                                                 \
        marix_ld,                                                                     \
        alpha,                                                                        \
        beta,                                                                         \
        bias_stride,                                                                  \
        rows_here,                                                                    \
        vec_row_stride,                                                               \
        bias_row_stride,                                                              \
        rows_k::tgp_mem_size == 0 ? nullptr : tgp_memory,                             \
        tid,                                                                          \
        simd_gid,                                                                     \
        simd_lid);                                                                    \
  }

NUVEC_GEMV_ROWS_KERNEL(gemv_rows, GEMVRowsKernel)
NUVEC_GEMV_ROWS_KERNEL(gemv_t_rows, GEMVTRowsKernel)


///////////////////////////////////////////////////////////////////////////////
/// Gathered matrix vector multiplication (MLX's `gemv_gather`)
///////////////////////////////////////////////////////////////////////////////

// Output batch `i` is the vector `in_vec[lhs_indices[i]]` times the matrix
// `mat[rhs_indices[i]]`, each batch computed by the one-row kernel, so every row of a
// gathered (mixture-of-experts) product has the bits of the same row computed alone.
template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const bool kTransposed>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv_gather(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& marix_ld [[buffer(5)]],
    const constant int& batch_ndim [[buffer(6)]],
    const constant int* batch_shape [[buffer(7)]],
    const constant int64_t* lhs_strides [[buffer(8)]],
    const constant int64_t* rhs_strides [[buffer(9)]],
    const constant int& vec_ndim [[buffer(10)]],
    const constant int* vec_shape [[buffer(11)]],
    const constant int64_t* vec_strides [[buffer(12)]],
    const constant int& mat_ndim [[buffer(13)]],
    const constant int* mat_shape [[buffer(14)]],
    const constant int64_t* mat_strides [[buffer(15)]],
    const device uint32_t* lhs_indices [[buffer(16)]],
    const device uint32_t* rhs_indices [[buffer(17)]],
    const constant float& alpha [[buffer(18)]],
    const constant float& beta [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = metal::conditional_t<
      kTransposed,
      GEMVTKernel<T, BM, BN, SM, SN, TM, TN, false>,
      GEMVKernel<T, BM, BN, SM, SN, TM, TN, false>>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  uint32_t indx_vec =
      lhs_indices[elem_to_loc(tid.z, batch_shape, lhs_strides, batch_ndim)];
  uint32_t indx_mat =
      rhs_indices[elem_to_loc(tid.z, batch_shape, rhs_strides, batch_ndim)];
  in_vec += elem_to_loc(indx_vec, vec_shape, vec_strides, vec_ndim);
  mat += elem_to_loc(indx_mat, mat_shape, mat_strides, mat_ndim);
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      in_vec,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      marix_ld,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

} // namespace matvec
} // namespace mlx
