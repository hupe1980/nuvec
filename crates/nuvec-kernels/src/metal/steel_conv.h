// SPDX-License-Identifier: MIT
//
// MLX's implicit-GEMM convolutions ("steel" conv: the 2-D kernel with its channel and filter
// loaders, the general 2-D kernel for strides, dilations and flips, and the 3-D kernel),
// ported from MLX v0.32.4 (mlx/backend/metal/kernels/steel/conv) into one header on top of
// steel_gemm.h's BlockMMA. As there, `mlx::steel` is not opened at file scope.
//
// Copyright © 2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#pragma once

#include <metal_stdlib>

#include "steel_gemm.h"

// ── conv/params.h ──
template <int NDIM>
struct MLXConvParams {
  int N; // Batch size
  int C; // In channels
  int O; // Out channels
  int iS[NDIM]; // Input spatial dim
  int wS[NDIM]; // Weight spatial dim
  int oS[NDIM]; // Output spatial dim
  int str[NDIM]; // Kernel strides
  int pad[NDIM]; // Input padding
  int kdil[NDIM]; // Kernel dilation
  int idil[NDIM]; // Input dilation
  int64_t in_strides[NDIM + 2]; // In strides
  int64_t wt_strides[NDIM + 2]; // Wt strides
  int64_t out_strides[NDIM + 2]; // Out strides
  int groups; // Input channel groups
  bool flip;

  static MLXConvParams<NDIM>
  with_padded_channels(MLXConvParams<NDIM> other, int pad_out, int pad_in) {
    MLXConvParams<NDIM> params = other;

    // Update strides
    for (int i = 0; i < NDIM + 1; i++) {
      params.in_strides[i] =
          (params.in_strides[i] / params.C) * (params.C + pad_in);
      params.wt_strides[i] =
          (params.wt_strides[i] / params.C) * (params.C + pad_in);
      params.out_strides[i] =
          (params.out_strides[i] / params.O) * (params.O + pad_out);
    }
    params.in_strides[NDIM + 1] = 1;
    params.wt_strides[NDIM + 1] = 1;
    params.out_strides[NDIM + 1] = 1;

    // Update channels
    params.C += pad_in;
    params.O += pad_out;

    return params;
  };
};

namespace mlx {
namespace steel {

struct ImplicitGemmConv2DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct ImplicitGemmConv3DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_d;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct Conv2DGeneralJumpParams {
  const int f_wgt_jump_h;
  const int f_wgt_jump_w;

  const int f_out_jump_h;
  const int f_out_jump_w;

  const int adj_out_h;
  const int adj_out_w;
  const int adj_out_hw;
  const int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

} // namespace steel
} // namespace mlx

// ── conv/loaders/loader_channel_l.h ──
///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderLargeFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  int weight_h;
  int weight_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderLargeFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int ih = read_ih[i] + weight_h * params->kdil[0];
      int iw = read_iw[i] + weight_w * params->kdil[1];

      // Read from input if in bounds
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  using mask_t = short;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  int weight_h;
  int weight_w;

  const device T* src[n_rows];

  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderSmallFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    int read_n[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_h[i] = 0;
      mask_w[i] = 0;
    }

    for (short kh = 0; kh < params->wS[0]; kh++) {
      short flip_h = params->flip ? params->wS[0] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int ih = read_ih[i] + flip_h * params->kdil[0];

        bool in_bounds = n < params->N && ih >= 0 && ih < params->iS[0];

        mask_h[i] |= (in_bounds << kh);
      }
    }

    for (short kw = 0; kw < params->wS[1]; kw++) {
      short flip_w = params->flip ? params->wS[1] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[1];

        bool in_bounds = iw >= 0 && iw < params->iS[1];

        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Read from input if in bounds
      if ((mask_h[i] & h_mask) && (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoader {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;

  int weight_hw;
  int weight_step;

  const int read_n;
  const int N;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_hw(0),
        weight_step(params->C / params->groups),
        read_n(offsets.y + bi),
        N(gemm_params_->N),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < N) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_hw < (params->wS[1] * params->wS[0])) {
      src += weight_step;
      return;
    }

    weight_hw = 0;

    src += BK - (params->wS[1] * params->wS[0] - 1) * weight_step;
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv3DInputBlockLoaderLargeFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<3>* params;
  const constant ImplicitGemmConv3DParams* gemm_params;

  int weight_d;
  int weight_h;
  int weight_w;

  short kdil_d;
  short kdil_h;
  short kdil_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_id[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv3DInputBlockLoaderLargeFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_d(0),
        weight_h(0),
        weight_w(0),
        kdil_d(params_->flip ? -params_->kdil[0] : params_->kdil[0]),
        kdil_h(params_->flip ? -params_->kdil[1] : params_->kdil[1]),
        kdil_w(params_->flip ? -params_->kdil[2] : params_->kdil[2]) {
    int out_n_pixels = params->oS[0] * params->oS[1] * params->oS[2];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_ndhw = offsets.y + bi + i * TROWS;
      int n = offset_ndhw / out_n_pixels;
      int dhw = offset_ndhw % out_n_pixels;
      int od = dhw / (params->oS[1] * params->oS[2]);
      int hw = dhw % (params->oS[1] * params->oS[2]);
      int oh = hw / params->oS[2];
      int ow = hw % params->oS[2];

      int id = od * params->str[0] - params->pad[0];
      int ih = oh * params->str[1] - params->pad[1];
      int iw = ow * params->str[2] - params->pad[2];

      read_n[i] = n;

      if (params->flip) {
        read_id[i] = id + (params->wS[0] - 1) * params->kdil[0];
        read_ih[i] = ih + (params->wS[1] - 1) * params->kdil[1];
        read_iw[i] = iw + (params->wS[2] - 1) * params->kdil[2];
      } else {
        read_id[i] = id;
        read_ih[i] = ih;
        read_iw[i] = iw;
      }

      // Adjust for flip
      if (params->flip) {
        id += (params->wS[0] - 1) * params->kdil[0];
        ih += (params->wS[1] - 1) * params->kdil[1];
        iw += (params->wS[2] - 1) * params->kdil[2];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + id * params->in_strides[1] +
          ih * params->in_strides[2] + iw * params->in_strides[3] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int id = read_id[i] + weight_d * kdil_d;
      int ih = read_ih[i] + weight_h * kdil_h;
      int iw = read_iw[i] + weight_w * kdil_w;

      // Read from input if in bounds
      if ((n < params->N) && (id >= 0 && id < params->iS[0]) &&
          (ih >= 0 && ih < params->iS[1]) && (iw >= 0 && iw < params->iS[2])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[2]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    if (++weight_d < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_d;
      }

      return;
    }

    weight_d = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv3DInputBlockLoaderSmallFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  using mask_t = short;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<3>* params;
  const constant ImplicitGemmConv3DParams* gemm_params;

  int weight_d;
  int weight_h;
  int weight_w;

  const device T* src[n_rows];

  mask_t mask_d[n_rows];
  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  /* Constructor */
  METAL_FUNC Conv3DInputBlockLoaderSmallFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_d(0),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1] * params->oS[2];

    int read_n[n_rows];
    int read_id[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_ndhw = offsets.y + bi + i * TROWS;
      int n = offset_ndhw / out_n_pixels;
      int dhw = offset_ndhw % out_n_pixels;
      int od = dhw / (params->oS[1] * params->oS[2]);
      int hw = dhw % (params->oS[1] * params->oS[2]);
      int oh = hw / params->oS[2];
      int ow = hw % params->oS[2];

      int id = od * params->str[0] - params->pad[0];
      int ih = oh * params->str[1] - params->pad[1];
      int iw = ow * params->str[2] - params->pad[2];

      read_n[i] = n;
      read_id[i] = id;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        id += (params->wS[0] - 1) * params->kdil[0];
        ih += (params->wS[1] - 1) * params->kdil[1];
        iw += (params->wS[2] - 1) * params->kdil[2];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + id * params->in_strides[1] +
          ih * params->in_strides[2] + iw * params->in_strides[3] + bj;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_d[i] = 0;
      mask_h[i] = 0;
      mask_w[i] = 0;
    }

    for (short kd = 0; kd < params->wS[0]; kd++) {
      short flip_d = params->flip ? params->wS[0] - kd - 1 : kd;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int id = read_id[i] + flip_d * params->kdil[0];

        bool in_bounds = n < params->N && id >= 0 && id < params->iS[0];

        mask_d[i] |= (in_bounds << kd);
      }
    }

    for (short kh = 0; kh < params->wS[1]; kh++) {
      short flip_h = params->flip ? params->wS[1] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int ih = read_ih[i] + flip_h * params->kdil[1];

        bool in_bounds = ih >= 0 && ih < params->iS[1];

        mask_h[i] |= (in_bounds << kh);
      }
    }

    for (short kw = 0; kw < params->wS[2]; kw++) {
      short flip_w = params->flip ? params->wS[2] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[2];

        bool in_bounds = iw >= 0 && iw < params->iS[2];

        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    mask_t d_mask = mask_t(1) << weight_d;
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Read from input if in bounds
      if ((mask_d[i] & d_mask) && (mask_h[i] & h_mask) &&
          (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[2]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    if (++weight_d < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_d;
      }

      return;
    }

    weight_d = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv3DWeightBlockLoader {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<3>* params;

  int weight_dhw;
  int weight_step;

  const int read_n;
  const int N;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv3DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<3>* params_,
      const constant ImplicitGemmConv3DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_dhw(0),
        weight_step(params->C / params->groups),
        read_n(offsets.y + bi),
        N(gemm_params_->N),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < N) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    if (++weight_dhw < (params->wS[0] * params->wS[1] * params->wS[2])) {
      src += weight_step;
      return;
    }

    weight_dhw = 0;

    src +=
        BK - (params->wS[0] * params->wS[1] * params->wS[2] - 1) * weight_step;
  }
};

} // namespace steel
} // namespace mlx

// ── conv/loaders/loader_channel_n.h ──
///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <short n_channels_>
struct ChannelHelper {
  STEEL_CONST short n_channels = n_channels_;
  STEEL_CONST short vec_size = n_channels_ <= 4 ? 4 : 8;
  STEEL_CONST short excess = vec_size - n_channels_;
};

template <>
struct ChannelHelper<1> {
  STEEL_CONST short n_channels = 1;
  STEEL_CONST short vec_size = 1;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<2> {
  STEEL_CONST short n_channels = 2;
  STEEL_CONST short vec_size = 2;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<3> {
  STEEL_CONST short n_channels = 3;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 1;
};

template <>
struct ChannelHelper<4> {
  STEEL_CONST short n_channels = 4;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 0;
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short n_channels,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallChannels {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;
  static_assert(n_rows > 0, "BM must be at least tgp_size * vec_size / BK");

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  int weight_hw;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderSmallChannels(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_hw(thread_idx % TCOLS) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    if (weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    int wh = (weight_hw / params->wS[1]);
    int ww = (weight_hw % params->wS[1]);

    int flip_h = params->flip ? params->wS[0] - wh - 1 : wh;
    int flip_w = params->flip ? params->wS[1] - ww - 1 : ww;

    int weight_h = flip_h * params->kdil[0];
    int weight_w = flip_w * params->kdil[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int ih = read_ih[i] + weight_h;
      int iw = read_iw[i] + weight_w;

      // Read from input if in bounds
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        const device T* curr_src = src[i] + weight_h * params->in_strides[1] +
            weight_w * params->in_strides[2];

        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; ++j) {
          dst[is * dst_ld + j] = curr_src[j];
        }

        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    weight_hw += TCOLS;
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short n_channels,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoaderSmallChannels {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;

  int weight_hw;

  const int read_n;
  const int N;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoaderSmallChannels(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld),
        params(params_),
        weight_hw(thread_idx % TCOLS),
        read_n(offsets.y + bi),
        N(gemm_params_->N),
        do_read(read_n + BN <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    if (bi >= BROWS || bj >= BCOLS)
      return;

    if (read_n >= N || weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }

      return;
    }

    const device T* curr_src = src + weight_hw * (params->C / params->groups);

    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }

        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
    } else {
      for (short i = 0; i < BROWS; i += TROWS) {
        if (((read_n + i) < N)) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < n_channels; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }

          STEEL_PRAGMA_UNROLL
          for (short j = n_channels; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    weight_hw += TCOLS;
  }
};

} // namespace steel
} // namespace mlx

// ── conv/loaders/loader_general.h ──
///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderGeneral {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant Conv2DGeneralJumpParams* jump_params;

  const short base_wh;
  const short base_ww;

  int weight_h;
  int weight_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int4 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / jump_params->adj_out_hw;
      int hw = offset_nhw % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + offsets.z;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + offsets.w;

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];

      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;

      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];

      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];

      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];

      // Read from input if in bounds
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = (src[i])[offset + j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  METAL_FUNC void load_safe(const short remaining_k) const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];

      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;

      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];

      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];

      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];

      // Read from input if in bounds
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
        if (bj + vec_size <= remaining_k) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; ++j) {
            dst[is * dst_ld + j] = (src[i])[offset + j];
          }
        } else {
          for (short j = 0; j < vec_size; ++j) {
            if (bj + j < remaining_k) {
              dst[is * dst_ld + j] = (src[i])[offset + j];
            } else {
              dst[is * dst_ld + j] = T(0);
            }
          }
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }

    weight_w = base_ww;

    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }

    weight_h = base_wh;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += BK;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoaderGeneral {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;
  const constant Conv2DGeneralJumpParams* jump_params;

  const short base_wh;
  const short base_ww;

  int weight_h;
  int weight_w;

  const int start_row;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size*(thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_),
        start_row(offsets.y + bi) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
        weight_w * params->wt_strides[2];

    if ((start_row + BN <= params->O)) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((start_row + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  METAL_FUNC void load_safe(const short remaining_k) const thread {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
        weight_w * params->wt_strides[2];

    if ((start_row + BN <= params->O)) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        if (bj + vec_size <= remaining_k) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }
        } else {
          for (short j = 0; j < vec_size; j++) {
            if (bj + j < remaining_k) {
              dst[i * dst_ld + j] = curr_src[i * src_ld + j];
            } else {
              dst[i * dst_ld + j] = T(0);
            }
          }
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((start_row + i) < params->O) {
          if (bj + vec_size <= remaining_k) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
              dst[i * dst_ld + j] = curr_src[i * src_ld + j];
            }
          } else {
            for (short j = 0; j < vec_size; j++) {
              if (bj + j < remaining_k) {
                dst[i * dst_ld + j] = curr_src[i * src_ld + j];
              } else {
                dst[i * dst_ld + j] = T(0);
              }
            }
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() thread {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }

    weight_w = base_ww;

    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }

    weight_h = base_wh;

    src += BK;
  }
};

} // namespace steel
} // namespace mlx

// ── conv/kernels/steel_conv.h ──
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int N_CHANNELS = 0,
    bool SMALL_FILTER = false>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
implicit_gemm_conv_2d(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant mlx::steel::ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;

  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);

  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;

  constexpr short tgp_size = WM * WN * 32;

  // Input loader

  using loader_a_t = typename metal::conditional_t<
      // Check for small channel specialization
      N_CHANNELS != 0 && N_CHANNELS <= 8,

      // Go to small channel specialization
      Conv2DInputBlockLoaderSmallChannels<
          T,
          BM,
          BN,
          BK,
          tgp_size,
          N_CHANNELS,
          tgp_padding_a>,

      // Else go to general loader
      typename metal::conditional_t<
          // Check if filter size is small enough
          SMALL_FILTER,

          // Go to small filter specialization
          Conv2DInputBlockLoaderSmallFilter<
              T,
              BM,
              BN,
              BK,
              tgp_size,
              tgp_padding_a>,

          // Else go to large filter generalization
          Conv2DInputBlockLoaderLargeFilter<
              T,
              BM,
              BN,
              BK,
              tgp_size,
              tgp_padding_a>>>;

  // Weight loader
  using loader_b_t = typename metal::conditional_t<
      // Check for small channel specialization
      N_CHANNELS != 0 && N_CHANNELS <= 8,

      // Go to small channel specialization
      Conv2DWeightBlockLoaderSmallChannels<
          T,
          BM,
          BN,
          BK,
          tgp_size,
          N_CHANNELS,
          tgp_padding_b>,

      // Else go to general loader
      Conv2DWeightBlockLoader<T, BM, BN, BK, tgp_size, tgp_padding_b>>;

  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
      ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;

  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  const int N = gemm_params->N;
  const int C_per_group = params->C / params->groups;

  // Groups
  A += tid.z * C_per_group;
  B += tid.z * N * K;
  C += tid.z * N;

  B += c_col * K;
  C += static_cast<size_t>(c_row) * N * params->groups + c_col;

  const int2 offsets_a(0, c_row);
  const int2 offsets_b(0, c_col);

  // Prepare threadgroup loading operations
  loader_a_t loader_a(
      A, As, offsets_a, params, gemm_params, simd_gid, simd_lid);
  loader_b_t loader_b(
      B, Bs, offsets_b, params, gemm_params, simd_gid, simd_lid);

  // Prepare threadgroup mma operation
  mma_t mma_op(simd_gid, simd_lid);

  int gemm_k_iterations = gemm_params->gemm_k_iterations;
  for (int k = 0; k < gemm_k_iterations; k++) {
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

  threadgroup_barrier(mem_flags::mem_none);

  // Store results to device memory
  short tgp_bm = min(BM, gemm_params->M - c_row);
  short tgp_bn = min(BN, gemm_params->N - c_col);
  const int ldc = N * params->groups;
  mma_op.store_result_safe(C, ldc, short2(tgp_bn, tgp_bm));
}

// ── conv/kernels/steel_conv_general.h ──
constant bool align_C [[function_constant(400)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float,
    typename Epilogue = mlx::steel::TransformNone<T, AccumType>>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
implicit_gemm_conv_2d_general(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant mlx::steel::ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    const constant mlx::steel::Conv2DGeneralJumpParams* jump_params [[buffer(5)]],
    const constant mlx::steel::Conv2DGeneralBaseInfo* base_h [[buffer(6)]],
    const constant mlx::steel::Conv2DGeneralBaseInfo* base_w [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;
  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);

  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;

  constexpr short tgp_size = WM * WN * 32;

  // Input loader
  using loader_a_t =
      Conv2DInputBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_a>;

  // Weight loader
  using loader_b_t =
      Conv2DWeightBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_b>;

  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
      ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;

  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }

  const int tid_z = tid.z;

  const int base_oh = tid_z / jump_params->f_out_jump_w;
  const int base_ow = tid_z % jump_params->f_out_jump_w;

  const int base_wh = base_h[base_oh].weight_base;
  const int base_ww = base_w[base_ow].weight_base;

  const int base_wh_size = base_h[base_oh].weight_size;
  const int base_ww_size = base_w[base_ow].weight_size;

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;

  B += c_col * K;

  const int4 offsets_a(0, c_row, base_oh, base_ow);
  const int2 offsets_b(0, c_col);

  // Prepare threadgroup loading operations
  loader_a_t loader_a(
      A,
      As,
      offsets_a,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);
  loader_b_t loader_b(
      B,
      Bs,
      offsets_b,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);

  // Prepare threadgroup mma operation
  mma_t mma_op(simd_gid, simd_lid);

  if (align_C) {
    int gemm_k_iterations =
        base_wh_size * base_ww_size * gemm_params->gemm_k_iterations;

    for (int k = 0; k < gemm_k_iterations; k++) {
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
  }

  else {
    for (int k = 1; k < gemm_params->gemm_k_iterations; k++) {
      for (int j = 0; j < base_wh_size * base_ww_size; j++) {
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
    }
    const short remaining_k = params->C % BK;
    for (int j = 0; j < base_wh_size * base_ww_size; j++) {
      // Load elements into threadgroup
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(remaining_k);
      loader_b.load_safe(remaining_k);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);
      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results to device memory
  {
    // Adjust for simdgroup and thread location
    int offset_m = c_row + mma_op.sm;
    int offset_n = c_col + mma_op.sn;
    C += offset_n;

    if (offset_n >= gemm_params->N)
      return;

    int diff = gemm_params->N - offset_n;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < mma_t::TM; i++) {
      int cm = offset_m + i * mma_t::TM_stride;

      int n = cm / jump_params->adj_out_hw;
      int hw = cm % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + base_oh;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + base_ow;

      if (n < params->N && oh < params->oS[0] && ow < params->oS[1]) {
        size_t offset_cm = static_cast<size_t>(n) * params->out_strides[0] +
            oh * params->out_strides[1] + ow * params->out_strides[2];

        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < mma_t::TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = mma_op.Ctile.frag_at(i, j);
          size_t offset = offset_cm + (j * mma_t::TN_stride);

          constexpr short kelems = decltype(mma_op.Ctile)::kElemsPerFrag;

          // Apply epilogue and output C
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * mma_t::TN_stride + k) < diff) {
              C[offset + k] = Epilogue::apply(accum[k]);
            }
          }
        }
      }
    }
  }
}

// ── conv/kernels/steel_conv_3d.h ──
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool SMALL_FILTER = false>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
implicit_gemm_conv_3d(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<3>* params [[buffer(3)]],
    const constant mlx::steel::ImplicitGemmConv3DParams* gemm_params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;

  (void)lid;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);

  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;

  constexpr short tgp_size = WM * WN * 32;

  // Input loader
  using loader_a_t = typename metal::conditional_t<
      // If the filter is small we can precompute masks for bounds checking
      SMALL_FILTER,
      Conv3DInputBlockLoaderSmallFilter<T, BM, BN, BK, tgp_size, tgp_padding_a>,
      Conv3DInputBlockLoaderLargeFilter<
          T,
          BM,
          BN,
          BK,
          tgp_size,
          tgp_padding_a>>;

  // Weight loader
  using loader_b_t =
      Conv3DWeightBlockLoader<T, BM, BN, BK, tgp_size, tgp_padding_b>;

  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
      ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;

  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  const int N = gemm_params->N;
  const int C_per_group = params->C / params->groups;

  // Groups
  A += tid.z * C_per_group;
  B += tid.z * N * K;
  C += tid.z * N;

  B += c_col * K;
  C += static_cast<size_t>(c_row) * N * params->groups + c_col;

  const int2 offsets_a(0, c_row);
  const int2 offsets_b(0, c_col);

  // Prepare threadgroup loading operations
  loader_a_t loader_a(
      A, As, offsets_a, params, gemm_params, simd_gid, simd_lid);
  loader_b_t loader_b(
      B, Bs, offsets_b, params, gemm_params, simd_gid, simd_lid);

  // Prepare threadgroup mma operation
  mma_t mma_op(simd_gid, simd_lid);

  int gemm_k_iterations = gemm_params->gemm_k_iterations;
  for (int k = 0; k < gemm_k_iterations; k++) {
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

  threadgroup_barrier(mem_flags::mem_none);

  // Store results to device memory
  short tgp_bm = min(BM, gemm_params->M - c_row);
  short tgp_bn = min(BN, gemm_params->N - c_col);
  const int ldc = N * params->groups;
  mma_op.store_result_safe(C, ldc, short2(tgp_bn, tgp_bm));
}
