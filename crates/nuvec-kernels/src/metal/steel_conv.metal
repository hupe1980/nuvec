// SPDX-License-Identifier: MIT
//
// MLX's convolution kernels, ported from MLX v0.32.4: the implicit-GEMM convolutions
// (steel_conv.h), and the depthwise 2-D kernel and the Winograd F(6×6, 3×3) transforms
// (mlx/backend/metal/kernels/conv.metal), with MLX's tile shapes. Compiled with
// contraction, as MLX compiles them.
//
// Copyright © 2023-2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include <metal_stdlib>

#include "steel_conv.h"

using namespace metal;

typedef bfloat bfloat16_t;

#define NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, channel_name, n_channels, filter_name, small_filter) \
  template [[host_name("nuvec_implicit_gemm_conv_2d_" #name "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm      \
                       "_wn" #wn "_channel_" #channel_name "_filter_" #filter_name)]] [[kernel]] void    \
  implicit_gemm_conv_2d<itype, bm, bn, bk, wm, wn, n_channels, small_filter>(                            \
      const device itype* A [[buffer(0)]],                                                              \
      const device itype* B [[buffer(1)]],                                                              \
      device itype* C [[buffer(2)]],                                                                    \
      const constant MLXConvParams<2>* params [[buffer(3)]],                                            \
      const constant mlx::steel::ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],                   \
      uint3 tid [[threadgroup_position_in_grid]],                                                       \
      uint3 lid [[thread_position_in_threadgroup]],                                                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                                                 \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define NUVEC_CONV2D_FILTERS(name, itype, bm, bn, bk, wm, wn)           \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, l, 0, s, true)          \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, l, 0, l, false)         \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, 1, 1, l, false)         \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, 2, 2, l, false)         \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, 3, 3, l, false)         \
  NUVEC_CONV2D(name, itype, bm, bn, bk, wm, wn, 4, 4, l, false)

// The 8-channel loader reads one row per thread and needs bm = 64.
#define NUVEC_CONV2D_BLOCKS(name, itype)                                 \
  NUVEC_CONV2D_FILTERS(name, itype, 32, 8, 16, 4, 1)                     \
  NUVEC_CONV2D_FILTERS(name, itype, 64, 8, 16, 4, 1)                     \
  NUVEC_CONV2D_FILTERS(name, itype, 32, 32, 16, 2, 2)                    \
  NUVEC_CONV2D_FILTERS(name, itype, 32, 64, 16, 2, 2)                    \
  NUVEC_CONV2D_FILTERS(name, itype, 64, 32, 16, 2, 2)                    \
  NUVEC_CONV2D_FILTERS(name, itype, 64, 64, 16, 2, 2)                    \
  NUVEC_CONV2D(name, itype, 64, 8, 16, 4, 1, 8, 8, l, false)             \
  NUVEC_CONV2D(name, itype, 64, 32, 16, 2, 2, 8, 8, l, false)            \
  NUVEC_CONV2D(name, itype, 64, 64, 16, 2, 2, 8, 8, l, false)

NUVEC_CONV2D_BLOCKS(float32, float)
NUVEC_CONV2D_BLOCKS(float16, half)
NUVEC_CONV2D_BLOCKS(bfloat16, bfloat16_t)

#define NUVEC_CONV2D_GENERAL(name, itype, bm, bn, bk, wm, wn)                                          \
  template [[host_name("nuvec_implicit_gemm_conv_2d_general_" #name "_bm" #bm "_bn" #bn "_bk" #bk     \
                       "_wm" #wm "_wn" #wn)]] [[kernel]] void                                          \
  implicit_gemm_conv_2d_general<itype, bm, bn, bk, wm, wn>(                                            \
      const device itype* A [[buffer(0)]],                                                             \
      const device itype* B [[buffer(1)]],                                                             \
      device itype* C [[buffer(2)]],                                                                   \
      const constant MLXConvParams<2>* params [[buffer(3)]],                                           \
      const constant mlx::steel::ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],                  \
      const constant mlx::steel::Conv2DGeneralJumpParams* jump_params [[buffer(5)]],                   \
      const constant mlx::steel::Conv2DGeneralBaseInfo* base_h [[buffer(6)]],                          \
      const constant mlx::steel::Conv2DGeneralBaseInfo* base_w [[buffer(7)]],                          \
      uint3 tid [[threadgroup_position_in_grid]],                                                      \
      uint3 lid [[thread_position_in_threadgroup]],                                                    \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                                                \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define NUVEC_CONV2D_GENERAL_BLOCKS(name, itype)                \
  NUVEC_CONV2D_GENERAL(name, itype, 32, 8, 16, 4, 1)            \
  NUVEC_CONV2D_GENERAL(name, itype, 64, 8, 16, 4, 1)            \
  NUVEC_CONV2D_GENERAL(name, itype, 32, 32, 16, 2, 2)           \
  NUVEC_CONV2D_GENERAL(name, itype, 32, 64, 16, 2, 2)           \
  NUVEC_CONV2D_GENERAL(name, itype, 64, 32, 16, 2, 2)           \
  NUVEC_CONV2D_GENERAL(name, itype, 64, 64, 16, 2, 2)

NUVEC_CONV2D_GENERAL_BLOCKS(float32, float)
NUVEC_CONV2D_GENERAL_BLOCKS(float16, half)
NUVEC_CONV2D_GENERAL_BLOCKS(bfloat16, bfloat16_t)

#define NUVEC_CONV3D(name, itype, bm, bn, bk, wm, wn, fn, f)                                           \
  template [[host_name("nuvec_implicit_gemm_conv_3d_" #name "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm  \
                       "_wn" #wn "_filter_" #fn)]] [[kernel]] decltype(implicit_gemm_conv_3d<itype,  \
                       bm, bn, bk, wm, wn, f>) implicit_gemm_conv_3d<itype, bm, bn, bk, wm, wn, f>;

#define NUVEC_CONV3D_FILTERS(name, itype, bm, bn, bk, wm, wn)   \
  NUVEC_CONV3D(name, itype, bm, bn, bk, wm, wn, s, true)        \
  NUVEC_CONV3D(name, itype, bm, bn, bk, wm, wn, l, false)

#define NUVEC_CONV3D_BLOCKS(name, itype)                        \
  NUVEC_CONV3D_FILTERS(name, itype, 32, 8, 16, 4, 1)            \
  NUVEC_CONV3D_FILTERS(name, itype, 64, 8, 16, 4, 1)            \
  NUVEC_CONV3D_FILTERS(name, itype, 32, 32, 16, 2, 2)           \
  NUVEC_CONV3D_FILTERS(name, itype, 32, 64, 16, 2, 2)           \
  NUVEC_CONV3D_FILTERS(name, itype, 64, 32, 16, 2, 2)           \
  NUVEC_CONV3D_FILTERS(name, itype, 64, 64, 16, 2, 2)

NUVEC_CONV3D_BLOCKS(float32, float)
NUVEC_CONV3D_BLOCKS(float16, half)
NUVEC_CONV3D_BLOCKS(bfloat16, bfloat16_t)

// ── Depthwise 2-D (conv.metal) ──
// One change: products are float32 (MLX multiplies in the input type, rounding each product
// of half-precision inputs before the float32 sum).
// Function constants: kernel size (410, 411), stride (412, 413), tile (414, 415), flip (416).

constant int dw_ker_h [[function_constant(410)]];
constant int dw_ker_w [[function_constant(411)]];
constant int dw_str_h [[function_constant(412)]];
constant int dw_str_w [[function_constant(413)]];
constant int dw_tgp_h [[function_constant(414)]];
constant int dw_tgp_w [[function_constant(415)]];
constant bool dw_flip [[function_constant(416)]];

constant int dw_span_h = dw_tgp_h * dw_str_h + dw_ker_h - 1;
constant int dw_span_w = dw_tgp_w * dw_str_w + dw_ker_w - 1;
constant int dw_span_hw = dw_span_h * dw_span_w;

template <typename T>
[[kernel]] void depthwise_conv_2d(
    const device T* in [[buffer(0)]],
    const device T* wt [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int tc = 8;
  constexpr int tw = 8;
  constexpr int th = 4;

  constexpr int c_per_thr = 8;

  constexpr int TGH = th * 2 + 6;
  constexpr int TGW = tw * 2 + 6;
  constexpr int TGC = tc;

  threadgroup T ins[TGH * TGW * TGC];

  const int n_tgblocks_h = (params.oS[0] + th - 1) / th;
  const int n = tid.z / n_tgblocks_h;
  const int tghid = tid.z % n_tgblocks_h;
  const int oh = tghid * th + lid.z;
  const int ow = gid.y;
  const int c = gid.x;

  in += n * params.in_strides[0];

  // Load in
  {
    constexpr int n_threads = th * tw * tc;
    const int tg_oh = (tghid * th) * dw_str_h - params.pad[0];
    const int tg_ow = (tid.y * tw) * dw_str_w - params.pad[1];
    const int tg_c = tid.x * tc;

    const int thread_idx = simd_gid * 32 + simd_lid;
    constexpr int thr_per_hw = tc / c_per_thr;
    constexpr int hw_per_group = n_threads / thr_per_hw;

    const int thr_c = thread_idx % thr_per_hw;
    const int thr_hw = thread_idx / thr_per_hw;

    for (int hw = thr_hw; hw < dw_span_hw; hw += hw_per_group) {
      const int h = hw / dw_span_w;
      const int w = hw % dw_span_w;

      const int ih = tg_oh + h;
      const int iw = tg_ow + w;

      const int in_s_offset = h * dw_span_w * TGC + w * TGC;

      if (ih >= 0 && ih < params.iS[0] && iw >= 0 && iw < params.iS[1]) {
        const auto in_load =
            in + ih * params.in_strides[1] + iw * params.in_strides[2] + tg_c;

        STEEL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] =
              in_load[c_per_thr * thr_c + cc];
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] = T(0);
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  wt += c * params.wt_strides[0];

  const auto ins_ptr =
      &ins[lid.z * dw_str_h * dw_span_w * TGC + lid.y * dw_str_w * TGC + lid.x];
  float o = 0.;
  for (int h = 0; h < dw_ker_h; ++h) {
    for (int w = 0; w < dw_ker_w; ++w) {
      int wt_h = h;
      int wt_w = w;
      if (dw_flip) {
        wt_h = dw_ker_h - h - 1;
        wt_w = dw_ker_w - w - 1;
      }
      auto inv = ins_ptr[h * dw_span_w * TGC + w * TGC];
      auto wtv = wt[wt_h * dw_ker_w + wt_w];
      o += static_cast<float>(inv) * static_cast<float>(wtv);
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  if (oh >= params.oS[0] || ow >= params.oS[1]) {
    return;
  }

  out += n * params.out_strides[0] + oh * params.out_strides[1] +
      ow * params.out_strides[2];
  out[c] = static_cast<T>(o);
}

#define NUVEC_DEPTHWISE2D(iname, itype) \
  template [[host_name("nuvec_depthwise_conv_2d_" #iname)]] [[kernel]] decltype(depthwise_conv_2d<itype>) depthwise_conv_2d<itype>;

NUVEC_DEPTHWISE2D(float32, float)
NUVEC_DEPTHWISE2D(float16, half)
NUVEC_DEPTHWISE2D(bfloat16, bfloat16_t)

// ── Winograd F(6×6, 3×3) transforms (conv.metal) ──

template <int M, int R, int S>
struct WinogradTransforms {};

template <>
struct WinogradTransforms<6, 3, 8> {
  static constant constexpr const int OUT_TILE_SIZE = 6;
  static constant constexpr const int FILTER_SIZE = 3;
  static constant constexpr const int IN_TILE_SIZE = OUT_TILE_SIZE + FILTER_SIZE - 1;
  static constant constexpr const int SIMD_MATRIX_SIZE = 8;
  static constant constexpr const float in_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 0.50f, -0.50f, 2.00f, -2.00f, -1.00f},
      {-5.25f, 1.00f, 1.00f, 0.25f, 0.25f, 4.00f, 4.00f, 0.00f},
      {0.00f, -4.25f, 4.25f, -2.50f, 2.50f, -2.50f, 2.50f, 5.25f},
      {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, -5.25f},
      {-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  static constant constexpr const float out_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
      {1.00f, -1.00f, 1.00f, -1.00f, 1.00f, -1.00f},
      {1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f},
      {1.00f, -2.00f, 4.00f, -8.00f, 16.00f, -32.00f},
      {1.00f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.03125f},
      {1.00f, -0.50f, 0.25f, -0.125f, 0.0625f, -0.03125f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  static constant constexpr const float wt_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00, 0.00, 0.00},
      {-2.0 / 9.00, -2.0 / 9.00, -2.0 / 9.00},
      {-2.0 / 9.00, 2.0 / 9.00, -2.0 / 9.00},
      {1.0 / 90.0, 1.0 / 45.0, 2.0 / 45.0},
      {1.0 / 90.0, -1.0 / 45.0, 2.0 / 45.0},
      {32.0 / 45.0, 16.0 / 45.0, 8.0 / 45.0},
      {32.0 / 45.0, -16.0 / 45.0, 8.0 / 45.0},
      {0.00, 0.00, 1.00},
  };
};

constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::in_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::out_transform[8][8];

template <typename T, int BC = 32, int BO = 4, int M = 6, int R = 3, typename TT = T>
[[kernel, max_total_threads_per_threadgroup(BO * 32)]] void
winograd_conv_2d_weight_transform(
    const device T* wt_in [[buffer(0)]],
    device TT* wt_out [[buffer(1)]],
    const constant int& C [[buffer(2)]],
    const constant int& O [[buffer(3)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  using WGT = WinogradTransforms<M, R, 8>;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize G matrix
  simdgroup_matrix<float, 8, 8> G;
  G.thread_elements()[0] = WGT::wt_transform[sm][sn];
  G.thread_elements()[1] = WGT::wt_transform[sm][sn + 1];

  // Initialize Gt matrix
  simdgroup_matrix<float, 8, 8> Gt;
  Gt.thread_elements()[0] = WGT::wt_transform[sn][sm];
  Gt.thread_elements()[1] = WGT::wt_transform[sn + 1][sm];

  // Move to the correct output filter
  size_t ko = BO * tid + simd_group_id;
  wt_in += ko * R * R * C;

  // wt_out is stored transposed (A x A x C x O)
  short ohw_0 = sm * 8 + sn;
  short ohw_1 = sm * 8 + sn + 1;
  device TT* wt_out_0 = wt_out + size_t(ohw_0) * C * O + ko;
  device TT* wt_out_1 = wt_out + size_t(ohw_1) * C * O + ko;

  // Prepare shared memory
  threadgroup T Ws[BO][R][R][BC];

  // Loop over C
  for (int bc = 0; bc < C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int kh = 0; kh < R; ++kh) {
      for (int kw = 0; kw < R; ++kw) {
        for (int kc = simd_lane_id; kc < BC; kc += 32) {
          Ws[simd_group_id][kh][kw][kc] = wt_in[kh * R * C + kw * C + kc];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = 0; c < BC; ++c) {
      simdgroup_matrix<float, 8, 8> g;
      g.thread_elements()[0] =
          sm < R && sn < R ? Ws[simd_group_id][sm][sn][c] : T(0);
      g.thread_elements()[1] =
          sm < R && sn + 1 < R ? Ws[simd_group_id][sm][sn + 1][c] : T(0);

      simdgroup_matrix<float, 8, 8> g_out = (G * g) * Gt;
      wt_out_0[c * O] = static_cast<TT>(g_out.thread_elements()[0]);
      wt_out_1[c * O] = static_cast<TT>(g_out.thread_elements()[1]);
    }

    wt_in += BC;
    wt_out_0 += BC * O;
    wt_out_1 += BC * O;
  }
}

#define instantiate_winograd_conv_2d_weight_transform_base(name, itype, bc)   \
  template [[host_name(                                                       \
      "nuvec_winograd_conv_2d_weight_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_weight_transform<itype, bc>(                               \
      const device itype* wt_in [[buffer(0)]],                                \
      device itype* wt_out [[buffer(1)]],                                     \
      const constant int& C [[buffer(2)]],                                    \
      const constant int& O [[buffer(3)]],                                    \
      uint tid [[threadgroup_position_in_grid]],                              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BC, int WM, int WN, int M = 6, int R = 3, typename TT = T>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_input_transform(
    const device T* inp_in [[buffer(0)]],
    device TT* inp_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int A = WGT::IN_TILE_SIZE;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize B matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::in_transform[sm][sn];
  B.thread_elements()[1] = WGT::in_transform[sm][sn + 1];

  // Initialize Bt matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::in_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::in_transform[sn + 1][sm];

  // Resolve input tile
  constexpr int TH = (A / WM);
  constexpr int TW = (A / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  inp_in += tid.z * params.in_strides[0] + bh * params.in_strides[1] +
      bw * params.in_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      jump_in[h][w] = h * params.in_strides[1] + w * params.in_strides[2];
    }
  }

  // inp_out is stored interleaved (A x A x tiles x C)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  device TT* inp_out_0 =
      inp_out + ohw_0 * N_TILES * params.C + tile_id * params.C;
  device TT* inp_out_1 =
      inp_out + ohw_1 * N_TILES * params.C + tile_id * params.C;

  // Prepare shared memory
  threadgroup T Is[A][A][BC];

  // Loop over C
  for (int bc = 0; bc < params.C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        const device T* in_ptr = inp_in + jump_in[h][w];
        for (int c = simd_lane_id; c < BC; c += 32) {
          Is[kh + h][kw + w][c] = in_ptr[c];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BC; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> I;
      I.thread_elements()[0] = Is[sm][sn][c];
      I.thread_elements()[1] = Is[sm][sn + 1][c];

      simdgroup_matrix<float, 8, 8> I_out = (Bt * I) * B;
      inp_out_0[c] = static_cast<TT>(I_out.thread_elements()[0]);
      inp_out_1[c] = static_cast<TT>(I_out.thread_elements()[1]);
    }

    inp_in += BC;
    inp_out_0 += BC;
    inp_out_1 += BC;
  }
}

#define instantiate_winograd_conv_2d_input_transform(name, itype, bc)        \
  template [[host_name(                                                      \
      "nuvec_winograd_conv_2d_input_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_input_transform<itype, bc, 2, 2>(                         \
      const device itype* inp_in [[buffer(0)]],                              \
      device itype* inp_out [[buffer(1)]],                                   \
      const constant MLXConvParams<2>& params [[buffer(2)]],                 \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BO, int WM, int WN, int M = 6, int R = 3, typename TT = T>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_output_transform(
    const device TT* out_in [[buffer(0)]],
    device T* out_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize A matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::out_transform[sm][sn];
  B.thread_elements()[1] = WGT::out_transform[sm][sn + 1];

  // Initialize At matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::out_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::out_transform[sn + 1][sm];

  // Out_in comes in shape (A x A x tiles x O)
  // We do transform and then write out to out_out in shape (N, H, W, O)

  // Resolve output tile
  constexpr int TH = (M / WM);
  constexpr int TW = (M / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  out_out += tid.z * params.out_strides[0] + bh * params.out_strides[1] +
      bw * params.out_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      bool valid = ((bh + h) < params.oS[0]) && ((bw + w) < params.oS[1]);
      jump_in[h][w] =
          valid ? h * params.out_strides[1] + w * params.out_strides[2] : -1;
    }
  }

  // out_in is stored interleaved (A x A x tiles x O)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  const device TT* out_in_0 =
      out_in + ohw_0 * N_TILES * params.O + tile_id * params.O;
  const device TT* out_in_1 =
      out_in + ohw_1 * N_TILES * params.O + tile_id * params.O;

  // Prepare shared memory
  threadgroup T Os[M][M][BO];

  // Loop over O
  for (int bo = 0; bo < params.O; bo += BO) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BO; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> O_mat;
      O_mat.thread_elements()[0] = out_in_0[c];
      O_mat.thread_elements()[1] = out_in_1[c];

      simdgroup_matrix<float, 8, 8> O_out = (Bt * (O_mat * B));
      if ((sm < M) && (sn < M)) {
        Os[sm][sn][c] = static_cast<T>(O_out.thread_elements()[0]);
      }
      if ((sm < M) && ((sn + 1) < M)) {
        Os[sm][sn + 1][c] = static_cast<T>(O_out.thread_elements()[1]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read out from shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        if (jump_in[h][w] >= 0) {
          device T* out_ptr = out_out + jump_in[h][w];
          for (int c = simd_lane_id; c < BO; c += 32) {
            out_ptr[c] = Os[kh + h][kw + w][c];
          }
        }
      }
    }

    out_out += BO;
    out_in_0 += BO;
    out_in_1 += BO;
  }
}

#define instantiate_winograd_conv_2d_output_transform(name, itype, bo)        \
  template [[host_name(                                                       \
      "nuvec_winograd_conv_2d_output_transform_" #name "_bo" #bo)]] [[kernel]] void \
  winograd_conv_2d_output_transform<itype, bo, 2, 2>(                         \
      const device itype* out_in [[buffer(0)]],                               \
      device itype* out_out [[buffer(1)]],                                    \
      const constant MLXConvParams<2>& params [[buffer(2)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                           \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

#define NUVEC_WINOGRAD(name, itype)                                   \
  instantiate_winograd_conv_2d_weight_transform_base(name, itype, 32) \
  instantiate_winograd_conv_2d_input_transform(name, itype, 32)       \
  instantiate_winograd_conv_2d_output_transform(name, itype, 32)

NUVEC_WINOGRAD(float32, float)
NUVEC_WINOGRAD(float16, half)
NUVEC_WINOGRAD(bfloat16, bfloat16_t)

// nuvec's half-precision Winograd: float32 tiles and a float32 product (the inputs and
// outputs stay half), as accurate as the direct convolution.
#define NUVEC_WINOGRAD_TF32(name, T)                                                          \
  template [[host_name("nuvec_winograd_conv_2d_weight_transform_" #name "_tf32_bc32")]]       \
  [[kernel]] decltype(winograd_conv_2d_weight_transform<T, 32, 4, 6, 3, float>)                \
      winograd_conv_2d_weight_transform<T, 32, 4, 6, 3, float>;                                \
  template [[host_name("nuvec_winograd_conv_2d_input_transform_" #name "_tf32_bc32")]]        \
  [[kernel]] decltype(winograd_conv_2d_input_transform<T, 32, 2, 2, 6, 3, float>)              \
      winograd_conv_2d_input_transform<T, 32, 2, 2, 6, 3, float>;                              \
  template [[host_name("nuvec_winograd_conv_2d_output_transform_" #name "_tf32_bo32")]]       \
  [[kernel]] decltype(winograd_conv_2d_output_transform<T, 32, 2, 2, 6, 3, float>)             \
      winograd_conv_2d_output_transform<T, 32, 2, 2, 6, 3, float>;

NUVEC_WINOGRAD_TF32(float16, half)
NUVEC_WINOGRAD_TF32(bfloat16, bfloat16_t)
