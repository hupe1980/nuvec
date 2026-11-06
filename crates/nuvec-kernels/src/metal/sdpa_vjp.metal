// SPDX-License-Identifier: MIT
//
// The attention backward's kernels, ported from MLX v0.32.4
// (mlx/backend/metal/kernels/scaled_dot_product_attention_vjp.metal): `δ = rowsum(O ⊙ dO)`
// (`sdpa_vjp_odo`, the value head size a parameter), one tile's `P = exp(scale · S − lse)`
// and `dS = P ⊙ (dP − δ) · scale` (`sdpa_vjp_ds`), and the float32 accumulation of a
// tile's product, its query group summed (`sdpa_vjp_reduce`).
//
// Copyright © 2026 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "fast.h"

template <typename T>
[[kernel]] void sdpa_vjp_odo(
    const device T* o [[buffer(0)]],      // [B, H, qL, Dv]
    const device T* cot_o [[buffer(1)]],  // [B, H, qL, Dv]
    device float* odo [[buffer(2)]],      // [B, H, qL]
    constant int& qL [[buffer(3)]],
    constant int& Dv [[buffer(4)]],
    uint3 tpg [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  const uint bh = tpg.z;
  const int i = int(tpg.y);
  if (i >= qL) {
    return;
  }
  const size_t row = size_t(bh) * qL + i;
  auto o_i = o + row * Dv;
  auto co_i = cot_o + row * Dv;
  float acc = 0.0f;
  for (int d = int(simd_lane_id); d < Dv; d += 32) {
    acc += float(o_i[d]) * float(co_i[d]);
  }
  acc = metal::simd_sum(acc);
  if (simd_lane_id == 0) {
    odo[row] = acc;
  }
}

struct SdpaVjpTileParams {
  int bq;
  int bk;
  int qL;
  int i0;
  int j0;
  float scale;
  int diag_off;
  int causal;
};

template <typename T>
[[kernel]] void sdpa_vjp_ds(
    const device T* S [[buffer(0)]],        // [BH, bq, bk]
    const device T* dP [[buffer(1)]],       // [BH, bq, bk]
    const device float* lse [[buffer(2)]],  // [B, H, qL]
    const device float* odo [[buffer(3)]],  // [B, H, qL]
    device T* dS [[buffer(4)]],             // [BH, bq, bk]
    device T* P [[buffer(5)]],              // [BH, bq, bk]
    const constant SdpaVjpTileParams& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int col = int(gid.x), row = int(gid.y), bh = int(gid.z);
  if (row >= p.bq || col >= p.bk) {
    return;
  }
  const size_t sbase = (size_t(bh) * size_t(p.bq) + size_t(row)) * size_t(p.bk);
  const size_t qi = size_t(bh) * size_t(p.qL) + size_t(p.i0 + row);
  const float l = lse[qi];
  const float dlt = odo[qi];
  // Fully masked rows carry lse == -inf; exp(-inf - -inf) would be NaN.
  const bool dead = (l < 0.0f) && metal::isinf(l);
  const int lim = p.i0 + row + p.diag_off;
  float pv;
  if (dead || (p.causal != 0 && (p.j0 + col) > lim)) {
    pv = 0.0f;
  } else {
    pv = metal::fast::exp(static_cast<float>(S[sbase + col]) * p.scale - l);
  }
  const float dsv = pv * (static_cast<float>(dP[sbase + col]) - dlt) * p.scale;
  dS[sbase + col] = static_cast<T>(dsv);
  P[sbase + col] = static_cast<T>(pv);
}

template <typename T, bool Accum>
[[kernel]] void sdpa_vjp_reduce(
    const device T* src [[buffer(0)]],
    device float* acc [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& rows [[buffer(3)]],
    const constant int& dim [[buffer(4)]],
    const constant int& group [[buffer(5)]],
    const constant int& acc_rows [[buffer(6)]],
    const constant int& row_off [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x), row = int(gid.y), bh = int(gid.z);
  if (c >= dim || row >= rows) {
    return;
  }
  const size_t sbase = size_t(bh) * size_t(group) * size_t(rows) * size_t(dim);
  float sum = 0.0f;
  for (int g = 0; g < group; ++g) {
    const size_t si = sbase + (size_t(g) * size_t(rows) + size_t(row)) * size_t(dim) + size_t(c);
    sum += static_cast<float>(src[si]);
  }
  const size_t ai = (size_t(bh) * size_t(acc_rows) + size_t(row_off + row)) * size_t(dim) + size_t(c);
  const float total = Accum ? acc[ai] + sum : sum;
  acc[ai] = total;
  out[ai] = static_cast<T>(total);
}

#define instantiate_sdpa_vjp(tname, type)                                                     \
  template [[host_name("nuvec_sdpa_vjp_odo_" #tname)]] [[kernel]] decltype(sdpa_vjp_odo<type>) \
      sdpa_vjp_odo<type>;                                                                      \
  template [[host_name("nuvec_sdpa_vjp_ds_" #tname)]] [[kernel]] decltype(sdpa_vjp_ds<type>)   \
      sdpa_vjp_ds<type>;                                                                       \
  template [[host_name("nuvec_sdpa_vjp_reduce_add_" #tname)]] [[kernel]]                       \
      decltype(sdpa_vjp_reduce<type, true>) sdpa_vjp_reduce<type, true>;                       \
  template [[host_name("nuvec_sdpa_vjp_reduce_set_" #tname)]] [[kernel]]                       \
      decltype(sdpa_vjp_reduce<type, false>) sdpa_vjp_reduce<type, false>;

instantiate_sdpa_vjp(float32, float)
instantiate_sdpa_vjp(float16, half)
instantiate_sdpa_vjp(bfloat16, bfloat16_t)
