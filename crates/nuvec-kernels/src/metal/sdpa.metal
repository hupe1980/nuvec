// SPDX-License-Identifier: MIT
//
// Scaled dot-product attention, ported from MLX v0.32.4: the vector kernels for short
// queries (`sdpa_vector`, the two-pass `sdpa_vector_2pass_1`/`_2` and its GQA variant;
// kernels/sdpa_vector.h) and the tiled steel attention for long queries
// (kernels/steel/attn/kernels/steel_attention.h), without the NAX and blocked head-dim-512
// kernels. Function constants are renamed and renumbered (30–36, 210–211, 310–313) to stay
// unique in the single translation unit of source mode.
//
// Copyright © 2024-26 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "steel_gemm.h"
#include "fast.h"
#include "steel_attn.h"

constant bool sv_has_mask [[function_constant(30)]];
constant bool sv_query_transposed [[function_constant(31)]];
constant bool sv_do_causal [[function_constant(32)]];
constant bool sv_bool_mask [[function_constant(33)]];
constant bool sv_float_mask [[function_constant(34)]];
constant bool sv_has_sinks [[function_constant(35)]];
constant int sv_blocks [[function_constant(36)]];

constant bool att_align_Q [[function_constant(210)]];
constant bool att_align_K [[function_constant(211)]];
constant bool att_has_mask [[function_constant(310)]];
constant bool att_do_causal [[function_constant(311)]];
constant bool att_has_sinks [[function_constant(312)]];
constant bool att_save_lse [[function_constant(313)]];

namespace mlx {
namespace sdpa_vec {

// The declared limit keeps 1,024 threads per threadgroup available where instrumented
// builds (shader validation) or smaller register files would otherwise lower it.
template <typename T, int D, int V = D>
[[kernel, max_total_threads_per_threadgroup(1024)]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(sv_bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(sv_float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(sv_has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(sv_has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(sv_has_mask)]],
    const device T* sinks [[buffer(16), function_constant(sv_has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(sv_has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int kv_seq_idx = simd_gid;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      sv_query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (sv_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (sv_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (sv_has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (sv_do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (sv_bool_mask) {
      use_key = bmask[0];
    } else if (sv_float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (sv_float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (sv_bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (sv_float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D, int TK = 4>
[[kernel, max_total_threads_per_threadgroup(1024)]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(sv_bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(sv_float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(sv_has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(sv_has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(sv_has_mask)]],
    const device T* sinks [[buffer(18), function_constant(sv_has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      sv_query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * sv_blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (sv_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (sv_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * sv_blocks + block_idx;
  maxs += o_offset * sv_blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (sv_has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  const int stride_k_tok = sv_blocks * int(k_seq_stride);
  const int stride_v_tok = sv_blocks * int(v_seq_stride);
  const int stride_mask_tok =
      (sv_bool_mask || sv_float_mask) ? sv_blocks * mask_kv_seq_stride : 0;

  // For each key
  int i = block_idx;
  for (; i + (TK - 1) * sv_blocks < N; i += sv_blocks * TK) {
    // Prefetch: issue all TK K/V loads first so the HW can overlap them.
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; t++) {
      const device T* kp = keys + t * stride_k_tok;
      for (int j = 0; j < qk_per_thread; j++) {
        k_reg[t][j] = kp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = values + t * stride_v_tok;
      for (int j = 0; j < v_per_thread; j++) {
        v_reg[t][j] = vp[j];
      }
    }

    for (int t = 0; t < TK; t++) {
      int ii = i + t * sv_blocks;
      bool use_key = true;
      if (sv_do_causal) {
        use_key = ii <= (N - q_seq_len + int(q_seq_idx));
      } else if (sv_bool_mask) {
        use_key = bmask[t * stride_mask_tok];
      } else if (sv_float_mask) {
        use_key = (fmask[t * stride_mask_tok] >= Limits<T>::finite_min);
      }
      if (use_key) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        if (sv_float_mask) {
          score += static_cast<U>(fmask[t * stride_mask_tok]);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
    }

    keys += TK * stride_k_tok;
    values += TK * stride_v_tok;
    if (sv_bool_mask) {
      bmask += TK * stride_mask_tok;
    }
    if (sv_float_mask) {
      fmask += TK * stride_mask_tok;
    }
  }

  // Tail (fewer than TK remaining keys)
  for (; i < N; i += sv_blocks) {
    bool use_key = true;
    if (sv_do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (sv_bool_mask) {
      use_key = bmask[0];
    } else if (sv_float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * keys[j];
      }
      score = simd_sum(score);

      if (sv_float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += stride_k_tok;
    values += stride_v_tok;
    if (sv_bool_mask) {
      bmask += stride_mask_tok;
    }
    if (sv_float_mask) {
      fmask += stride_mask_tok;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries without mask or sinks only; the partials layout matches
// sdpa_vector_2pass_2.
template <typename T, int D, int V, int G, int HPT>
[[kernel, max_total_threads_per_threadgroup(1024)]] void sdpa_vector_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int NT = G / HPT;

  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int sv_blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * G;
  const int base_head = batch_idx * num_q_heads + kv_head_idx * G;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  const int chunk = (N + sv_blocks - 1) / sv_blocks;
  const int kstart = block_idx * chunk;
  const int kend = min(N, kstart + chunk);
  const int sub = (chunk + HPT - 1) / HPT;
  const int s0 = kstart + cchunk * sub;
  const int s1 = min(kend, s0 + sub);

  const device T* kp = keys + kv_batch_head_idx * k_head_stride +
      s0 * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = values + kv_batch_head_idx * v_head_stride +
      s0 * v_seq_stride + simd_lid * v_per_thread;

  U q[HPT][qk_per_thread];
  for (int j = 0; j < HPT; j++) {
    const device T* qp =
        queries + (base_head + h0 + j) * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[j][i] = static_cast<U>(scale) * qp[i];
    }
  }

  U max_score[HPT];
  U sum_exp_score[HPT];
  U o[HPT][v_per_thread];
  for (int j = 0; j < HPT; j++) {
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) {
      o[j][i] = 0;
    }
  }

  for (int t = s0; t < s1; t++) {
    U kr[qk_per_thread];
    U vr[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) {
      kr[i] = kp[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      vr[i] = vp[i];
    }
    kp += k_seq_stride;
    vp += v_seq_stride;
    for (int j = 0; j < HPT; j++) {
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[j][i] * kr[i];
      }
      score = simd_sum(score);
      U new_max = max(max_score[j], score);
      U factor = fast::exp(max_score[j] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[j] = new_max;
      sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
      for (int i = 0; i < v_per_thread; i++) {
        o[j][i] = o[j][i] * factor + exp_score * vr[i];
      }
    }
  }

  threadgroup U o_sh[G * HPT * V];
  threadgroup U se_sh[G * HPT];
  threadgroup U mx_sh[G * HPT];
  for (int j = 0; j < HPT; j++) {
    int slot = (h0 + j) * HPT + cchunk;
    U inv = sum_exp_score[j] > 0 ? 1 / sum_exp_score[j] : 0;
    for (int i = 0; i < v_per_thread; i++) {
      o_sh[slot * V + simd_lid * v_per_thread + i] = o[j][i] * inv;
    }
    if (simd_lid == 0) {
      se_sh[slot] = sum_exp_score[j];
      mx_sh[slot] = max_score[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U gmax = Limits<U>::finite_min;
  for (int s = 0; s < HPT; s++) {
    gmax = max(gmax, mx_sh[g * HPT + s]);
  }
  U denom = 0;
  U acc[v_per_thread] = {0};
  for (int s = 0; s < HPT; s++) {
    U w = se_sh[g * HPT + s] * fast::exp(mx_sh[g * HPT + s] - gmax);
    denom += w;
    for (int i = 0; i < v_per_thread; i++) {
      acc[i] += w * o_sh[(g * HPT + s) * V + simd_lid * v_per_thread + i];
    }
  }

  const int o_offset = base_head + g;
  device T* op =
      out + o_offset * sv_blocks * V + block_idx * V + simd_lid * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) {
    op[i] = static_cast<T>(acc[i]);
  }
  if (simd_lid == 0) {
    sums[o_offset * sv_blocks + block_idx] = denom;
    maxs[o_offset * sv_blocks + block_idx] = gmax;
  }
}

template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(1024)]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& sv_blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * sv_blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * sv_blocks;
  maxs += q_offset * sv_blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < sv_blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < sv_blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < sv_blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

} // namespace sdpa_vec

namespace steel_attn {


///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////



struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BV,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(att_has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(att_has_mask)]],
    const device T* sinks [[buffer(7), function_constant(att_has_sinks)]],
    device float* lse [[buffer(8), function_constant(att_save_lse)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (att_save_lse) {
    lse += tidl.z * params->H * params->qL + // Batch
        tidl.y * params->qL + // Head
        tidl.x * BQ; // Sequence
  }

  if (att_has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;
  constexpr bool q_in_registers = (WN == 2) && (BK == 32 || sizeof(T) == 4);
  constexpr short q_tgp_mem_s = q_in_registers ? 1 : BQ * (BD + padQ);

  threadgroup T Q_smem[q_tgp_mem_s];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BV,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MMAInType = metal::conditional_t<WN == 2, T, AccumType>;
  using MMAFrag_in_t = BaseMMAFrag<MMAInType, kFragSize, kFragSize>;

  static_assert(WN == 1 || WN == 2, "WN must be 1 or 2");
  static_assert(
      BD % WN == 0 && BV % WN == 0,
      "The head dims must split evenly across WN");

  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;
  constexpr int TVh = BV / (kFragSize * WN);
  constexpr int BVh = BV / WN;

  static_assert(TQ == 1, "Check TQ");

  MMATile<MMAInType, TQ, 1, MMAFrag_in_t> Qtile;
  MMATile<MMAInType, 1, TK, MMAFrag_in_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<MMAInType, 1, 1, MMAFrag_in_t> Vtile;
  MMATile<AccumType, TQ, TVh, MMAFrag_acc_t> Otile;
  MMATile<MMAInType, 1, 1, MMAFrag_in_t> Qtiles[q_in_registers ? TDh : 1];

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;
  const short tm = kFragSize * TQ * row_group;

  const short Qs_offset = (tm + sm) * LDQ_tgp + d_half * BDh + sn;
  const short Ks_offset = d_half * BDh * LDK_tgp + sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + d_half * BVh + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (!q_in_registers) {
    if (!att_align_Q && int(tid.x) == (params->NQ_aligned)) {
      loader_q.load_safe(short2(BD, params->qL_rem));
    } else {
      loader_q.load_unsafe();
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      const device T* q_src = Q + (tm + sm) * params->Q_strides[2] +
          d_half * BDh + dd * kFragSize + sn;
      if (!att_align_Q && int(tid.x) == (params->NQ_aligned)) {
        MMAFrag_in_t::load_safe(
            Qtiles[dd].frag_at(0, 0),
            q_src,
            params->Q_strides[2],
            Int<1>{},
            params->qL_rem - (tm + sm),
            BDh - (dd * kFragSize + sn),
            Int<0>{},
            Int<0>{});
      } else {
        MMAFrag_in_t::load(
            Qtiles[dd].frag_at(0, 0), q_src, params->Q_strides[2], Int<1>{});
      }
    }
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (att_has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (att_do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!att_align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      if constexpr (!q_in_registers) {
        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
      }
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      if constexpr (!q_in_registers) {
        tile_matmad(Stile, Qtile, Ktile, Stile);
      } else {
        tile_matmad(Stile, Qtiles[dd], Ktile, Stile);
      }
    }

    if constexpr (WN == 2) {
      constexpr int s_xchg_stride = decltype(Stile)::kElemsPerTile * 32;
      constexpr int s_xchg_size = WM * WN * s_xchg_stride;
      static_assert(
          s_xchg_size * sizeof(AccumType) <= tgp_mem_s * sizeof(T),
          "The score exchange must fit in KV_smem");

      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup AccumType* s_xchg =
          reinterpret_cast<threadgroup AccumType*>(KV_smem);
      threadgroup AccumType* slot =
          s_xchg + (row_group * WN + d_half) * s_xchg_stride;
      const short base = short(simd_lane_id) * decltype(Stile)::kElemsPerTile;

      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        slot[base + ii] = Stile.elems()[ii];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const threadgroup AccumType* peer =
          s_xchg + (row_group * WN + 1 - d_half) * s_xchg_stride;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] += peer[base + ii];
      }
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out length sequence
    if (!att_align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (att_do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (att_has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!att_align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BV, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TVh; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store the logsumexp for the backward pass
  if (att_save_lse && sn == 0) {
    using stile_t = decltype(Stile);
    const bool is_last_q = int(tid.x) == (params->NQ_aligned);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const short r = tm + sm + i * stile_t::kFragRows;
      if (att_align_Q || !is_last_q || r < params->qL_rem) {
        lse[r] = M_LN2_F * (max_score[i] + metal::log2(sum_score[i]));
      }
    }
  }

  // Store results
  O += (tm + sm) * params->O_strides[2] + d_half * BVh + sn;

  if (!att_align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BVh - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

} // namespace steel_attn
} // namespace mlx

template [[host_name("nuvec_sdpa_vector_float32_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 64, 64>) mlx::sdpa_vec::sdpa_vector<float, 64, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 64, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 64, 64>;
template [[host_name("nuvec_sdpa_vector_float32_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 96, 64>) mlx::sdpa_vec::sdpa_vector<float, 96, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 96, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 96, 64>;
template [[host_name("nuvec_sdpa_vector_float32_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 96, 96>) mlx::sdpa_vec::sdpa_vector<float, 96, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 96, 96>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 96, 96>;
template [[host_name("nuvec_sdpa_vector_float32_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 128, 128>) mlx::sdpa_vec::sdpa_vector<float, 128, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 128, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 128, 128>;
template [[host_name("nuvec_sdpa_vector_float32_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 192, 128>) mlx::sdpa_vec::sdpa_vector<float, 192, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 128>;
template [[host_name("nuvec_sdpa_vector_float32_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 192, 192>) mlx::sdpa_vec::sdpa_vector<float, 192, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 192>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 192>;
template [[host_name("nuvec_sdpa_vector_float32_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 256, 256>) mlx::sdpa_vec::sdpa_vector<float, 256, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 256, 256>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 256, 256>;
template [[host_name("nuvec_sdpa_vector_float32_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<float, 512, 512>) mlx::sdpa_vec::sdpa_vector<float, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 512, 512>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_float32_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 64, 64, 8, 8>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 64, 64, 8, 8>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_float32_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 8, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 8, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_12_float32_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 12, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 12, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_16_float32_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 16, 2>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<float, 128, 128, 16, 2>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 64>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 96>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 128>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 192>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 256>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float32_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<float, 512>) mlx::sdpa_vec::sdpa_vector_2pass_2<float, 512>;
template [[host_name("nuvec_sdpa_vector_bfloat16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 64, 64>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 64, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 64, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 64, 64>;
template [[host_name("nuvec_sdpa_vector_bfloat16_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 96, 64>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 96, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 96, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 96, 64>;
template [[host_name("nuvec_sdpa_vector_bfloat16_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 96, 96>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 96, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 96, 96>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 96, 96>;
template [[host_name("nuvec_sdpa_vector_bfloat16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 128, 128>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 128, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 128, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 128, 128>;
template [[host_name("nuvec_sdpa_vector_bfloat16_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 192, 128>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 192, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 192, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 192, 128>;
template [[host_name("nuvec_sdpa_vector_bfloat16_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 192, 192>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 192, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 192, 192>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 192, 192>;
template [[host_name("nuvec_sdpa_vector_bfloat16_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 256, 256>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 256, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 256, 256>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 256, 256>;
template [[host_name("nuvec_sdpa_vector_bfloat16_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<bfloat16_t, 512, 512>) mlx::sdpa_vec::sdpa_vector<bfloat16_t, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 512, 512>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_bfloat16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 64, 64, 8, 8>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 64, 64, 8, 8>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_bfloat16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 8, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 8, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_12_bfloat16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 12, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 12, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_16_bfloat16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 16, 2>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<bfloat16_t, 128, 128, 16, 2>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 64>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 96>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 128>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 192>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 256>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_2_bfloat16_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 512>) mlx::sdpa_vec::sdpa_vector_2pass_2<bfloat16_t, 512>;
template [[host_name("nuvec_sdpa_vector_float16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 64, 64>) mlx::sdpa_vec::sdpa_vector<half, 64, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 64, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 64, 64>;
template [[host_name("nuvec_sdpa_vector_float16_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 96, 64>) mlx::sdpa_vec::sdpa_vector<half, 96, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_96_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 96, 64>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 96, 64>;
template [[host_name("nuvec_sdpa_vector_float16_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 96, 96>) mlx::sdpa_vec::sdpa_vector<half, 96, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_96_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 96, 96>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 96, 96>;
template [[host_name("nuvec_sdpa_vector_float16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 128, 128>) mlx::sdpa_vec::sdpa_vector<half, 128, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 128, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 128, 128>;
template [[host_name("nuvec_sdpa_vector_float16_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 192, 128>) mlx::sdpa_vec::sdpa_vector<half, 192, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_192_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 192, 128>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 192, 128>;
template [[host_name("nuvec_sdpa_vector_float16_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 192, 192>) mlx::sdpa_vec::sdpa_vector<half, 192, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_192_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 192, 192>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 192, 192>;
template [[host_name("nuvec_sdpa_vector_float16_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 256, 256>) mlx::sdpa_vec::sdpa_vector<half, 256, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_256_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 256, 256>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 256, 256>;
template [[host_name("nuvec_sdpa_vector_float16_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector<half, 512, 512>) mlx::sdpa_vec::sdpa_vector<half, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float16_512_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<half, 512, 512>) mlx::sdpa_vec::sdpa_vector_2pass_1<half, 512, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_float16_64_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 64, 64, 8, 8>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 64, 64, 8, 8>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_8_float16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 8, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 8, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_12_float16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 12, 4>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 12, 4>;
template [[host_name("nuvec_sdpa_vector_2pass_1_gqa_16_float16_128_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 16, 2>) mlx::sdpa_vec::sdpa_vector_2pass_1_gqa<half, 128, 128, 16, 2>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_64")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 64>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 64>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_96")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 96>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 96>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_128")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 128>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 128>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_192")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 192>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 192>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_256")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 256>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 256>;
template [[host_name("nuvec_sdpa_vector_2pass_2_float16_512")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_2<half, 512>) mlx::sdpa_vec::sdpa_vector_2pass_2<half, 512>;
template [[host_name("nuvec_sdpa_vector_2pass_1_bfloat16_256_256_tk1")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 256, 256, 1>) mlx::sdpa_vec::sdpa_vector_2pass_1<bfloat16_t, 256, 256, 1>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_192_192_tk1")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 192, 1>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 192, 192, 1>;
template [[host_name("nuvec_sdpa_vector_2pass_1_float32_256_256_tk1")]] [[kernel]] decltype(mlx::sdpa_vec::sdpa_vector_2pass_1<float, 256, 256, 1>) mlx::sdpa_vec::sdpa_vector_2pass_1<float, 256, 256, 1>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd192_bv192_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 192, 192, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 16, 192, 192, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd192_bv192_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 192, 192, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 16, 192, 192, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd128_bv128_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 128, 128, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 16, 128, 128, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd128_bv128_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 128, 128, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 16, 128, 128, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd96_bv96_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 96, 96, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 32, 96, 96, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd96_bv96_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 96, 96, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 32, 96, 96, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd96_bv64_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 96, 64, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 32, 96, 64, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd96_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 96, 64, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 32, 96, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd80_bv80_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 80, 80, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 32, 80, 80, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd80_bv80_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 80, 80, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 32, 80, 80, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd72_bv72_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 72, 72, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 32, 72, 72, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd72_bv72_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 72, 72, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 32, 72, 72, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 64, 64, 4, 1, half, float>) mlx::steel_attn::attention<half, 32, 32, 64, 64, 4, 1, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd64_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 64, 64, 4, 1, bool, float>) mlx::steel_attn::attention<half, 32, 32, 64, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd256_bv256_wm4_wn2_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 256, 256, 4, 2, half, float>) mlx::steel_attn::attention<half, 32, 16, 256, 256, 4, 2, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk16_bd256_bv256_wm4_wn2_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 16, 256, 256, 4, 2, bool, float>) mlx::steel_attn::attention<half, 32, 16, 256, 256, 4, 2, bool, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd256_bv256_wm4_wn2_maskfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 256, 256, 4, 2, half, float>) mlx::steel_attn::attention<half, 32, 32, 256, 256, 4, 2, half, float>;
template [[host_name("nuvec_steel_attention_float16_bq32_bk32_bd256_bv256_wm4_wn2_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<half, 32, 32, 256, 256, 4, 2, bool, float>) mlx::steel_attn::attention<half, 32, 32, 256, 256, 4, 2, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd192_bv192_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 192, 192, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 192, 192, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd192_bv192_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 192, 192, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 192, 192, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd128_bv128_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 128, 128, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 128, 128, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd128_bv128_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 128, 128, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 128, 128, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd96_bv96_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 96, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 96, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd96_bv96_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 96, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 96, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd96_bv64_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 64, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 64, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd96_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 64, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 96, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd80_bv80_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 80, 80, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 80, 80, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd80_bv80_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 80, 80, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 80, 80, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd72_bv72_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 72, 72, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 72, 72, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd72_bv72_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 72, 72, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 72, 72, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd64_bv64_wm4_wn1_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 64, 64, 4, 1, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 64, 64, 4, 1, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd64_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 64, 64, 4, 1, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 64, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd256_bv256_wm4_wn2_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 256, 256, 4, 2, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 256, 256, 4, 2, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk16_bd256_bv256_wm4_wn2_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 16, 256, 256, 4, 2, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 16, 256, 256, 4, 2, bool, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd256_bv256_wm4_wn2_maskbfloat16")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 256, 256, 4, 2, bfloat16_t, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 256, 256, 4, 2, bfloat16_t, float>;
template [[host_name("nuvec_steel_attention_bfloat16_bq32_bk32_bd256_bv256_wm4_wn2_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<bfloat16_t, 32, 32, 256, 256, 4, 2, bool, float>) mlx::steel_attn::attention<bfloat16_t, 32, 32, 256, 256, 4, 2, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd192_bv192_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 192, 192, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 16, 192, 192, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd192_bv192_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 192, 192, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 16, 192, 192, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd128_bv128_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 128, 128, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 16, 128, 128, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd128_bv128_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 128, 128, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 16, 128, 128, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd96_bv96_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 96, 96, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 32, 96, 96, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd96_bv96_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 96, 96, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 32, 96, 96, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd96_bv64_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 96, 64, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 32, 96, 64, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd96_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 96, 64, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 32, 96, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd80_bv80_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 80, 80, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 32, 80, 80, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd80_bv80_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 80, 80, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 32, 80, 80, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd72_bv72_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 72, 72, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 32, 72, 72, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd72_bv72_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 72, 72, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 32, 72, 72, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 64, 64, 4, 1, float, float>) mlx::steel_attn::attention<float, 32, 32, 64, 64, 4, 1, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 32, 64, 64, 4, 1, bool, float>) mlx::steel_attn::attention<float, 32, 32, 64, 64, 4, 1, bool, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd256_bv256_wm4_wn2_maskfloat32")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 256, 256, 4, 2, float, float>) mlx::steel_attn::attention<float, 32, 16, 256, 256, 4, 2, float, float>;
template [[host_name("nuvec_steel_attention_float32_bq32_bk16_bd256_bv256_wm4_wn2_maskbool")]] [[kernel]] decltype(mlx::steel_attn::attention<float, 32, 16, 256, 256, 4, 2, bool, float>) mlx::steel_attn::attention<float, 32, 16, 256, 256, 4, 2, bool, float>;
