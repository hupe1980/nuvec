// SPDX-License-Identifier: MIT
//
// Paged attention (nuvec's own; head sizes 64, 128, 256 and 512) on MLX v0.32.4's vector attention (`sdpa_vector` and the
// first pass of `sdpa_vector_2pass`, sdpa.metal): keys and values live in blocks of
// `block_size` (a power of two) positions,
// `[blocks, kv_heads, block_size, D]`, and each sequence of a batch reads its keys through
// its row of a block table, up to its own length. The arithmetic is MLX's, so a
// sequence's result equals the same kernel over the same keys laid out contiguously. Also
// the write of new keys or values into their slots.
//
// Copyright © 2024-26 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "fast.h"

// Attention sinks (one logit per query head in the softmax) and a sliding window (each
// query attends its last `window` keys; earlier keys, whose blocks may already belong to
// other sequences, are never read), as `sdpa_vector`'s.
constant bool pa_has_sinks [[function_constant(40)]];
constant bool pa_has_window [[function_constant(41)]];

// Out-of-range inputs never reach outside the caches (`indexing.metal`'s clamp rule): a
// sequence's length is clamped to the table's capacity, `max_blocks * block_size` (so the
// last key a query attends is at most `capacity - 1`), and block ids to `0..num_blocks`.
// The host keeps `2 * capacity` within an `int`.
inline int paged_last(int n, int lq, int l, int max_blocks, int shift) {
  return clamp(n, 0, max_blocks << shift) - lq + l;
}

inline uint paged_block(const device int* table, int i, int shift, int num_blocks) {
  return uint(clamp(table[i >> shift], 0, num_blocks - 1));
}

// Queries `[B, Hq, L, D]` (row-contiguous), output likewise. `params`: Hq, Hkv, L,
// log2(block_size), max_blocks, num_blocks. Query `l` of a sequence of length `N` attends
// keys `0..=N - L + l` (causal at the end, as `sdpa_vector`). One threadgroup of 32
// SIMD-groups per (sequence, query head, query).
// The declared limit keeps 1,024 threads per threadgroup available where instrumented
// builds (shader validation) or smaller register files would otherwise lower it.
template <typename T, int D, bool W = false>
[[kernel, max_total_threads_per_threadgroup(1024)]] void paged_attention(
    const device T* queries [[buffer(0)]],
    const device T* k_cache [[buffer(1)]],
    const device T* v_cache [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device int* block_tables [[buffer(4)]],
    const device int* lengths [[buffer(5)]],
    const constant int* params [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const device T* sinks [[buffer(8), function_constant(pa_has_sinks)]],
    const constant int& window [[buffer(9), function_constant(pa_has_window)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int hq = params[0], hkv = params[1], lq = params[2], shift = params[3],
            max_blocks = params[4], num_blocks = params[5];
  const int block_size = 1 << shift, within = block_size - 1;
  // 32-bit offsets (the host keeps caches under 2^32 elements): Apple GPUs emulate 64-bit
  // integer products, which would cost more than the keys' arithmetic.
  const uint page_stride = uint(hkv) << (shift + __builtin_ctz(uint(D)));

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int b = tid.x / hq, head = tid.x % hq, l = tid.y;
  const int kv_head = head / (hq / hkv);
  const int last = paged_last(lengths[b], lq, l, max_blocks, shift);
  const device int* table = block_tables + size_t(b) * max_blocks;
  const uint head_offset = uint(kv_head) * uint(block_size * D);
  const size_t row = (size_t(b) * hq + head) * lq + l;
  queries += row * D + simd_lid * qk_per_thread;
  out += row * D + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = mlx::Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (pa_has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[head]);
    sum_exp_score = 1;
  }

  // A window's keys are assigned to SIMD groups from its first key, as `sdpa_vector`'s.
  int first = simd_gid;
  if (W) {
    first += max(last - window + 1, 0);
  }

  for (int i = first; i <= last; i += BN) {
    const uint at = paged_block(table, i, shift, num_blocks) * page_stride + head_offset +
        (uint(i & within) * uint(D)) + simd_lid * qk_per_thread;
    const device T* keys = k_cache + at;
    const device T* values = v_cache + at;
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// MLX's first pass of the two-pass vector attention (`sdpa_vector_2pass_1`) over a paged
// cache: threadgroup `(kv_head, sequence, block)` with one SIMD-group per query head of
// the group and query (`[32, G, L]` threads); key `i` of a sequence is visited by block
// `i % blocks`, read through the block table. Writes MLX's partials layout (outputs, sums
// and maxima per block), reduced by `sdpa_vector_2pass_2`. `params`: Hkv, L,
// log2(block_size), max_blocks, blocks, num_blocks.
template <typename T, int D, int TK = 4, bool W = false>
[[kernel, max_total_threads_per_threadgroup(1024)]] void paged_attention_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* k_cache [[buffer(1)]],
    const device T* v_cache [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const device int* block_tables [[buffer(6)]],
    const device int* lengths [[buffer(7)]],
    const constant int* params [[buffer(8)]],
    const constant float& scale [[buffer(9)]],
    const device T* sinks [[buffer(10), function_constant(pa_has_sinks)]],
    const constant int& window [[buffer(11), function_constant(pa_has_window)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int hkv = params[0], lq = params[1], shift = params[2], max_blocks = params[3],
            blocks = params[4], num_blocks = params[5];
  const int block_size = 1 << shift, within = block_size - 1;
  const uint page_stride = uint(hkv) << (shift + __builtin_ctz(uint(D)));

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  const int kv_head = tid.x, b = tid.y, block_idx = tid.z;
  const int gqa = tptg.y, l = tidtg.z;
  const int head = gqa * kv_head + tidtg.y;
  const int hq = hkv * gqa;
  const int last = paged_last(lengths[b], lq, l, max_blocks, shift);
  const device int* table = block_tables + size_t(b) * max_blocks;
  const size_t row = (size_t(b) * hq + head) * lq + l;

  queries += row * D + simd_lid * qk_per_thread;
  out += row * blocks * D + block_idx * D + simd_lid * v_per_thread;
  sums += row * blocks + block_idx;
  maxs += row * blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = mlx::Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (pa_has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[head]);
    sum_exp_score = 1;
  }

  // The offset of key `i` (this head) in the caches.
  const uint head_offset = uint(kv_head) * uint(block_size * D) + simd_lid * qk_per_thread;
  auto at = [&](int i) {
    return paged_block(table, i, shift, num_blocks) * page_stride + head_offset +
        uint(i & within) * uint(D);
  };

  // A window's keys are partitioned from its first key, `lo`, as `sdpa_vector_2pass_1`'s.
  int i = block_idx;
  const int lo = W ? max(last - window + 1, 0) : 0;
  i += lo;
  for (; i + (TK - 1) * blocks <= last; i += blocks * TK) {
    // Prefetch: issue all TK K/V loads first so the hardware can overlap them.
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; t++) {
      const device T* kp = k_cache + at(i + t * blocks);
      for (int j = 0; j < qk_per_thread; j++) {
        k_reg[t][j] = kp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = v_cache + at(i + t * blocks);
      for (int j = 0; j < v_per_thread; j++) {
        v_reg[t][j] = vp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      score = simd_sum(score);
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

  // Tail (fewer than TK remaining keys).
  for (; i <= last; i += blocks) {
    const device T* keys = k_cache + at(i);
    const device T* values = v_cache + at(i);
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * keys[j];
    }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int j = 0; j < v_per_thread; j++) {
    out[j] = static_cast<T>(o[j]);
  }
}

// `cache[slot / block_size, h, slot % block_size, :] = new[b, t, h, :]` for each new
// position `(b, t)` with its `slots[b · T + t]`; one thread per element, `new` read through
// its strides (`src_strides`: b, t, h, e; a strided view needs no copy). `params`:
// kv_heads, block_size, D, num_blocks, T. Slots outside `0..num_blocks * block_size` are
// skipped.
template <typename T>
[[kernel]] void paged_write(
    const device T* src [[buffer(0)]],
    const device int* slots [[buffer(1)]],
    device T* cache [[buffer(2)]],
    const constant int* params [[buffer(3)]],
    const constant int64_t* src_strides [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int hkv = params[0], block_size = params[1], d = params[2], num_blocks = params[3];
  const int length = params[4];
  const int bt = pos.z, h = pos.y, e = pos.x;
  const int b = bt / length, t = bt % length;
  const int slot = slots[bt];
  if (slot < 0 || ulong(slot) >= ulong(num_blocks) * ulong(block_size)) {
    return;
  }
  const size_t at =
      ((size_t(slot / block_size) * hkv + h) * block_size + slot % block_size) * d + e;
  cache[at] = src[b * src_strides[0] + t * src_strides[1] + h * src_strides[2] +
                  e * src_strides[3]];
}

#define instantiate_paged_w(tname, type, w, suffix)                                    \
  template [[host_name("nuvec_paged_attention_" #tname "_64" suffix)]] [[kernel]]      \
  decltype(paged_attention<type, 64, w>) paged_attention<type, 64, w>;                 \
  template [[host_name("nuvec_paged_attention_" #tname "_128" suffix)]] [[kernel]]     \
  decltype(paged_attention<type, 128, w>) paged_attention<type, 128, w>;               \
  template [[host_name("nuvec_paged_attention_" #tname "_256" suffix)]] [[kernel]]     \
  decltype(paged_attention<type, 256, w>) paged_attention<type, 256, w>;               \
  template [[host_name("nuvec_paged_attention_" #tname "_512" suffix)]] [[kernel]]     \
  decltype(paged_attention<type, 512, w>) paged_attention<type, 512, w>;               \
  template [[host_name("nuvec_paged_attention_2pass_1_" #tname "_64" suffix)]]         \
  [[kernel]] decltype(paged_attention_2pass_1<type, 64, 4, w>)                         \
  paged_attention_2pass_1<type, 64, 4, w>;                                             \
  template [[host_name("nuvec_paged_attention_2pass_1_" #tname "_128" suffix)]]        \
  [[kernel]] decltype(paged_attention_2pass_1<type, 128, 4, w>)                        \
  paged_attention_2pass_1<type, 128, 4, w>;                                            \
  template [[host_name("nuvec_paged_attention_2pass_1_" #tname "_256" suffix)]]        \
  [[kernel]] decltype(paged_attention_2pass_1<type, 256, 1, w>)                        \
  paged_attention_2pass_1<type, 256, 1, w>;                                            \
  template [[host_name("nuvec_paged_attention_2pass_1_" #tname "_512" suffix)]]        \
  [[kernel]] decltype(paged_attention_2pass_1<type, 512, 1, w>)                        \
  paged_attention_2pass_1<type, 512, 1, w>;

// The window is a template parameter (`_w` kernels), not a function constant in the loops.
#define instantiate_paged(tname, type)                                                 \
  instantiate_paged_w(tname, type, false, "")                                          \
  instantiate_paged_w(tname, type, true, "_w")                                         \
  template [[host_name("nuvec_paged_write_" #tname)]] [[kernel]]                       \
  decltype(paged_write<type>) paged_write<type>;

instantiate_paged(float32, float)
instantiate_paged(float16, half)
instantiate_paged(bfloat16, bfloat16_t)
