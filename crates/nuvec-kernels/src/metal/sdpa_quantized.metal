// SPDX-License-Identifier: MIT
//
// nuvec's own: one query row's attention over an affine-quantized key/value cache (mlx-lm's
// `kv_bits` layout: per position, `D·bits/32` words of codes (3-bit codes packed 8 to 3
// bytes) and one scale and bias per `GS` values), the first of two passes; MLX's `sdpa_vector_2pass_2` reduces the partials.
//
// One threadgroup of `TG` threads serves the `G` query heads of a key head over a range of
// keys, in tiles of `TG` keys, without a SIMD reduction per key:
// - scores: each thread takes one key of the tile whole and scores it against every head
//   (`s · Σ q·code + b · Σ q` per group; the scaled queries and their group sums sit in
//   threadgroup memory, read as broadcasts);
// - softmax: the tile's maximum per head (one `simd_max` per SIMD-group and head), then each
//   thread's probabilities into threadgroup memory; the running sums stay per thread;
// - values: thread `t` takes one word of the value rows (`PACK` features, dequantized once
//   for all heads) for a stripe of the tile's keys (`t / WORDS`, every `STRIPES`-th key);
//   the stripes are summed once per block (the rescaling is the same for every thread).
//
// Grid: (key heads, batch, blocks); block `b` takes keys `b·chunk .. (b + 1)·chunk`.
// Partials: `out` `[B·Hq, blocks, D]` (unnormalised), `sums`, `maxs` `[B·Hq, blocks]`, as
// `sdpa_vector_2pass_1`'s. With `SINKS`, block 0 starts each head's softmax at its sink.

#include <metal_stdlib>
using namespace metal;

// A unit of codes: a 32-bit word (`32 / BITS` codes) at 4 and 8 bits; 3 bytes (8 codes,
// MLX's 3-bit packing) at 3 bits. Code `c` of a unit is bits `BITS·c …` of its value.
template <int BITS>
struct Unit {
  static constant constexpr int values = 32 / BITS;
  static constant constexpr int bytes = 4;
  static uint32_t load(const device uint8_t* p) {
    return *reinterpret_cast<const device uint32_t*>(p);
  }
};

template <>
struct Unit<3> {
  static constant constexpr int values = 8;
  static constant constexpr int bytes = 3;
  static uint32_t load(const device uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
  }
};

template <typename T, int D, int BITS, int GS, int G, bool SINKS>
[[kernel, max_total_threads_per_threadgroup(128)]] void sdpa_vector_q_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* kw [[buffer(1)]],
    const device T* ks [[buffer(2)]],
    const device T* kb [[buffer(3)]],
    const device uint32_t* vw [[buffer(4)]],
    const device T* vs [[buffer(5)]],
    const device T* vb [[buffer(6)]],
    device T* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    // Per (batch, key head) and per position: codes in 32-bit words, scales/biases in
    // elements.
    const constant int64_t* strides [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device T* sinks [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int TG = 128;
  constexpr int SG = TG / 32;
  constexpr int PACK = Unit<BITS>::values;  // codes per unit
  constexpr int UB = Unit<BITS>::bytes;     // bytes per unit
  constexpr int WORDS = D / PACK;           // units per key
  constexpr int GROUPS = D / GS;            // scale groups per key
  constexpr int WPG = GS / PACK;            // units per group
  constexpr int STRIPES = TG / WORDS;       // key stripes of the value pass
  constexpr uint MASK = (1u << BITS) - 1;
  typedef float U;

  threadgroup U q_sh[G][D];
  threadgroup U qsum_sh[G][GROUPS];
  threadgroup U p_sh[G][TG];
  threadgroup U max_sh[G][SG];
  threadgroup U red_sh[G][D];

  const int kv_head = tid.x;
  const int batch = tid.y;
  const int block = tid.z;
  const int blocks = tpg.z;
  const int hkv = tpg.x;
  const int hq = hkv * G;
  const int head0 = batch * hq + kv_head * G;
  const int kvh = batch * hkv + kv_head;

  for (int i = t; i < G * D; i += TG) {
    q_sh[i / D][i % D] = static_cast<U>(scale) * static_cast<U>(queries[head0 * D + i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = t; i < G * GROUPS; i += TG) {
    U s = 0;
    for (int e = 0; e < GS; e++) {
      s += q_sh[i / GROUPS][(i % GROUPS) * GS + e];
    }
    qsum_sh[i / GROUPS][i % GROUPS] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The value unit and key stripe this thread takes.
  const int vword = int(t) % WORDS;
  const int stripe = int(t) / WORDS;
  const int vgroup = (vword * PACK) / GS;

  const int chunk = (N + blocks - 1) / blocks;
  const int s0 = block * chunk;
  const int s1 = min(N, s0 + chunk);

  U m[G];
  U l[G];
  for (int j = 0; j < G; j++) {
    m[j] = -INFINITY;
    l[j] = 0;
  }
  if (SINKS && block == 0) {
    for (int j = 0; j < G; j++) {
      m[j] = static_cast<U>(sinks[(head0 + j) % hq]);
      l[j] = t == 0 ? U(1) : U(0);
    }
  }
  U o[G][PACK];
  for (int j = 0; j < G; j++) {
    for (int c = 0; c < PACK; c++) {
      o[j][c] = 0;
    }
  }

  const device uint8_t* kbase =
      reinterpret_cast<const device uint8_t*>(kw + kvh * strides[0]);
  const device T* ksb = ks + kvh * strides[2];
  const device T* kbb = kb + kvh * strides[2];
  const device uint8_t* vbase =
      reinterpret_cast<const device uint8_t*>(vw + kvh * strides[4]) + vword * UB;
  const device T* vsb = vs + kvh * strides[6] + vgroup;
  const device T* vbb = vb + kvh * strides[6] + vgroup;

  for (int base = s0; base < s1; base += TG) {
    const int key = base + int(t);
    const bool valid = key < s1;
    // Scores of this thread's key for every head.
    U score[G];
    for (int j = 0; j < G; j++) {
      score[j] = 0;
    }
    if (valid) {
      const device uint8_t* kp = kbase + key * strides[1] * 4;
      for (int g = 0; g < GROUPS; g++) {
        U acc[G];
        for (int j = 0; j < G; j++) {
          acc[j] = 0;
        }
        for (int w = 0; w < WPG; w++) {
          const uint32_t word = Unit<BITS>::load(kp + (g * WPG + w) * UB);
          for (int c = 0; c < PACK; c++) {
            const U code = static_cast<U>((word >> (BITS * c)) & MASK);
            const int e = (g * WPG + w) * PACK + c;
            for (int j = 0; j < G; j++) {
              acc[j] += q_sh[j][e] * code;
            }
          }
        }
        const U sc = static_cast<U>(ksb[key * strides[3] + g]);
        const U bi = static_cast<U>(kbb[key * strides[3] + g]);
        for (int j = 0; j < G; j++) {
          score[j] += sc * acc[j] + bi * qsum_sh[j][g];
        }
      }
    }
    // The tile's maximum per head.
    for (int j = 0; j < G; j++) {
      const U mx = simd_max(valid ? score[j] : -INFINITY);
      if (simd_lid == 0) {
        max_sh[j][simd_gid] = mx;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U factor[G];
    for (int j = 0; j < G; j++) {
      U mx = m[j];
      for (int s = 0; s < SG; s++) {
        mx = max(mx, max_sh[j][s]);
      }
      factor[j] = mx == m[j] ? U(1) : fast::exp(m[j] - mx);
      m[j] = mx;
      const U p = valid ? fast::exp(score[j] - mx) : U(0);
      l[j] = l[j] * factor[j] + p;
      p_sh[j][t] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Values: this thread's word, over its stripe of the tile's keys.
    for (int j = 0; j < G; j++) {
      for (int c = 0; c < PACK; c++) {
        o[j][c] *= factor[j];
      }
    }
    const int n = min(TG, s1 - base);
    for (int k = stripe; k < n; k += STRIPES) {
      const int kk = base + k;
      const U sc = static_cast<U>(vsb[kk * strides[7]]);
      const U bi = static_cast<U>(vbb[kk * strides[7]]);
      const uint32_t word = Unit<BITS>::load(vbase + kk * strides[5] * 4);
      U v[PACK];
      for (int c = 0; c < PACK; c++) {
        v[c] = sc * static_cast<U>((word >> (BITS * c)) & MASK) + bi;
      }
      for (int j = 0; j < G; j++) {
        const U p = p_sh[j][k];
        for (int c = 0; c < PACK; c++) {
          o[j][c] += p * v[c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Each head's sum over the threads.
  for (int j = 0; j < G; j++) {
    const U s = simd_sum(l[j]);
    if (simd_lid == 0) {
      max_sh[j][simd_gid] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < uint(G)) {
    U s = 0;
    for (int k = 0; k < SG; k++) {
      s += max_sh[t][k];
    }
    const int row = (head0 + int(t)) * blocks + block;
    sums[row] = s;
    maxs[row] = m[t] == -INFINITY ? -3.40282347e38f : m[t];
  }
  // The stripes' sums, one stripe at a time through threadgroup memory into stripe 0's.
  for (int r = 1; r < STRIPES; r++) {
    if (stripe == r) {
      for (int j = 0; j < G; j++) {
        for (int c = 0; c < PACK; c++) {
          red_sh[j][vword * PACK + c] = o[j][c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stripe == 0) {
      for (int j = 0; j < G; j++) {
        for (int c = 0; c < PACK; c++) {
          o[j][c] += red_sh[j][vword * PACK + c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (stripe == 0) {
    for (int j = 0; j < G; j++) {
      const int row = (head0 + j) * blocks + block;
      for (int c = 0; c < PACK; c++) {
        out[row * D + vword * PACK + c] = static_cast<T>(o[j][c]);
      }
    }
  }
}

#define NUVEC_SDPA_Q(tn, T, d, b, gs, g)                                                       \
  template [[host_name("nuvec_sdpa_vector_q_2pass_1_" #tn "_" #d "_b" #b "_gs" #gs "_g" #g)]] \
  [[kernel]] decltype(sdpa_vector_q_2pass_1<T, d, b, gs, g, false>)                            \
      sdpa_vector_q_2pass_1<T, d, b, gs, g, false>;                                            \
  template [[host_name("nuvec_sdpa_vector_q_2pass_1_" #tn "_" #d "_b" #b "_gs" #gs "_g" #g    \
                       "_sinks")]] [[kernel]]                                                  \
  decltype(sdpa_vector_q_2pass_1<T, d, b, gs, g, true>)                                        \
      sdpa_vector_q_2pass_1<T, d, b, gs, g, true>;

#define NUVEC_SDPA_Q_G(tn, T, d, b, gs) \
  NUVEC_SDPA_Q(tn, T, d, b, gs, 1)      \
  NUVEC_SDPA_Q(tn, T, d, b, gs, 2)      \
  NUVEC_SDPA_Q(tn, T, d, b, gs, 4)      \
  NUVEC_SDPA_Q(tn, T, d, b, gs, 8)

#define NUVEC_SDPA_Q_GS(tn, T, d, b) \
  NUVEC_SDPA_Q_G(tn, T, d, b, 32)    \
  NUVEC_SDPA_Q_G(tn, T, d, b, 64)

// Groups of 128 only for head size 128 (a group never spans heads).
#define NUVEC_SDPA_Q_D(tn, T)              \
  NUVEC_SDPA_Q_GS(tn, T, 64, 3)            \
  NUVEC_SDPA_Q_GS(tn, T, 64, 4)            \
  NUVEC_SDPA_Q_GS(tn, T, 64, 8)            \
  NUVEC_SDPA_Q_GS(tn, T, 128, 3)           \
  NUVEC_SDPA_Q_GS(tn, T, 128, 4)           \
  NUVEC_SDPA_Q_GS(tn, T, 128, 8)           \
  NUVEC_SDPA_Q_G(tn, T, 128, 3, 128)       \
  NUVEC_SDPA_Q_G(tn, T, 128, 4, 128)       \
  NUVEC_SDPA_Q_G(tn, T, 128, 8, 128)

NUVEC_SDPA_Q_D(float32, float)
NUVEC_SDPA_Q_D(float16, half)
NUVEC_SDPA_Q_D(bfloat16, bfloat)
