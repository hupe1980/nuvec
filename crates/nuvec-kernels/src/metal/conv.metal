// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Depthwise 1-D convolution (nuvec's own): `out[n, t, c] = Σ_j x[n, t·stride + j·dilation −
// padding, c] · w[c, j]`, positions outside the input reading zero. Products and sums are
// float32, in tap order, without contraction (this file is compiled with
// `-ffp-contract=off`), so the result equals the composition of `conv1d`'s fallback.
// One thread per output element; channels are the fastest grid dimension, so reads of
// neighbouring threads are contiguous.

#include <metal_stdlib>

#include "common.h"

using namespace metal;

typedef bfloat bfloat16_t;

template <typename T>
[[kernel]] void depthwise_conv1d(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int* sizes [[buffer(3)]], // L, C, K, L_out
    constant int* params [[buffer(4)]], // stride, padding, dilation
    uint3 pos [[thread_position_in_grid]]) {
  const int l = sizes[0], c = sizes[1], k = sizes[2], l_out = sizes[3];
  const int stride = params[0], padding = params[1], dilation = params[2];
  const int ch = pos.x, t = pos.y, n = pos.z;
  const device T* xn = x + size_t(n) * l * c + ch;
  const device T* wc = w + size_t(ch) * k;
  float sum = 0;
  for (int j = 0; j < k; ++j) {
    int p = t * stride + j * dilation - padding;
    float xv = (p >= 0 && p < l) ? static_cast<float>(xn[size_t(p) * c]) : 0.0f;
    float term = xv * static_cast<float>(wc[j]);
    sum = j == 0 ? term : sum + term;
  }
  out[(size_t(n) * l_out + t) * c + ch] = static_cast<T>(sum);
}

// `causal_conv1d_slots` (nuvec's own): for sequence `n` at pool slot `slots[n]`, the window
// `pool[slot, 0..K − 1, c]` then `x[n, 0..L, c]`, convolved as `depthwise_conv1d` (taps in
// order, float32, uncontracted, rounded once) and, with `silu`, `v · sigmoid(v)` as the
// element-wise kernels round it; then the window's last `K − 1` entries written back to
// the slot. One thread per (channel, sequence) owns its channel's window, so the slot is
// read before it is written. Slots outside `0..S` read slot `S − 1` clamped and write
// nothing (the paged cache never passes them).
static constant constexpr int CONV_SLOTS_WINDOW = 64;

template <typename T>
[[kernel]] void causal_conv1d_slots(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* pool [[buffer(2)]],
    const device int* slots [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int* sizes [[buffer(5)]], // L, C, K, S
    constant int& silu [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int l = sizes[0], c = sizes[1], k = sizes[2], s = sizes[3];
  const int ch = pos.x, n = pos.y;
  const int slot = slots[n];
  const bool valid = slot >= 0 && slot < s;
  const size_t at = size_t(metal::clamp(slot, 0, s - 1)) * (k - 1) * c + ch;
  T win[CONV_SLOTS_WINDOW];
  for (int i = 0; i < k - 1; ++i) {
    win[i] = pool[at + size_t(i) * c];
  }
  const device T* xn = x + size_t(n) * l * c + ch;
  for (int t = 0; t < l; ++t) {
    win[k - 1 + t] = xn[size_t(t) * c];
  }
  const device T* wc = w + size_t(ch) * k;
  for (int t = 0; t < l; ++t) {
    float sum = 0;
    for (int j = 0; j < k; ++j) {
      float term = static_cast<float>(win[t + j]) * static_cast<float>(wc[j]);
      sum = j == 0 ? term : sum + term;
    }
    T v = static_cast<T>(sum);
    if (silu) {
      T g = T(nuvec::sigmoid_(float(v)));
      v = T(float(v) * float(g));
    }
    out[(size_t(n) * l + t) * c + ch] = v;
  }
  if (valid) {
    for (int i = 0; i < k - 1; ++i) {
      pool[at + size_t(i) * c] = win[l + i];
    }
  }
}

template [[host_name("nuvec_causal_conv1d_slots_float32")]] [[kernel]] decltype(causal_conv1d_slots<float>) causal_conv1d_slots<float>;
template [[host_name("nuvec_causal_conv1d_slots_float16")]] [[kernel]] decltype(causal_conv1d_slots<half>) causal_conv1d_slots<half>;
template [[host_name("nuvec_causal_conv1d_slots_bfloat16")]] [[kernel]] decltype(causal_conv1d_slots<bfloat16_t>) causal_conv1d_slots<bfloat16_t>;

template [[host_name("nuvec_depthwise_conv1d_float32")]] [[kernel]] decltype(depthwise_conv1d<float>) depthwise_conv1d<float>;
template [[host_name("nuvec_depthwise_conv1d_float16")]] [[kernel]] decltype(depthwise_conv1d<half>) depthwise_conv1d<half>;
template [[host_name("nuvec_depthwise_conv1d_bfloat16")]] [[kernel]] decltype(depthwise_conv1d<bfloat16_t>) depthwise_conv1d<bfloat16_t>;
