// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Depthwise 1-D convolution (nuvec's own): `out[n, t, c] = Σ_j x[n, t·stride + j·dilation −
// padding, c] · w[c, j]`, positions outside the input reading zero. Products and sums are
// float32, in tap order, without contraction (this file is compiled with
// `-ffp-contract=off`), so the result equals the composition of `conv1d`'s fallback.
// One thread per output element; channels are the fastest grid dimension, so reads of
// neighbouring threads are contiguous.

#include <metal_stdlib>

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

template [[host_name("nuvec_depthwise_conv1d_float32")]] [[kernel]] decltype(depthwise_conv1d<float>) depthwise_conv1d<float>;
template [[host_name("nuvec_depthwise_conv1d_float16")]] [[kernel]] decltype(depthwise_conv1d<half>) depthwise_conv1d<half>;
template [[host_name("nuvec_depthwise_conv1d_bfloat16")]] [[kernel]] decltype(depthwise_conv1d<bfloat16_t>) depthwise_conv1d<bfloat16_t>;
