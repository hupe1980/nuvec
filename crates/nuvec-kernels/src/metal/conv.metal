// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Convolutions (nuvec's own), channels last: `x` `[N, spatial…, C_in]`, `w` `[C_out, kernel…,
// C_in / groups]`, one to three spatial axes (unused axes have size 1). Per axis, kernel
// offset `k` of output `o` reads the position `q = o·stride + k·kernel_dilation − pad_lo` of
// the input dilated by `input_dilation`: input element `q / input_dilation` when it divides
// and lies inside the input, else zero. `flip` reverses the kernel. Products and sums are
// float32 without contraction (this file is compiled with `-ffp-contract=off`).

#include <metal_stdlib>

#include "common.h"

using namespace metal;

typedef bfloat bfloat16_t;

// The geometry, as `int`s: input sizes, output sizes, kernel sizes, then per-axis stride,
// pad_lo, kernel dilation and input dilation (three each), then C_in, C_out, groups, flip.
struct ConvGeometry {
  int in[3];
  int out[3];
  int ksize[3];
  int stride[3];
  int pad[3];
  int kdil[3];
  int idil[3];
  int c_in;
  int c_out;
  int groups;
  int flip;
};

// The input offset (in pixels, `-1` for a zero) of output pixel `o` at kernel offset `k`.
static inline int conv_source(constant ConvGeometry& g, const thread int* o, const thread int* k) {
  int at = 0;
  for (int i = 0; i < 3; ++i) {
    int kk = g.flip ? g.ksize[i] - 1 - k[i] : k[i];
    int q = o[i] * g.stride[i] + kk * g.kdil[i] - g.pad[i];
    if (g.idil[i] != 1) {
      if (q % g.idil[i] != 0) {
        return -1;
      }
      q /= g.idil[i];
    }
    if (q < 0 || q >= g.in[i]) {
      return -1;
    }
    at = at * g.in[i] + q;
  }
  return at;
}

// Depthwise (`groups = C_in`, `C_out = m · C_in`): output channel `co` convolves input channel
// `co / m` with its own kernel, the taps summed in order (zeros included), so the result
// equals the composition's bit for bit. One thread per output element; channels are the
// fastest grid dimension, so neighbouring threads read contiguously.
template <typename T>
[[kernel]] void depthwise_conv(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant ConvGeometry& g [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int co = pos.x, n = pos.z;
  int pix = pos.y;
  int o[3];
  for (int i = 2; i >= 0; --i) {
    o[i] = pix % g.out[i];
    pix /= g.out[i];
  }
  const int m = g.c_out / g.c_in;
  const int ci = co / m;
  const int in_pixels = g.in[0] * g.in[1] * g.in[2];
  const int taps = g.ksize[0] * g.ksize[1] * g.ksize[2];
  const device T* xn = x + size_t(n) * in_pixels * g.c_in + ci;
  const device T* wc = w + size_t(co) * taps;
  float sum = 0;
  int k[3] = {0, 0, 0};
  for (int t = 0; t < taps; ++t) {
    int at = conv_source(g, o, k);
    float xv = at >= 0 ? static_cast<float>(xn[size_t(at) * g.c_in]) : 0.0f;
    float term = xv * static_cast<float>(wc[t]);
    sum = t == 0 ? term : sum + term;
    for (int i = 2; i >= 0; --i) {
      if (++k[i] < g.ksize[i]) {
        break;
      }
      k[i] = 0;
    }
  }
  const int out_pixels = g.out[0] * g.out[1] * g.out[2];
  out[(size_t(n) * out_pixels + pos.y) * g.c_out + co] = static_cast<T>(sum);
}

// The columns of the general convolution (im2col): `cols[group, row, tap, c]` is input channel
// `group · C_in / groups + c` at tap `tap` of output row `row` (`n` and the output pixel),
// zero outside the input; one GEMM per group with `w` then gives the output. A copy: no
// arithmetic, so the GEMM's float32 accumulation is the only rounding.
template <typename T>
[[kernel]] void conv_unfold(
    const device T* x [[buffer(0)]],
    device T* cols [[buffer(1)]],
    constant ConvGeometry& g [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // x: the channel in its group; y: the tap; z: the row (groups · rows rows).
  const int cg = g.c_in / g.groups;
  const int taps = g.ksize[0] * g.ksize[1] * g.ksize[2];
  const int c = pos.x;
  int t = pos.y;
  const int out_pixels = g.out[0] * g.out[1] * g.out[2];
  const int rows = int(grid.z) / g.groups;
  const int group = int(pos.z) / rows, row = int(pos.z) % rows;
  const int n = row / out_pixels;
  int pix = row % out_pixels;
  int o[3], k[3];
  for (int i = 2; i >= 0; --i) {
    o[i] = pix % g.out[i];
    pix /= g.out[i];
    k[i] = t % g.ksize[i];
    t /= g.ksize[i];
  }
  const int in_pixels = g.in[0] * g.in[1] * g.in[2];
  int at = conv_source(g, o, k);
  T v = at >= 0 ? x[(size_t(n) * in_pixels + at) * g.c_in + group * cg + c] : T(0);
  cols[(size_t(pos.z) * taps + pos.y) * cg + c] = v;
}

// Zero padding for the convolution kernels: `y[n, h, w, c]` is `x[n, h − top, w − left, c]`
// inside `x` (and `c < C_x`), else zero. Channel padding (MLX's `pad_in_channels`) and the
// Winograd input's spatial padding, in one pass with no fill. Grid: `[C_y, W_y, N · H_y]`.
template <typename T>
[[kernel]] void conv_pad(
    const device T* x [[buffer(0)]],
    device T* y [[buffer(1)]],
    constant int* s [[buffer(2)]], // H_x, W_x, C_x, H_y, W_y, C_y, top, left
    uint3 pos [[thread_position_in_grid]]) {
  const int c = pos.x, w = pos.y, h = int(pos.z) % s[3], n = int(pos.z) / s[3];
  const int hx = h - s[6], wx = w - s[7];
  T v = T(0);
  if (c < s[2] && hx >= 0 && hx < s[0] && wx >= 0 && wx < s[1]) {
    v = x[((size_t(n) * s[0] + hx) * s[1] + wx) * s[2] + c];
  }
  y[((size_t(n) * s[3] + h) * s[4] + w) * s[5] + c] = v;
}

// `causal_conv1d_slots` (nuvec's own): for sequence `n` at pool slot `slots[n]`, the window
// `pool[slot, 0..K − 1, c]` then `x[n, 0..L, c]`, convolved as `depthwise_conv` (taps in
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

#define CONV(NAME, T) \
  template [[host_name("nuvec_depthwise_conv_" NAME)]] [[kernel]] decltype(depthwise_conv<T>) depthwise_conv<T>; \
  template [[host_name("nuvec_conv_unfold_" NAME)]] [[kernel]] decltype(conv_unfold<T>) conv_unfold<T>; \
  template [[host_name("nuvec_conv_pad_" NAME)]] [[kernel]] decltype(conv_pad<T>) conv_pad<T>;
CONV("float32", float)
CONV("float16", half)
CONV("bfloat16", bfloat16_t)
