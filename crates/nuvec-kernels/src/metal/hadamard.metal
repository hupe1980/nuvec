// SPDX-License-Identifier: MIT
//
// MLX's Hadamard transform kernels (mlx/backend/metal/kernels/hadamard.h and the codelets of
// mlx/backend/metal/hadamard.cpp), ported from MLX v0.32.4: butterflies in float32 within
// a radix pass, rounded to the input type between passes and stages, as MLX's (so the
// bits are MLX's). One change: the size of a pass is a template parameter and its stride
// an argument (no run-time compilation).
//
// Copyright © 2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include <metal_common>
#include <metal_compute>

using namespace metal;

typedef bfloat bfloat16_t;

#define HADAMARD_UNROLL _Pragma("clang loop unroll(full)")

// Thread-local Hadamard transform for 2^R.
template <short R>
METAL_FUNC void hadamard_radix(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  HADAMARD_UNROLL
  for (short s = 0; s < logR; s++) {
    HADAMARD_UNROLL
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

// MLX's generated codelets for orders 12, 20 and 28 (the Hadamard matrices of
// http://neilsloane.com/hadamard/): each output a signed sum of the inputs in order.
template <short M>
METAL_FUNC void hadamard_radix_m(thread float* x);

template <>
METAL_FUNC void hadamard_radix_m<12>(thread float* x) {
  float tmp[12];
  tmp[0] = x[0] - x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11];
  tmp[1] = -x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] + x[10] - x[11];
  tmp[2] = x[0] + x[1] + x[2] - x[3] + x[4] + x[5] - x[6] - x[7] - x[8] - x[9] + x[10] + x[11];
  tmp[3] = x[0] - x[1] - x[2] - x[3] + x[4] - x[5] - x[6] + x[7] - x[8] + x[9] + x[10] - x[11];
  tmp[4] = x[0] + x[1] + x[2] + x[3] + x[4] - x[5] + x[6] + x[7] - x[8] - x[9] - x[10] - x[11];
  tmp[5] = x[0] - x[1] + x[2] - x[3] - x[4] - x[5] + x[6] - x[7] - x[8] + x[9] - x[10] + x[11];
  tmp[6] = x[0] + x[1] - x[2] - x[3] + x[4] + x[5] + x[6] - x[7] + x[8] + x[9] - x[10] - x[11];
  tmp[7] = x[0] - x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] + x[8] - x[9] - x[10] + x[11];
  tmp[8] = x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7] + x[8] - x[9] + x[10] + x[11];
  tmp[9] = x[0] - x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7] - x[8] - x[9] + x[10] - x[11];
  tmp[10] = x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] + x[8] + x[9] + x[10] - x[11];
  tmp[11] = x[0] - x[1] + x[2] - x[3] - x[4] + x[5] - x[6] + x[7] + x[8] - x[9] - x[10] - x[11];
  for (short i = 0; i < 12; i++) {
    x[i] = tmp[i];
  }
}

template <>
METAL_FUNC void hadamard_radix_m<20>(thread float* x) {
  float tmp[20];
  tmp[0] = x[0] - x[1] - x[2] - x[3] - x[4] + x[5] - x[6] - x[7] - x[8] - x[9] + x[10] + x[11] - x[12] - x[13] + x[14] + x[15] - x[16] + x[17] + x[18] - x[19];
  tmp[1] = -x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] - x[7] - x[8] - x[9] + x[10] + x[11] + x[12] - x[13] - x[14] - x[15] + x[16] - x[17] + x[18] + x[19];
  tmp[2] = -x[0] - x[1] + x[2] - x[3] - x[4] - x[5] - x[6] + x[7] - x[8] - x[9] - x[10] + x[11] + x[12] + x[13] - x[14] + x[15] - x[16] + x[17] - x[18] + x[19];
  tmp[3] = -x[0] - x[1] - x[2] + x[3] - x[4] - x[5] - x[6] - x[7] + x[8] - x[9] - x[10] - x[11] + x[12] + x[13] + x[14] + x[15] + x[16] - x[17] + x[18] - x[19];
  tmp[4] = -x[0] - x[1] - x[2] - x[3] + x[4] - x[5] - x[6] - x[7] - x[8] + x[9] + x[10] - x[11] - x[12] + x[13] + x[14] - x[15] + x[16] + x[17] - x[18] + x[19];
  tmp[5] = -x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - x[6] - x[7] - x[8] - x[9] - x[10] + x[11] - x[12] - x[13] + x[14] + x[15] + x[16] - x[17] - x[18] + x[19];
  tmp[6] = x[0] - x[1] + x[2] + x[3] + x[4] - x[5] + x[6] - x[7] - x[8] - x[9] + x[10] - x[11] + x[12] - x[13] - x[14] + x[15] + x[16] + x[17] - x[18] - x[19];
  tmp[7] = x[0] + x[1] - x[2] + x[3] + x[4] - x[5] - x[6] + x[7] - x[8] - x[9] - x[10] + x[11] - x[12] + x[13] - x[14] - x[15] + x[16] + x[17] + x[18] - x[19];
  tmp[8] = x[0] + x[1] + x[2] - x[3] + x[4] - x[5] - x[6] - x[7] + x[8] - x[9] - x[10] - x[11] + x[12] - x[13] + x[14] - x[15] - x[16] + x[17] + x[18] + x[19];
  tmp[9] = x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] - x[8] + x[9] + x[10] - x[11] - x[12] + x[13] - x[14] + x[15] - x[16] - x[17] + x[18] + x[19];
  tmp[10] = -x[0] - x[1] + x[2] + x[3] - x[4] + x[5] - x[6] + x[7] + x[8] - x[9] + x[10] - x[11] - x[12] - x[13] - x[14] - x[15] + x[16] + x[17] + x[18] + x[19];
  tmp[11] = -x[0] - x[1] - x[2] + x[3] + x[4] - x[5] + x[6] - x[7] + x[8] + x[9] - x[10] + x[11] - x[12] - x[13] - x[14] + x[15] - x[16] + x[17] + x[18] + x[19];
  tmp[12] = x[0] - x[1] - x[2] - x[3] + x[4] + x[5] - x[6] + x[7] - x[8] + x[9] - x[10] - x[11] + x[12] - x[13] - x[14] + x[15] + x[16] - x[17] + x[18] + x[19];
  tmp[13] = x[0] + x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] + x[8] - x[9] - x[10] - x[11] - x[12] + x[13] - x[14] + x[15] + x[16] + x[17] - x[18] + x[19];
  tmp[14] = -x[0] + x[1] + x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] + x[9] - x[10] - x[11] - x[12] - x[13] + x[14] + x[15] + x[16] + x[17] + x[18] - x[19];
  tmp[15] = -x[0] + x[1] - x[2] - x[3] + x[4] - x[5] - x[6] + x[7] + x[8] - x[9] + x[10] - x[11] - x[12] - x[13] - x[14] + x[15] - x[16] - x[17] - x[18] - x[19];
  tmp[16] = x[0] - x[1] + x[2] - x[3] - x[4] - x[5] - x[6] - x[7] + x[8] + x[9] - x[10] + x[11] - x[12] - x[13] - x[14] - x[15] + x[16] - x[17] - x[18] - x[19];
  tmp[17] = -x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] - x[7] - x[8] + x[9] - x[10] - x[11] + x[12] - x[13] - x[14] - x[15] - x[16] + x[17] - x[18] - x[19];
  tmp[18] = -x[0] - x[1] + x[2] - x[3] + x[4] + x[5] + x[6] - x[7] - x[8] - x[9] - x[10] - x[11] - x[12] + x[13] - x[14] - x[15] - x[16] - x[17] + x[18] - x[19];
  tmp[19] = x[0] - x[1] - x[2] + x[3] - x[4] - x[5] + x[6] + x[7] - x[8] - x[9] - x[10] - x[11] - x[12] - x[13] + x[14] - x[15] - x[16] - x[17] - x[18] + x[19];
  for (short i = 0; i < 20; i++) {
    x[i] = tmp[i];
  }
}

template <>
METAL_FUNC void hadamard_radix_m<28>(thread float* x) {
  float tmp[28];
  tmp[0] = x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] + x[7] + x[8] - x[9] - x[10] - x[11] - x[12] + x[13] + x[14] - x[15] + x[16] - x[17] - x[18] + x[19] - x[20] + x[21] - x[22] - x[23] + x[24] + x[25] - x[26] - x[27];
  tmp[1] = -x[0] + x[1] - x[2] - x[3] - x[4] - x[5] - x[6] + x[7] + x[8] + x[9] - x[10] - x[11] - x[12] - x[13] - x[14] + x[15] - x[16] + x[17] - x[18] - x[19] + x[20] - x[21] + x[22] - x[23] - x[24] + x[25] + x[26] - x[27];
  tmp[2] = -x[0] - x[1] + x[2] - x[3] - x[4] - x[5] - x[6] - x[7] + x[8] + x[9] + x[10] - x[11] - x[12] - x[13] + x[14] - x[15] + x[16] - x[17] + x[18] - x[19] - x[20] - x[21] - x[22] + x[23] - x[24] - x[25] + x[26] + x[27];
  tmp[3] = -x[0] - x[1] - x[2] + x[3] - x[4] - x[5] - x[6] - x[7] - x[8] + x[9] + x[10] + x[11] - x[12] - x[13] - x[14] + x[15] - x[16] + x[17] - x[18] + x[19] - x[20] + x[21] - x[22] - x[23] + x[24] - x[25] - x[26] + x[27];
  tmp[4] = -x[0] - x[1] - x[2] - x[3] + x[4] - x[5] - x[6] - x[7] - x[8] - x[9] + x[10] + x[11] + x[12] - x[13] - x[14] - x[15] + x[16] - x[17] + x[18] - x[19] + x[20] + x[21] + x[22] - x[23] - x[24] + x[25] - x[26] - x[27];
  tmp[5] = -x[0] - x[1] - x[2] - x[3] - x[4] + x[5] - x[6] - x[7] - x[8] - x[9] - x[10] + x[11] + x[12] + x[13] + x[14] - x[15] - x[16] + x[17] - x[18] + x[19] - x[20] - x[21] + x[22] + x[23] - x[24] - x[25] + x[26] - x[27];
  tmp[6] = -x[0] - x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7] - x[8] - x[9] - x[10] - x[11] + x[12] + x[13] - x[14] + x[15] - x[16] - x[17] + x[18] - x[19] + x[20] - x[21] - x[22] + x[23] + x[24] - x[25] - x[26] + x[27];
  tmp[7] = -x[0] - x[1] + x[2] + x[3] + x[4] + x[5] - x[6] + x[7] - x[8] - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] + x[15] + x[16] - x[17] - x[18] + x[19] + x[20] + x[21] - x[22] + x[23] - x[24] - x[25] + x[26] - x[27];
  tmp[8] = -x[0] - x[1] - x[2] + x[3] + x[4] + x[5] + x[6] - x[7] + x[8] - x[9] - x[10] - x[11] - x[12] - x[13] + x[14] - x[15] + x[16] + x[17] - x[18] - x[19] + x[20] - x[21] + x[22] - x[23] + x[24] - x[25] - x[26] + x[27];
  tmp[9] = x[0] - x[1] - x[2] - x[3] + x[4] + x[5] + x[6] - x[7] - x[8] + x[9] - x[10] - x[11] - x[12] - x[13] + x[14] + x[15] - x[16] + x[17] + x[18] - x[19] - x[20] + x[21] - x[22] + x[23] - x[24] + x[25] - x[26] - x[27];
  tmp[10] = x[0] + x[1] - x[2] - x[3] - x[4] + x[5] + x[6] - x[7] - x[8] - x[9] + x[10] - x[11] - x[12] - x[13] - x[14] + x[15] + x[16] - x[17] + x[18] + x[19] - x[20] - x[21] + x[22] - x[23] + x[24] - x[25] + x[26] - x[27];
  tmp[11] = x[0] + x[1] + x[2] - x[3] - x[4] - x[5] + x[6] - x[7] - x[8] - x[9] - x[10] + x[11] - x[12] - x[13] - x[14] - x[15] + x[16] + x[17] - x[18] + x[19] + x[20] - x[21] - x[22] + x[23] - x[24] + x[25] - x[26] + x[27];
  tmp[12] = x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] - x[8] - x[9] - x[10] - x[11] + x[12] - x[13] + x[14] - x[15] - x[16] + x[17] + x[18] - x[19] + x[20] + x[21] - x[22] - x[23] + x[24] - x[25] + x[26] - x[27];
  tmp[13] = -x[0] + x[1] + x[2] + x[3] + x[4] - x[5] - x[6] - x[7] - x[8] - x[9] - x[10] - x[11] - x[12] + x[13] + x[14] + x[15] - x[16] - x[17] + x[18] + x[19] - x[20] - x[21] + x[22] - x[23] - x[24] + x[25] - x[26] + x[27];
  tmp[14] = -x[0] + x[1] - x[2] + x[3] + x[4] - x[5] + x[6] + x[7] - x[8] - x[9] + x[10] + x[11] - x[12] - x[13] + x[14] - x[15] - x[16] - x[17] - x[18] - x[19] - x[20] - x[21] - x[22] + x[23] + x[24] + x[25] + x[26] - x[27];
  tmp[15] = x[0] - x[1] + x[2] - x[3] + x[4] + x[5] - x[6] - x[7] + x[8] - x[9] - x[10] + x[11] + x[12] - x[13] - x[14] + x[15] - x[16] - x[17] - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] + x[24] + x[25] + x[26] + x[27];
  tmp[16] = -x[0] + x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7] - x[8] + x[9] - x[10] - x[11] + x[12] + x[13] - x[14] - x[15] + x[16] - x[17] - x[18] - x[19] - x[20] + x[21] - x[22] - x[23] - x[24] + x[25] + x[26] + x[27];
  tmp[17] = x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] + x[7] - x[8] - x[9] + x[10] - x[11] - x[12] + x[13] - x[14] - x[15] - x[16] + x[17] - x[18] - x[19] - x[20] + x[21] + x[22] - x[23] - x[24] - x[25] + x[26] + x[27];
  tmp[18] = x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] + x[7] + x[8] - x[9] - x[10] + x[11] - x[12] - x[13] - x[14] - x[15] - x[16] - x[17] + x[18] - x[19] - x[20] + x[21] + x[22] + x[23] - x[24] - x[25] - x[26] + x[27];
  tmp[19] = -x[0] + x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] + x[9] - x[10] - x[11] + x[12] - x[13] - x[14] - x[15] - x[16] - x[17] - x[18] + x[19] - x[20] + x[21] + x[22] + x[23] + x[24] - x[25] - x[26] - x[27];
  tmp[20] = x[0] - x[1] + x[2] + x[3] - x[4] + x[5] - x[6] - x[7] - x[8] + x[9] + x[10] - x[11] - x[12] + x[13] - x[14] - x[15] - x[16] - x[17] - x[18] - x[19] + x[20] - x[21] + x[22] + x[23] + x[24] + x[25] - x[26] - x[27];
  tmp[21] = -x[0] + x[1] + x[2] - x[3] - x[4] + x[5] + x[6] - x[7] + x[8] - x[9] + x[10] + x[11] - x[12] + x[13] + x[14] + x[15] - x[16] - x[17] - x[18] - x[19] + x[20] + x[21] - x[22] - x[23] - x[24] - x[25] - x[26] - x[27];
  tmp[22] = x[0] - x[1] + x[2] + x[3] - x[4] - x[5] + x[6] + x[7] - x[8] + x[9] - x[10] + x[11] + x[12] - x[13] + x[14] + x[15] + x[16] - x[17] - x[18] - x[19] - x[20] - x[21] + x[22] - x[23] - x[24] - x[25] - x[26] - x[27];
  tmp[23] = x[0] + x[1] - x[2] + x[3] + x[4] - x[5] - x[6] - x[7] + x[8] - x[9] + x[10] - x[11] + x[12] + x[13] - x[14] + x[15] + x[16] + x[17] - x[18] - x[19] - x[20] - x[21] - x[22] + x[23] - x[24] - x[25] - x[26] - x[27];
  tmp[24] = -x[0] + x[1] + x[2] - x[3] + x[4] + x[5] - x[6] + x[7] - x[8] + x[9] - x[10] + x[11] - x[12] + x[13] - x[14] - x[15] + x[16] + x[17] + x[18] - x[19] - x[20] - x[21] - x[22] - x[23] + x[24] - x[25] - x[26] - x[27];
  tmp[25] = -x[0] - x[1] + x[2] + x[3] - x[4] + x[5] + x[6] + x[7] + x[8] - x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15] - x[16] + x[17] + x[18] + x[19] - x[20] - x[21] - x[22] - x[23] - x[24] + x[25] - x[26] - x[27];
  tmp[26] = x[0] - x[1] - x[2] + x[3] + x[4] - x[5] + x[6] - x[7] + x[8] + x[9] - x[10] + x[11] - x[12] + x[13] - x[14] - x[15] - x[16] - x[17] + x[18] + x[19] + x[20] - x[21] - x[22] - x[23] - x[24] - x[25] + x[26] - x[27];
  tmp[27] = x[0] + x[1] - x[2] - x[3] + x[4] + x[5] - x[6] + x[7] - x[8] + x[9] + x[10] - x[11] + x[12] - x[13] + x[14] - x[15] - x[16] - x[17] - x[18] + x[19] + x[20] - x[21] - x[22] - x[23] - x[24] - x[25] - x[26] + x[27];
  for (short i = 0; i < 28; i++) {
    x[i] = tmp[i];
  }
}

// A transform of size N = 2^k over every N elements (`stride` 1) or over every N elements
// `stride` apart.
template <typename TI, typename TO, int N, int max_radix, int read_width>
[[kernel]] void hadamard_n(
    const device TI* in [[buffer(0)]],
    device TO* out [[buffer(1)]],
    constant const float& scale [[buffer(2)]],
    constant const int& stride [[buffer(3)]],
    uint3 elem [[thread_position_in_grid]]) {
  constexpr short num_threads = N / max_radix;
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = __builtin_ctz(max_radix);
  constexpr short num_steps = logN / logR;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << (logFinal);

  size_t batch_idx = size_t(elem.y) * N * stride + elem.z;
  short i = elem.x;

  threadgroup TO buf[N];

  if (stride == 1) {
    HADAMARD_UNROLL
    for (short j = 0; j < max_radix / read_width; j++) {
      short index = j * read_width * num_threads + i * read_width;
      HADAMARD_UNROLL
      for (short r = 0; r < read_width; r++) {
        buf[index + r] = TO(in[batch_idx + index + r]);
      }
    }
  } else {
    HADAMARD_UNROLL
    for (short j = 0; j < max_radix; j++) {
      buf[j * num_threads + i] = TO(in[batch_idx + size_t(j * num_threads + i) * stride]);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float x[max_radix];
  short h = 1;

  HADAMARD_UNROLL
  for (short s = 0; s < num_steps; s++) {
    short k = i & (h - 1);
    short j = ((i - k) << logR) + k;
    HADAMARD_UNROLL
    for (short r = 0; r < max_radix; r++) {
      x[r] = buf[j + h * r];
    }
    hadamard_radix<max_radix>(x);
    HADAMARD_UNROLL
    for (short r = 0; r < max_radix; r++) {
      buf[j + h * r] = TO(x[r]);
    }
    h <<= logR;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (final_radix > 1) {
    HADAMARD_UNROLL
    for (int t = 0; t < max_radix / final_radix; t++) {
      short index = i + t * num_threads;
      short k = index & (h - 1);
      short j = ((index - k) << logFinal) + k;
      HADAMARD_UNROLL
      for (short r = 0; r < final_radix; r++) {
        x[r] = buf[j + h * r];
      }
      hadamard_radix<final_radix>(x);
      HADAMARD_UNROLL
      for (short r = 0; r < final_radix; r++) {
        buf[j + h * r] = TO(x[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (stride == 1) {
    HADAMARD_UNROLL
    for (short j = 0; j < max_radix / read_width; j++) {
      short index = j * read_width * num_threads + i * read_width;
      HADAMARD_UNROLL
      for (short r = 0; r < read_width; r++) {
        out[batch_idx + index + r] = TO(float(buf[index + r]) * scale);
      }
    }
  } else {
    HADAMARD_UNROLL
    for (short j = 0; j < max_radix; j++) {
      out[batch_idx + size_t(j * num_threads + i) * stride] = TO(float(buf[j * num_threads + i]) * scale);
    }
  }
}

// The order-M stage of a transform of size M · n (n = 2^k, an argument): a naive O(M²)
// codelet over elements n apart.
template <typename TI, typename TO, int M, int read_width>
[[kernel]] void hadamard_m(
    const device TI* in [[buffer(0)]],
    device TO* out [[buffer(1)]],
    constant const float& scale [[buffer(2)]],
    constant const int& n [[buffer(3)]],
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  size_t index = size_t(elem.x) * grid.y + elem.y;
  int i = index % (n / read_width);
  size_t batch_idx = index / (n / read_width) * M * n;

  float x[read_width][M];
  HADAMARD_UNROLL
  for (short c = 0; c < M; c++) {
    HADAMARD_UNROLL
    for (short r = 0; r < read_width; r++) {
      x[r][c] = float(in[batch_idx + size_t(c) * n + i * read_width + r]);
    }
  }
  HADAMARD_UNROLL
  for (short r = 0; r < read_width; r++) {
    hadamard_radix_m<M>(x[r]);
  }
  HADAMARD_UNROLL
  for (short c = 0; c < M; c++) {
    HADAMARD_UNROLL
    for (short r = 0; r < read_width; r++) {
      out[batch_idx + size_t(c) * n + i * read_width + r] = TO(x[r][c] * scale);
    }
  }
}

template [[host_name("nuvec_hadamard_n2_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 2, 2, 2>) hadamard_n<float, float, 2, 2, 2>;
template [[host_name("nuvec_hadamard_n4_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 4, 4, 4>) hadamard_n<float, float, 4, 4, 4>;
template [[host_name("nuvec_hadamard_n8_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 8, 8, 4>) hadamard_n<float, float, 8, 8, 4>;
template [[host_name("nuvec_hadamard_n16_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 16, 16, 4>) hadamard_n<float, float, 16, 16, 4>;
template [[host_name("nuvec_hadamard_n32_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 32, 16, 4>) hadamard_n<float, float, 32, 16, 4>;
template [[host_name("nuvec_hadamard_n64_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 64, 16, 4>) hadamard_n<float, float, 64, 16, 4>;
template [[host_name("nuvec_hadamard_n128_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 128, 16, 4>) hadamard_n<float, float, 128, 16, 4>;
template [[host_name("nuvec_hadamard_n256_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 256, 16, 4>) hadamard_n<float, float, 256, 16, 4>;
template [[host_name("nuvec_hadamard_n512_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 512, 16, 4>) hadamard_n<float, float, 512, 16, 4>;
template [[host_name("nuvec_hadamard_n1024_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 1024, 16, 4>) hadamard_n<float, float, 1024, 16, 4>;
template [[host_name("nuvec_hadamard_n2048_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 2048, 16, 4>) hadamard_n<float, float, 2048, 16, 4>;
template [[host_name("nuvec_hadamard_n4096_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 4096, 16, 4>) hadamard_n<float, float, 4096, 16, 4>;
template [[host_name("nuvec_hadamard_n8192_float32_float32")]] [[kernel]] decltype(hadamard_n<float, float, 8192, 16, 4>) hadamard_n<float, float, 8192, 16, 4>;
template [[host_name("nuvec_hadamard_m12_r1_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 12, 1>) hadamard_m<float, float, 12, 1>;
template [[host_name("nuvec_hadamard_m12_r2_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 12, 2>) hadamard_m<float, float, 12, 2>;
template [[host_name("nuvec_hadamard_m12_r4_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 12, 4>) hadamard_m<float, float, 12, 4>;
template [[host_name("nuvec_hadamard_m20_r1_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 20, 1>) hadamard_m<float, float, 20, 1>;
template [[host_name("nuvec_hadamard_m20_r2_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 20, 2>) hadamard_m<float, float, 20, 2>;
template [[host_name("nuvec_hadamard_m20_r4_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 20, 4>) hadamard_m<float, float, 20, 4>;
template [[host_name("nuvec_hadamard_m28_r1_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 28, 1>) hadamard_m<float, float, 28, 1>;
template [[host_name("nuvec_hadamard_m28_r2_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 28, 2>) hadamard_m<float, float, 28, 2>;
template [[host_name("nuvec_hadamard_m28_r4_float32_float32")]] [[kernel]] decltype(hadamard_m<float, float, 28, 4>) hadamard_m<float, float, 28, 4>;
template [[host_name("nuvec_hadamard_n2_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 2, 2, 2>) hadamard_n<half, half, 2, 2, 2>;
template [[host_name("nuvec_hadamard_n4_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 4, 4, 4>) hadamard_n<half, half, 4, 4, 4>;
template [[host_name("nuvec_hadamard_n8_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 8, 8, 4>) hadamard_n<half, half, 8, 8, 4>;
template [[host_name("nuvec_hadamard_n16_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 16, 16, 4>) hadamard_n<half, half, 16, 16, 4>;
template [[host_name("nuvec_hadamard_n32_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 32, 16, 4>) hadamard_n<half, half, 32, 16, 4>;
template [[host_name("nuvec_hadamard_n64_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 64, 16, 4>) hadamard_n<half, half, 64, 16, 4>;
template [[host_name("nuvec_hadamard_n128_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 128, 16, 4>) hadamard_n<half, half, 128, 16, 4>;
template [[host_name("nuvec_hadamard_n256_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 256, 16, 4>) hadamard_n<half, half, 256, 16, 4>;
template [[host_name("nuvec_hadamard_n512_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 512, 16, 4>) hadamard_n<half, half, 512, 16, 4>;
template [[host_name("nuvec_hadamard_n1024_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 1024, 16, 4>) hadamard_n<half, half, 1024, 16, 4>;
template [[host_name("nuvec_hadamard_n2048_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 2048, 16, 4>) hadamard_n<half, half, 2048, 16, 4>;
template [[host_name("nuvec_hadamard_n4096_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 4096, 16, 4>) hadamard_n<half, half, 4096, 16, 4>;
template [[host_name("nuvec_hadamard_n8192_float16_float16")]] [[kernel]] decltype(hadamard_n<half, half, 8192, 16, 4>) hadamard_n<half, half, 8192, 16, 4>;
template [[host_name("nuvec_hadamard_m12_r1_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 12, 1>) hadamard_m<half, half, 12, 1>;
template [[host_name("nuvec_hadamard_m12_r2_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 12, 2>) hadamard_m<half, half, 12, 2>;
template [[host_name("nuvec_hadamard_m12_r4_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 12, 4>) hadamard_m<half, half, 12, 4>;
template [[host_name("nuvec_hadamard_m20_r1_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 20, 1>) hadamard_m<half, half, 20, 1>;
template [[host_name("nuvec_hadamard_m20_r2_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 20, 2>) hadamard_m<half, half, 20, 2>;
template [[host_name("nuvec_hadamard_m20_r4_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 20, 4>) hadamard_m<half, half, 20, 4>;
template [[host_name("nuvec_hadamard_m28_r1_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 28, 1>) hadamard_m<half, half, 28, 1>;
template [[host_name("nuvec_hadamard_m28_r2_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 28, 2>) hadamard_m<half, half, 28, 2>;
template [[host_name("nuvec_hadamard_m28_r4_float16_float16")]] [[kernel]] decltype(hadamard_m<half, half, 28, 4>) hadamard_m<half, half, 28, 4>;
template [[host_name("nuvec_hadamard_n2_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 2, 2, 2>) hadamard_n<bfloat16_t, bfloat16_t, 2, 2, 2>;
template [[host_name("nuvec_hadamard_n4_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 4, 4, 4>) hadamard_n<bfloat16_t, bfloat16_t, 4, 4, 4>;
template [[host_name("nuvec_hadamard_n8_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 8, 8, 4>) hadamard_n<bfloat16_t, bfloat16_t, 8, 8, 4>;
template [[host_name("nuvec_hadamard_n16_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 16, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 16, 16, 4>;
template [[host_name("nuvec_hadamard_n32_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 32, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 32, 16, 4>;
template [[host_name("nuvec_hadamard_n64_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 64, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 64, 16, 4>;
template [[host_name("nuvec_hadamard_n128_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 128, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 128, 16, 4>;
template [[host_name("nuvec_hadamard_n256_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 256, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 256, 16, 4>;
template [[host_name("nuvec_hadamard_n512_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 512, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 512, 16, 4>;
template [[host_name("nuvec_hadamard_n1024_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 1024, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 1024, 16, 4>;
template [[host_name("nuvec_hadamard_n2048_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 2048, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 2048, 16, 4>;
template [[host_name("nuvec_hadamard_n4096_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 4096, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 4096, 16, 4>;
template [[host_name("nuvec_hadamard_n8192_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_n<bfloat16_t, bfloat16_t, 8192, 16, 4>) hadamard_n<bfloat16_t, bfloat16_t, 8192, 16, 4>;
template [[host_name("nuvec_hadamard_m12_r1_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 12, 1>) hadamard_m<bfloat16_t, bfloat16_t, 12, 1>;
template [[host_name("nuvec_hadamard_m12_r2_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 12, 2>) hadamard_m<bfloat16_t, bfloat16_t, 12, 2>;
template [[host_name("nuvec_hadamard_m12_r4_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 12, 4>) hadamard_m<bfloat16_t, bfloat16_t, 12, 4>;
template [[host_name("nuvec_hadamard_m20_r1_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 20, 1>) hadamard_m<bfloat16_t, bfloat16_t, 20, 1>;
template [[host_name("nuvec_hadamard_m20_r2_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 20, 2>) hadamard_m<bfloat16_t, bfloat16_t, 20, 2>;
template [[host_name("nuvec_hadamard_m20_r4_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 20, 4>) hadamard_m<bfloat16_t, bfloat16_t, 20, 4>;
template [[host_name("nuvec_hadamard_m28_r1_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 28, 1>) hadamard_m<bfloat16_t, bfloat16_t, 28, 1>;
template [[host_name("nuvec_hadamard_m28_r2_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 28, 2>) hadamard_m<bfloat16_t, bfloat16_t, 28, 2>;
template [[host_name("nuvec_hadamard_m28_r4_bfloat16_bfloat16")]] [[kernel]] decltype(hadamard_m<bfloat16_t, bfloat16_t, 28, 4>) hadamard_m<bfloat16_t, bfloat16_t, 28, 4>;
