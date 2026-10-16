// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Contiguous float32 addition: proves the runtime end to end (numerics, barriers) until the
// generated kernel library (F-007) replaces it.

#include "common.h"

using namespace metal;

[[kernel]] void nuvec_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
