// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Placeholder kernel: proves the build pipeline end to end (sources -> .air ->
// nuvect.metallib, with header tracking). Replaced by real kernels from F-007.

#include "common.h"

using namespace metal;

[[kernel]] void nuvect_placeholder(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = nuvect::twice(float(i));
}
