// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Placeholder kernel: proves the build pipeline end to end (sources -> .air ->
// bessemer.metallib, with header tracking). Replaced by real kernels from F-007.

#include "common.h"

using namespace metal;

[[kernel]] void bessemer_placeholder(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = bessemer::twice(float(i));
}
