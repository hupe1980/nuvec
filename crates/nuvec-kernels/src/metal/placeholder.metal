// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Placeholder kernel: proves the build pipeline end to end (sources -> .air ->
// nuvec.metallib, with header tracking). Replaced by real kernels from F-007.

#include "common.h"

using namespace metal;

[[kernel]] void nuvec_placeholder(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = nuvec::twice(float(i));
}

// Function-constant specialisation test: writes 1 or 2 depending on constant 300.
constant bool nuvec_test_flag [[function_constant(300)]];

[[kernel]] void nuvec_function_constant_test(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = nuvec_test_flag ? 2.0f : 1.0f;
}
