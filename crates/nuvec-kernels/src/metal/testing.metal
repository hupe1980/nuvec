// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Kernels for the runtime's own tests (independent of the array library's kernels).

// Function-constant specialisation: writes 1 or 2 depending on constant 300.

#include <metal_stdlib>

using namespace metal;

constant bool nuvec_test_flag [[function_constant(300)]];

[[kernel]] void nuvec_function_constant_test(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = nuvec_test_flag ? 2.0f : 1.0f;
}

// Contiguous float32 addition: the runtime's own tests (numerics, barriers, hazards) and
// its documentation example use it, independently of the array library's kernels.
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
