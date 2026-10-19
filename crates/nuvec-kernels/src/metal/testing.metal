// SPDX-License-Identifier: MIT OR Apache-2.0
//
// A kernel for the runtime's own tests: function-constant specialisation writes 1 or 2
// depending on constant 300.

#include <metal_stdlib>

using namespace metal;

constant bool nuvec_test_flag [[function_constant(300)]];

[[kernel]] void nuvec_function_constant_test(
    device float* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = nuvec_test_flag ? 2.0f : 1.0f;
}
