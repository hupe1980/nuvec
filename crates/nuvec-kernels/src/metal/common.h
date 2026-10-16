// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Helpers shared by nuvec's Metal kernels.

#pragma once

#include <metal_stdlib>

namespace nuvec {

inline float twice(float x) { return x * 2.0f; }

} // namespace nuvec
