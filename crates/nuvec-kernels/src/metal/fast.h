// SPDX-License-Identifier: MIT
//
// Shared pieces of the fast-op kernels (softmax, logsumexp, normalisation, RoPE, cross
// entropy), from MLX v0.32.4 (mlx/backend/metal/kernels/utils.h and defines.h): type
// limits, `ceildiv`, the per-thread read counts, and the function
// constants (indices are unique across nuvec's kernel library).
//
// Copyright © 2023-2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#pragma once

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;

static constant constexpr int SOFTMAX_N_READS = 4;
static constant constexpr int RMS_N_READS = 4;
static constant constexpr int RMS_LOOPED_LIMIT = 4096;
static constant constexpr int CROSS_ENTROPY_N_READS = 4;

// Normalisation gradients: whether there is a weight.
constant bool has_w [[function_constant(20)]];
// RoPE: direction, interleaved pairs, and heads/sequence transposed in memory.
constant bool rope_forward [[function_constant(21)]];
constant bool rope_traditional [[function_constant(22)]];
constant bool rope_transpose [[function_constant(23)]];

namespace mlx {

template <typename U>
struct Limits;

#define NUVEC_FLOAT_LIMIT(type)                                \
  template <>                                                  \
  struct Limits<type> {                                        \
    static constexpr constant type max =                       \
        metal::numeric_limits<type>::infinity();               \
    static constexpr constant type min =                       \
        -metal::numeric_limits<type>::infinity();              \
    static constexpr constant type finite_max =                \
        metal::numeric_limits<type>::max();                    \
    static constexpr constant type finite_min =                \
        -metal::numeric_limits<type>::max();                   \
  };

#define NUVEC_INT_LIMIT(type)                                  \
  template <>                                                  \
  struct Limits<type> {                                        \
    static constexpr constant type max =                       \
        metal::numeric_limits<type>::max();                    \
    static constexpr constant type min =                       \
        metal::numeric_limits<type>::min();                    \
    static constexpr constant type finite_max =                \
        metal::numeric_limits<type>::max();                    \
    static constexpr constant type finite_min =                \
        metal::numeric_limits<type>::min();                    \
  };

NUVEC_INT_LIMIT(uint8_t)
NUVEC_INT_LIMIT(uint16_t)
NUVEC_INT_LIMIT(uint32_t)
NUVEC_INT_LIMIT(uint64_t)
NUVEC_INT_LIMIT(int8_t)
NUVEC_INT_LIMIT(int16_t)
NUVEC_INT_LIMIT(int32_t)
NUVEC_INT_LIMIT(int64_t)

NUVEC_FLOAT_LIMIT(half)
NUVEC_FLOAT_LIMIT(float)
NUVEC_FLOAT_LIMIT(bfloat16_t)

template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

} // namespace mlx
