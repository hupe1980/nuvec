// SPDX-License-Identifier: MIT OR Apache-2.0
//
// MLX's `to_fp8` and `from_fp8` (`fast::ConvertFP8`): float16, bfloat16 or float32 to E4M3
// bytes (uint8, saturating at ±448) and back, with MLX's `fp8_e4m3` (fp8.h). The layout
// variants are the cast kernels' (`_c`, `_g`, `_g3`, `_g3s`).

#include "common.h"
#include "fp8.h"

using namespace metal;
using namespace nuvec;

// `S` to `D` where one of them is the E4M3 byte (`uchar`).
template <typename S, typename D>
inline D fp8_convert(S x) {
    if constexpr (metal::is_same_v<D, uchar>) {
        return mlx::fpq::fp8_e4m3(x).bits;
    } else {
        uint8_t bits = x;
        return static_cast<D>(*(thread mlx::fpq::fp8_e4m3*)(&bits));
    }
}

template <typename S, typename D>
[[kernel]] void fp8_c(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant uint& a_scalar [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = fp8_convert<S, D>(a[a_off + (a_scalar ? 0 : i)]);
}

template <typename S, typename D>
[[kernel]] void fp8_g(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = fp8_convert<S, D>(a[a_off + loc(i, shape, a_strides, ndim)]);
}

template <typename S, typename D>
[[kernel]] void fp8_g3(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3(p, shape)] = fp8_convert<S, D>(a[a_off + loc3(p, a_strides)]);
}

template <typename S, typename D>
[[kernel]] void fp8_g3s(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3s(p, shape)] = fp8_convert<S, D>((a + a_off)[loc3s(p, a_strides)]);
}

#define NUVEC_FP8(name, S, D)                                                                        \
    template [[host_name(name "_c")]] [[kernel]] void fp8_c<S, D>(                                  \
        device const S*, device D*, constant long&, constant uint&, uint);                          \
    template [[host_name(name "_g")]] [[kernel]] void fp8_g<S, D>(                                  \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint); \
    template [[host_name(name "_g3")]] [[kernel]] void fp8_g3<S, D>(                                \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint3); \
    template [[host_name(name "_g3s")]] [[kernel]] void fp8_g3s<S, D>(                              \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint3);

NUVEC_FP8("nuvec_to_fp8_float16", half, uchar)
NUVEC_FP8("nuvec_to_fp8_bfloat16", bfloat, uchar)
NUVEC_FP8("nuvec_to_fp8_float32", float, uchar)
NUVEC_FP8("nuvec_from_fp8_float16", uchar, half)
NUVEC_FP8("nuvec_from_fp8_bfloat16", uchar, bfloat)
NUVEC_FP8("nuvec_from_fp8_float32", uchar, float)
