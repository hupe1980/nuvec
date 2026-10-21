// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Element type conversions between every pair of GPU dtypes (all but float64), with the CPU
// backend's semantics (see `convert` in common.h).

#include "common.h"

using namespace metal;
using namespace nuvec;

template <typename S, typename D>
[[kernel]] void cast_c(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant uint& a_scalar [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = convert<S, D>(a[a_off + (a_scalar ? 0 : i)]);
}

template <typename S, typename D>
[[kernel]] void cast_g(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = convert<S, D>(a[a_off + loc(i, shape, a_strides, ndim)]);
}

template <typename S, typename D>
[[kernel]] void cast_g3(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3(p, shape)] = convert<S, D>(a[a_off + loc3(p, a_strides)]);
}

// `cast_g3` with 32-bit offsets (see elementwise.metal's `_g3s`).
template <typename S, typename D>
[[kernel]] void cast_g3s(
    device const S* a [[buffer(0)]],
    device D* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3s(p, shape)] = convert<S, D>((a + a_off)[loc3s(p, a_strides)]);
}

#define NUVEC_CAST(sname, S, dname, D)                                                               \
    template [[host_name("nuvec_cast_" #sname "_" #dname "_c")]] [[kernel]] void cast_c<S, D>(       \
        device const S*, device D*, constant long&, constant uint&, uint);                          \
    template [[host_name("nuvec_cast_" #sname "_" #dname "_g")]] [[kernel]] void cast_g<S, D>(       \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint); \
    template [[host_name("nuvec_cast_" #sname "_" #dname "_g3")]] [[kernel]] void cast_g3<S, D>(     \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint3); \
    template [[host_name("nuvec_cast_" #sname "_" #dname "_g3s")]] [[kernel]] void cast_g3s<S, D>(   \
        device const S*, device D*, constant long&, constant int*, constant long*, constant int&, uint3);

#define NUVEC_CAST_FROM(sname, S)                    \
    NUVEC_CAST(sname, S, bool, nbool)                \
    NUVEC_CAST(sname, S, uint8, uchar)               \
    NUVEC_CAST(sname, S, uint16, ushort)             \
    NUVEC_CAST(sname, S, uint32, uint)               \
    NUVEC_CAST(sname, S, uint64, ulong)              \
    NUVEC_CAST(sname, S, int8, char)                 \
    NUVEC_CAST(sname, S, int16, short)               \
    NUVEC_CAST(sname, S, int32, int)                 \
    NUVEC_CAST(sname, S, int64, long)                \
    NUVEC_CAST(sname, S, float16, half)              \
    NUVEC_CAST(sname, S, bfloat16, bfloat)           \
    NUVEC_CAST(sname, S, float32, float)             \
    NUVEC_CAST(sname, S, complex64, complex64_t)

NUVEC_CAST_FROM(bool, nbool)
NUVEC_CAST_FROM(uint8, uchar)
NUVEC_CAST_FROM(uint16, ushort)
NUVEC_CAST_FROM(uint32, uint)
NUVEC_CAST_FROM(uint64, ulong)
NUVEC_CAST_FROM(int8, char)
NUVEC_CAST_FROM(int16, short)
NUVEC_CAST_FROM(int32, int)
NUVEC_CAST_FROM(int64, long)
NUVEC_CAST_FROM(float16, half)
NUVEC_CAST_FROM(bfloat16, bfloat)
NUVEC_CAST_FROM(float32, float)
NUVEC_CAST_FROM(complex64, complex64_t)
