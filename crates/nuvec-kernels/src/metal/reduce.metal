// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Reductions (threadgroups of a [outputs, slices] grid, tree combine in threadgroup memory)
// and arg-reductions (one thread per output element). With several slices, threadgroup
// (o, s) folds slice s of output o's reduced range into `out[o * slices + s]`, and a second
// launch folds those partial results. Inputs may have any layout: kept and
// reduced dimensions each come with their own shape and signed strides.

#include "common.h"

using namespace metal;
using namespace nuvec;

constant constexpr uint MAX_THREADS = 256;

// A reduction: `A` accumulates, `load` lifts an element, `combine` folds, `store` lowers,
// `empty` is the value of an empty reduction.
template <typename T, typename A> struct Sum {
    static A load(T x) { return A(x); }
    static A combine(A a, A b) { return a + b; }
    static T store(A a) { return T(a); }
    static T empty() { return T(0); }
};
template <typename T, typename A> struct Prod {
    static A load(T x) { return A(x); }
    static A combine(A a, A b) { return a * b; }
    static T store(A a) { return T(a); }
    static T empty() { return T(1); }
};
// Integers wrap: accumulate in the element type with wrapping operations.
template <typename T> struct ISum {
    static T load(T x) { return x; }
    static T combine(T a, T b) { return op_add(a, b); }
    static T store(T a) { return a; }
    static T empty() { return T(0); }
};
template <typename T> struct IProd {
    static T load(T x) { return x; }
    static T combine(T a, T b) { return op_mul(a, b); }
    static T store(T a) { return a; }
    static T empty() { return T(1); }
};
template <typename T> struct Max {
    static T load(T x) { return x; }
    static T combine(T a, T b) { return op_max(a, b); }
    static T store(T a) { return a; }
    static T empty() { return T(0); }
};
template <typename T> struct Min {
    static T load(T x) { return x; }
    static T combine(T a, T b) { return op_min(a, b); }
    static T store(T a) { return a; }
    static T empty() { return T(0); }
};
struct And {
    static uchar load(nbool x) { return x.v; }
    static uchar combine(uchar a, uchar b) { return a & b; }
    static nbool store(uchar a) { return {a}; }
    static nbool empty() { return {1}; }
};
struct Or {
    static uchar load(nbool x) { return x.v; }
    static uchar combine(uchar a, uchar b) { return a | b; }
    static nbool store(uchar a) { return {a}; }
    static nbool empty() { return {0}; }
};
struct BMax {
    static uchar load(nbool x) { return x.v; }
    static uchar combine(uchar a, uchar b) { return a | b; }
    static nbool store(uchar a) { return {a}; }
    static nbool empty() { return {0}; }
};
struct BMin {
    static uchar load(nbool x) { return x.v; }
    static uchar combine(uchar a, uchar b) { return a & b; }
    static nbool store(uchar a) { return {a}; }
    static nbool empty() { return {0}; }
};

template <typename T, typename A, typename Op>
[[kernel]] void reduce(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant long& offset [[buffer(2)]],
    constant int* kept_shape [[buffer(3)]],
    constant long* kept_strides [[buffer(4)]],
    constant int& kept_ndim [[buffer(5)]],
    constant int* red_shape [[buffer(6)]],
    constant long* red_strides [[buffer(7)]],
    constant int& red_ndim [[buffer(8)]],
    constant uint& n [[buffer(9)]],
    constant uint& chunk [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 slices [[threadgroups_per_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 threads2 [[threads_per_threadgroup]]) {
    threadgroup A acc_s[MAX_THREADS];
    threadgroup bool valid_s[MAX_THREADS];
    uint tid = tid2.x;
    uint threads = threads2.x;
    long base = offset + loc(group.x, kept_shape, kept_strides, kept_ndim);
    uint end = min(n, (group.y + 1) * chunk);
    A acc = A();
    bool valid = false;
    for (uint j = group.y * chunk + tid; j < end; j += threads) {
        A x = Op::load(in[base + loc(j, red_shape, red_strides, red_ndim)]);
        acc = valid ? Op::combine(acc, x) : x;
        valid = true;
    }
    acc_s[tid] = acc;
    valid_s[tid] = valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s && valid_s[tid + s]) {
            acc_s[tid] = valid_s[tid] ? Op::combine(acc_s[tid], acc_s[tid + s]) : acc_s[tid + s];
            valid_s[tid] = true;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[group.x * slices.y + group.y] = valid_s[0] ? Op::store(acc_s[0]) : Op::empty();
    }
}

// Index of the maximum (or minimum) along one axis: first occurrence wins, the first NaN
// wins over any number.
template <typename T, bool IS_MAX>
[[kernel]] void arg_reduce(
    device const T* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant long& offset [[buffer(2)]],
    constant int* kept_shape [[buffer(3)]],
    constant long* kept_strides [[buffer(4)]],
    constant int& kept_ndim [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant long& axis_stride [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    long base = offset + loc(i, kept_shape, kept_strides, kept_ndim);
    T best = in[base];
    uint best_i = 0;
    for (uint j = 1; j < n && !is_nan_(best); ++j) {
        T x = in[base + long(j) * axis_stride];
        bool better = IS_MAX ? op_lt(best, x) : op_lt(x, best);
        if (is_nan_(x) || better) {
            best = x;
            best_i = j;
        }
    }
    out[i] = best_i;
}

#define NUVEC_REDUCE(op, Op, name, T, A)                                                        \
    template [[host_name("nuvec_reduce_" #op "_" #name)]] [[kernel]] void reduce<T, A, Op>(     \
        device const T*, device T*, constant long&, constant int*, constant long*, constant int&, \
        constant int*, constant long*, constant int&, constant uint&, constant uint&, uint2, uint2, \
        uint2, uint2);

#define NUVEC_ARG_REDUCE(name, T)                                                                 \
    template [[host_name("nuvec_argmax_" #name)]] [[kernel]] void arg_reduce<T, true>(            \
        device const T*, device uint*, constant long&, constant int*, constant long*, constant int&, \
        constant uint&, constant long&, uint);                                                    \
    template [[host_name("nuvec_argmin_" #name)]] [[kernel]] void arg_reduce<T, false>(           \
        device const T*, device uint*, constant long&, constant int*, constant long*, constant int&, \
        constant uint&, constant long&, uint);

#define NUVEC_INT_REDUCE(name, T)                       \
    NUVEC_REDUCE(sum, ISum<T>, name, T, T)              \
    NUVEC_REDUCE(prod, IProd<T>, name, T, T)            \
    NUVEC_REDUCE(max, Max<T>, name, T, T)               \
    NUVEC_REDUCE(min, Min<T>, name, T, T)               \
    NUVEC_ARG_REDUCE(name, T)

// Floats accumulate in float32 (as in MLX on the GPU). Single-parameter aliases keep the
// template commas out of the macro arguments.
template <typename T> using FSum = Sum<T, float>;
template <typename T> using FProd = Prod<T, float>;

#define NUVEC_FLOAT_REDUCE(name, T)                     \
    NUVEC_REDUCE(sum, FSum<T>, name, T, float)          \
    NUVEC_REDUCE(prod, FProd<T>, name, T, float)        \
    NUVEC_REDUCE(max, Max<T>, name, T, T)               \
    NUVEC_REDUCE(min, Min<T>, name, T, T)               \
    NUVEC_ARG_REDUCE(name, T)

NUVEC_INT_REDUCE(uint8, uchar)
NUVEC_INT_REDUCE(uint16, ushort)
NUVEC_INT_REDUCE(uint32, uint)
NUVEC_INT_REDUCE(uint64, ulong)
NUVEC_INT_REDUCE(int8, char)
NUVEC_INT_REDUCE(int16, short)
NUVEC_INT_REDUCE(int32, int)
NUVEC_INT_REDUCE(int64, long)
NUVEC_FLOAT_REDUCE(float16, half)
NUVEC_FLOAT_REDUCE(bfloat16, bfloat)
NUVEC_FLOAT_REDUCE(float32, float)

NUVEC_REDUCE(and, And, bool, nbool, uchar)
NUVEC_REDUCE(or, Or, bool, nbool, uchar)
NUVEC_REDUCE(max, BMax, bool, nbool, uchar)
NUVEC_REDUCE(min, BMin, bool, nbool, uchar)
NUVEC_ARG_REDUCE(bool, nbool)
