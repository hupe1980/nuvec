// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Element-wise kernels: unary, binary (including comparisons), select, and strided copy.
//
// Variants: `_c` for contiguous operands (each operand is either laid out like the output or
// a broadcast scalar, chosen by a flag), `_g` for any layout (shape plus signed strides), and
// `_g3` for layouts of at most three (collapsed) dimensions on a 3-D grid, whose offsets need
// no divisions (MLX's `_nd1` to `_nd3` kernels).
// Every operand also takes an element offset, so views bind their buffer at offset 0.

#include "common.h"

using namespace metal;
using namespace nuvec;

// ── Operations ───────────────────────────────────────────────────────────────

#define NUVEC_UNARY_OP(Name, fn)                                \
    struct Name {                                               \
        template <typename T> T operator()(T x) { return fn(x); } \
    };
NUVEC_UNARY_OP(Neg, op_neg)
NUVEC_UNARY_OP(Abs, op_abs)
NUVEC_UNARY_OP(Exp, op_exp)
NUVEC_UNARY_OP(Log, op_log)
NUVEC_UNARY_OP(Sqrt, op_sqrt)
NUVEC_UNARY_OP(Sin, op_sin)
NUVEC_UNARY_OP(Cos, op_cos)
NUVEC_UNARY_OP(Tanh, op_tanh)
NUVEC_UNARY_OP(Sigmoid, op_sigmoid)
NUVEC_UNARY_OP(Conj, op_conj)
NUVEC_UNARY_OP(ErfInv, op_erfinv)

#define NUVEC_BINARY_OP(Name, expr)                                          \
    struct Name {                                                            \
        template <typename T> auto operator()(T a, T b) { return expr; }     \
    };
NUVEC_BINARY_OP(Add, op_add(a, b))
NUVEC_BINARY_OP(Sub, op_sub(a, b))
NUVEC_BINARY_OP(Mul, op_mul(a, b))
NUVEC_BINARY_OP(Div, op_div(a, b))
NUVEC_BINARY_OP(Max, op_max(a, b))
NUVEC_BINARY_OP(Min, op_min(a, b))
NUVEC_BINARY_OP(LogAddExp, op_logaddexp(a, b))
NUVEC_BINARY_OP(Eq, nbool{uchar(op_eq(a, b))})
NUVEC_BINARY_OP(Ne, nbool{uchar(!op_eq(a, b))})
NUVEC_BINARY_OP(Lt, nbool{uchar(op_lt(a, b))})
NUVEC_BINARY_OP(Le, nbool{uchar(op_lt(a, b) || op_eq(a, b))})
NUVEC_BINARY_OP(Gt, nbool{uchar(op_lt(b, a))})
NUVEC_BINARY_OP(Ge, nbool{uchar(op_lt(b, a) || op_eq(a, b))})

// ── Kernels ──────────────────────────────────────────────────────────────────

template <typename T, typename U, typename Op>
[[kernel]] void unary_c(
    device const T* a [[buffer(0)]],
    device U* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant uint& a_scalar [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = Op()(a[a_off + (a_scalar ? 0 : i)]);
}

template <typename T, typename U, typename Op>
[[kernel]] void unary_g(
    device const T* a [[buffer(0)]],
    device U* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = Op()(a[a_off + loc(i, shape, a_strides, ndim)]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_c(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device U* out [[buffer(2)]],
    constant long& a_off [[buffer(3)]],
    constant long& b_off [[buffer(4)]],
    constant uint& scalars [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = Op()(a[a_off + ((scalars & 1) ? 0 : i)], b[b_off + ((scalars & 2) ? 0 : i)]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device U* out [[buffer(2)]],
    constant long& a_off [[buffer(3)]],
    constant long& b_off [[buffer(4)]],
    constant int* shape [[buffer(5)]],
    constant long* a_strides [[buffer(6)]],
    constant long* b_strides [[buffer(7)]],
    constant int& ndim [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = Op()(a[a_off + loc(i, shape, a_strides, ndim)], b[b_off + loc(i, shape, b_strides, ndim)]);
}

// Select moves elements by size: the element type does not matter.
template <typename T>
[[kernel]] void select_c(
    device const uchar* c [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant long& c_off [[buffer(4)]],
    constant long& a_off [[buffer(5)]],
    constant long& b_off [[buffer(6)]],
    constant uint& scalars [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    bool take = c[c_off + ((scalars & 1) ? 0 : i)] != 0;
    out[i] = take ? a[a_off + ((scalars & 2) ? 0 : i)] : b[b_off + ((scalars & 4) ? 0 : i)];
}

template <typename T>
[[kernel]] void select_g(
    device const uchar* c [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant long& c_off [[buffer(4)]],
    constant long& a_off [[buffer(5)]],
    constant long& b_off [[buffer(6)]],
    constant int* shape [[buffer(7)]],
    constant long* c_strides [[buffer(8)]],
    constant long* a_strides [[buffer(9)]],
    constant long* b_strides [[buffer(10)]],
    constant int& ndim [[buffer(11)]],
    uint i [[thread_position_in_grid]]) {
    bool take = c[c_off + loc(i, shape, c_strides, ndim)] != 0;
    out[i] = take ? a[a_off + loc(i, shape, a_strides, ndim)] : b[b_off + loc(i, shape, b_strides, ndim)];
}

template <typename T, typename U, typename Op>
[[kernel]] void unary_g3(
    device const T* a [[buffer(0)]],
    device U* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3(p, shape)] = Op()(a[a_off + loc3(p, a_strides)]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_g3(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device U* out [[buffer(2)]],
    constant long& a_off [[buffer(3)]],
    constant long& b_off [[buffer(4)]],
    constant int* shape [[buffer(5)]],
    constant long* a_strides [[buffer(6)]],
    constant long* b_strides [[buffer(7)]],
    constant int& ndim [[buffer(8)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3(p, shape)] = Op()(a[a_off + loc3(p, a_strides)], b[b_off + loc3(p, b_strides)]);
}

template <typename T>
[[kernel]] void copy_g3(
    device const T* a [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out3(p, shape)] = a[a_off + loc3(p, a_strides)];
}

template <typename T>
[[kernel]] void select_g3(
    device const uchar* c [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant long& c_off [[buffer(4)]],
    constant long& a_off [[buffer(5)]],
    constant long& b_off [[buffer(6)]],
    constant int* shape [[buffer(7)]],
    constant long* c_strides [[buffer(8)]],
    constant long* a_strides [[buffer(9)]],
    constant long* b_strides [[buffer(10)]],
    constant int& ndim [[buffer(11)]],
    uint3 p [[thread_position_in_grid]]) {
    bool take = c[c_off + loc3(p, c_strides)] != 0;
    out[out3(p, shape)] = take ? a[a_off + loc3(p, a_strides)] : b[b_off + loc3(p, b_strides)];
}

// Gathers any layout into a contiguous buffer (by element size).
template <typename T>
[[kernel]] void copy_g(
    device const T* a [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant long* a_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = a[a_off + loc(i, shape, a_strides, ndim)];
}

// ── Instantiations (host names: nuvec_<family>_<op>_<dtype>_<variant>) ──────

#define NUVEC_UNARY(op, Op, name, T, U)                                                        \
    template [[host_name("nuvec_unary_" #op "_" #name "_c")]] [[kernel]] void unary_c<T, U, Op>( \
        device const T*, device U*, constant long&, constant uint&, uint);                    \
    template [[host_name("nuvec_unary_" #op "_" #name "_g")]] [[kernel]] void unary_g<T, U, Op>( \
        device const T*, device U*, constant long&, constant int*, constant long*, constant int&, uint); \
    template [[host_name("nuvec_unary_" #op "_" #name "_g3")]] [[kernel]] void unary_g3<T, U, Op>( \
        device const T*, device U*, constant long&, constant int*, constant long*, constant int&, uint3);

#define NUVEC_BINARY(op, Op, name, T, U)                                                                \
    template [[host_name("nuvec_binary_" #op "_" #name "_c")]] [[kernel]] void binary_c<T, U, Op>(      \
        device const T*, device const T*, device U*, constant long&, constant long&, constant uint&, uint); \
    template [[host_name("nuvec_binary_" #op "_" #name "_g")]] [[kernel]] void binary_g<T, U, Op>(      \
        device const T*, device const T*, device U*, constant long&, constant long&, constant int*,     \
        constant long*, constant long*, constant int&, uint);                                          \
    template [[host_name("nuvec_binary_" #op "_" #name "_g3")]] [[kernel]] void binary_g3<T, U, Op>(    \
        device const T*, device const T*, device U*, constant long&, constant long&, constant int*,     \
        constant long*, constant long*, constant int&, uint3);

// Integers: arithmetic, ordering, negation, abs.
#define NUVEC_INT_KERNELS(name, T)              \
    NUVEC_UNARY(neg, Neg, name, T, T)           \
    NUVEC_UNARY(abs, Abs, name, T, T)           \
    NUVEC_BINARY(add, Add, name, T, T)          \
    NUVEC_BINARY(sub, Sub, name, T, T)          \
    NUVEC_BINARY(mul, Mul, name, T, T)          \
    NUVEC_BINARY(max, Max, name, T, T)          \
    NUVEC_BINARY(min, Min, name, T, T)          \
    NUVEC_BINARY(eq, Eq, name, T, nbool)        \
    NUVEC_BINARY(ne, Ne, name, T, nbool)        \
    NUVEC_BINARY(lt, Lt, name, T, nbool)        \
    NUVEC_BINARY(le, Le, name, T, nbool)        \
    NUVEC_BINARY(gt, Gt, name, T, nbool)        \
    NUVEC_BINARY(ge, Ge, name, T, nbool)

// Floats: everything integers have, plus division and transcendental functions.
#define NUVEC_FLOAT_KERNELS(name, T)            \
    NUVEC_INT_KERNELS(name, T)                  \
    NUVEC_BINARY(div, Div, name, T, T)          \
    NUVEC_BINARY(logaddexp, LogAddExp, name, T, T) \
    NUVEC_UNARY(exp, Exp, name, T, T)           \
    NUVEC_UNARY(log, Log, name, T, T)           \
    NUVEC_UNARY(sqrt, Sqrt, name, T, T)         \
    NUVEC_UNARY(sin, Sin, name, T, T)           \
    NUVEC_UNARY(cos, Cos, name, T, T)           \
    NUVEC_UNARY(tanh, Tanh, name, T, T)         \
    NUVEC_UNARY(sigmoid, Sigmoid, name, T, T)    \
    NUVEC_UNARY(erfinv, ErfInv, name, T, T)

NUVEC_INT_KERNELS(uint8, uchar)
NUVEC_INT_KERNELS(uint16, ushort)
NUVEC_INT_KERNELS(uint32, uint)
NUVEC_INT_KERNELS(uint64, ulong)
NUVEC_INT_KERNELS(int8, char)
NUVEC_INT_KERNELS(int16, short)
NUVEC_INT_KERNELS(int32, int)
NUVEC_INT_KERNELS(int64, long)
NUVEC_FLOAT_KERNELS(float16, half)
NUVEC_FLOAT_KERNELS(bfloat16, bfloat)
NUVEC_FLOAT_KERNELS(float32, float)

// bool: add/sub/mul are OR/XOR/AND, max/min are OR/AND; negation is rejected by the ops layer.
NUVEC_UNARY(abs, Abs, bool, nbool, nbool)
NUVEC_BINARY(add, Add, bool, nbool, nbool)
NUVEC_BINARY(sub, Sub, bool, nbool, nbool)
NUVEC_BINARY(mul, Mul, bool, nbool, nbool)
NUVEC_BINARY(max, Max, bool, nbool, nbool)
NUVEC_BINARY(min, Min, bool, nbool, nbool)
NUVEC_BINARY(eq, Eq, bool, nbool, nbool)
NUVEC_BINARY(ne, Ne, bool, nbool, nbool)
NUVEC_BINARY(lt, Lt, bool, nbool, nbool)
NUVEC_BINARY(le, Le, bool, nbool, nbool)
NUVEC_BINARY(gt, Gt, bool, nbool, nbool)
NUVEC_BINARY(ge, Ge, bool, nbool, nbool)

// complex64: arithmetic, conjugation and equality (ordering is rejected by the ops layer; abs
// runs on CPU).
NUVEC_UNARY(neg, Neg, complex64, complex64_t, complex64_t)
NUVEC_UNARY(conj, Conj, complex64, complex64_t, complex64_t)
NUVEC_BINARY(add, Add, complex64, complex64_t, complex64_t)
NUVEC_BINARY(sub, Sub, complex64, complex64_t, complex64_t)
NUVEC_BINARY(mul, Mul, complex64, complex64_t, complex64_t)
NUVEC_BINARY(div, Div, complex64, complex64_t, complex64_t)
NUVEC_BINARY(eq, Eq, complex64, complex64_t, nbool)
NUVEC_BINARY(ne, Ne, complex64, complex64_t, nbool)

#define NUVEC_MOVE(bits, T)                                                                         \
    template [[host_name("nuvec_select_" #bits "_c")]] [[kernel]] void select_c<T>(                \
        device const uchar*, device const T*, device const T*, device T*, constant long&,           \
        constant long&, constant long&, constant uint&, uint);                                      \
    template [[host_name("nuvec_select_" #bits "_g")]] [[kernel]] void select_g<T>(                \
        device const uchar*, device const T*, device const T*, device T*, constant long&,           \
        constant long&, constant long&, constant int*, constant long*, constant long*,              \
        constant long*, constant int&, uint);                                                       \
    template [[host_name("nuvec_select_" #bits "_g3")]] [[kernel]] void select_g3<T>(              \
        device const uchar*, device const T*, device const T*, device T*, constant long&,           \
        constant long&, constant long&, constant int*, constant long*, constant long*,              \
        constant long*, constant int&, uint3);                                                      \
    template [[host_name("nuvec_copy_" #bits "_g")]] [[kernel]] void copy_g<T>(                    \
        device const T*, device T*, constant long&, constant int*, constant long*, constant int&, uint); \
    template [[host_name("nuvec_copy_" #bits "_g3")]] [[kernel]] void copy_g3<T>(                  \
        device const T*, device T*, constant long&, constant int*, constant long*, constant int&, uint3);

NUVEC_MOVE(8, uchar)
NUVEC_MOVE(16, ushort)
NUVEC_MOVE(32, uint)
NUVEC_MOVE(64, ulong)

// 16-byte units, for layouts whose innermost dimension is contiguous and aligned.
template [[host_name("nuvec_copy_128_g3")]] [[kernel]] void copy_g3<uint4>(
    device const uint4*, device uint4*, constant long&, constant int*, constant long*,
    constant int&, uint3);
