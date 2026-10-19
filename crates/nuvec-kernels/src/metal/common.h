// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared definitions for nuvec's Metal kernels: element types, per-type arithmetic with the
// same semantics as the CPU backend, conversions, and strided indexing.

#pragma once

// No implicit fused multiply-add (also when compiled from source at runtime, where the
// `-ffp-contract=off` build flag does not apply): results round like the CPU backend.
#pragma clang fp contract(off)

#include <metal_stdlib>

using namespace metal;

namespace nuvec {

// ── Element types ────────────────────────────────────────────────────────────
// bool and complex64 get distinct types so that overloads never confuse bool with uint8
// or complex numbers with float2 vectors.

struct nbool {
    uchar v;
};

struct complex64_t {
    float re;
    float im;
};

// ── Strided indexing ─────────────────────────────────────────────────────────

// Element offset of row-major index `i` in a layout with `shape` and signed `strides`.
inline long loc(uint i, constant int* shape, constant long* strides, int ndim) {
    long offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        uint n = uint(shape[d]);
        offset += long(i % n) * strides[d];
        i /= n;
    }
    return offset;
}

// ── Arithmetic: integers wrap (computed in unsigned types) ──────────────────

#define NUVEC_INT_OPS(T, U)                                                   \
    inline T op_add(T a, T b) { return T(U(a) + U(b)); }                     \
    inline T op_sub(T a, T b) { return T(U(a) - U(b)); }                     \
    inline T op_mul(T a, T b) { return T(U(a) * U(b)); }                     \
    inline T op_neg(T a) { return T(U(0) - U(a)); }                          \
    inline T op_abs(T a) { return a < T(0) ? op_neg(a) : a; }                \
    inline T op_max(T a, T b) { return a > b ? a : b; }                      \
    inline T op_min(T a, T b) { return a < b ? a : b; }                      \
    inline bool op_eq(T a, T b) { return a == b; }                           \
    inline bool op_lt(T a, T b) { return a < b; }                            \
    inline bool is_nan_(T) { return false; }

NUVEC_INT_OPS(uchar, uint)
NUVEC_INT_OPS(ushort, uint)
NUVEC_INT_OPS(uint, uint)
NUVEC_INT_OPS(ulong, ulong)
NUVEC_INT_OPS(char, uint)
NUVEC_INT_OPS(short, uint)
NUVEC_INT_OPS(int, uint)
NUVEC_INT_OPS(long, ulong)

// ── Arithmetic: half precision computes in float and rounds once ────────────

inline float sigmoid_(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + precise::exp(-x));
    }
    float e = precise::exp(x);
    return e / (1.0f + e);
}

// MLX's inverse error function (Giles' single-precision polynomial).
inline float erfinv_(float a) {
    float t = precise::log(fma(a, 0.0f - a, 1.0f));
    float p;
    if (fabs(t) > 6.125f) {
        p = 3.03697567e-10f;
        p = fma(p, t, 2.93243101e-8f);
        p = fma(p, t, 1.22150334e-6f);
        p = fma(p, t, 2.84108955e-5f);
        p = fma(p, t, 3.93552968e-4f);
        p = fma(p, t, 3.02698812e-3f);
        p = fma(p, t, 4.83185798e-3f);
        p = fma(p, t, -2.64646143e-1f);
        p = fma(p, t, 8.40016484e-1f);
    } else {
        p = 5.43877832e-9f;
        p = fma(p, t, 1.43285448e-7f);
        p = fma(p, t, 1.22774793e-6f);
        p = fma(p, t, 1.12963626e-7f);
        p = fma(p, t, -5.61530760e-5f);
        p = fma(p, t, -1.47697632e-4f);
        p = fma(p, t, 2.31468678e-3f);
        p = fma(p, t, 1.15392581e-2f);
        p = fma(p, t, -2.32015476e-1f);
        p = fma(p, t, 8.86226892e-1f);
    }
    return a * p;
}

#define NUVEC_FLOAT_OPS(T)                                                                     \
    inline T op_add(T a, T b) { return T(float(a) + float(b)); }                              \
    inline T op_sub(T a, T b) { return T(float(a) - float(b)); }                              \
    inline T op_mul(T a, T b) { return T(float(a) * float(b)); }                              \
    inline T op_div(T a, T b) { return T(precise::divide(float(a), float(b))); }              \
    inline T op_neg(T a) { return T(-float(a)); }                                             \
    inline T op_abs(T a) { return T(precise::fabs(float(a))); }                               \
    inline bool is_nan_(T a) { return isnan(float(a)); }                                      \
    inline T op_max(T a, T b) {                                                               \
        return (is_nan_(a) || is_nan_(b)) ? T(NAN) : (float(a) > float(b) ? a : b);           \
    }                                                                                         \
    inline T op_min(T a, T b) {                                                               \
        return (is_nan_(a) || is_nan_(b)) ? T(NAN) : (float(a) < float(b) ? a : b);           \
    }                                                                                         \
    inline bool op_eq(T a, T b) { return float(a) == float(b); }                              \
    inline bool op_lt(T a, T b) { return float(a) < float(b); }                               \
    inline T op_exp(T a) { return T(precise::exp(float(a))); }                                \
    inline T op_log(T a) { return T(precise::log(float(a))); }                                \
    inline T op_sqrt(T a) { return T(precise::sqrt(float(a))); }                              \
    inline T op_sin(T a) { return T(precise::sin(float(a))); }                                \
    inline T op_cos(T a) { return T(precise::cos(float(a))); }                                \
    inline T op_tanh(T a) { return T(precise::tanh(float(a))); }                              \
    inline T op_sigmoid(T a) { return T(sigmoid_(float(a))); }                                \
    inline T op_erfinv(T a) { return T(erfinv_(float(a))); }

NUVEC_FLOAT_OPS(half)
NUVEC_FLOAT_OPS(bfloat)
NUVEC_FLOAT_OPS(float)

// ── Arithmetic: bool (add = OR, subtract = XOR, multiply = AND, as in MLX) ──

inline nbool op_add(nbool a, nbool b) { return nbool{uchar(a.v | b.v)}; }
inline nbool op_sub(nbool a, nbool b) { return nbool{uchar(a.v ^ b.v)}; }
inline nbool op_mul(nbool a, nbool b) { return nbool{uchar(a.v & b.v)}; }
inline nbool op_max(nbool a, nbool b) { return nbool{uchar(a.v | b.v)}; }
inline nbool op_min(nbool a, nbool b) { return nbool{uchar(a.v & b.v)}; }
inline nbool op_abs(nbool a) { return a; }
inline bool op_eq(nbool a, nbool b) { return a.v == b.v; }
inline bool op_lt(nbool a, nbool b) { return a.v < b.v; }
inline bool is_nan_(nbool) { return false; }

// ── Arithmetic: complex64 ────────────────────────────────────────────────────

inline complex64_t op_add(complex64_t a, complex64_t b) { return {a.re + b.re, a.im + b.im}; }
inline complex64_t op_sub(complex64_t a, complex64_t b) { return {a.re - b.re, a.im - b.im}; }
inline complex64_t op_mul(complex64_t a, complex64_t b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline complex64_t op_div(complex64_t a, complex64_t b) {
    float d = b.re * b.re + b.im * b.im;
    return {precise::divide(a.re * b.re + a.im * b.im, d), precise::divide(a.im * b.re - a.re * b.im, d)};
}
inline complex64_t op_neg(complex64_t a) { return {-a.re, -a.im}; }
inline complex64_t op_conj(complex64_t a) { return {a.re, -a.im}; }
inline bool op_eq(complex64_t a, complex64_t b) { return a.re == b.re && a.im == b.im; }

// ── Conversions (the CPU backend's rules) ────────────────────────────────────
// The source is decomposed into an exact canonical value, which is then converted to the
// destination: integers wrap, floats round once, float->integer saturates (NaN -> 0),
// anything->bool tests non-zero, complex->real keeps the real part.

struct CInt { long v; };
struct CUint { ulong v; };
struct CFloat { float v; };
struct CCplx { float re; float im; };

inline CInt canon(nbool x) { return {long(x.v != 0)}; }
inline CInt canon(char x) { return {long(x)}; }
inline CInt canon(short x) { return {long(x)}; }
inline CInt canon(int x) { return {long(x)}; }
inline CInt canon(long x) { return {x}; }
inline CUint canon(uchar x) { return {ulong(x)}; }
inline CUint canon(ushort x) { return {ulong(x)}; }
inline CUint canon(uint x) { return {ulong(x)}; }
inline CUint canon(ulong x) { return {x}; }
inline CFloat canon(half x) { return {float(x)}; }
inline CFloat canon(bfloat x) { return {float(x)}; }
inline CFloat canon(float x) { return {x}; }
inline CCplx canon(complex64_t x) { return {x.re, x.im}; }

template <typename D> D saturate(float f) {
    if (isnan(f)) {
        return D(0);
    }
    if (f >= float(numeric_limits<D>::max())) {
        return numeric_limits<D>::max();
    }
    if (f <= float(numeric_limits<D>::min())) {
        return numeric_limits<D>::min();
    }
    return D(f);
}

template <typename D> struct To;

#define NUVEC_TO_INT(D)                                                    \
    template <> struct To<D> {                                             \
        static D from(CInt c) { return D(c.v); }                           \
        static D from(CUint c) { return D(c.v); }                          \
        static D from(CFloat c) { return saturate<D>(c.v); }               \
        static D from(CCplx c) { return saturate<D>(c.re); }               \
    };
NUVEC_TO_INT(uchar)
NUVEC_TO_INT(ushort)
NUVEC_TO_INT(uint)
NUVEC_TO_INT(ulong)
NUVEC_TO_INT(char)
NUVEC_TO_INT(short)
NUVEC_TO_INT(int)
NUVEC_TO_INT(long)

#define NUVEC_TO_FLOAT(D)                                                  \
    template <> struct To<D> {                                             \
        static D from(CInt c) { return D(float(c.v)); }                    \
        static D from(CUint c) { return D(float(c.v)); }                   \
        static D from(CFloat c) { return D(c.v); }                         \
        static D from(CCplx c) { return D(c.re); }                         \
    };
NUVEC_TO_FLOAT(half)
NUVEC_TO_FLOAT(bfloat)
NUVEC_TO_FLOAT(float)

template <> struct To<nbool> {
    static nbool from(CInt c) { return {uchar(c.v != 0)}; }
    static nbool from(CUint c) { return {uchar(c.v != 0)}; }
    static nbool from(CFloat c) { return {uchar(c.v != 0.0f)}; }
    static nbool from(CCplx c) { return {uchar(c.re != 0.0f || c.im != 0.0f)}; }
};

template <> struct To<complex64_t> {
    static complex64_t from(CInt c) { return {float(c.v), 0.0f}; }
    static complex64_t from(CUint c) { return {float(c.v), 0.0f}; }
    static complex64_t from(CFloat c) { return {c.v, 0.0f}; }
    static complex64_t from(CCplx c) { return {c.re, c.im}; }
};

template <typename S, typename D> inline D convert(S x) { return To<D>::from(canon(x)); }

// The element of a 3-D grid position in a layout of three dimensions (outermost first).
inline long loc3(uint3 p, constant long* strides) {
    return long(p.z) * strides[0] + long(p.y) * strides[1] + long(p.x) * strides[2];
}

// The contiguous output index of a 3-D grid position.
inline long out3(uint3 p, constant int* shape) {
    return (long(p.z) * shape[1] + p.y) * shape[2] + p.x;
}

} // namespace nuvec
