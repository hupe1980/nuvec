// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared definitions for nuvec's Metal kernels: element types, per-type arithmetic with the
// same semantics as the CPU backend, conversions, and strided indexing.

#pragma once

// No implicit fused multiply-add (also when compiled from source at runtime, where the
// `-ffp-contract=off` build flag does not apply): results round like the CPU backend.
#pragma clang fp contract(off)

#include <metal_stdlib>

// Elements per thread of the flat `_cn` kernels: 8 bytes' worth (MLX's work per thread,
// used from 2^16 elements).
#define NUVEC_PER_THREAD(T) (sizeof(T) < 8 ? int(8 / sizeof(T)) : 1)

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
    inline bool is_nan_(T) { return false; }                                 \
    /* Toward negative infinity (MLX's FloorDivide); 0 for a zero divisor, */ \
    /* and MIN / -1 wraps. */                                                 \
    inline T op_floordiv(T a, T b) {                                         \
        if (b == T(0)) return T(0);                                          \
        if (T(-1) < T(0) && b == T(-1)) return op_neg(a);                    \
        T q = a / b;                                                         \
        if (T(-1) < T(0) && a % b != T(0) && ((a < T(0)) != (b < T(0)))) {  \
            q = T(q - T(1));                                                 \
        }                                                                    \
        return q;                                                            \
    }                                                                        \
    /* Squaring (MLX's Power); 0 for a negative exponent; wraps. */           \
    inline T op_pow(T a, T b) {                                              \
        if (b < T(0)) return T(0);                                           \
        U r = U(1), x = U(a), e = U(b);                                      \
        while (e) {                                                          \
            if (e & U(1)) r = r * x;                                         \
            e >>= 1;                                                         \
            x = x * x;                                                       \
        }                                                                    \
        return T(r);                                                         \
    }                                                                        \
    /* The divisor's sign (MLX's Remainder); 0 for a zero divisor. */         \
    inline T op_remainder(T a, T b) {                                        \
        if (b == T(0)) return T(0);                                          \
        if (T(-1) < T(0) && b == T(-1)) return T(0);                         \
        T r = a % b;                                                         \
        if (r != T(0) && ((r < T(0)) != (b < T(0)))) r = op_add(r, b);       \
        return r;                                                            \
    }                                                                        \
    inline T op_bitwise_and(T a, T b) { return a & b; }                      \
    inline T op_bitwise_or(T a, T b) { return a | b; }                       \
    inline T op_bitwise_xor(T a, T b) { return a ^ b; }                      \
    inline T op_bitwise_invert(T a) { return ~a; }                           \
    /* C++'s shifts: types narrower than int are promoted first; the shift */ \
    /* is taken modulo the (promoted) width. */                               \
    inline T op_left_shift(T a, T b) {                                       \
        if (sizeof(T) < 4) return T(int(a) << (int(b) & 31));                \
        return T(a << (b & T(sizeof(T) * 8 - 1)));                           \
    }                                                                        \
    inline T op_right_shift(T a, T b) {                                      \
        if (sizeof(T) < 4) return T(int(a) >> (int(b) & 31));                \
        return T(a >> (b & T(sizeof(T) * 8 - 1)));                           \
    }

NUVEC_INT_OPS(uchar, uint)
NUVEC_INT_OPS(ushort, uint)
NUVEC_INT_OPS(uint, uint)
NUVEC_INT_OPS(ulong, ulong)
NUVEC_INT_OPS(char, uint)
NUVEC_INT_OPS(short, uint)
NUVEC_INT_OPS(int, uint)
NUVEC_INT_OPS(long, ulong)

// ── Arithmetic: half precision computes in float and rounds once ────────────

// MLX's log1p (`utils.h`; its `Limits<float>::max` is infinity).
inline float log1p_(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == INFINITY) {
        return INFINITY;
    }
    if (xp1 == 1.0f) {
        return x;
    }
    return x * (precise::log(xp1) / (xp1 - 1.0f));
}

// expm1f and erf: Norbert Juffa's single-precision approximations (BSD-2-Clause; see
// THIRD_PARTY_NOTICES), maximum errors 0.997 and 0.995 units in the last place, as MLX's
// `expm1f.h` and `erf.h` but with the original's two-step reduction by ln 2 (MLX's drops
// the low part, an error of up to tens of units in the last place).
inline float expm1f_(float a) {
    float j = fma(1.442695f, a, 12582912.f);
    j = j - 12582912.0f;
    int i = int(j);
    float f = fma(j, -6.93145752e-1f, a);
    f = fma(j, -1.42860677e-6f, f);
    float s = f * f;
    if (a == 0.0f) {
        s = a;
    }
    float r = 1.97350979e-4f;
    r = fma(r, f, 1.39309070e-3f);
    r = fma(r, f, 8.33343994e-3f);
    r = fma(r, f, 4.16668020e-2f);
    r = fma(r, f, 1.66666716e-1f);
    r = fma(r, f, 4.99999970e-1f);
    float u = (j == 1) ? (f + 0.5f) : f;
    float v = fma(r, s, u);
    s = 0.5f;
    float t = ldexp(s, i);
    float y = t - s;
    float x = (t - y) - s;
    r = fma(v, t, x) + y;
    r = r + r;
    if (j == 0) {
        r = v;
    }
    if (j == 1) {
        r = v + v;
    }
    if (fabs(a - 1.0f) > 88.0f) {
        r = pow(2.0f, a);
        r = fma(r, r, -1.0f);
    }
    return r;
}

inline float erf_(float a) {
    float r, s, t, u;
    t = fabs(a);
    s = a * a;
    if (t > 0.927734375f) {
        r = fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = fma(r, s, u);
        r = fma(r, t, -1.06777877e-1f);
        r = fma(r, t, -6.34846687e-1f);
        r = fma(r, t, -1.28717512e-1f);
        r = fma(r, t, -t);
        r = -expm1f_(r);
        r = copysign(r, a);
    } else {
        r = -5.96761703e-4f;
        r = fma(r, s, 4.99119423e-3f);
        r = fma(r, s, -2.67681349e-2f);
        r = fma(r, s, 1.12819925e-1f);
        r = fma(r, s, -3.76125336e-1f);
        r = fma(r, s, 1.28379166e-1f);
        r = fma(r, a, a);
    }
    return r;
}

// MLX's Remainder: `fmod`, then the divisor's sign.
inline float remainder_(float a, float b) {
    float r = fmod(a, b);
    if (r != 0 && ((r < 0) != (b < 0))) {
        r += b;
    }
    return r;
}

// MLX's LogAddExp: the larger operand plus log1p(exp(smaller - larger)).
inline float logaddexp_(float x, float y) {
    if (isnan(x) || isnan(y)) {
        return NAN;
    }
    float mx = max(x, y);
    float mn = min(x, y);
    return (mn == -INFINITY || mx == INFINITY) ? mx : mx + log1p_(precise::exp(mn - mx));
}

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
    inline T op_erfinv(T a) { return T(erfinv_(float(a))); }                                  \
    inline T op_logaddexp(T a, T b) { return T(logaddexp_(float(a), float(b))); }            \
    inline T op_floordiv(T a, T b) { return T(precise::floor(float(op_div(a, b)))); }    \
    inline T op_erf(T a) { return T(erf_(float(a))); }                                        \
    inline T op_expm1(T a) { return T(expm1f_(float(a))); }                                   \
    inline T op_log1p(T a) { return T(log1p_(float(a))); }                                    \
    inline T op_log2(T a) { return T(precise::log2(float(a))); }                              \
    inline T op_log10(T a) { return T(precise::log10(float(a))); }                            \
    inline T op_tan(T a) { return T(precise::tan(float(a))); }                                \
    inline T op_sinh(T a) { return T(precise::sinh(float(a))); }                              \
    inline T op_cosh(T a) { return T(precise::cosh(float(a))); }                              \
    inline T op_arccos(T a) { return T(precise::acos(float(a))); }                            \
    inline T op_arccosh(T a) { return T(precise::acosh(float(a))); }                          \
    inline T op_arcsin(T a) { return T(precise::asin(float(a))); }                            \
    inline T op_arcsinh(T a) { return T(precise::asinh(float(a))); }                          \
    inline T op_arctan(T a) { return T(precise::atan(float(a))); }                            \
    inline T op_arctanh(T a) { return T(precise::atanh(float(a))); }                          \
    inline T op_rsqrt(T a) { return T(precise::rsqrt(float(a))); }                            \
    inline T op_floor(T a) { return T(metal::floor(float(a))); }                              \
    inline T op_ceil(T a) { return T(metal::ceil(float(a))); }                                \
    inline T op_round(T a) { return T(metal::rint(float(a))); }                               \
    inline T op_arctan2(T a, T b) { return T(precise::atan2(float(a), float(b))); }

NUVEC_FLOAT_OPS(half)
NUVEC_FLOAT_OPS(bfloat)
NUVEC_FLOAT_OPS(float)

inline float op_remainder(float a, float b) { return remainder_(a, b); }
inline bfloat op_remainder(bfloat a, bfloat b) { return bfloat(remainder_(float(a), float(b))); }

// MLX's Remainder in half precision: `fmod` is exact, so computing it in the type gives the
// bits of the float computation, at the speed of half arithmetic (MLX's).
inline half op_remainder(half a, half b) {
    half r = fmod(a, b);
    if (r != half(0) && ((r < half(0)) != (b < half(0)))) {
        r += b;
    }
    return r;
}

// MLX's Power: `metal::pow` in the type (half natively; bfloat through float, as MLX's
// `bfloat16_t`).
inline half op_pow(half a, half b) { return metal::pow(a, b); }
inline float op_pow(float a, float b) { return metal::pow(a, b); }
inline bfloat op_pow(bfloat a, bfloat b) { return bfloat(metal::pow(float(a), float(b))); }

// ── Arithmetic: bool (add = OR, subtract = XOR, multiply = AND, as in MLX) ──

inline nbool op_add(nbool a, nbool b) { return nbool{uchar(a.v | b.v)}; }
inline nbool op_sub(nbool a, nbool b) { return nbool{uchar(a.v ^ b.v)}; }
inline nbool op_mul(nbool a, nbool b) { return nbool{uchar(a.v & b.v)}; }
inline nbool op_max(nbool a, nbool b) { return nbool{uchar(a.v | b.v)}; }
inline nbool op_min(nbool a, nbool b) { return nbool{uchar(a.v & b.v)}; }
inline nbool op_abs(nbool a) { return a; }
// Bitwise operations of booleans (MLX: `bitwise_invert` negates; shifts run as uint8).
inline nbool op_bitwise_and(nbool a, nbool b) { return nbool{uchar(a.v & b.v)}; }
inline nbool op_bitwise_or(nbool a, nbool b) { return nbool{uchar(a.v | b.v)}; }
inline nbool op_bitwise_xor(nbool a, nbool b) { return nbool{uchar(a.v ^ b.v)}; }
inline nbool op_bitwise_invert(nbool a) { return nbool{uchar(a.v ^ 1)}; }
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

// `loc3` and `out3` in 32-bit arithmetic, for layouts whose offsets stay below 2^31: Apple
// GPUs emulate 64-bit integer products (MLX's general kernels take `int` indices then).
inline int loc3s(uint3 p, constant long* strides) {
    return int(p.z) * int(strides[0]) + int(p.y) * int(strides[1]) + int(p.x) * int(strides[2]);
}
inline int out3s(uint3 p, constant int* shape) {
    return (int(p.z) * shape[1] + int(p.y)) * shape[2] + int(p.x);
}

} // namespace nuvec
