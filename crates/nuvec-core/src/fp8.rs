//! The 8- and 4-bit floating-point codes of MLX's microscaling quantization modes: E4M3
//! (mxfp8 values and nvfp4 scales), E2M1 (mxfp4 and nvfp4 values) and E8M0 (mx scales),
//! transcribed from MLX v0.32.4's Metal headers (`fp8.h`, `fp4.h`) so that the CPU writes
//! the bytes MLX's GPU writes.
//!
//! Copyright © 2025-2026 Apple Inc. Licensed under the MIT License:
//! <https://github.com/ml-explore/mlx/blob/main/LICENSE>

use half::f16;

/// The largest finite E4M3 value.
pub(crate) const E4M3_MAX: f32 = 448.0;
/// The largest E2M1 value.
pub(crate) const E2M1_MAX: f32 = 6.0;

/// `x` as E4M3, rounding to nearest even and saturating at ±448 (NaN and ±Inf included,
/// as MLX's `fp8_e4m3`).
pub(crate) fn e4m3_from_f32(x: f32) -> u8 {
    const FP8_MAX: u32 = 543 << 21;
    const DENORM_MASK: u32 = 141 << 23;
    let mut bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    bits ^= sign;
    let code = if bits >= FP8_MAX {
        0x7E
    } else if bits < (121 << 23) {
        let sum = f32::from_bits(bits) + f32::from_bits(DENORM_MASK);
        sum.to_bits().wrapping_sub(DENORM_MASK) as u8
    } else {
        let odd = (bits >> 20) & 1;
        bits = bits.wrapping_add((7u32.wrapping_sub(127) << 23).wrapping_add(0x7FFFF));
        bits = bits.wrapping_add(odd);
        (bits >> 20) as u8
    };
    code | (sign >> 24) as u8
}

/// The value of E4M3 code `code` (through float16, as MLX; `0x7F`/`0xFF` are NaN).
pub(crate) fn e4m3_to_f32(code: u8) -> f32 {
    let v = u16::from(code & 127);
    let sign = u16::from(code >> 7) << 15;
    let u = (v << 7) | (((v + 1) >> 7) << 14) | sign;
    f16::from_f32(f16::from_bits(u).to_f32() * 256.0).to_f32()
}

/// `x` as E2M1 (MLX's `fp4_e2m1`: ties as listed, saturating at ±6, NaN as `0x7`, the sign
/// of zero kept).
pub(crate) fn e2m1_from_f32(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7;
    }
    let sign = if x.is_sign_negative() { 0x8 } else { 0x0 };
    let a = x.abs();
    let code = if a > 5.0 {
        0x7
    } else if a >= 3.5 {
        0x6
    } else if a > 2.5 {
        0x5
    } else if a >= 1.75 {
        0x4
    } else if a > 1.25 {
        0x3
    } else if a >= 0.75 {
        0x2
    } else if a > 0.25 {
        0x1
    } else {
        0x0
    };
    code | sign
}

/// The value of E2M1 code `code` (low nibble): 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives.
pub(crate) fn e2m1_to_f32(code: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = VALUES[usize::from(code & 7)];
    if code & 8 != 0 { -magnitude } else { magnitude }
}

/// `x` as E8M0, a power of two `2^(code - 127)`: the nearest exponent (`round(log2 x)`,
/// halves away from zero), clamped to ±127; non-finite values give `0xFF`, negatives `0x00`.
pub(crate) fn e8m0_from_f32(x: f32) -> u8 {
    if !x.is_finite() {
        return 0xFF;
    }
    if x < 0.0 {
        return 0x00;
    }
    let n = (x.log2().round() as i32).clamp(-127, 127);
    (n + 127) as u8
}

/// The value of E8M0 code `code`: `2^(code - 127)`, with `0x00` as 2⁻¹²⁷ and `0xFF` as +Inf.
pub(crate) fn e8m0_to_f32(code: u8) -> f32 {
    if code == 0 {
        f32::from_bits(0x40_0000)
    } else {
        f32::from_bits(u32::from(code) << 23)
    }
}

/// `x`, or zero (with its sign) if subnormal: Apple GPUs flush float32 subnormals to zero
/// in arithmetic and comparisons, so the CPU applies this wherever MLX's Metal kernels
/// compute in float32 to write the GPU's bytes.
pub(crate) fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// `x`, with a NaN made the canonical (positive, quiet) NaN, as Apple GPUs produce from
/// float32 arithmetic and from float16-to-float32 conversion (bit operations keep the sign).
pub(crate) fn canonical(x: f32) -> f32 {
    if x.is_nan() { f32::NAN } else { x }
}

/// The smallest E8M0 value at least `x` (MLX's `mx_scale_round_up`): scales are
/// `amax / max_element`, so rounding one down would push a group's largest values out of
/// the element range.
pub(crate) fn mx_scale_round_up(x: f32) -> f32 {
    let mut code = e8m0_from_f32(x);
    if code < 0xFE && ftz(e8m0_to_f32(code)) < x {
        code += 1;
    }
    e8m0_to_f32(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
    fn e4m3_matches_mlx_probes() {
        // MLX 0.32.4 `mx.to_fp8` on the GPU.
        let values = [
            0.1,
            1.0,
            448.0,
            500.0,
            -3.3,
            1e-4,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            2f32.powi(-9),
            2f32.powi(-10),
            0.002_929_687_5,
            464.0,
            465.0,
            1.0625,
            1.1875,
        ];
        let expected = [
            29, 56, 126, 126, 197, 0, 0, 128, 126, 254, 126, 1, 0, 2, 126, 126, 56, 58,
        ];
        let got: Vec<u8> = values.iter().map(|&v| e4m3_from_f32(v)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
    fn e4m3_codes_round_trip() {
        // MLX 0.32.4 `mx.from_fp8(codes, mx.float32)`.
        for (code, value) in [
            (0, 0.0),
            (1, 0.001_953_125),
            (7, 0.013_671_875),
            (8, 0.015_625),
            (56, 1.0),
            (126, 448.0),
            (128, -0.0),
            (254, -448.0),
        ] {
            let v = e4m3_to_f32(code);
            assert!(v == value && v.is_sign_negative() == value.is_sign_negative());
        }
        assert!(e4m3_to_f32(0x7F).is_nan() && e4m3_to_f32(0xFF).is_nan());
        for code in (0..=255u8).filter(|c| c & 0x7F != 0x7F) {
            assert_eq!(e4m3_from_f32(e4m3_to_f32(code)), code, "code {code:#x}");
        }
    }

    #[test]
    fn e2m1_round_trips_and_ties() {
        for code in 0..16u8 {
            assert_eq!(e2m1_from_f32(e2m1_to_f32(code)), code, "code {code:#x}");
        }
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, v) in values.iter().enumerate() {
            assert_eq!(e2m1_to_f32(code as u8), *v);
            assert_eq!(e2m1_to_f32(code as u8 | 8), -*v);
        }
        // Ties: up at 0.75, 1.75 and 3.5, down at 0.25, 1.25, 2.5 and 5 (MLX's thresholds).
        let ties = [
            (0.25, 0x0),
            (0.75, 0x2),
            (1.25, 0x2),
            (1.75, 0x4),
            (2.5, 0x4),
            (3.5, 0x6),
            (5.0, 0x6),
            (7.0, 0x7),
        ];
        for (x, code) in ties {
            assert_eq!(e2m1_from_f32(x), code, "{x}");
        }
        assert_eq!(e2m1_from_f32(-0.0), 0x8);
        assert_eq!(e2m1_from_f32(f32::NAN), 0x7);
    }

    #[test]
    fn e8m0_rounds_and_clamps() {
        assert_eq!(e8m0_from_f32(1.0), 127);
        assert_eq!(e8m0_from_f32(0.25), 125);
        assert_eq!(e8m0_from_f32(0.0), 0);
        assert_eq!(e8m0_from_f32(-1.0), 0);
        assert_eq!(e8m0_from_f32(f32::INFINITY), 0xFF);
        assert_eq!(e8m0_from_f32(f32::NAN), 0xFF);
        // 2⁻¹²⁷ (a subnormal), from its bits: Miri perturbs `powi`.
        let tiny = f32::from_bits(0x40_0000);
        assert_eq!(e8m0_to_f32(0), tiny);
        assert_eq!(e8m0_to_f32(0xFF), f32::INFINITY);
        // 1/6 rounds to 2⁻³ but up to 2⁻² (MLX: a group [-1, 0, …] gets scale code 125).
        assert_eq!(mx_scale_round_up(1.0 / 6.0), 0.25);
        assert_eq!(mx_scale_round_up(0.0), tiny);
        assert_eq!(mx_scale_round_up(0.5), 0.5);
    }
}
