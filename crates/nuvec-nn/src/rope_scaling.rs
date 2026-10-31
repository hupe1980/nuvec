//! Rotary-embedding frequency scalings (mlx-lm 0.32.2's `rope_utils.py`): linear, Llama 3,
//! YaRN, LongRoPE (`SuScaledRoPE`) and the proportional RoPE of Gemma 4's global layers.
//! [`crate::Rope::scaled`] builds a rotary embedding from one.

use nuvec::{Array, Static};

/// How a rotary embedding's frequencies (or positions) are scaled, as mlx-lm's RoPE
/// classes. Frequencies are computed in float32 with mlx-lm's formulas.
#[derive(Debug, Clone, PartialEq, Static)]
#[non_exhaustive]
pub enum RopeScaling {
    /// The plain `base^(-2i/dims)` frequencies (`default`).
    None,
    /// Positions divided by `factor` (`linear`).
    Linear {
        /// The division factor.
        factor: f32,
    },
    /// Llama 3's frequency-dependent scaling (`Llama3RoPE`).
    Llama3(Llama3),
    /// YaRN (`YarnRoPE`): interpolated frequencies and an input scale (`mscale`).
    Yarn(Yarn),
    /// LongRoPE (`SuScaledRoPE`): per-pair long factors and an input scale.
    LongRope(LongRope),
    /// Gemma 4's proportional RoPE (`ProportionalRoPE`): the first `rotated_dims` of the
    /// head's features rotate with exponents over the whole head, scaled by `factor`; the
    /// rest keep their values (infinite wavelength).
    Proportional {
        /// Features that rotate (a multiple of 2, at most the head's).
        rotated_dims: i32,
        /// The frequencies' scale.
        factor: f32,
    },
}

/// [`RopeScaling::Llama3`]'s parameters (mlx-lm's defaults).
#[derive(Debug, Clone, Copy, PartialEq, Static)]
pub struct Llama3 {
    /// Scale of low frequencies.
    pub factor: f32,
    /// Wavelength bound (in original contexts) of scaled low frequencies.
    pub low_freq_factor: f32,
    /// Wavelength bound of unscaled high frequencies.
    pub high_freq_factor: f32,
    /// The context length the model was trained with.
    pub original_max_position_embeddings: f32,
}

impl Default for Llama3 {
    fn default() -> Self {
        Self {
            factor: 1.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        }
    }
}

/// [`RopeScaling::Yarn`]'s parameters (mlx-lm's defaults; Hugging Face's `truncate` is
/// ignored, as by mlx-lm).
#[derive(Debug, Clone, Copy, PartialEq, Static)]
pub struct Yarn {
    /// The context's scale (`factor`).
    pub factor: f32,
    /// The context length the model was trained with.
    pub original_max_position_embeddings: f32,
    /// Rotations above which frequencies are kept (`beta_fast`).
    pub beta_fast: f32,
    /// Rotations below which frequencies are interpolated (`beta_slow`).
    pub beta_slow: f32,
    /// `mscale`.
    pub mscale: f32,
    /// `mscale_all_dim`.
    pub mscale_all_dim: f32,
}

impl Default for Yarn {
    fn default() -> Self {
        Self {
            factor: 1.0,
            original_max_position_embeddings: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        }
    }
}

/// [`RopeScaling::LongRope`]'s parameters (mlx-lm's defaults, long factors).
#[derive(Debug, Clone, PartialEq, Static)]
pub struct LongRope {
    /// Per-pair factors (`long_factor`; one value applies to every pair).
    pub long_factor: Vec<f32>,
    /// The maximum context length.
    pub max_position_embeddings: f32,
    /// The context length the model was trained with.
    pub original_max_position_embeddings: f32,
    /// An explicit input scale (`long_mscale`).
    pub long_mscale: Option<f32>,
}

impl Default for LongRope {
    fn default() -> Self {
        Self {
            long_factor: vec![1.0],
            max_position_embeddings: 131_072.0,
            original_max_position_embeddings: 4096.0,
            long_mscale: None,
        }
    }
}

/// `base^(i / dims)` for `i = 0, 2, …, upto − 2`, in float32 (mlx-lm's `base ** (arange(0,
/// dims, 2) / dims)`).
fn powers(base: f32, upto: i32, dims: i32) -> Vec<f32> {
    (0..upto)
        .step_by(2)
        .map(|i| base.powf(i as f32 / dims as f32))
        .collect()
}

impl RopeScaling {
    /// The per-pair frequencies (`[dims / 2]`, float32) for `dims` rotated features and
    /// `base`, or `None` when the plain `base` (and [`RopeScaling::position_scale`])
    /// applies.
    pub fn frequencies(&self, dims: i32, base: f32) -> Option<Array> {
        let freqs: Vec<f32> = match self {
            Self::None | Self::Linear { .. } => return None,
            Self::Llama3(p) => {
                let old = p.original_max_position_embeddings;
                let (low_wavelen, high_wavelen) =
                    (old / p.low_freq_factor, old / p.high_freq_factor);
                powers(base, dims, dims)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2.0 * std::f32::consts::PI * freq;
                        let scaled = if wavelen > low_wavelen {
                            freq * p.factor
                        } else {
                            freq
                        };
                        if wavelen > high_wavelen && wavelen < low_wavelen {
                            let smooth = (old / wavelen - p.low_freq_factor)
                                / (p.high_freq_factor - p.low_freq_factor);
                            scaled / ((1.0 - smooth) / p.factor + smooth)
                        } else {
                            scaled
                        }
                    })
                    .collect()
            }
            Self::Yarn(p) => {
                let correction = |rotations: f32| {
                    (dims as f32
                        * (p.original_max_position_embeddings
                            / (rotations * 2.0 * std::f32::consts::PI))
                            .ln())
                        / (2.0 * base.ln())
                };
                let low = correction(p.beta_fast).floor().max(0.0);
                let mut high = correction(p.beta_slow).ceil().min((dims - 1) as f32);
                if low == high {
                    high += 0.001;
                }
                powers(base, dims, dims)
                    .into_iter()
                    .enumerate()
                    .map(|(i, extra)| {
                        let inter = p.factor * extra;
                        let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                        let mask = 1.0 - ramp;
                        (inter * extra) / (inter * mask + extra * (1.0 - mask))
                    })
                    .collect()
            }
            Self::LongRope(p) => {
                let f = powers(base, dims, dims);
                let factor = |i: usize| match p.long_factor.as_slice() {
                    [one] => *one,
                    all => all.get(i).copied().unwrap_or(1.0),
                };
                f.into_iter()
                    .enumerate()
                    .map(|(i, x)| factor(i) * x)
                    .collect()
            }
            Self::Proportional {
                rotated_dims,
                factor,
            } => {
                let mut f: Vec<f32> = powers(base, *rotated_dims, dims)
                    .into_iter()
                    .map(|x| factor * x)
                    .collect();
                f.resize((dims / 2) as usize, f32::INFINITY);
                f
            }
        };
        Some(Array::from(freqs))
    }

    /// The scale mlx-lm applies to the rotated features before the rotation (YaRN's
    /// `mscale`, LongRoPE's; 1 otherwise).
    pub fn mscale(&self) -> f32 {
        let yarn = |scale: f32, m: f32| {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * m * scale.ln() + 1.0
            }
        };
        match self {
            Self::Yarn(p) => yarn(p.factor, p.mscale) / yarn(p.factor, p.mscale_all_dim),
            Self::LongRope(p) => {
                let factor = p.max_position_embeddings / p.original_max_position_embeddings;
                p.long_mscale.unwrap_or(if factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / p.original_max_position_embeddings.ln()).sqrt()
                })
            }
            _ => 1.0,
        }
    }

    /// The scale of positions (`1 / factor` for [`RopeScaling::Linear`]; 1 otherwise).
    pub fn position_scale(&self) -> f32 {
        match self {
            Self::Linear { factor } => 1.0 / factor,
            _ => 1.0,
        }
    }
}
