//! Metal kernel sources and the precompiled `nuvec.metallib`.
//!
//! The build script compiles every shader under `src/metal` with Apple's Metal toolchain
//! and embeds the resulting library. If the toolchain is missing, or precompilation is
//! disabled with `NUVEC_METALLIB=off`, the sources are embedded instead so they can
//! be compiled at runtime. [`MODE`] reports which of these happened.
//!
//! The kernels:
//!
//! - element-wise operations, casts, copies, gather and scatter along an axis;
//! - MLX's reductions, sort (block and multi-block merge) and scans;
//! - MLX's GEMM (steel tiles, split-K, fused `addmm`), GEMV and gather GEMM (`gather_mm`);
//! - MLX's quantization, affine and floating-point (MXFP4, MXFP8, NVFP4): quantize,
//!   dequantize and the quantized products, including the mixture-of-experts gather
//!   kernels, and E4M3 conversions;
//! - MLX's fused kernels: softmax, log-sum-exp, RMS and layer normalisation, RoPE, cross
//!   entropy, scaled dot-product attention (vector, two-pass and tiled);
//! - mlx-lm's Gated DeltaNet kernels;
//! - nuvec's own: residual add + RMS norm, norm + RoPE + cache write, few-row quantized
//!   products, depthwise 1-D convolution and random bits;
//! - in a second library for macOS 26.2 and later ([`NAX_MODE`]), GEMM on the GPU's matrix
//!   units through Metal 4 TensorOps (the Neural Accelerators of M5-class GPUs).
//!
//! With feature `cuda`, [`cuda`] holds the CUDA kernels: generated PTX, and Tile kernels
//! (cutile) on Linux.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.
//!
//! **Status:** pre-alpha.

#[cfg(feature = "cuda")]
pub mod cuda;

/// How the kernel library is provided in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetallibMode {
    /// Kernels were compiled to `nuvec.metallib` at build time.
    Precompiled,
    /// Shader sources are embedded for compilation at runtime.
    RuntimeSource,
    /// The target has no Metal support.
    Unavailable,
}

/// The mode this build was configured with.
pub const MODE: MetallibMode = if cfg!(nuvec_metallib = "precompiled") {
    MetallibMode::Precompiled
} else if cfg!(nuvec_metallib = "source") {
    MetallibMode::RuntimeSource
} else {
    MetallibMode::Unavailable
};

#[cfg(nuvec_metallib = "precompiled")]
const METALLIB: Option<&[u8]> = Some(include_bytes!(concat!(env!("OUT_DIR"), "/nuvec.metallib")));
#[cfg(not(nuvec_metallib = "precompiled"))]
const METALLIB: Option<&[u8]> = None;

const SOURCES: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/sources.rs"));
const NAX_SOURCES: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/nax_sources.rs"));

#[cfg(nuvec_nax_metallib = "precompiled")]
const NAX_METALLIB: Option<&[u8]> = Some(include_bytes!(concat!(
    env!("OUT_DIR"),
    "/nuvec_nax.metallib"
)));
#[cfg(not(nuvec_nax_metallib = "precompiled"))]
const NAX_METALLIB: Option<&[u8]> = None;

/// How the Neural Accelerator (NAX) kernel library is provided: Metal 4.0 kernels for
/// macOS 26.2 and later, in a library of their own so the baseline keeps macOS 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NaxMode {
    /// Compiled to `nuvec_nax.metallib` at build time.
    Precompiled,
    /// Sources embedded for compilation at runtime (with the baseline's runtime mode).
    RuntimeSource,
    /// Not built: an SDK older than 26.2, no Metal Performance Primitives, or no Metal.
    Absent,
}

/// The NAX library's mode in this build.
pub const NAX_MODE: NaxMode = if cfg!(nuvec_nax_metallib = "precompiled") {
    NaxMode::Precompiled
} else if cfg!(nuvec_nax_metallib = "source") {
    NaxMode::RuntimeSource
} else {
    NaxMode::Absent
};

/// The precompiled NAX library bytes, if [`NAX_MODE`] is [`NaxMode::Precompiled`].
pub fn nax_metallib() -> Option<&'static [u8]> {
    NAX_METALLIB
}

/// The embedded NAX shader sources, if [`NAX_MODE`] is [`NaxMode::RuntimeSource`].
pub fn nax_sources() -> &'static [(&'static str, &'static str)] {
    NAX_SOURCES
}

include!("contracted.rs");

/// Whether the shader file `path` (relative to `src/metal`) is compiled with floating-point
/// contraction (MLX's default) rather than `-ffp-contract=off`, as the build script compiles
/// it; a runtime compiler must do the same to get the same results.
pub fn contracted(path: &str) -> bool {
    CONTRACTED.contains(&path)
}

/// The precompiled library bytes, if [`MODE`] is [`MetallibMode::Precompiled`].
pub fn metallib() -> Option<&'static [u8]> {
    METALLIB
}

/// `common.h`, the header of nuvec's element types, arithmetic, conversions and strided
/// indexing, for kernels generated at runtime. Embedded in every mode.
pub const COMMON_H: &str = include_str!("metal/common.h");

/// The embedded shader sources and headers as `(path relative to src/metal, source)`, if
/// [`MODE`] is [`MetallibMode::RuntimeSource`]. Empty in other modes.
pub fn sources() -> &'static [(&'static str, &'static str)] {
    SOURCES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_match_mode() {
        assert_eq!(MODE == MetallibMode::Precompiled, metallib().is_some());
        assert_eq!(MODE == MetallibMode::RuntimeSource, !sources().is_empty());
    }

    #[test]
    fn precompiled_library_is_a_metallib() {
        // Its kernels are looked up by name in `nuvec-gpu`'s runtime tests.
        if MODE != MetallibMode::Precompiled {
            return;
        }
        let lib = metallib().expect("precompiled mode has a library");
        assert!(lib.starts_with(b"MTLB"), "not a metallib");
    }

    #[test]
    fn runtime_sources_include_kernels_and_headers() {
        if MODE != MetallibMode::RuntimeSource {
            return;
        }
        let paths: Vec<_> = sources().iter().map(|(p, _)| *p).collect();
        assert!(paths.contains(&"elementwise.metal"));
        assert!(paths.contains(&"common.h"));
    }

    #[test]
    fn nax_accessors_match_mode() {
        assert_eq!(NAX_MODE == NaxMode::Precompiled, nax_metallib().is_some());
        assert_eq!(
            NAX_MODE == NaxMode::RuntimeSource,
            !nax_sources().is_empty()
        );
        if option_env!("NUVEC_REQUIRE_NAX") == Some("1") {
            assert_eq!(NAX_MODE, NaxMode::Precompiled);
        }
        if let Some(lib) = nax_metallib() {
            assert!(lib.starts_with(b"MTLB"), "not a metallib");
        }
    }

    #[test]
    fn required_mode_is_precompiled() {
        if option_env!("NUVEC_REQUIRE_METALLIB") == Some("1") {
            assert_eq!(MODE, MetallibMode::Precompiled);
        }
    }
}
