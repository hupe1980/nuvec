//! Metal kernel sources and the precompiled `bessemer.metallib`.
//!
//! The build script compiles every shader under `src/metal` with Apple's Metal toolchain
//! and embeds the resulting library. If the toolchain is missing, or precompilation is
//! disabled with `BESSEMER_METALLIB=off`, the sources are embedded instead so they can
//! be compiled at runtime. [`MODE`] reports which of these happened.
//!
//! **Status:** pre-alpha. The only kernel is a placeholder that proves the pipeline.

/// How the kernel library is provided in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetallibMode {
    /// Kernels were compiled to `bessemer.metallib` at build time.
    Precompiled,
    /// Shader sources are embedded for compilation at runtime.
    RuntimeSource,
    /// The target has no Metal support.
    Unavailable,
}

/// The mode this build was configured with.
pub const MODE: MetallibMode = if cfg!(bessemer_metallib = "precompiled") {
    MetallibMode::Precompiled
} else if cfg!(bessemer_metallib = "source") {
    MetallibMode::RuntimeSource
} else {
    MetallibMode::Unavailable
};

#[cfg(bessemer_metallib = "precompiled")]
const METALLIB: Option<&[u8]> = Some(include_bytes!(concat!(
    env!("OUT_DIR"),
    "/bessemer.metallib"
)));
#[cfg(not(bessemer_metallib = "precompiled"))]
const METALLIB: Option<&[u8]> = None;

const KERNEL_NAMES: &[&str] = include!(concat!(env!("OUT_DIR"), "/kernel_names.rs"));
const SOURCES: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/sources.rs"));

/// The precompiled library bytes, if [`MODE`] is [`MetallibMode::Precompiled`].
pub fn metallib() -> Option<&'static [u8]> {
    METALLIB
}

/// The kernel function names in the precompiled library, sorted. Empty in other modes.
pub fn kernel_names() -> &'static [&'static str] {
    KERNEL_NAMES
}

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
        assert_eq!(
            MODE == MetallibMode::Precompiled,
            !kernel_names().is_empty()
        );
        assert_eq!(MODE == MetallibMode::RuntimeSource, !sources().is_empty());
    }

    #[test]
    fn precompiled_library_contains_placeholder() {
        if MODE != MetallibMode::Precompiled {
            return;
        }
        let lib = metallib().expect("precompiled mode has a library");
        assert!(lib.starts_with(b"MTLB"), "not a metallib");
        assert!(kernel_names().contains(&"bessemer_placeholder"));
    }

    #[test]
    fn runtime_sources_include_placeholder_and_headers() {
        if MODE != MetallibMode::RuntimeSource {
            return;
        }
        let paths: Vec<_> = sources().iter().map(|(p, _)| *p).collect();
        assert!(paths.contains(&"placeholder.metal"));
        assert!(paths.contains(&"common.h"));
    }

    #[test]
    fn required_mode_is_precompiled() {
        if option_env!("BESSEMER_REQUIRE_METALLIB") == Some("1") {
            assert_eq!(MODE, MetallibMode::Precompiled);
        }
    }
}
