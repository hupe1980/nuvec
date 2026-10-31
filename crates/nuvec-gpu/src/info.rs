//! Device facts.

/// What the GPU is and what it can do.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// Marketing name, e.g. `Apple M4 Pro`.
    pub name: String,
    /// Metal's architecture string, e.g. `applegpu_g16s`.
    pub architecture: String,
    /// GPU generation parsed from the architecture string (16 for `applegpu_g16s`), if any.
    /// Neural-accelerator kernels need generation 17 or later.
    pub generation: Option<u32>,
    /// Class letter after the generation (`s` for `applegpu_g16s`), if any. Kept verbatim:
    /// the letter-to-tier mapping is not documented by Apple.
    pub class: Option<char>,
    /// Bytes the GPU can use without hurting performance.
    pub recommended_working_set: u64,
    /// Maximum threads per threadgroup along one dimension.
    pub max_threads_per_threadgroup: usize,
    /// The machine's physical memory in bytes (unified memory: shared by the CPU and the
    /// GPU; `hw.memsize`, MLX's `device_info()["memory_size"]`).
    pub physical_memory: u64,
}

/// Generation and class letter of an `applegpu_g<NN><class>` architecture string.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn parse_architecture(arch: &str) -> (Option<u32>, Option<char>) {
    let Some(rest) = arch.strip_prefix("applegpu_g") else {
        return (None, None);
    };
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    let generation = rest[..digits].parse().ok();
    let class = rest[digits..]
        .chars()
        .next()
        .filter(char::is_ascii_lowercase);
    (generation, class)
}

#[cfg(test)]
mod tests {
    use super::parse_architecture;

    #[test]
    fn parses_apple_gpu_architectures() {
        assert_eq!(parse_architecture("applegpu_g16s"), (Some(16), Some('s')));
        assert_eq!(parse_architecture("applegpu_g17p"), (Some(17), Some('p')));
        assert_eq!(parse_architecture("applegpu_g13"), (Some(13), None));
        assert_eq!(parse_architecture("Paravirtual"), (None, None));
    }
}

/// Why the Neural Accelerator (NAX) kernels are not used ([`NaxStatus::Off`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NaxReason {
    /// `NUVEC_NAX=off`.
    Override,
    /// The OS is older than macOS 26.2 (Metal 4 TensorOps).
    Os,
    /// The GPU is not an M5-class one (MLX's rule: generation 17 or later, 18 for the `p`
    /// class), or its generation is unknown.
    Generation,
    /// This build has no NAX kernel library (an SDK older than 26.2 or no Metal).
    NoLibrary,
    /// An eligible GPU, but nobody has measured nuvec's NAX kernels on one yet: `auto`
    /// keeps them off until an M5 measurement is recorded (D-078). `NUVEC_NAX=on` forces
    /// them.
    Unmeasured,
    /// A NAX pipeline failed to build; the existing kernels are used from then on.
    PipelineError,
}

/// Whether products run on the Neural Accelerator kernels ([`crate::Device::nax`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NaxStatus {
    /// They run where MLX would route to its NAX kernels; `forced` when `NUVEC_NAX=on`
    /// enabled them on a GPU MLX would not (TensorOps run on every Apple GPU of macOS 26.2,
    /// not on matrix units).
    Active {
        /// Enabled by `NUVEC_NAX=on` rather than MLX's eligibility rule.
        forced: bool,
    },
    /// They do not run.
    Off(NaxReason),
}

impl NaxStatus {
    /// Whether NAX kernels run.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }
}

/// The `NUVEC_NAX` override: `auto` (default), `on` or `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum NaxOverride {
    Auto,
    On,
    Off,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl NaxOverride {
    /// From `NUVEC_NAX` (case-insensitive; anything else is `auto`, with a warning).
    pub(crate) fn from_env() -> Self {
        match std::env::var("NUVEC_NAX") {
            Err(_) => Self::Auto,
            Ok(v) => match v.to_ascii_lowercase().as_str() {
                "" | "auto" => Self::Auto,
                "on" | "1" => Self::On,
                "off" | "0" => Self::Off,
                other => {
                    eprintln!("nuvec: ignoring NUVEC_NAX={other:?} (auto, on or off)");
                    Self::Auto
                }
            },
        }
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
/// MLX's NAX eligibility (`is_nax_available`): macOS 26.2 or later and GPU generation 17 or
/// later (18 or later for the `p` class).
pub(crate) fn nax_eligible(generation: Option<u32>, class: Option<char>, os: (u32, u32)) -> bool {
    let min = if class == Some('p') { 18 } else { 17 };
    os >= (26, 2) && generation.is_some_and(|g| g >= min)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
/// The NAX decision: the override, the OS (TensorOps need macOS 26.2 even when forced),
/// the library, then MLX's rule, with `auto` off on eligible GPUs until measured.
pub(crate) fn nax_status(
    generation: Option<u32>,
    class: Option<char>,
    os: (u32, u32),
    overriding: NaxOverride,
    library: bool,
) -> NaxStatus {
    let eligible = nax_eligible(generation, class, os);
    match overriding {
        NaxOverride::Off => NaxStatus::Off(NaxReason::Override),
        _ if os < (26, 2) => NaxStatus::Off(NaxReason::Os),
        _ if !library => NaxStatus::Off(NaxReason::NoLibrary),
        NaxOverride::On => NaxStatus::Active { forced: !eligible },
        NaxOverride::Auto if eligible => NaxStatus::Off(NaxReason::Unmeasured),
        NaxOverride::Auto => NaxStatus::Off(NaxReason::Generation),
    }
}

#[cfg(test)]
mod nax_tests {
    use super::*;

    #[test]
    fn the_nax_decision_follows_mlx_and_the_override() {
        use NaxOverride::{Auto, Off, On};
        let off = |r| NaxStatus::Off(r);
        // MLX's rule over generations, classes and OS versions.
        for generation in [None, Some(13), Some(16), Some(17), Some(18)] {
            for class in ['g', 's', 'd', 'p', 'c'] {
                for os in [(15, 6), (26, 0), (26, 1), (26, 2), (26, 6), (27, 0)] {
                    let min = if class == 'p' { 18 } else { 17 };
                    let eligible = os >= (26, 2) && generation.is_some_and(|g| g >= min);
                    assert_eq!(nax_eligible(generation, Some(class), os), eligible);
                    let auto = nax_status(generation, Some(class), os, Auto, true);
                    let want = match (os >= (26, 2), eligible) {
                        (false, _) => off(NaxReason::Os),
                        (true, true) => off(NaxReason::Unmeasured),
                        (true, false) => off(NaxReason::Generation),
                    };
                    assert_eq!(auto, want, "{generation:?} {class} {os:?}");
                    let on = nax_status(generation, Some(class), os, On, true);
                    let want = match os >= (26, 2) {
                        true => NaxStatus::Active { forced: !eligible },
                        false => off(NaxReason::Os),
                    };
                    assert_eq!(on, want);
                    assert_eq!(
                        nax_status(generation, Some(class), os, Off, true),
                        off(NaxReason::Override)
                    );
                }
            }
        }
        // No library: never active.
        assert_eq!(
            nax_status(Some(17), Some('s'), (26, 2), On, false),
            off(NaxReason::NoLibrary)
        );
    }
}
