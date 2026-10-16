//! Device facts.

/// What the GPU is and what it can do.
#[derive(Debug, Clone, PartialEq, Eq)]
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
