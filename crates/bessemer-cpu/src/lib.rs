//! Pure-Rust CPU backend for bessemer.
//!
//! **Status:** pre-alpha. This crate has no functional API yet.

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::VERSION.is_empty());
    }
}
