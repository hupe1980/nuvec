//! Pure-Rust CPU kernels for bessemer.
//!
//! Kernels work on typed slices and know nothing about arrays or graphs. `bessemer-core`
//! dispatches to them.
//!
//! **Status:** pre-alpha. Only the contiguous element-wise binary kernel exists.

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Applies `f` element-wise: `out[i] = f(a[i], b[i])`.
///
/// # Panics
///
/// If the three slices differ in length. Callers validate shapes beforehand.
pub fn binary_map<T: Copy>(a: &[T], b: &[T], out: &mut [T], f: impl Fn(T, T) -> T) {
    assert!(
        a.len() == out.len() && b.len() == out.len(),
        "binary_map: length mismatch"
    );
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = f(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn binary_map_applies_elementwise() {
        let mut out = [0i32; 3];
        binary_map(&[1, 2, 3], &[10, 20, 30], &mut out, |x, y| x * y);
        assert_eq!(out, [10, 40, 90]);
    }
}
