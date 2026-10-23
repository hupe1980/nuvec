//! Unannotated tuple patterns cannot be inferred either.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn main() {
    let _ = value_and_grad(|p| p.0.matmul(&p.1).sum())(&(Array, Array));
}
