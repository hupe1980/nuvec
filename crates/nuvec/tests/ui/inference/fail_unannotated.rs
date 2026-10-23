//! Unannotated closure parameters cannot be inferred through a generic `P: Tree`.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn main() {
    let x = Array;
    let _ = grad(|x| x.sum())(&x);
}
