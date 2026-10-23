//! `?` inside a closure whose return type is the generic `R: IntoResult`, ending in `Ok(..)`.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn checked(x: &Array) -> Result<Array> {
    Ok(x.clone())
}

/// Type-checked only: the stub transforms are `unimplemented!()`.
fn check() {
    let x = Array;
    let _ = grad(|x: &Array| {
        let y = checked(x)?;
        std::result::Result::Ok(y.sum())
    })(&x);
}

fn main() {
    let _ = check as fn();
}
