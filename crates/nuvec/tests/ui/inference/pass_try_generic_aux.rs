//! `?` inside a generic-`R` closure ending in `Ok((loss, aux))`.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn checked(x: &Array) -> Result<Array> {
    Ok(x.clone())
}

/// Type-checked only: the stub transforms are `unimplemented!()`.
fn check() {
    let x = Array;
    let _ = value_and_grad(|x: &Array| {
        let y = checked(x)?;
        std::result::Result::Ok((y.sum(), y))
    })(&x);
}

fn main() {
    let _ = check as fn();
}
