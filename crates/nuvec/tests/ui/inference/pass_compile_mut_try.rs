//! The compiled step with the generic `R: IntoResult` bound and `?` inside.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

/// Type-checked only: the stub transforms are `unimplemented!()`.
fn check() {
    let step = compile_mut(|(model, opt): &mut (Model, Opt), (x, y): &(Array, Array)| {
        let (loss, grads) = value_and_grad(|m: &Model| m.w.matmul(x).multiply(y).sum())(model)?;
        opt.update(model, &grads)?;
        std::result::Result::Ok(loss)
    });
    let _ = step(&mut (Model { w: Array }, Opt), &(Array, Array));
}

fn main() {
    let _ = check as fn();
}
