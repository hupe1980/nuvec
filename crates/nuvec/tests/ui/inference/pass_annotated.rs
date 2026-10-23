//! Annotated closures infer through the generic transforms.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn loss(p: &(Array, Array)) -> Array {
    p.0.matmul(&p.1).sum()
}

/// Type-checked only: the stub transforms are `unimplemented!()`.
fn check() {
    let x = Array;
    let _ = grad(|x: &Array| x.multiply(x).sum())(&x);
    let batch = Array;
    let _ = value_and_grad(|p: &(Array, Array)| p.0.matmul(&batch).sum())(&(Array, Array));
    // A fn item passed by name.
    let _ = grad(loss)(&(Array, Array));
    // Returning `Result<Array>` without `?`.
    let _ = grad(|x: &Array| -> Result<Array> { Ok(x.sum()) })(&x);
    // Aux outputs chosen by the return type.
    let _: Result<((Array, Array), Array)> =
        value_and_grad(|x: &Array| (x.sum(), x.clone()))(&x);
    let _: Result<((Array, Vec<Array>), Array)> =
        value_and_grad(|x: &Array| (x.sum(), vec![x.clone()]))(&x);
    // The compiled step, with `?` inside.
    let step = compile_mut(|(model, opt): &mut (Model, Opt), (x, y): &(Array, Array)| {
        let (loss, grads) = value_and_grad(|m: &Model| m.w.matmul(x).multiply(y).sum())(model)?;
        opt.update(model, &grads)?;
        Ok(loss)
    });
    let _ = step(&mut (Model { w: Array }, Opt), &(Array, Array));
}

fn main() {
    let _ = check as fn();
}
