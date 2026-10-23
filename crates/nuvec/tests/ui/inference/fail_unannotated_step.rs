//! An unannotated compiled step does not infer its state type.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn main() {
    let step = compile_mut(|(model, opt), (x, y): &(Array, Array)| {
        let _ = (x, y);
        opt.update(model, model)?;
        Ok(Array)
    });
    let _ = step(&mut (Model { w: Array }, Opt), &(Array, Array));
}
