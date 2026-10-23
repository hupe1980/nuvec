//! `vmap(..).in_axes((None, 0))`: `None` must infer as `Option<i32>`.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

/// Type-checked only: the stub transforms are `unimplemented!()`.
fn check() {
    let batched = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 0)).out_axes(0);
    let _: Result<Array> = batched.call(&(Array, Array));
}

fn main() {
    let _ = check as fn();
}
