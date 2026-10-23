//! `checkpoint` stores its closure in the graph: borrowing a local is rejected.
#[path = "stub/mod.rs"]
mod stub;
use stub::*;

fn main() {
    let w = Array;
    let block = checkpoint(|x: &Array| x.matmul(&w));
    let _ = block(&Array);
}
