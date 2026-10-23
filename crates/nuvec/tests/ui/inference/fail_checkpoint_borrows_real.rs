//! The real `checkpoint` stores its closure in the graph: borrowing a local is rejected.
use nuvec::{Array, array, checkpoint};

fn main() {
    let w = array![1.0];
    let block = checkpoint(|x: &Array| x.multiply(&w));
    let _ = block(&array![2.0]);
}
