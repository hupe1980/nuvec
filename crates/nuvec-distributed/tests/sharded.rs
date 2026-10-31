//! Tensor-parallel layers across processes: sharded linear layers (full precision and
//! quantized) equal the unsharded layer, and a small transformer block split across 2 and
//! 4 ranks gives the single-process output and greedy tokens.

mod common;

use common::ranks;
use nuvec::distributed::Group;
use nuvec::{Array, DType, Quant, random};
use nuvec_nn::sharded::{AllToShardedLinear, ShardedToAllLinear};
use nuvec_nn::{Linear, QuantizedLinear};

fn close(name: &str, got: &Array, want: &Array, tol: f32) {
    let (g, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    assert_eq!(got.shape(), want.shape(), "{name}");
    let worst = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(worst <= tol, "{name}: off by {worst}");
}

/// Every rank draws the same layers (same seed) and checks its sharded result.
fn layers(g: Group) {
    random::seed(3);
    let x = random::normal([3, 64], DType::F32, 0.0, 1.0, None);
    let up = Linear::new(64, 128);
    let down = Linear::new(128, 64);
    // Column then row parallel: the pair equals the unsharded pair.
    let up_r = AllToShardedLinear::from_linear(&up, &g).unwrap();
    let down_r = ShardedToAllLinear::from_linear(&down, &g).unwrap();
    let want = down.forward(&up.forward(&x).sigmoid());
    let got = down_r.forward(&up_r.forward(&x).sigmoid());
    close("linear pair", &got, &want, 1e-4);
    // This rank's slice of the first layer's output.
    let (r, n) = (g.rank() as i32, g.size() as i32);
    let per = 128 / n;
    close(
        "all-to-sharded slice",
        &up_r.forward(&x),
        &up.forward(&x).i((.., r * per..(r + 1) * per)),
        1e-5,
    );
    // Quantized: the same pair from the quantized layers.
    let (qup, qdown) = (
        QuantizedLinear::from_linear_with(&up, Quant::affine(32, 4)),
        QuantizedLinear::from_linear_with(&down, Quant::affine(32, 4)),
    );
    let want = qdown.forward(&qup.forward(&x).sigmoid());
    let got = ShardedToAllLinear::from_quantized(&qdown, &g)
        .unwrap()
        .forward(
            &AllToShardedLinear::from_quantized(&qup, &g)
                .unwrap()
                .forward(&x)
                .sigmoid(),
        );
    close("quantized pair", &got, &want, 1e-4);
    // Shards that do not divide: an error naming the layer and sizes.
    let odd = Linear::new(64, 6 * n + 1);
    let err = AllToShardedLinear::from_linear(&odd, &g)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("AllToShardedLinear") && err.contains("output features"),
        "{err}"
    );
    let err = ShardedToAllLinear::from_quantized(
        &QuantizedLinear::from_linear_with(&Linear::new(32 * n + 32, 8), Quant::affine(32, 4)),
        &g,
    );
    if n > 1 {
        assert!(err.unwrap_err().to_string().contains("ShardedToAllLinear"));
    }
}

#[test]
fn sharded_layers_equal_the_unsharded_layer() {
    for n in [2, 4] {
        if ranks("sharded_layers_equal_the_unsharded_layer", n, layers).is_none() {
            return;
        }
    }
}

/// A pre-norm transformer block (4 heads of 8, an MLP of 64) with its projections split by
/// heads and hidden units: q, k, v and the MLP's first layer all-to-sharded, the attention
/// output and the MLP's second layer sharded-to-all.
struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}

fn attention(q: &Array, k: &Array, v: &Array, heads: i32) -> Array {
    let (l, d) = (q.shape()[0], 8);
    let split = |a: &Array| a.reshape([l, heads, d]).transpose([1, 0, 2]);
    let (q, k, v) = (split(q), split(k), split(v));
    let pos = nuvec::ops::arange(0, l, 1);
    let causal = pos.reshape([l, 1]).greater_equal(pos.reshape([1, l]));
    let scores = q
        .matmul(&k.transpose([0, 2, 1]))
        .multiply(1.0 / (d as f32).sqrt());
    let scores = causal.select(&scores, f32::NEG_INFINITY);
    let p = nuvec::ops::softmax(&scores, -1);
    p.matmul(&v).transpose([1, 0, 2]).reshape([l, heads * d])
}

impl Block {
    fn forward(&self, x: &Array) -> Array {
        let a = attention(
            &self.q.forward(x),
            &self.k.forward(x),
            &self.v.forward(x),
            4,
        );
        let h = x.add(self.o.forward(&a));
        h.add(self.down.forward(&nuvec_nn::silu(&self.up.forward(&h))))
    }

    fn forward_sharded(&self, x: &Array, g: &Group) -> Array {
        let col = |l: &Linear| AllToShardedLinear::from_linear(l, g).unwrap();
        let row = |l: &Linear| ShardedToAllLinear::from_linear(l, g).unwrap();
        let heads = 4 / g.size() as i32;
        let a = attention(
            &col(&self.q).forward(x),
            &col(&self.k).forward(x),
            &col(&self.v).forward(x),
            heads,
        );
        let h = x.add(row(&self.o).forward(&a));
        h.add(row(&self.down).forward(&nuvec_nn::silu(&col(&self.up).forward(&h))))
    }
}

fn transformer(g: Group) {
    random::seed(11);
    let block = Block {
        q: Linear::new(32, 32),
        k: Linear::new(32, 32),
        v: Linear::new(32, 32),
        o: Linear::new(32, 32),
        up: Linear::new(32, 64),
        down: Linear::new(64, 32),
    };
    let head = Linear::new(32, 50);
    let x = random::normal([6, 32], DType::F32, 0.0, 1.0, None);
    let want = head.forward(&block.forward(&x));
    let got = head.forward(&block.forward_sharded(&x, &g));
    close("block logits", &got, &want, 1e-4);
    assert_eq!(
        got.argmax(-1).to_vec::<u32>().unwrap(),
        want.argmax(-1).to_vec::<u32>().unwrap(),
        "greedy tokens"
    );
}

#[test]
fn a_tensor_parallel_block_equals_the_single_process_block() {
    for n in [2, 4] {
        if ranks(
            "a_tensor_parallel_block_equals_the_single_process_block",
            n,
            transformer,
        )
        .is_none()
        {
            return;
        }
    }
}
