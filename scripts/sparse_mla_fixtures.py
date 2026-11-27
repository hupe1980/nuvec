# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "mlx-lm==0.32.2"]
# ///
"""Record a tiny DeepSeek-V3.2 attention layer from the released mlx-lm oracle."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention, ModelArgs

OUT = Path(__file__).resolve().parents[1] / "crates/nuvec-lm/tests/fixtures/sparse_mla"
SEED = 1307
TINY = {
    "hidden_size": 16,
    "num_attention_heads": 2,
    "q_lora_rank": 8,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 6,
    "qk_rope_head_dim": 4,
    "v_head_dim": 6,
    "index_n_heads": 4,
    "index_head_dim": 8,
    "index_topk": 3,
    "max_position_embeddings": 32,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16,
        "beta_fast": 8,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}


def record(directory, name, dtype):
    mx.random.seed(SEED)
    layer = DeepseekV32Attention(ModelArgs(**TINY))
    layer.set_dtype(dtype)
    x = mx.random.uniform(-0.75, 0.75, (1, 8, 16)).astype(dtype)
    mask = mx.arange(8)[:, None] >= mx.arange(8)

    def output(h):
        return layer(h, mask=mask)

    # The released prefill path does the unabsorbed calculation.
    expected = output(x)
    rows = []
    for length in range(1, 9):
        prefix = x[:, :length]
        qr = layer.q_a_layernorm(layer.q_a_proj(prefix))
        selected = layer.indexer(prefix, qr, mask[:length, :length])
        ids = sorted(selected[0, 0, -1].tolist()) if selected is not None else list(range(length))
        rows.append(ids + [-1] * (3 - len(ids)))
    tensors = dict(tree_flatten(layer.parameters()))
    tensors.update(input=x, expected=expected, picks=mx.array(rows))
    # The first query block is dense and the second crosses the selection limit.
    wide = DeepseekV32Attention(
        ModelArgs(**(TINY | {"index_topk": 66, "max_position_embeddings": 96}))
    )
    wide.load_weights(tree_flatten(layer.parameters()))
    long_input = mx.random.uniform(-0.75, 0.75, (1, 67, 16)).astype(dtype)
    causal = mx.arange(67)[:, None] >= mx.arange(67)
    tensors.update(long_input=long_input, long_expected=wide(long_input, mask=causal))
    if dtype == mx.float32:
        # Central differences on one direction exercise the continuous path through the layer.
        dx = mx.random.uniform(-0.5, 0.5, x.shape)
        step = 1e-3
        tensors.update(
            direction=dx,
            directional=(output(x + step * dx).sum() - output(x - step * dx).sum()) / (2 * step),
        )
    path = directory / f"{name}.safetensors"
    mx.save_safetensors(str(path), tensors)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="destination directory")
    directory = parser.parse_args().out
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        name: record(directory, name, dtype)
        for name, dtype in (("f32", mx.float32), ("bf16", mx.bfloat16))
    }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "config": TINY,
                "mlx": mx.__version__,
                "mlx_lm": version("mlx-lm"),
                "sha256": files,
            },
            indent=2,
        )
        + "\n"
    )
