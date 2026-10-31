# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Requantizes a cached MLX 4-bit (affine) checkpoint to a floating-point mode, for
benchmarks of the mxfp4, mxfp8 and nvfp4 kernels on real model shapes when the original
bf16 weights are not cached.

Every 4-bit affine layer is dequantized and quantized again in the mode (layer by layer,
shard by shard, so memory stays near the checkpoint's size); other layers (8-bit routers,
unquantized ones) are kept, with per-layer entries. The result is a valid mlx-lm
checkpoint: `mlx_lm generate --model DST` and nuvec-lm load it. Its quality is that of a
requantized model, which a speed benchmark and token parity between the two do not need.

With `--global-scale` (nvfp4), mixture-of-experts layers get one float32 scale per expert,
as mlx-lm's `quantize_model(..., global_scale=True)` makes them.

Usage: `uv run scripts/requantize.py mlx-community/Qwen3-8B-4bit DST [mxfp4|mxfp8|nvfp4]
[--global-scale]`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import _download

SRC, DST = sys.argv[1], Path(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "mxfp4"
GLOBAL_SCALE = "--global-scale" in sys.argv
GROUP = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}[MODE]


def main() -> None:
    src = _download(SRC)
    config = json.loads((src / "config.json").read_text())
    old = config.get("quantization") or {}
    default = (old.get("group_size", 64), old.get("bits", 4))
    layers = {}
    DST.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("model*.safetensors")):
        weights = mx.load(str(path))
        out = {}
        for name, value in weights.items():
            if not name.endswith(".scales"):
                continue
            prefix = name[: -len(".scales")]
            entry = old.get(prefix)
            gs, bits = (entry["group_size"], entry["bits"]) if isinstance(entry, dict) else default
            w, s, b = weights[prefix + ".weight"], value, weights.get(prefix + ".biases")
            in_dims = w.shape[-1] * 32 // bits
            if bits == 4 and b is not None and in_dims % GROUP == 0:
                dense = mx.dequantize(w, s, b, group_size=gs, bits=bits)
                if GLOBAL_SCALE and MODE == "nvfp4" and dense.ndim == 3:
                    # mlx-lm's `_quantize_experts`: each expert's largest magnitude.
                    scale = mx.max(mx.abs(dense), axis=(-2, -1)).astype(mx.float32)
                    qs = [mx.quantize(dense[e], mode=MODE, global_scale=scale[e])
                          for e in range(dense.shape[0])]
                    out[prefix + ".weight"] = mx.stack([q for q, _ in qs])
                    out[prefix + ".scales"] = mx.stack([q for _, q in qs])
                    out[prefix + ".global_scale"] = scale
                    layers[prefix] = {"group_size": 16, "bits": 4, "mode": "nvfp4",
                                      "global_scale": True}
                else:
                    out[prefix + ".weight"], out[prefix + ".scales"] = mx.quantize(dense, mode=MODE)
            else:
                out[prefix + ".weight"], out[prefix + ".scales"] = w, s
                if b is not None:
                    out[prefix + ".biases"] = b
                layers[prefix] = {"group_size": gs, "bits": bits, "mode": "affine"}
            mx.eval(out[prefix + ".weight"], out[prefix + ".scales"])
        for name, value in weights.items():
            prefix = name.rsplit(".", 1)[0]
            if f"{prefix}.scales" not in weights:
                out[name] = value
        mx.save_safetensors(str(DST / path.name), out, metadata={"format": "mlx"})
        print(f"{path.name}: {len(out)} tensors")
        del weights, out
    quantization = {"group_size": GROUP, "bits": 8 if MODE == "mxfp8" else 4, "mode": MODE}
    quantization.update(layers)
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    (DST / "config.json").write_text(json.dumps(config, indent=2))
    index = src / "model.safetensors.index.json"
    if index.exists():
        shutil.copy(index, DST / index.name)
    for extra in src.iterdir():
        if extra.suffix in (".json", ".txt", ".model", ".jinja") and not (DST / extra.name).exists():
            shutil.copy(extra, DST / extra.name)
    print(f"wrote {DST} ({MODE}, {len(layers)} per-layer entries)")


if __name__ == "__main__":
    main()
