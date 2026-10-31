# /// script
# requires-python = ">=3.11"
# ///
"""Extracts the multi-token-prediction (MTP) tensors of a Hugging Face checkpoint into one
safetensors file, downloading only their bytes (HTTP range requests on the shards).

mlx-community conversions drop `mtp.*`; nuvec-lm attaches them from this file
(`Model::attach_mtp`). Qwen 3.5/3.6/3.8 store one MTP decoder layer (0.3–2 GB in bf16).

Usage: `uv run scripts/extract_mtp.py Qwen/Qwen3.5-9B [out.safetensors]` (default:
`~/.cache/nuvec/mtp/<repo>.safetensors`). Set `HF_TOKEN` for gated repositories.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

CHUNK = 64 << 20


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path.home() / ".cache" / "nuvec" / "mtp" / (repo.replace("/", "--") + ".safetensors")
    )
    base = f"https://huggingface.co/{repo}/resolve/main/"
    try:
        index = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    except Exception:
        index = None
    wanted: dict[str, list[str]] = {}
    if index is None:
        wanted["model.safetensors"] = []
    else:
        for name, shard in index.items():
            if name.startswith("mtp.") or ".mtp." in name:
                wanted.setdefault(shard, []).append(name)
    if not wanted:
        sys.exit(f"{repo}: no mtp.* tensors")
    tensors: list[tuple[str, dict, bytes]] = []
    for shard, names in wanted.items():
        url = base + shard
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        header = json.loads(get(url, 8, 8 + n - 1))
        header.pop("__metadata__", None)
        names = names or [k for k in header if k.startswith("mtp.") or ".mtp." in k]
        for name in sorted(names):
            info = header[name]
            a, b = info["data_offsets"]
            data = bytearray()
            for at in range(a, b, CHUNK):
                data += get(url, 8 + n + at, 8 + n + min(b, at + CHUNK) - 1)
            print(f"{name}: {info['dtype']} {info['shape']} ({(b - a) / 1e6:.1f} MB)", flush=True)
            tensors.append((name, info, bytes(data)))
    meta: dict[str, object] = {"__metadata__": {"source": repo, "format": "pt"}}
    offset = 0
    for name, info, data in tensors:
        meta[name] = {"dtype": info["dtype"], "shape": info["shape"],
                      "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(meta, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # 8-byte aligned data
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, data in tensors:
            f.write(data)
    tmp.replace(out)
    print(f"{out}: {len(tensors)} tensors, {offset / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
