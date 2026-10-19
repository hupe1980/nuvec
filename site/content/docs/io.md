+++
title = "Checkpoints and I/O"
description = "nuvec-io loads safetensors and NumPy npy files without a copy by memory-mapping them into Metal buffers, and saves them in the reference layout."
weight = 110
[extra]
group = "Libraries"
+++

## Zero-copy loading

A 20 GB checkpoint should not need 20 GB of copying before the first token. nuvec maps files read-only and wraps the mapping as a Metal buffer (`newBufferWithBytesNoCopy`), so every tensor is a view of the file. Pages are read from disk when the CPU or GPU first touches them, and the operating system shares them with every process that maps the same file.

```rust
use nuvec::io::{load_safetensors, save_safetensors};

let file = load_safetensors("model.safetensors")?;
let w = &file.tensors["model.layers.0.self_attn.q_proj.weight"]; // a view, no copy
println!("{:?} {:?}", w.shape(), w.dtype());

save_safetensors("out.safetensors", &[("weight", &w)], &[("format", "mlx")])?;
```

Files whose tensors are not aligned to their element size — MLX writes headers without padding, so mlx-community checkpoints are among them — are copied once, as a whole, into one aligned buffer; every tensor is a view of it.

Saving follows the reference implementation: tensors by decreasing element size after a header padded to 8 bytes, so files nuvec writes always load without a copy.

## Formats

- **safetensors:** every dtype, including MLX's `C64` complex type; validation as the reference implementation (header size, metadata, tensors tiling the data).
- **npy:** `load_npy` and `save_npy` for NumPy's format (little-endian, C or Fortran order).
- **IDX:** `read_idx` for MNIST-style datasets.

## Measured

Loading a 2 GiB safetensors file and summing every tensor (warm page cache, `just bench-load`): 48 ms with nuvec against 89 ms with MLX, with no GPU allocations against MLX's 2 GiB.
