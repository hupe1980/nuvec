+++
title = "Checkpoints and I/O"
description = "nuvec-io loads aligned safetensors and NumPy npy files without a copy by memory-mapping them into Metal buffers, and saves them in the reference layout."
weight = 110
[extra]
group = "Libraries"
+++

## Zero-copy loading

A 20 GB checkpoint should not need 20 GB of copying before the first token. For aligned files, nuvec maps the file read-only and wraps the mapping as a Metal buffer (`newBufferWithBytesNoCopy`), so every tensor is a view of the file. Pages are read from disk when the CPU or GPU first touches them, and the operating system shares them with every process that maps the same file.

```rust,no_run
use nuvec::io::{load_safetensors, save_safetensors};

# fn main() -> nuvec::Result<()> {
let file = load_safetensors("model.safetensors")?;
let w = &file.tensors["model.layers.0.self_attn.q_proj.weight"]; // a view, no copy
println!("{:?} {:?}", w.shape(), w.dtype());

save_safetensors("out.safetensors", &[("weight", w)], &[("format", "mlx")])?;
# Ok(())
# }
```

Files whose tensors are not aligned to their element size — MLX writes headers without padding, so mlx-community checkpoints are among them — are not mapped: their tensors are read once, with parallel positioned reads (up to 8 threads), into one buffer per file in which each is aligned, as MLX reads them. `load_safetensors_filtered(path, keep)` returns only the tensors `keep` accepts and does not read the others from such files (nuvec-lm skips vision and multi-token-prediction weights this way).

`safetensors_footprint` reads only the headers of a file or a directory and reports what loading will hold: the bytes kept mapped, the bytes read into memory, and the transient peak (their sum: no file is mapped and copied at once). A load whose transient would pass the memory limit fails before mapping or reading anything.

Saving follows the reference implementation: tensors by decreasing element size after a header padded to 8 bytes, so files nuvec writes always load without a copy. Files are written to `path.tmp` and renamed into place; `save_safetensors_async` writes in the background and returns a handle (`wait()` gives the result), for snapshots taken with `copy` while work continues.

Beyond single files:

- `nuvec::save(path, &tree)` and `nuvec::load_into(path, &mut tree)` write and read any tree of arrays (a model, an optimizer, a tuple of both) as one safetensors file, keyed by path.
- `safetensors_footprint(path)` reports the memory a load will hold before loading.
- `save_safetensors_async` saves in the background and returns a `SaveHandle`.

## Formats

- **safetensors:** every dtype, including MLX's `C64` complex type; validation as the reference implementation (header size, metadata, tensors tiling the data).
- **npy:** `load_npy` and `save_npy` for NumPy's format (little-endian, C or Fortran order).
- **IDX:** `read_idx` for MNIST-style datasets.

## Measured

Loading a 2 GiB safetensors file and summing every tensor (warm page cache, `just bench-load`; Apple M4 Pro, nuvec 0.2, 2026-10-20): 48 ms with nuvec against 89 ms with MLX, with no GPU allocations against MLX's 2 GiB.
