+++
title = "Checkpoints and I/O"
description = "nuvec-io loads aligned safetensors and NumPy npy files without a copy by memory-mapping them into Metal buffers, and saves them in the reference layout."
weight = 110
[extra]
group = "Libraries"
+++

## Loading without a copy

Aligned files are mapped read-only and wrapped as a Metal buffer (`newBufferWithBytesNoCopy`), so every tensor is a view of the file. Pages are read from disk when the CPU or GPU first touches them and are shared with every process that maps the same file.

```rust,no_run
use nuvec::io::{load_safetensors, save_safetensors};

# fn main() -> nuvec::Result<()> {
let file = load_safetensors("model.safetensors")?;
let w = &file.tensors["model.layers.0.self_attn.q_proj.weight"]; // a view of the file when aligned
println!("{:?} {:?}", w.shape(), w.dtype());

save_safetensors("out.safetensors", &[("weight", w)], &[("format", "mlx")])?;
# Ok(())
# }
```

Files whose tensors are not aligned to their element size are not mapped. mlx-community checkpoints are among them (MLX writes headers without padding). Their tensors are read once, with parallel positioned reads (up to 8 threads), into one buffer per file in which each tensor starts at a multiple of 256 bytes.

- `load_safetensors_filtered(path, keep)` returns only the tensors `keep` accepts: from aligned files it maps only their pages, from unaligned files it reads only them (nuvec-lm skips vision and multi-token-prediction weights this way).
- `align_safetensors(src, dst)` writes a copy of a file with every tensor at a multiple of 256 bytes, which then maps without a copy. `load_safetensors_cached(path, cache_dir, keep)` loads an unaligned file through such a copy in `cache_dir`, written on first use and rewritten when the source's size or modification time changes.
- `safetensors_tensors(path)` lists a file's tensors (name, dtype, shape, bytes) from its header alone.
- `safetensors_footprint(path)` reads only the headers of a file or a directory and reports what loading will hold: the bytes kept mapped, the bytes read into memory, and the transient peak (their sum). `safetensors_footprint_filtered(path, keep)` does the same for a filtered load. A load whose transient would pass the memory limit fails before mapping or reading anything.

## Saving

Saving follows the reference implementation: tensors by decreasing element size after a header padded to 8 bytes, so files nuvec writes always load without a copy. Files are written to `path.tmp` and renamed into place. `save_safetensors_async` writes in the background and returns a `SaveHandle` (`wait()` gives the result), for snapshots taken with `copy` while work continues.

`nuvec::save(path, &tree)` and `nuvec::load_into(path, &mut tree)` write and read any tree of arrays (a model, an optimizer, a tuple of both) as one safetensors file, keyed by path.

## Formats

- **safetensors:** every dtype, including MLX's `C64` complex type; validation as the reference implementation (header size, metadata, tensors tiling the data).
- **npy:** `load_npy` and `save_npy` for NumPy's format (little-endian, C or Fortran order).
- **IDX:** `read_idx` for MNIST-style datasets.
