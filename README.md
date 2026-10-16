# nuvect

A pure-Rust array framework for Apple Silicon in the spirit of
[MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory, composable function
transforms (`grad`, `vjp`, `jvp`, `vmap`, `compile`) and Metal kernels. It is written in
Rust from the ground up, with **no bindings** to MLX or any other C/C++ library.

> **Status: pre-alpha.** Lazy arrays with 14 dtypes, MLX type promotion, broadcasting,
> zero-copy views, element-wise operations, reductions and matmul evaluate on the CPU.
> The Metal backend and function transforms are next.

## Platforms

- **macOS 15+ on Apple Silicon:** the primary target, with the full Metal backend.
- **Linux:** the core and CPU crates build for development. The GPU crates compile as stubs.

## Prerequisites

| Tool | Install |
|---|---|
| Rust (pinned in `rust-toolchain.toml`) | [rustup](https://rustup.rs). The pinned toolchain installs automatically |
| Xcode and the Metal toolchain | `xcodebuild -downloadComponent MetalToolchain` (check with `xcrun -sdk macosx -f metal`) |
| [just](https://just.systems) | `brew install just` |
| [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) | `cargo install cargo-deny --locked` |
| Miri (nightly) | `rustup +nightly component add miri` |
| jq | `brew install jq` |

Without the Metal toolchain the build still succeeds. It prints one warning and embeds the
shader sources for runtime compilation. Set `NUVECT_METALLIB=off` to choose this mode
explicitly.

## Quick start

```sh
git clone https://github.com/hupe1980/rusty-metal && cd rusty-metal
just ci
```

`just` lists every task. `just ci` runs the same gates as CI: formatting, clippy, tests,
docs, Miri, cargo-deny, the Metal kernel library check, and the crate-layering and
repository-hygiene checks.

## Testing against MLX

The test suite replays outputs recorded from real MLX (`crates/nuvect-core/tests/fixtures/mlx.bin`).
Python is not needed to build or test. With [uv](https://docs.astral.sh/uv/) installed:

- `just fixtures`: re-records the fixtures with the pinned MLX version on its CPU device
- `just bench-mlx`: prints a same-machine benchmark table, nuvect vs MLX

## Crates

| Crate | Responsibility |
|---|---|
| `nuvect` | Public facade: re-exports the layers below |
| `nuvect-core` | Lazy array graph, primitives, dtypes, streams and evaluation |
| `nuvect-compile` | Tracing, graph simplification, fusion and buffer planning |
| `nuvect-gpu` | Metal backend: device, command encoding, memory, dispatch |
| `nuvect-kernels` | Metal kernel sources and the precompiled `nuvect.metallib` |
| `nuvect-cpu` | Pure-Rust CPU backend |
| `nuvect-io` | Zero-copy safetensors, GGUF and npy loading |
| `nuvect-distributed` | Collectives over TCP rings and Thunderbolt RDMA |
| `nuvect-nn` | Neural-network modules, layers and optimizers |
| `nuvect-lm` | LLM runtime: model loading, KV caches, sampling |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
