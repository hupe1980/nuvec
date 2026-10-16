# bessemer

A pure-Rust array framework for Apple Silicon in the spirit of
[MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory, composable function
transforms (`grad`, `vjp`, `jvp`, `vmap`, `compile`) and Metal kernels. It is written in
Rust from the ground up, with **no bindings** to MLX or any other C/C++ library.

> **Status: pre-alpha.** Lazy arrays with 14 dtypes and element-wise `add`/`multiply`
> evaluate on the CPU. Broadcasting, more operations, Metal and transforms are next.

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
shader sources for runtime compilation. Set `BESSEMER_METALLIB=off` to choose this mode
explicitly.

## Quick start

```sh
git clone https://github.com/hupe1980/rusty-metal && cd rusty-metal
just ci
```

`just` lists every task. `just ci` runs the same gates as CI: formatting, clippy, tests,
docs, Miri, cargo-deny, the Metal kernel library check, and the crate-layering and
repository-hygiene checks.

## Crates

| Crate | Responsibility |
|---|---|
| `bessemer` | Public facade: re-exports the layers below |
| `bessemer-core` | Lazy array graph, primitives, dtypes, streams and evaluation |
| `bessemer-compile` | Tracing, graph simplification, fusion and buffer planning |
| `bessemer-gpu` | Metal backend: device, command encoding, memory, dispatch |
| `bessemer-kernels` | Metal kernel sources and the precompiled `bessemer.metallib` |
| `bessemer-cpu` | Pure-Rust CPU backend |
| `bessemer-io` | Zero-copy safetensors, GGUF and npy loading |
| `bessemer-distributed` | Collectives over TCP rings and Thunderbolt RDMA |
| `bessemer-nn` | Neural-network modules, layers and optimizers |
| `bessemer-lm` | LLM runtime: model loading, KV caches, sampling |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
