# nuvec — development task runner (https://just.systems). `just ci` mirrors CI.

default:
    @just --list --unsorted

# ── Code quality ──────────────────────────────────────────────────────────────

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

# ── Tests ─────────────────────────────────────────────────────────────────────

# On the pinned toolchain (rust-toolchain.toml), which is the MSRV.
test:
    RUSTFLAGS="-D warnings" cargo test --workspace --all-targets --all-features

# On the latest stable toolchain, to catch upcoming breakage early.
test-stable:
    RUSTUP_TOOLCHAIN=stable RUSTFLAGS="-D warnings" cargo test --workspace --all-targets --all-features

# Undefined-behaviour check of the core graph logic (`rustup +nightly component add miri`).
miri:
    cargo +nightly miri test -p nuvec-core

# ── Metal kernels ─────────────────────────────────────────────────────────────

# Build the kernel library (precompiled when the Metal toolchain is installed).
metallib:
    cargo build -p nuvec-kernels

# Fail unless the kernel library is precompiled and contains the expected kernels.
metallib-check:
    NUVEC_REQUIRE_METALLIB=1 cargo test -p nuvec-kernels

# GPU runtime tests with Metal API validation (skip without a Metal device).
metal-check:
    MTL_DEBUG_LAYER=1 cargo test -p nuvec-gpu
    NUVEC_METALLIB=off MTL_DEBUG_LAYER=1 cargo test -p nuvec-gpu --test runtime

# ── MLX reference (needs uv; never part of `ci`) ──────────────────────────────

# Re-record the MLX fixtures replayed by `tests/mlx.rs` (pinned MLX, CPU device).
fixtures:
    uv run scripts/mlx_fixtures.py

# Re-records the safetensors and npy files written by MLX and NumPy for nuvec-io's tests.
io-fixtures:
    uv run scripts/io_fixtures.py

# Same-machine benchmark table: nuvec vs MLX (CPU device).
bench-mlx:
    uv run scripts/bench_vs_mlx.py

# GPU GEMM shape matrix: nuvec vs MLX (TFLOP/s).
bench-gemm:
    uv run scripts/bench_gemm.py

# GPU quantized matrix products (4-bit decode and prefill shapes): nuvec vs MLX.
bench-quant:
    uv run scripts/bench_quant.py

# GPU fused operations (norms, RoPE, softmax, log-sum-exp, cross entropy): nuvec vs MLX.
bench-fast:
    uv run scripts/bench_fast.py

# Checkpoint loading (2 GiB safetensors, warm page cache): nuvec vs MLX.
bench-load:
    uv run scripts/bench_load.py

# Downloads MNIST and trains MLX's example model with nuvec and MLX (M1 exit criterion).
mnist:
    uv run scripts/mnist_vs_mlx.py

# ── Constitution gates ────────────────────────────────────────────────────────

# Licenses, advisories, sources, and the ban on native C/C++ builds (Principle I).
deny:
    cargo deny --all-features check

# Tracked files must not cite the untracked internal planning folders.
internal-paths:
    #!/usr/bin/env bash
    set -euo pipefail
    if git ls-files -z --cached --others --exclude-standard | grep -zv '^\.gitignore$' \
        | xargs -0 grep -nE '(^|[^[:alnum:]_.-])(concepts|specs|\.specify)/'; then
        echo "✗ tracked files reference internal planning folders (above)" >&2
        exit 1
    fi
    echo "✓ no internal planning paths in tracked files"

# Crate dependencies follow the layering: lower layers never depend on higher ones.
layering:
    #!/usr/bin/env bash
    set -euo pipefail
    allowed='
    nuvec-core -> nuvec-cpu
    nuvec-core -> nuvec-gpu
    nuvec-compile -> nuvec-core
    nuvec-io -> nuvec-core
    nuvec-distributed -> nuvec-core
    nuvec-gpu -> nuvec-kernels
    nuvec -> nuvec-core
    nuvec -> nuvec-compile
    nuvec -> nuvec-cpu
    nuvec -> nuvec-gpu
    nuvec -> nuvec-io
    nuvec -> nuvec-distributed
    nuvec-nn -> nuvec
    nuvec-lm -> nuvec
    nuvec-lm -> nuvec-nn
    '
    edges=$(cargo metadata --format-version 1 --no-deps | jq -r '
        .packages[] | .name as $from | .dependencies[]
        | select(.name | startswith("nuvec")) | select(.kind == null)
        | "\($from) -> \(.name)"')
    bad=0
    while IFS= read -r edge; do
        [ -z "$edge" ] && continue
        if ! sed "s/^ *//" <<<"$allowed" | grep -qxF "$edge"; then
            echo "✗ disallowed dependency: $edge" >&2
            bad=1
        fi
    done <<<"$edges"
    [ "$bad" -eq 0 ] && echo "✓ crate layering"
    exit "$bad"

# ── Everything ────────────────────────────────────────────────────────────────

ci: fmt-check lint test doc miri deny metallib-check metal-check internal-paths layering
