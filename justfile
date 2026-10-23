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

# On the pinned toolchain (rust-toolchain.toml), which is the MSRV. `--all-targets` skips
# doctests, so they run separately.
test:
    RUSTFLAGS="-D warnings" cargo test --workspace --all-targets --all-features
    RUSTDOCFLAGS="-D warnings" cargo test --workspace --doc --all-features

# On the latest stable toolchain, to catch upcoming breakage early.
test-stable:
    RUSTUP_TOOLCHAIN=stable RUSTFLAGS="-D warnings" cargo test --workspace --all-targets --all-features

# Undefined-behaviour check of the graph logic, parsers and CPU kernels
# (`rustup +nightly component add miri`). For nuvec-io and nuvec-cpu only the unit tests run
# (the integration tests evaluate through `half`'s inline assembly, which Miri cannot run),
# and file access is allowed for the parser tests.
# Miri on the library unit tests (graph logic, CPU kernels, parsers): minutes, part of `ci`.
# Thread-stress and GPU suites run natively in `test`; under Miri they take far longer
# and the unsafe code they would reach (Metal FFI) cannot run there at all.
miri:
    cargo +nightly miri test -p nuvec-core --lib
    MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test -p nuvec-io -p nuvec-cpu --lib

# Miri on every CPU test, integration suites included (slow; run before releases).
miri-full:
    cargo +nightly miri test -p nuvec-core
    MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test -p nuvec-io -p nuvec-cpu --lib

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
    # Runtime-compiled kernels give the precompiled results (MLX fixtures, bit-exact fused ops).
    NUVEC_METALLIB=off cargo test -p nuvec-core --test gpu --test fast
    # Out-of-bounds GPU reads and writes do not panic: shader validation turns them into
    # errors (paged and indexing kernels, random primitives).
    MTL_SHADER_VALIDATION=1 cargo test -p nuvec-core --test fast --test primitive_fuzz
    MTL_DEBUG_LAYER=1 cargo test -p nuvec-core --test gpu --test fast --test donation --test determinism
    # Determinism across processes with runtime-compiled kernels (and the hash for
    # `just determinism-xbuild`).
    NUVEC_METALLIB=off cargo test -p nuvec-core --test determinism
    # A long decode loop leaves no Objective-C object outside an autorelease pool.
    scripts/leak_check.sh

# Build the website (Zola) into site/public.
site:
    zola --root site build

# Serve the website locally with live reload.
site-serve:
    zola --root site serve

# ── MLX reference (needs uv; never part of `ci`) ──────────────────────────────

# Re-record the MLX fixtures replayed by `tests/mlx.rs` (pinned MLX, CPU device).
fixtures:
    uv run scripts/mlx_fixtures.py

# Re-records the nuvec-lm fixtures (tiny models and their outputs) with mlx-lm (needs uv).
lm-fixtures:
    uv run scripts/lm_fixtures.py

# Re-records the nuvec-nn fixtures from MLX (needs uv).
nn-fixtures:
    uv run scripts/nn_fixtures.py

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
bench-quant *filter:
    uv run scripts/bench_quant.py {{filter}}

# GPU fused operations (norms, RoPE, softmax, log-sum-exp, cross entropy): nuvec vs MLX.
bench-fast *filter:
    uv run scripts/bench_fast.py {{filter}}

# GPU compiled element-wise functions (GELU, SwiGLU gating, Adam step): nuvec vs MLX.
bench-compile:
    uv run scripts/bench_compile.py

# Text generation (prefill and decode) with a cached model: nuvec-lm vs mlx-lm.
bench-lm model="mlx-community/Qwen3-8B-4bit" tokens="256":
    uv run scripts/bench_lm.py {{model}} {{tokens}}

# Parity table over a model roster: nuvec-lm vs mlx-lm (cached models only).
bench-parity *args:
    uv run scripts/bench_parity.py {{args}}

# Validates the committed benchmark records (default: everything under site/static/benchmarks).
bench-validate *paths:
    uv run scripts/validate_parity.py {{paths}}

# Compares two parity records: fails if a nuvec / mlx-lm ratio fell by more than 2%.
bench-compare base candidate:
    uv run scripts/validate_parity.py --compare {{base}} {{candidate}}

# Checkpoint loading (2 GiB safetensors, warm page cache): nuvec vs MLX.
bench-load:
    uv run scripts/bench_load.py

# Attention under a score-memory cap (tiled kernel vs blocked composition) at long-prefill chunk
# shapes: nuvec vs MLX; the table behind the capped routing rule.
bench-attention-route *filter:
    uv run scripts/bench_attention_route.py {{filter}}

# Few-row products (B = 1..16: dense, quantized, gathers) against 0.3.0's routing and MLX;
# prints the largest row count that keeps up (`ROW_INDEPENDENT_ROWS`).
bench-rows *filter:
    uv run scripts/bench_rows.py {{filter}}

# ── Downstream and long-running checks ────────────────────────────────────────

# A server-like workload on a caller-owned thread for HOURS hours: fails if the process
# footprint grows by 64 MiB or more after warm-up (CSV in target/soak/).
soak HOURS="1":
    cargo run --release -p nuvec-lm --example soak -- --hours {{HOURS}}

# Determinism across build modes: the precompiled and the runtime-compiled kernels give
# the same output hash.
determinism-xbuild:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test -q -p nuvec-core --test determinism -- results_are_bit_identical_across_processes
    NUVEC_METALLIB=off cargo test -q -p nuvec-core --test determinism -- results_are_bit_identical_across_processes
    a=$(cat target/determinism/precompiled.hash) b=$(cat target/determinism/runtime.hash)
    echo "precompiled $a"
    echo "runtime     $b"
    [ "$a" = "$b" ] || { echo "✗ build modes differ" >&2; exit 1; }
    echo "✓ precompiled and runtime-compiled kernels agree"

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

# The crates CI builds on Linux (no Metal) compile without warnings there.
portable-check:
    rustup target add x86_64-unknown-linux-gnu >/dev/null 2>&1 || true
    RUSTFLAGS="-D warnings" cargo check --target x86_64-unknown-linux-gnu -p nuvec-core -p nuvec-cpu -p nuvec-io -p nuvec-kernels -p nuvec-gpu --all-targets

layering:
    #!/usr/bin/env bash
    set -euo pipefail
    allowed='
    nuvec-core -> nuvec-cpu
    nuvec-core -> nuvec-gpu
    nuvec-compile -> nuvec-core
    nuvec-io -> nuvec-core
    nuvec-gpu -> nuvec-kernels
    nuvec -> nuvec-core
    nuvec -> nuvec-compile
    nuvec -> nuvec-cpu
    nuvec -> nuvec-gpu
    nuvec -> nuvec-io
    nuvec -> nuvec-macros
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

# The array API hard cut (0.4): no 0.3 names anywhere, graph-level types only through
# `graph::` in user-facing code, and no `?` applied to an array op in user-facing code.
api-hygiene:
    #!/usr/bin/env bash
    set -euo pipefail
    if rg -n --glob '!*.bin' 'legacy|Array::(scalar|full)\b|split_pair|shape_usize' crates README.md site/content; then
        echo "✗ 0.3 API names (above)" >&2
        exit 1
    fi
    if rg -n 'SliceArg' README.md site/content crates/*/src/lib.rs crates/*/examples | grep -v 'graph::' | grep -v 'pinned absent'; then
        echo "✗ SliceArg outside graph:: in user-facing code (above)" >&2
        exit 1
    fi
    # The positional tree API of 0.3: hand-written module visitors, positional
    # updates and `nn::value_and_grad` are gone.
    if rg -n 'update_trainable|Parameters<|Frozen<|nuvec_nn::value_and_grad|value_and_grad\(&' crates README.md site/content; then
        echo "✗ 0.3 tree/module API names (above)" >&2
        exit 1
    fi
    scripts/check_no_op_try.sh
    echo "✓ API hygiene"

# Models derive their trees: no hand-written `visit`/`visit_mut` in nuvec-nn and
# nuvec-lm, except the one for boxed optimizers.
no-hand-visits:
    #!/usr/bin/env bash
    set -euo pipefail
    hits=$(grep -rnE "fn visit(_mut)?\(" crates/nuvec-nn crates/nuvec-lm | grep -v "^crates/nuvec-nn/src/optim.rs:" || true)
    if [ -n "$hits" ]; then
        echo "$hits"
        echo "✗ hand-written tree visits in nuvec-nn/nuvec-lm (derive Module or Tree instead)" >&2
        exit 1
    fi
    echo "✓ no hand-written visits"

# Every crate that ships ported MLX or mlx-lm code carries THIRD_PARTY_NOTICES, identical to
# the root copy and included in its package.
notices:
    #!/usr/bin/env bash
    set -euo pipefail
    for c in nuvec-core nuvec-gpu nuvec-kernels nuvec-nn nuvec-lm nuvec-compile; do
        cmp -s THIRD_PARTY_NOTICES "crates/$c/THIRD_PARTY_NOTICES" || { echo "✗ crates/$c/THIRD_PARTY_NOTICES differs from the root" >&2; exit 1; }
        cargo package -p "$c" --list --allow-dirty 2>/dev/null | grep -qx THIRD_PARTY_NOTICES || { echo "✗ $c does not package THIRD_PARTY_NOTICES" >&2; exit 1; }
    done
    echo "✓ third-party notices"

# The docs build as docs.rs runs it: no Metal toolchain call (a fake `xcrun` on PATH records
# any `metal`/`metallib` call and passes everything else, such as rustc's SDK lookup, to the
# real one), and every published crate documents the Apple target.
docs-rs:
    #!/usr/bin/env bash
    set -euo pipefail
    fake="$PWD/target/fake-xcrun"
    rm -rf "$fake" && mkdir -p "$fake"
    printf '#!/bin/sh\nfor a in "$@"; do case "$a" in metal|metallib) touch "%s/called"; exit 1;; esac; done\nexec /usr/bin/xcrun "$@"\n' "$fake" > "$fake/xcrun"
    chmod +x "$fake/xcrun"
    DOCS_RS=1 PATH="$fake:$PATH" RUSTDOCFLAGS="-D warnings" CARGO_TARGET_DIR=target/docs-rs \
        cargo doc -q --workspace --no-deps --all-features
    [ ! -e "$fake/called" ] || { echo "✗ the docs build called xcrun" >&2; exit 1; }
    scripts/check_docs_rs_meta.sh
    echo "✓ docs.rs build"

# Public docs make no claim the code does not keep (rejected phrases).
docs-claims:
    scripts/check_docs_claims.sh

# Semver check of the published facade and nuvec-lm against crates.io (network).
semver:
    cargo semver-checks --package nuvec --package nuvec-lm

# The release's API gate: the pinned surface, then semver.
api-check:
    cargo test -p nuvec --test api_surface
    just semver

# ── Everything ────────────────────────────────────────────────────────────────

# GPU tests fail instead of skipping without a Metal device (`NUVEC_REQUIRE_GPU=1`).
ci:
    NUVEC_REQUIRE_GPU=1 just fmt-check lint test doc miri deny metallib-check metal-check internal-paths layering api-hygiene no-hand-visits notices docs-rs docs-claims portable-check
