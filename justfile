# bessemer — development task runner (https://just.systems). `just ci` mirrors CI.

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
    cargo +nightly miri test -p bessemer-core

# ── Metal kernels ─────────────────────────────────────────────────────────────

# Build the kernel library (precompiled when the Metal toolchain is installed).
metallib:
    cargo build -p bessemer-kernels

# Fail unless the kernel library is precompiled and contains the expected kernels.
metallib-check:
    BESSEMER_REQUIRE_METALLIB=1 cargo test -p bessemer-kernels

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
    bessemer-compile -> bessemer-core
    bessemer-cpu -> bessemer-core
    bessemer-io -> bessemer-core
    bessemer-distributed -> bessemer-core
    bessemer-gpu -> bessemer-core
    bessemer-gpu -> bessemer-kernels
    bessemer -> bessemer-core
    bessemer -> bessemer-compile
    bessemer -> bessemer-cpu
    bessemer -> bessemer-gpu
    bessemer -> bessemer-io
    bessemer -> bessemer-distributed
    bessemer-nn -> bessemer
    bessemer-lm -> bessemer
    bessemer-lm -> bessemer-nn
    '
    edges=$(cargo metadata --format-version 1 --no-deps | jq -r '
        .packages[] | .name as $from | .dependencies[]
        | select(.name | startswith("bessemer")) | select(.kind == null)
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

ci: fmt-check lint test doc miri deny metallib-check internal-paths layering
