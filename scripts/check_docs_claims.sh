#!/usr/bin/env bash
# Rejected phrases in the public docs: claims the code does not keep (weights of
# unaligned checkpoints are copied once, and quantization has no 7-bit width).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if git grep -n -E 'Zero-copy weights\.|Weights are memory-mapped, not copied|1–8 bit' \
    -- README.md 'crates/*/src/lib.rs' site/content site/templates; then
    echo "✗ rejected claims in the docs (above)" >&2
    exit 1
fi
echo "✓ docs claims"
