#!/usr/bin/env bash
# Every published crate documents the Apple target on docs.rs.
set -euo pipefail
bad=$(cargo metadata --format-version 1 --no-deps | jq -r '
    .packages[] | select(.publish != [])
    | select((.metadata.docs.rs.targets // []) != ["aarch64-apple-darwin"]) | .name')
if [ -n "$bad" ]; then
    echo "✗ missing [package.metadata.docs.rs] targets = [\"aarch64-apple-darwin\"]: $bad" >&2
    exit 1
fi
echo "✓ docs.rs metadata"
