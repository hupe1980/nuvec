#!/usr/bin/env bash
# Shared helpers for scripts that build mrun (the downstream inference server) against this
# nuvec workspace: a throwaway checkout of mrun at a pinned commit whose nuvec dependency is
# rewritten to a path dependency. The operator's mrun checkout is only read (cloned from);
# its tracked files and its .git are never modified.
#
# Source it:  . "$(dirname "$0")/lib/mrun_worktree.sh"

# The nuvec repository root.
nuvec_root() {
    git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel
}

# The pinned mrun commit: the first word of ci/mrun.ref (or $MRUN_REF when set).
mrun_ref() {
    if [ -n "${MRUN_REF:-}" ]; then
        echo "$MRUN_REF"
    else
        cut -d' ' -f1 "$(nuvec_root)/ci/mrun.ref"
    fi
}

# Whether ci/mrun.ref marks the pinned commit as predating mrun's port to the v0.4 API.
mrun_pre_port() {
    [ -z "${MRUN_REF:-}" ] && grep -qw "pre-port" "$(nuvec_root)/ci/mrun.ref"
}

# mrun_worktree NAME SHA [MRUN_DIR] — prints the path of a fresh checkout of mrun at SHA
# under target/NAME/SHA, cloned from MRUN_DIR (shared objects, no hard links into its
# working tree) or from GitHub when MRUN_DIR is empty.
mrun_worktree() {
    local name=$1 sha=$2 src=${3:-}
    local dir
    dir="$(nuvec_root)/target/$name/$sha"
    rm -rf "$dir"
    mkdir -p "$(dirname "$dir")"
    if [ -n "$src" ]; then
        git clone --quiet --shared --no-checkout "$src" "$dir" >&2
    else
        git clone --quiet --no-checkout https://github.com/hupe1980/mrun.git "$dir" >&2
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --quiet "$sha" >&2
    echo "$dir"
}

# mrun_use_workspace_nuvec DIR — rewrites DIR's crates/mrun-engine/Cargo.toml so that its
# `nuvec = { version = …` dependency points at this workspace's facade crate. Fails when
# the line is missing (a commit without the nuvec backend) or the rewrite did not apply.
mrun_use_workspace_nuvec() {
    local dir=$1
    local toml="$dir/crates/mrun-engine/Cargo.toml"
    local facade
    facade="$(nuvec_root)/crates/nuvec"
    if ! grep -qE '^nuvec = \{ *version *=' "$toml"; then
        echo "mrun at $(git -C "$dir" rev-parse --short HEAD) has no 'nuvec = { version = …' dependency" >&2
        return 1
    fi
    sed -E -i.bak "s|^nuvec = \{ *version *= *\"[^\"]*\"|nuvec = { path = \"$facade\"|" "$toml"
    rm -f "$toml.bak"
    grep -qF "nuvec = { path = \"$facade\"" "$toml"
}
