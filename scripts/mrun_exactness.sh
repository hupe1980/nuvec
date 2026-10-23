#!/usr/bin/env bash
# Downstream exactness check: builds mrun (the inference server that uses nuvec as its
# engine) at the commit pinned in ci/mrun.ref, against this nuvec workspace, and runs its
# tiny-model exactness suites with the nuvec backend.
#
#   scripts/mrun_exactness.sh [MRUN_DIR]
#
# MRUN_DIR is an mrun checkout to clone from (read only; default: GitHub). Environment:
#   MRUN_REF  the mrun commit (default: the first word of ci/mrun.ref)
#   DEVICES   devices to test (default: "cpu gpu" with Metal, else "cpu")
#
# Prints `build pass|build-failed`, then `<suite> <device> pass|fail|absent`. Exits 0 iff
# the build passes and every present suite passes on every device; while ci/mrun.ref is
# marked `pre-port` (the pinned commit predates mrun's port to the v0.4 API), a failed
# build prints `build-failed (pre-port, expected)` and exits 0.
#
# Works in target/mrun-exactness/<sha> (a fresh clone): MRUN_DIR is never modified.
set -euo pipefail

. "$(dirname "$0")/lib/mrun_worktree.sh"

SUITES="exactness tiny_qwen3 tiny_qwen3_moe tiny_qwen3_5 tiny_qwen3_5_moe blocked_attention segmented_scan failure determinism"
FEATURES="--no-default-features --features nuvec,test-support"

src=${1:-}
sha=$(mrun_ref)
if [ -z "${DEVICES:-}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
        DEVICES="cpu gpu"
    else
        DEVICES="cpu"
    fi
fi

build_failed() {
    if mrun_pre_port; then
        echo "build-failed (pre-port, expected)"
        echo "note: ci/mrun.ref pins an mrun commit before its v0.4 port; remove 'pre-port' once it points at a ported commit" >&2
        exit 0
    fi
    echo "build-failed"
    exit 1
}

dir=$(mrun_worktree mrun-exactness "$sha" "$src") || build_failed
echo "mrun $sha in $dir" >&2
mrun_use_workspace_nuvec "$dir" || build_failed
# shellcheck disable=SC2086
if ! (cd "$dir" && cargo test -p mrun-engine $FEATURES --tests --no-run) >&2; then
    build_failed
fi
echo "build pass"

status=0
for suite in $SUITES; do
    if [ ! -f "$dir/crates/mrun-engine/tests/$suite.rs" ]; then
        for device in $DEVICES; do echo "$suite $device absent"; done
        continue
    fi
    for device in $DEVICES; do
        # shellcheck disable=SC2086
        if (cd "$dir" && MRUN_TEST_DEVICE=$device cargo test -q -p mrun-engine $FEATURES --test "$suite") >&2; then
            echo "$suite $device pass"
        else
            echo "$suite $device fail"
            status=1
        fi
    done
done
exit "$status"
