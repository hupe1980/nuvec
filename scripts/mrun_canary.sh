#!/usr/bin/env bash
# Release canary: builds mrun (the inference server built on nuvec) at the commit pinned in
# ci/mrun.ref twice, with its MLX backend and with its nuvec backend pointed at this tree,
# runs mrun's synthetic benchmark (qwen3 and qwen3.6; one server at a time) on both, and
# writes site/static/benchmarks/<version>/mrun-canary-<machine>.json.
#
#   scripts/mrun_canary.sh [MRUN_DIR]
#
# MRUN_DIR is an mrun checkout to clone from (read only; default: GitHub); MRUN_REF
# overrides the pinned commit. Manual and per release; never part of CI. A build that
# fails is recorded as `build-failed` with the first compiler errors (exit 0); the
# result is written whenever the run gets that far.
set -euo pipefail

. "$(dirname "$0")/lib/mrun_worktree.sh"

root=$(nuvec_root)
src=${1:-}
sha=$(mrun_ref)
run="$root/target/mrun-canary/run-$sha"
rm -rf "$run" && mkdir -p "$run"
dir=""
cleanup() { [ -n "$dir" ] && rm -rf "$dir"; }
trap cleanup EXIT

write_meta() { # status reason
    python3 - "$run/canary.json" "$sha" "$1" "$2" <<'PY'
import json, sys
path, sha, status, reason = sys.argv[1:]
meta = {"mrun_commit": sha, "status": status}
if reason:
    meta["reason"] = reason
json.dump(meta, open(path, "w"))
PY
}

report() {
    uv run "$root/scripts/bench_parity.py" --mrun-canary-report "$run"
}

dir=$(mrun_worktree mrun-canary "$sha" "$src") || { write_meta not-run "cannot check out mrun $sha"; report; exit 0; }
pre_port=""
mrun_pre_port && pre_port=" (ci/mrun.ref is marked pre-port: expected until mrun is ported)"
if ! mrun_use_workspace_nuvec "$dir"; then
    write_meta build-failed "mrun $sha has no nuvec dependency to point at this tree$pre_port"
    report
    exit 0
fi

build() { # name cargo-args...
    local name=$1; shift
    if ! (cd "$dir" && cargo build --release -p mrun "$@" --target-dir "$dir/target-$name") \
        >"$run/build-$name.log" 2>&1; then
        local errors
        errors=$(grep -E '^(error|  -->)' "$run/build-$name.log" | head -40 || true)
        write_meta build-failed "$name build failed$pre_port: ${errors:-see the build log}"
        report
        exit 0
    fi
}
build mlx --features mlx
build nuvec --no-default-features --features nuvec
if (cd "$dir" && cargo tree -p mrun --no-default-features --features nuvec -e normal) \
    | grep -qE '(^|[^a-z-])mlx(-sys)? v'; then
    write_meta build-failed "the nuvec build links an mlx crate"
    report
    exit 0
fi

for alias in qwen3 qwen3.6; do
    for name in mlx nuvec; do
        bin="$dir/target-$name/release/mrun"
        out="$run/$name-$alias"
        "$bin" serve --detach --model "$alias"
        if ! "$bin" bench --suite synthetic --reps 5 --out "$out"; then
            "$bin" stop || true
            write_meta bench-failed "$name $alias: mrun bench failed"
            report
            exit 0
        fi
        "$bin" stop
    done
done
write_meta ok ""
report
