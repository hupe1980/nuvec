#!/usr/bin/env bash
# The quiet-machine measurements that need a person at the Mac: `sudo purge` for cold
# loads, and Instruments traces. Run after `scripts/quiet_session.sh`, on the same quiet
# machine; it asks for your password once.
#
#     scripts/quiet/manual.sh            # all three
#     scripts/quiet/manual.sh cold       # only these: cold, memory, tensor
#
#   cold   - spec 075 SC-002: load time from a cold page cache, plain mlx-community
#            checkpoint against its aligned copy (3 rounds each, purged before every load)
#   memory - the copied-weight footprint (research/11 §6): one 2K-token prefill of
#            Qwen3-8B, misaligned and through the aligned cache, under the Allocations
#            (VM Tracker) and Metal System Trace templates
#   tensor - the tensor-parallel prefill gap: 2 ranks over loopback, nuvec and mlx-lm, each
#            under a system-wide Metal System Trace
#
# Results: target/quiet/manual-<timestamp>/. Open a .trace with `open <file>.trace`.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT=${QUIET_OUT:-$ROOT/target/quiet/manual-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
B=$ROOT/target/release/examples/bench
PHASES=${*:-cold memory tensor}
snapshot() { ls -d ~/.cache/huggingface/hub/models--mlx-community--"$1"/snapshots/*/ 2>/dev/null | head -1; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }

cargo build -q --release -p nuvec-lm --example bench --example pipeline_generate || exit 1
# One password prompt, kept valid for the whole run; unattended (no terminal), `cold` needs
# passwordless sudo for `purge` and is skipped without it.
SUDO=1
if [ -t 0 ]; then sudo -v || SUDO=0; else sudo -n true 2>/dev/null || SUDO=0; fi
if [ $SUDO = 1 ]; then
    while true; do sudo -n true; sleep 50; done 2>/dev/null &
    KEEP=$!
    trap 'kill $KEEP 2>/dev/null' EXIT
fi
caffeinate -dimsu -w $$ &

# About 2,048 tokens of plain text, and token ids for the load phase.
PROMPT=$(printf 'word %.0s' $(seq 2048))
IDS=$OUT/ids.txt
seq -s, 100 600 | sed 's/,$//' >"$IDS"

cold() {
    local m cache
    if [ $SUDO = 0 ]; then log "cold: skipped (purge needs sudo)"; return; fi
    m=$(snapshot Qwen3.5-9B-4bit)
    cache=$OUT/aligned-cache
    log "cold: writing the aligned copy (warm, not measured)"
    NUVEC_ALIGNED_CACHE=$cache "$B" --model "$m" --phase load --first-token-ids "$IDS" >"$OUT/cold-warmup.log" 2>&1
    for r in 1 2 3; do
        for v in plain aligned; do
            sudo purge
            sleep 2
            if [ $v = aligned ]; then
                NUVEC_ALIGNED_CACHE=$cache "$B" --model "$m" --phase load --first-token-ids "$IDS"
            else
                "$B" --model "$m" --phase load --first-token-ids "$IDS"
            fi 2>>"$OUT/cold.err" | sed "s/^/$v $r /" | tee -a "$OUT/cold.log"
        done
    done
}

memory() {
    local m cache
    m=$(snapshot Qwen3-8B-4bit)
    cache=$OUT/aligned-cache-8b
    NUVEC_ALIGNED_CACHE=$cache "$B" --model "$m" --phase load --first-token-ids "$IDS" >/dev/null 2>&1
    # Allocations injects a library that waits for Developer Tools access (a prompt at the
    # screen); MEMORY_METAL_ONLY=1 skips it unattended.
    local templates=(Allocations "Metal System Trace")
    [ "${MEMORY_METAL_ONLY:-0}" = 1 ] && templates=("Metal System Trace")
    for template in "${templates[@]}"; do
        local t=${template// /-}
        for v in misaligned aligned; do
            log "memory: $template, $v"
            local env=()
            [ $v = aligned ] && env=(/usr/bin/env NUVEC_ALIGNED_CACHE="$cache")
            xcrun xctrace record --template "$template" --time-limit 180s --output "$OUT/memory-$t-$v.trace" \
                --launch -- ${env[@]+"${env[@]}"} "$B" --model "$m" --text "$PROMPT" --repeats 1 --runs 1 --max-tokens 1 \
                >>"$OUT/memory.log" 2>&1
        done
    done
}

tensor() {
    local m launch
    m=$(snapshot Qwen3-0.6B-4bit)
    launch=(uv run --python 3.13 --with mlx==0.32.4 --with mlx-lm==0.32.2 python -c
        "from mlx._distributed_utils.launch import main; main()" --backend ring -n 2 --)
    log "tensor: nuvec"
    xcrun xctrace record --template "Metal System Trace" --all-processes --time-limit 60s \
        --output "$OUT/tensor-nuvec.trace" >>"$OUT/tensor.log" 2>&1 &
    local trace=$!
    sleep 5
    "${launch[@]}" "$ROOT/target/release/examples/pipeline_generate" --model "$m" \
        --prompt "$PROMPT" --max-tokens 8 --tensor >>"$OUT/tensor.log" 2>&1
    wait $trace
    log "tensor: mlx-lm"
    xcrun xctrace record --template "Metal System Trace" --all-processes --time-limit 60s \
        --output "$OUT/tensor-mlx.trace" >>"$OUT/tensor.log" 2>&1 &
    trace=$!
    sleep 5
    "${launch[@]}" python -m mlx_lm.benchmark --model "$m" -p 2048 -g 8 -n 1 >>"$OUT/tensor.log" 2>&1
    wait $trace
}

for phase in $PHASES; do
    log "== $phase"
    "$phase"
done
log "done: $OUT"
