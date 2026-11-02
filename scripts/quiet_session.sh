#!/usr/bin/env bash
# The quiet-machine session (research/11): every pending measurement and the experiments that
# decide the next optimisations, in priority order. Each phase writes into one directory:
#
#     scripts/quiet_session.sh             # every phase, in order
#     scripts/quiet_session.sh gates serve # only these phases
#
# Phases:
#   preflight - refuse to start on a busy machine (load, swap, battery)
#   gates     - the open measurements of specs 070-074 (≈ 40 min)
#   serve     - Batcher vs mlx-lm: noise floor, paged tax, per-kernel shares (≈ 50 min)
#   beyond    - beyond-MLX probes: GEMV alignment, MoE glue, draft head, Gemma 26B, host vs GPU
#               bound (≈ 90 min)
#   decode    - decode and long-prefill gaps: per-kernel traces in both engines (≈ 45 min)
#   ab        - HEAD (the last commit) against the working tree, interleaved, on the models
#               specs 076-080 target (≈ 70 min; needs `serve`'s prompt files)
#   record    - a fresh quick-tier parity record (≈ 60 min)
#
# Results: target/quiet/<timestamp>/<phase>/*.{log,json,trace}. Nothing is committed.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${QUIET_OUT:-$ROOT/target/quiet/$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
B=$ROOT/target/release/examples/bench
P=$ROOT/crates/nuvec-lm/benches/prompts/natural.txt
TEXT="The lighthouse keeper climbed the spiral stairs every evening, counting the steps and listening to the sea against the rocks below. "
PHASES=${*:-preflight gates serve ab beyond decode record}
# HEAD's bench, built from a worktree of the last commit (see `ab`).
A=$ROOT/target/baseline/target/release/examples/bench

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
step() { # step <phase> <name> <command...>: run, log, keep going on failure
    local phase=$1 name=$2; shift 2
    mkdir -p "$OUT/$phase"
    log "$phase/$name: $*"
    local t=$SECONDS
    ( "$@" ) >"$OUT/$phase/$name.log" 2>&1
    local code=$?
    log "$phase/$name: exit $code after $((SECONDS - t)) s"
}
snapshot() { ls -d ~/.cache/huggingface/hub/models--mlx-community--"$1"/snapshots/*/ 2>/dev/null | head -1; }

preflight() {
    local load swap
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
    log "preflight: load $load, swap used ${swap} MB, $(pmset -g batt | grep -o "'.*'" | head -1)"
    if awk "BEGIN{exit !($load > 3)}"; then log "preflight: load above 3 — stop other work first"; exit 1; fi
    # Swapped pages cost nothing unless touched: refuse only when memory is short too. Paging
    # during the run is logged per phase (vm_stat), so a disturbed phase shows.
    local free; free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2+0}')
    if awk "BEGIN{exit !($swap > 512 && ${free:-0} < 60)}"; then log "preflight: swap in use and only ${free}% free — reboot first"; exit 1; fi
    log "preflight: ${free}% of memory free"
    pmset -g batt | grep -q "AC Power" || { log "preflight: not on AC power"; exit 1; }
    git rev-parse HEAD >"$OUT/commit.txt"; git status --short >"$OUT/dirty.txt"
    log "preflight: building"
    cargo build -q --release -p nuvec-lm --example bench --example pipeline_generate || exit 1
    baseline || exit 1
}

baseline() { # HEAD's bench in target/baseline (a detached worktree), rebuilt when HEAD moved
    local src=$ROOT/target/baseline
    if [ ! -d "$src" ]; then git worktree add --detach "$src" HEAD >/dev/null || return 1; fi
    (cd "$src" && git checkout -q --detach "$(git -C "$ROOT" rev-parse HEAD)" &&
        cargo build -q --release -p nuvec-lm --example bench) || return 1
    git -C "$src" rev-parse HEAD >"$OUT/baseline-commit.txt"
}

ab() {
    # Specs 076-080 against HEAD, alternating the two builds (three rounds) so drift hits
    # both alike. 077: MoE decode; 078: hybrid decode; 079: Gemma 4 chains; D-117: Gemma 4
    # 26B past 1K keys; 076: the Batcher at 4 requests; 080: long prefill with a wired limit.
    [ -x "$A" ] || { log "ab: $A missing (run preflight)"; return; }
    local r v exe
    for r in 1 2 3; do
        for v in A B; do
            exe=$A; [ $v = B ] && exe=$B
            step ab "e4b-$v$r" "$exe" --model "$(snapshot gemma-4-e4b-it-4bit)" --text "$TEXT" --repeats 1,40 --runs 1 --max-tokens 128 --ignore-eos
            step ab "g26-$v$r" "$exe" --model "$(snapshot gemma-4-26b-a4b-it-4bit)" --text-file "$P" --repeats 1,26,52 --runs 1 --max-tokens 64 --ignore-eos
            step ab "q30-$v$r" "$exe" --model "$(snapshot Qwen3-30B-A3B-4bit)" --text "$TEXT" --repeats 1,40 --runs 1 --max-tokens 128 --ignore-eos
            step ab "q36-$v$r" "$exe" --model "$(snapshot Qwen3.6-35B-A3B-4bit)" --text "$TEXT" --repeats 1,40 --runs 1 --max-tokens 128 --ignore-eos
            for model in Qwen3-0.6B-4bit Qwen3-8B-4bit; do
                local ids=$ROOT/target/bench-parity/$model/prompts/concurrency.txt
                if [ -f "$ids" ]; then
                    step ab "$model-batch4-$v$r" "$exe" --model "$(snapshot "$model")" --text "$TEXT" --batch 4 --ids-file "$ids" --max-tokens 128
                fi
            done
        done
    done
    # 080: a ~32K-token prefill with a wired limit (first token only), once per build.
    for v in A B; do
        exe=$A; [ $v = B ] && exe=$B
        step ab "q36-32k-$v" "$exe" --model "$(snapshot Qwen3.6-35B-A3B-4bit)" --text-file "$P" --repeats 230 --runs 1 --max-tokens 1
    done
}

gates() {
    # 070: the TCP ring with the new framing and JACCL code, against MLX's ring (no regression).
    step gates ring just bench-distributed
    # 073: the pre-shared key costs nothing after connecting.
    step gates ring-keyed env NUVEC_DISTRIBUTED_KEY=0123456789abcdef0123456789abcdef just bench-distributed
    # 071/072: pipeline and tensor parallelism on one Mac (2 ranks over loopback) vs mlx-lm.
    step gates pipeline uv run scripts/bench_pipeline.py --model mlx-community/Qwen3-0.6B-4bit -n 2
    step gates pipeline-batch uv run scripts/bench_pipeline.py --model mlx-community/Qwen3-0.6B-4bit -n 2 --batch 4
    step gates tensor uv run scripts/bench_pipeline.py --model mlx-community/Qwen3-0.6B-4bit -n 2 --tensor
    # 075 SC-002: an mlx-community checkpoint through the aligned cache: the first load
    # writes the copy, the second maps it (load time, then peak memory in the bench line).
    local cache=$OUT/gates/aligned-cache
    for i in 1 2; do
        step gates "aligned-cache-load-$i" env NUVEC_ALIGNED_CACHE="$cache" "$B" --model "$(snapshot Qwen3.5-9B-4bit)" --text "$TEXT" --repeats 1 --runs 1 --max-tokens 16
    done
    step gates plain-load "$B" --model "$(snapshot Qwen3.5-9B-4bit)" --text "$TEXT" --repeats 1 --runs 1 --max-tokens 16
    # 074 SC-003: 16 stop strings cost under 1% of decode.
    local m; m=$(snapshot Qwen3-0.6B-4bit)
    for s in 0 16 0 16; do step gates "stops-$s-$RANDOM" "$B" --model "$m" --text "$TEXT" --repeats 8 --runs 3 --max-tokens 256 --ignore-eos --stops $s; done
}

serve() {
    # E1: the noise floor. The 0.5.0 concurrency cells are one sample each, with swap in use.
    for i in 1 2 3 4 5; do
        step serve "conc-$i" uv run scripts/bench_parity.py mlx-community/Qwen3-0.6B-4bit mlx-community/Qwen3-8B-4bit mlx-community/Qwen3.6-35B-A3B-4bit mlx-community/Qwen3-30B-A3B-4bit \
            --sections concurrency --concurrency 1,4,8 --no-baseline --quiet --allow-dirty --out "$OUT/serve/conc-$i.json"
    done
    # E2/E3: the paged-path tax (Batcher at 1 request vs generate) and per-kernel shares.
    for model in Qwen3-0.6B-4bit Qwen3.6-35B-A3B-4bit; do
        local m ids; m=$(snapshot "$model"); ids=$ROOT/target/bench-parity/$model/prompts/concurrency.txt
        step serve "$model-generate" "$B" --model "$m" --text "$TEXT" --repeats 8 --runs 3 --max-tokens 128 --ignore-eos
        for n in 1 4 8; do step serve "$model-batch$n" "$B" --model "$m" --text "$TEXT" --batch $n --ids-file "$ids" --max-tokens 128; done
        step serve "$model-profile-batch4" env NUVEC_PROFILE=1 "$B" --model "$m" --text "$TEXT" --batch 4 --ids-file "$ids" --max-tokens 128
        step serve "$model-profile-generate" env NUVEC_PROFILE=1 "$B" --model "$m" --text "$TEXT" --repeats 8 --runs 1 --max-tokens 128 --ignore-eos
        # E5: prefill and insert at 8 requests.
        step serve "$model-profile-insert8" env NUVEC_PROFILE=1 "$B" --model "$m" --text "$TEXT" --batch 8 --ids-file "$ids" --max-tokens 1
    done
    step serve paged just bench-paged
}

beyond() {
    # Opportunity 2: MLX's GEMV at unaligned K (its ceiling for nuvec's port).
    step beyond qmv-alignment uv run scripts/quiet/qmv_alignment.py
    # The same out of the system cache (one row, N = 32,768), nuvec and MLX: the fast path
    # for unaligned K was dropped on a busy machine; this settles it.
    step beyond quant-dram just bench-quant dram
    # Opportunity 3: the MoE glue's upper bound.
    step beyond moe-glue uv run scripts/quiet/moe_glue.py
    step beyond moe-profile-129 env NUVEC_PROFILE=shapes "$B" --model "$(snapshot Qwen3-30B-A3B-4bit)" --text-file "$P" --repeats 1 --runs 1 --max-tokens 129 --ignore-eos
    step beyond moe-profile-1 env NUVEC_PROFILE=shapes "$B" --model "$(snapshot Qwen3-30B-A3B-4bit)" --text-file "$P" --repeats 1 --runs 1 --max-tokens 1
    # Opportunity 1: speculation today, and the draft head's share.
    step beyond spec just bench-spec
    local mtp=~/.cache/nuvec/mtp/Qwen--Qwen3.5-9B.safetensors
    if [ -f "$mtp" ]; then
        step beyond draft-head-on env NUVEC_PROFILE=shapes "$B" --model "$(snapshot Qwen3.5-9B-4bit)" --text-file "$P" --repeats 1 --runs 1 --max-tokens 128 --ignore-eos --draft 1 --drafter mtp --mtp-weights "$mtp"
        step beyond draft-head-off env NUVEC_PROFILE=shapes "$B" --model "$(snapshot Qwen3.5-9B-4bit)" --text-file "$P" --repeats 1 --runs 1 --max-tokens 128 --ignore-eos
    else
        log "beyond: $mtp missing (scripts/extract_mtp.py); skipping the draft-head profile"
    fi
    # Opportunity 5: Gemma 4 26B's decode drop past 1K keys.
    local g; g=$(snapshot gemma-4-26b-a4b-it-4bit)
    step beyond gemma26-context "$B" --model "$g" --text "$TEXT" --repeats 1,40,160,320 --runs 2 --max-tokens 128 --ignore-eos
    step beyond gemma26-profile-129 env NUVEC_PROFILE=shapes "$B" --model "$g" --text "$TEXT" --repeats 160 --runs 1 --max-tokens 129 --ignore-eos
    step beyond gemma26-profile-1 env NUVEC_PROFILE=shapes "$B" --model "$g" --text "$TEXT" --repeats 160 --runs 1 --max-tokens 1
    # Host- or GPU-bound: GPU idle between command buffers on the smallest model.
    step beyond host-bound xcrun xctrace record --template 'Metal System Trace' --time-limit 15s \
        --output "$OUT/beyond/q06.trace" --launch -- "$B" --model "$(snapshot Qwen3-0.6B-4bit)" --text "Hello. " --repeats 1 --runs 1 --max-tokens 512 --ignore-eos
}

decode() {
    # Filled from the decode/long-prefill study (research/11 §3): per-kernel traces of both
    # engines on gpt-oss-20b, Gemma 4 E4B and Qwen3.6-35B-A3B at 32K.
    if [ -x scripts/quiet/decode.sh ]; then step decode traces bash scripts/quiet/decode.sh "$OUT/decode"; else log "decode: scripts/quiet/decode.sh not present"; fi
}

record() {
    step record parity uv run scripts/bench_parity.py --tier quick --no-baseline --allow-dirty --out "$OUT/record/record.json"
}

caffeinate -dimsu -w $$ &
paging() { vm_stat | awk '/Pageins|Pageouts|Swapins|Swapouts/{gsub(/\./,"",$NF); sub(/:$/,"",$1); printf "%s=%s ", $1, $NF}'; }
for phase in $PHASES; do
    log "== $phase (paging before: $(paging))"
    "$phase"
    log "== $phase done (paging after: $(paging))"
done
log "done: $OUT"
