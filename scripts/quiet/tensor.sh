#!/usr/bin/env bash
# Tensor parallelism against MLX and mlx-lm on one quiet Mac (2 ranks over loopback). Waits
# for a quiet machine (5-minute load below 3 at two checks 5 minutes apart, at most 6 hours;
# nothing runs if it never is).
#
#     scripts/quiet/tensor.sh
#
#   decode  - decode-shaped all_sum (56 GPU ops + all_sums per evaluation), nuvec vs MLX
#   prefill - the same at prefill's shape (2,048 rows: 4 MB per all_sum)
#   tensor  - Qwen3-0.6B-4bit, random 2,048-id prompt, 128 tokens, EOS ignored: 3
#             interleaved rounds of `pipeline_generate --tensor --trials 3` and
#             `mlx_lm.benchmark -p 2048 -g 128 -n 3` (decode compared in mlx-lm's definition)
#   single  - single-process decode, Qwen3-0.6B and Qwen3-8B, against the baseline build in
#             target/baseline (3 interleaved rounds)
#   ring    - the ring all_sum bench against MLX
#
# Results: target/quiet/tensor-<timestamp>/.
set -u
cd "$(dirname "$0")/../.."
OUT=target/quiet/tensor-$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
snapshot() { ls -d ~/.cache/huggingface/hub/models--mlx-community--"$1"/snapshots/*/ 2>/dev/null | head -1; }

quiet=0
for _ in $(seq 72); do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk "BEGIN{exit !($load < 3)}"; then quiet=$((quiet + 1)); else quiet=0; fi
    log "load $load, quiet $quiet"
    [ $quiet -ge 2 ] && break
    sleep 300
done
[ $quiet -ge 2 ] || { log "never quiet: nothing measured"; exit 1; }
caffeinate -dimsu -w $$ &

cargo build -q --release -p nuvec-lm --example bench --example pipeline_generate || exit 1
M06=$(snapshot Qwen3-0.6B-4bit)
M8=$(snapshot Qwen3-8B-4bit)
LAUNCH=(uv run -q --python 3.13 --with mlx==0.32.4 --with mlx-lm==0.32.2 python -c
    "from mlx._distributed_utils.launch import main; main()" --backend ring -n 2 --)

log "decode"
uv run -q scripts/bench_distributed.py --decode 2>&1 | tee -a "$OUT/decode.log"
log "prefill"
NUVEC_BENCH_ROWS=2048 uv run -q scripts/bench_distributed.py --decode 2>&1 | tee -a "$OUT/prefill.log"

for r in 1 2 3; do
    log "tensor, round $r"
    "${LAUNCH[@]}" "$PWD/target/release/examples/pipeline_generate" --model "$M06" --tensor \
        --warmup --random-prompt 2048 --ignore-eos --max-tokens 128 --trials 3 2>/dev/null |
        grep -E "^(Prompt|Generation)" | sed "s/^/nuvec $r /" | tee -a "$OUT/tensor.log"
    "${LAUNCH[@]}" python -m mlx_lm.benchmark --model mlx-community/Qwen3-0.6B-4bit \
        -p 2048 -g 128 -n 3 2>&1 | grep Averages | sed "s/^/mlx $r /" | tee -a "$OUT/tensor.log"
done

if [ -x target/baseline/target/release/examples/bench ]; then
    for m in "$M06" "$M8"; do
        for r in 1 2 3; do
            log "single $(basename "$(dirname "$(dirname "$m")")"), round $r"
            for b in new base; do
                exe=target/release/examples/bench
                [ $b = base ] && exe=target/baseline/target/release/examples/bench
                "$exe" --model "$m" --text "Hello there. " --repeats 1 --runs 2 --max-tokens 128 \
                    --ignore-eos 2>/dev/null | sed "s/^/$b $r /" | tee -a "$OUT/single.log"
            done
        done
    done
else
    log "single: skipped (no baseline build in target/baseline)"
fi

log "ring"
uv run -q scripts/bench_distributed.py 2>&1 | tee -a "$OUT/ring.log"
log "done: $OUT"
