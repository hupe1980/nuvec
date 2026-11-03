#!/usr/bin/env bash
# The decode and long-prefill phase of scripts/quiet_session.sh: per-kernel
# traces of both engines (Metal System Trace), MLX-side switches that test hypotheses
# without code, and the Qwen3-30B-A3B 0.4.1 vs 0.5.0 A/B. Released code is measured from
# worktrees (the working tree carries v0.6 changes).
#
#     scripts/quiet/decode.sh OUT_DIR [step ...]   # steps: shaders ab traces prefill mlx-switches
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT=${1:?output directory}; shift
STEPS=${*:-shaders ab traces prefill mlx-switches}
mkdir -p "$OUT"
TOOLS=$ROOT/scripts/quiet
TEXT="The lighthouse keeper climbed the spiral stairs every evening, counting the steps and listening to the sea against the rocks below. "
MST=${MST:-Metal System Trace}
# A persistent environment (an ephemeral `uv run` interpreter is gone once it returns).
VENV=$ROOT/target/quiet/mlx-venv
[ -x "$VENV/bin/python" ] || { uv venv -q "$VENV" && uv pip install -q --python "$VENV/bin/python" mlx==0.32.4 mlx-lm==0.32.2; }
PY=$VENV/bin/python
snap() { local d=~/.cache/huggingface/hub/models--${1//\//--}; echo "$d/snapshots/$(cat "$d/refs/main")"; }
log() { echo "[$(date +%H:%M:%S)] decode: $*" | tee -a "$OUT/decode.log"; }

# Release binaries from worktrees, each with its own target directory.
build() { # build <tag>
    local tag=$1 wt=$ROOT/target/wt-$1
    [ -d "$wt" ] || git worktree add --detach "$wt" "$tag" >/dev/null
    cargo build -q --release -p nuvec-lm --example bench --manifest-path "$wt/Cargo.toml" --target-dir "$ROOT/target/wt-target-$tag"
    echo "$ROOT/target/wt-target-$tag/release/examples/bench"
}
B=$(build v0.5.0)

# Exports the tables the analysis reads, and prints the per-process and per-family tables.
export_trace() { # export_trace <trace> <process substring>
    local x=$1
    xcrun xctrace export --input "$x" --toc >"$x.toc.xml"
    for s in metal-shader-profiler-intervals metal-gpu-intervals metal-resource-allocations \
        metal-residency-set-usage-event metal-residency-set-resource-event gpu-performance-state-intervals; do
        xcrun xctrace export --input "$x" --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$s\"]" >"$x.$s.xml" 2>/dev/null
    done
    python3 -I "$TOOLS/mst_kernels.py" "$x.metal-gpu-intervals.xml" --process "$2" --segments 1 >"$x.segments.txt"
    python3 -I "$TOOLS/mst_kernels.py" "$x.metal-shader-profiler-intervals.xml" --process "$2" --family >"$x.families.txt"
    python3 -I "$TOOLS/mst_kernels.py" "$x.metal-shader-profiler-intervals.xml" --process "$2" --top 60 --csv "$x.kernels.csv" >"$x.kernels.txt"
}

# The same decode command for both engines: a warm-up, then the measured prompt; the last
# GPU-busy segment is the first token plus 128 decode steps.
trace_decode() { # trace_decode <model> <repeats> <tag>
    local m; m=$(snap "$1")
    xcrun xctrace record --template "$MST" --output "$OUT/nv-$3.trace" --launch -- \
        "$B" --model "$m" --text "$TEXT" --repeats "$2" --runs 1 --max-tokens 129 --ignore-eos >/dev/null
    export_trace "$OUT/nv-$3.trace" bench
    xcrun xctrace record --template "$MST" --output "$OUT/mlx-$3.trace" --launch -- "$PY" -I -c '
import sys; from mlx_lm import load, stream_generate
m, t = load(sys.argv[1]); t.eos_token_ids = []; T = sys.argv[2]
for _ in stream_generate(m, t, t.encode(T), max_tokens=129): pass
for _ in stream_generate(m, t, t.encode(T * int(sys.argv[3])), max_tokens=129): pass' "$m" "$TEXT" "$2" >/dev/null
    export_trace "$OUT/mlx-$3.trace" python
}

for step in $STEPS; do
    case $step in
    shaders)
        # E0: Shader Timeline must be on, or the per-kernel table is empty (then save a
        # template with it on in Instruments and rerun with MST=~/MST-shaders.tracetemplate).
        xcrun xctrace record --template "$MST" --output "$OUT/probe.trace" --launch -- \
            "$B" --model "$(snap mlx-community/Qwen3-0.6B-4bit)" --text "$TEXT" --repeats 1 --runs 1 --max-tokens 32 --ignore-eos >/dev/null
        n=$(xcrun xctrace export --input "$OUT/probe.trace" --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' | grep -c '<row')
        log "shader rows: $n"
        [ "$n" -gt 0 ] || { log "Shader Timeline is off: save a template with it on and set MST"; exit 1; }
        ;;
    ab)
        # E1: is Qwen3-30B-A3B slower than 0.4.1? Interleaved, medians per prompt length.
        A=$(build v0.4.1)
        for i in 1 2 3 4; do for b in "$A" "$B"; do
            echo "# $b"; "$b" --model "$(snap mlx-community/Qwen3-30B-A3B-4bit)" --text "$TEXT" --repeats 1,160 --runs 2 --max-tokens 128 --ignore-eos
        done; done >"$OUT/q30-ab.jsonl" 2>&1
        log "Qwen3-30B-A3B A/B written; compare medians, not best samples"
        ;;
    traces)
        # E2: decode traces where the gaps are (gpt-oss grows with context; Gemma 4 E4B).
        trace_decode mlx-community/gpt-oss-20b-MXFP4-Q8 1 gptoss-1
        trace_decode mlx-community/gpt-oss-20b-MXFP4-Q8 160 gptoss-160
        trace_decode mlx-community/gemma-4-e4b-it-4bit 160 e4b-160
        log "decode traces: compare *.families.txt per engine, last segment / 129 steps"
        ;;
    prefill)
        # E4: Qwen3.6-35B-A3B first token at 32K, both engines, the same ids.
        m=$(snap mlx-community/Qwen3.6-35B-A3B-4bit)
        "$PY" -I scripts/bench_parity.py --mlx-worker ids "$m" "$OUT/prompts-q36" 32768 >/dev/null
        f=$OUT/prompts-q36/ttft-32768.txt
        xcrun xctrace record --template "$MST" --output "$OUT/nv-q36-32k.trace" --launch -- "$B" --model "$m" --ids-file "$f" --runs 1 >/dev/null
        export_trace "$OUT/nv-q36-32k.trace" bench
        xcrun xctrace record --template "$MST" --output "$OUT/mlx-q36-32k.trace" --launch -- "$PY" -I scripts/bench_parity.py --mlx-worker ttft "$m" "$f" 1 >/dev/null
        export_trace "$OUT/mlx-q36-32k.trace" python
        log "32K traces: per-chunk families of the last 4 segments, idle = wall − kernel union"
        ;;
    mlx-switches)
        # E6/E7: hypotheses tested on MLX without code. If MLX loses the gap with one
        # residency set, port the capped sets (the parked patch no longer exists); if it slows
        # with nuvec-like command buffers, add a prefill byte budget.
        m=$(snap mlx-community/Qwen3.6-35B-A3B-4bit); f=$OUT/prompts-q36/ttft-32768.txt
        [ -f "$f" ] || "$PY" -I scripts/bench_parity.py --mlx-worker ids "$m" "$OUT/prompts-q36" 32768 >/dev/null
        for env in "" "MLX_RESIDENCY_SET_MAX_PCT=0" "MLX_MAX_MB_PER_BUFFER=100000 MLX_MAX_OPS_PER_BUFFER=256" "MLX_MAX_MB_PER_BUFFER=8 MLX_MAX_OPS_PER_BUFFER=8"; do
            echo "# ${env:-default}"; env $env "$PY" -I scripts/bench_parity.py --mlx-worker ttft "$m" "$f" 2
        done >"$OUT/mlx-switches.txt" 2>&1
        log "MLX switches written"
        ;;
    *) log "unknown step $step" ;;
    esac
done
