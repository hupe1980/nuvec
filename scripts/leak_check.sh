#!/usr/bin/env bash
# Autorelease-pool and leak check: runs a greedy decode loop on
# the tiny qwen3 test fixture, on a thread other than the main one, and fails on
#   - any "MISSING POOLS" line from OBJC_DEBUG_MISSING_POOLS=YES (an Objective-C object
#     autoreleased with no pool in place, which leaks on such threads), or
#   - leaked objects (`leaks --atExit`) growing between a 1 000-step and a 10 000-step run.
#
# Usage: scripts/leak_check.sh [steps]   (default 10000; macOS with a Metal GPU only)
set -euo pipefail

steps="${1:-10000}"
short=$((steps / 10))
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

cargo build --release -q -p nuvec-lm --example decode_loop
bin="$root/target/release/examples/decode_loop"

log="$(mktemp -t nuvec-leak-check)"
trap 'rm -f "$log"' EXIT

echo "decode loop, $steps steps, OBJC_DEBUG_MISSING_POOLS=YES"
OBJC_DEBUG_MISSING_POOLS=YES "$bin" --steps "$steps" 2>"$log"
if grep -q "MISSING POOLS" "$log"; then
    echo "error: objects autoreleased with no pool in place:" >&2
    grep "MISSING POOLS" "$log" | sed -E 's/^objc\[[0-9]+\]: //; s/0x[0-9a-f]+/0x…/g' | sort | uniq -c >&2
    exit 1
fi

leaked() {
    # "Process N: K leaks for B total leaked bytes."
    leaks --atExit -- "$bin" --steps "$1" 2>/dev/null | sed -nE 's/^Process [0-9]+: ([0-9]+) leaks? for.*/\1/p' | tail -1
}
echo "leaks --atExit: $short and $steps steps"
few="$(leaked "$short")"
many="$(leaked "$steps")"
echo "leaked objects: $few after $short steps, $many after $steps steps"
if [[ -z "$few" || -z "$many" ]]; then
    echo "error: could not read the leaks report" >&2
    exit 1
fi
if ((many > few)); then
    echo "error: leaked objects grow with the number of steps" >&2
    exit 1
fi
echo "ok"
