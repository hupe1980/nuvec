#!/usr/bin/env bash
# The CUDA runtime on a GPU pod: differential tests, nuvec's benchmark (with a
# GEMM tile sweep) and MLX's on the same GPU. Run through
# `scripts/cuda_remote.py session -- bash scripts/cuda_spike.sh` (TESTS_ONLY=1: the tests and
# memcheck only). Every step has a timeout.
set -u
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
cargo build -q --release -p nuvec-gpu --features cuda --tests --examples 2>&1 | tail -3
echo "== tests (one process each)"
T=$(ls -t target/release/deps/cuda_spike-* | grep -v '\.d$' | head -1)
for t in elementwise_matches_the_cpu strided_and_broadcast_operands_match_the_cpu row_sums_match_the_cpu tile_gemm_matches_the_cpu; do
  NUVEC_REQUIRE_GPU=1 timeout 600 "$T" "$t" --exact --nocapture 2>&1 | grep -vE "^$|^running" | tail -12
done
echo "== tests in one process (the state one test leaves for the next)"
NUVEC_REQUIRE_GPU=1 timeout 600 "$T" --test-threads=1 2>&1 | grep -vE "^$|^running" | tail -25
echo "== compute-sanitizer memcheck (FR-014)"
for t in elementwise_matches_the_cpu row_sums_match_the_cpu strided_and_broadcast_operands_match_the_cpu; do
  NUVEC_REQUIRE_GPU=1 timeout 900 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --print-limit 5 "$T" "$t" --exact 2>&1 \
    | grep -E "ERROR SUMMARY|Invalid|at 0x|by thread|Address|nuvec_" | head -12
done
[ "${TESTS_ONLY:-0}" = 1 ] && exit 0
echo "== nuvec benchmark (f32 exact: 3xTF32)"
timeout 900 ./target/release/examples/cuda_bench 2>&1 | grep -E '^\{|panicked|Error'
echo "== nuvec f32, TF32"
NUVEC_CUDA_TF32=1 timeout 600 ./target/release/examples/cuda_bench gemm f32 2>&1 | grep -E '^\{|panicked|Error'
echo "== tile sweep (4096 and 8192)"
sweep() { # sweep <dtype> <env> <tiles...>
  local d=$1 env=$2; shift 2
  for t in "$@"; do for n in 4096 8192; do
    env $env NUVEC_CUDA_GEMM_TILES=$t timeout 300 ./target/release/examples/cuda_bench sweep $d $n 2>&1 \
      | grep -E '"Gemm"|panicked|Error' | sed "s/^/$t $d ${env:-} /"
  done; done
}
sweep f16 "" 128,128,32 128,64,32 64,128,32 128,128,16 64,64,32
sweep f32 "NUVEC_CUDA_TF32=1" 64,64,32 64,128,32 128,64,32 128,128,16 64,64,16
sweep f32 "" 64,64,32 64,128,32 128,64,32 128,128,16 64,64,16
echo "== MLX benchmark"
python3 -m venv /root/mlx >/dev/null && timeout 600 /root/mlx/bin/pip install -q 'mlx[cuda13]==0.32.4' 2>&1 | tail -2
timeout 900 /root/mlx/bin/python scripts/bench_cuda.py 2>&1 | grep Gemm
echo "== MLX, TF32 off"
MLX_ENABLE_TF32=0 timeout 600 /root/mlx/bin/python scripts/bench_cuda.py 2>&1 | grep -E 'Gemm.*float32'
