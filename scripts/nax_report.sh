#!/usr/bin/env bash
# The Neural Accelerator (NAX) report for an M5-class Mac: correctness and speed of nuvec's
# NAX kernels against its SIMD kernels and MLX (whose own NAX path runs on an M5), written
# to `nax-report-<architecture>.md` to attach to a GitHub issue. Runs with
# `just nax-report` (about 10 minutes; close other apps first).
set -euo pipefail
cd "$(dirname "$0")/.."
arch=$(swift -e 'import Metal; print(MTLCreateSystemDefaultDevice()!.architecture.name)' 2>/dev/null || echo unknown)
out="nax-report-${arch}.md"
{
  echo "# nuvec NAX report"
  echo
  echo "- GPU: $(swift -e 'import Metal; print(MTLCreateSystemDefaultDevice()!.name)' 2>/dev/null) (\`$arch\`)"
  echo "- macOS: $(sw_vers -productVersion), SDK $(xcrun --sdk macosx --show-sdk-version)"
  echo "- nuvec: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo
  echo "## Correctness (NUVEC_NAX=on)"
  echo
  echo '```'
  NUVEC_NAX=on NUVEC_PROFILE=1 cargo test -q --release -p nuvec-core --test nax -- --nocapture 2>&1 | grep -E "NAX|test result|panicked|FAILED" || true
  echo '```'
  for mode in off on; do
    echo
    echo "## GEMM (NUVEC_NAX=$mode)"
    echo
    NUVEC_NAX=$mode uv run scripts/bench_gemm.py 2>&1 | grep -E "^\||^GPU" || true
    echo
    echo "## Quantized products (NUVEC_NAX=$mode)"
    echo
    NUVEC_NAX=$mode uv run scripts/bench_quant.py 2>&1 | grep -E "^\||^GPU" || true
  done
  # Time to first token at 2K and 32K tokens against mlx-lm (MLX's own NAX path on an M5),
  # a dense and a mixture-of-experts 4-bit model, with nuvec's NAX kernels off and on.
  for mode in off on; do
    echo
    echo "## Prefill against mlx-lm (NUVEC_NAX=$mode)"
    echo
    NUVEC_NAX=$mode uv run scripts/bench_parity.py mlx-community/Qwen3-8B-4bit \
      mlx-community/Qwen3-30B-A3B-4bit --sections ttft --prompt-tokens 2048,32768 \
      --no-baseline --out "target/nax-report-ttft-$mode.json" 2>/dev/null \
      | grep -E "^\| (model|---|Qwen)" || echo "(skipped: the models are not cached)"
  done
} | tee "$out"
echo
echo "Wrote $out: please attach it to an issue titled \"NAX report: $arch\"."
