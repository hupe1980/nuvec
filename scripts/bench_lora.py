# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""LoRA training, nuvec-lm against `mlx_lm.lora`: the same synthetic dataset (200 rows of
`--words` words, about 500 tokens by default; `text` rows, or with `--data-format chat` a
user message of a quarter of the words and the assistant's answer of the rest), Qwen3-0.6B-4bit by default, mlx-lm's
defaults (16 layers, rank 8, batch 4, Adam 1e-5), 50 iterations, interleaved rounds.
Compares Tokens/sec (mean of the reports after the first), peak memory, and each report's
training loss (first round).

    uv run scripts/bench_lora.py [--model mlx-community/Qwen3-0.6B-4bit] [--rounds 3]
        [--words 400] [--max-seq-length 2048] [--grad-checkpoint]
        [--data-format text|chat] [--mask-prompt] [--steps-per-report 10]
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORDS = ("the lighthouse keeper climbed the spiral stairs every evening counting steps "
         "while the sea below kept its own slow time against the rocks and the gulls").split()
# mlx-lm reports through a training callback; this runner prints each report as nuvec's
# `lora` example does: `REPORT {"iteration", "train_loss", "tokens_per_second", "peak_memory"}`.
MLX_RUNNER = """
import json, sys
import mlx_lm.lora as lora
from mlx_lm.tuner.callbacks import TrainingCallback
class Report(TrainingCallback):
    def on_train_loss_report(self, info):
        print("REPORT " + json.dumps(info), flush=True)
args = vars(lora.build_parser().parse_args(sys.argv[1:]))
for k, v in lora.CONFIG_DEFAULTS.items():
    if args.get(k) is None:
        args[k] = v
lora.get_reporting_callbacks = lambda *a, **k: Report()  # `run` builds its own otherwise
lora.run(lora.types.SimpleNamespace(**args))
"""


def snapshot(model):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    repo = hub / f"models--{model.replace('/', '--')}"
    ref = (repo / "refs" / "main").read_text().strip()
    return repo / "snapshots" / ref


def data(dirname, words, fmt):
    rng = random.Random(0)
    say = lambda n: " ".join(rng.choice(WORDS) for _ in range(n))
    for split, n in (("train", 200), ("valid", 20)):
        with open(Path(dirname) / f"{split}.jsonl", "w") as f:
            for _ in range(n):
                if fmt == "chat":
                    row = {"messages": [{"role": "user", "content": say(words // 4)},
                                        {"role": "assistant", "content": say(words - words // 4)}]}
                else:
                    row = {"text": say(words)}
                f.write(json.dumps(row) + "\n")


def parse(out):
    rows = [json.loads(l[7:]) for l in out.splitlines() if l.startswith("REPORT ")]
    if len(rows) < 2:
        sys.exit("no reports:\n" + out[-2000:])
    return (statistics.mean(r["tokens_per_second"] for r in rows[1:]),
            max(r["peak_memory"] for r in rows), [r["train_loss"] for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--engines", default="nuvec,mlx")
    p.add_argument("--words", type=int, default=400)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--data-format", choices=["text", "chat"], default="text")
    p.add_argument("--mask-prompt", action="store_true")
    p.add_argument("--steps-per-report", type=int, default=10)
    a = p.parse_args()
    snap = snapshot(a.model)
    engines = a.engines.split(",")
    if "nuvec" in engines:
        subprocess.run(["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example", "lora"],
                       cwd=ROOT, check=True)
    common = ["--iters", str(a.iters), "--batch-size", "4", "--steps-per-report", str(a.steps_per_report),
              "--steps-per-eval", "1000", "--val-batches", "1", "--seed", "0", "--train",
              "--max-seq-length", str(a.max_seq_length)] + (["--grad-checkpoint"] * a.grad_checkpoint)
    common += ["--mask-prompt"] * a.mask_prompt
    results = {e: [] for e in engines}
    with tempfile.TemporaryDirectory() as tmp:
        data(tmp, a.words, a.data_format)
        for r in range(a.rounds):
            for e in engines:
                adapters = Path(tmp) / f"adapters-{e}"
                if e == "mlx":
                    cmd = [sys.executable, "-c", MLX_RUNNER, "--model", str(snap),
                           "--data", tmp, "--adapter-path", str(adapters), *common]
                else:
                    cmd = [str(ROOT / "target/release/examples/lora"), "--model", str(snap),
                           "--data", tmp, "--adapter-path", str(adapters), *common]
                out = subprocess.run(cmd, capture_output=True, text=True)
                if out.returncode:
                    sys.exit(f"{e} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
                tps, peak, losses = parse(out.stdout + out.stderr)
                results[e].append((tps, peak, losses))
                print(f"round {r + 1} {e}: {tps:.1f} tok/s, peak {peak:.3f} GB, last loss {losses[-1]:.3f}",
                      flush=True)
    ckpt = ", gradient checkpointing" if a.grad_checkpoint else ""
    ckpt += f", {a.data_format} rows" + (", masked prompts" if a.mask_prompt else "")
    print(f"\n{a.model}, batch 4, {a.words} words per row{ckpt}, {a.iters} iterations, "
          f"median of {a.rounds} rounds")
    med = {e: (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
           for e, v in results.items()}
    for e, (t, m) in med.items():
        print(f"{e}: {t:.1f} tok/s, peak {m:.3f} GB")
    if {"nuvec", "mlx"} <= med.keys():
        print(f"nuvec / mlx-lm: speed {med['nuvec'][0] / med['mlx'][0]:.2f}×, "
              f"peak {med['nuvec'][1] / med['mlx'][1]:.2f}×")
        ours, theirs = results["nuvec"][0][2], results["mlx"][0][2]
        worst = max(abs(x - y) / abs(y) for x, y in zip(ours, theirs))
        print("losses (nuvec | mlx-lm): " + ", ".join(f"{x:.4f}|{y:.4f}" for x, y in zip(ours, theirs))
              + f"; largest relative difference {worst:.2e}")


if __name__ == "__main__":
    main()
