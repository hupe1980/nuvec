# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "mlx-lm==0.32.2", "numpy"]
# ///
"""Check nuvec's numerical fixtures and frozen greedy prompts against the pinned MLX."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from bench_parity import ROOT, snapshot, write_atomic

FIXTURE = ROOT / "crates/nuvec-core/tests/fixtures/elementwise.txt"
PROMPTS = ROOT / "scripts/fixtures/qualification"
MODELS = ("Qwen3-8B-4bit", "Qwen3.5-9B-4bit")


def check_models(work, report):
    for name in MODELS:
        corpus_path = PROMPTS / f"{name}.json"
        corpus = json.loads(corpus_path.read_bytes())
        model = corpus["model"]
        checkpoint = snapshot(model, corpus["snapshot"])
        if not checkpoint:
            raise ValueError(f"missing checkpoint {model} at {corpus['snapshot']}")
        reference = work / f"{name}.json"
        subprocess.run(
            [
                sys.executable,
                ROOT / "scripts/greedy_reference.py",
                checkpoint,
                "--prompts-from",
                corpus_path,
                "--out",
                reference,
            ],
            cwd=ROOT,
            check=True,
        )
        replay = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "-p",
                "nuvec-lm",
                "--example",
                "greedy_parity",
                "--",
                "--model",
                checkpoint,
                "--reference",
                reference,
                "--jsonl",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        cases = [json.loads(line) for line in replay.stdout.splitlines()]
        report.setdefault("greedy", []).append(
            {
                "model": model,
                "snapshot": checkpoint.name,
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "cases": cases,
            }
        )
        replay.check_returncode()
        expected = [len(case["prompt"]) for case in corpus["cases"]]
        if [c["prompt_tokens"] for c in cases] != expected or not all(
            c["status"] == "pass" and c["matched_tokens"] == 256 for c in cases
        ):
            raise ValueError(f"{model}: incomplete greedy replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-only", action="store_true", help="skip model generation")
    args = parser.parse_args()
    directory = ROOT / "target/qualification"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / ("fixtures.json" if args.fixture_only else "report.json")
    report = {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")}
    try:
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            work = Path(temp)
            fixture, meta = work / "elementwise.txt", work / "machine.json"
            subprocess.run(
                [
                    sys.executable,
                    ROOT / "scripts/elementwise_fixtures.py",
                    "--out",
                    fixture,
                    "--metadata",
                    meta,
                ],
                cwd=ROOT,
                check=True,
            )
            machine = json.loads(meta.read_bytes())
            report["machine"] = machine
            if not str(machine.get("architecture", "")).startswith("applegpu"):
                raise ValueError("qualification requires a physical Apple GPU")
            observed = fixture.read_bytes()
            if observed != FIXTURE.read_bytes():
                raise ValueError(
                    f"generated MLX {report['mlx']} fixture differs from {FIXTURE.name}"
                )
            report["fixture_rows"] = len(observed.splitlines()) - 1
            report["fixture_sha256"] = hashlib.sha256(observed).hexdigest()
            subprocess.run(
                ["cargo", "test", "-p", "nuvec-core", "--test", "elementwise", "--test", "mlx"],
                cwd=ROOT,
                env={**os.environ, "NUVEC_REQUIRE_GPU": "1", "NUVEC_REQUIRE_CPU_PROFILE": "1"},
                check=True,
            )
            if not args.fixture_only:
                check_models(work, report)
        report["status"] = "passed"
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report.update(status="failed", reason=str(error))
    write_atomic(dest, report)
    print(dest, report["status"], report.get("reason", ""))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    sys.exit(main())
