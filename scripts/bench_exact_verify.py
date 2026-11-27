"""Measure fixed DFlash chains and verify their acceptance decisions against serial decoding."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from bench_parity import ROOT, snapshot, source_identity, write_atomic

MODEL = "mlx-community/Qwen3.5-9B-4bit"
DRAFT = "z-lab/Qwen3.5-9B-DFlash"
PROMPTS = {
    "copy": "Copy exactly the following list twice, retaining punctuation:\napple, pear, lime, fig, plum, apple, pear, lime, fig, plum.\n",
    "code": "Write a Rust function that stably merges two sorted slices of integers. Include a unit test.\n",
    "natural": "Explain to a gardener why a clear night can bring frost even after a warm day.\n",
}


def check_rounds(row, reference, width):
    """Check coverage, the longest valid draft prefix and the emitted correction for each round."""
    if row["ids"] != reference:
        raise ValueError("emitted tokens differ from the reference")
    rounds = row["rounds"]
    drafted = accepted = 0
    end = 0  # The first DFlash block starts from the prompt’s last token.
    widths = []
    for r in rounds:
        start, draft = r["start"], r["draft"]
        if (
            start != end
            or start >= len(reference)
            or len(draft) != min(width, len(reference) - start - 1)
        ):
            raise ValueError("non-contiguous round or wrong effective width")
        expected = 0
        while expected < len(draft) and start + expected < len(reference):
            if draft[expected] != reference[start + expected]:
                break
            expected += 1
        end = start + expected
        if r["accepted"] != expected or end >= len(reference) or r["next"] != reference[end]:
            raise ValueError(f"wrong accepted prefix or correction at output {start}")
        end += 1
        drafted += len(draft)
        accepted += expected
        widths.append(len(draft) + 1)
    if (len(rounds), drafted, accepted, max(widths, default=0)) != (
        row["verifies"],
        row["drafted"],
        row["accepted"],
        row["max_verify_rows"],
    ):
        raise ValueError("trace and aggregate verification counts differ")
    if width and (width + 1 not in widths or len(reference) - end > 1):
        raise ValueError(f"incomplete coverage for {width + 1}-row verification")
    if not width and rounds:
        raise ValueError("serial generation reported a speculative round")


def main(run_dir):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=ROOT / "target/exact-verify/report.json")
    args = ap.parse_args()
    if args.runs < 3 or args.tokens < 16:
        ap.error("use at least three paired runs and sixteen generated tokens")
    if os.getenv("NUVEC_PROFILE") == "1":
        ap.error("GPU dispatch profiling changes throughput; use a separate run")
    model, draft = snapshot(MODEL), snapshot(DRAFT)
    if not model or not draft:
        ap.error("both target and DFlash snapshots must be in the local cache")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "nuvec-lm", "--example", "bench"],
        cwd=ROOT,
        check=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # A concurrent Cargo build can replace the example during a multi-minute run.
    binary = run_dir / "bench"
    shutil.copy2(ROOT / "target/release/examples/bench", binary)
    report = {
        "model": MODEL,
        "target_revision": model.name,
        "draft": DRAFT,
        "draft_revision": draft.name,
        "draft_format": "affine 4-bit group 64",
        "temperature": 0,
        "output_tokens": args.tokens,
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "macos": platform.mac_ver()[0],
        "load_average_at_start": os.getloadavg(),
        "source_sha256": source_identity(),
        "bench_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "status": "measuring",
        "samples": [],
        "summary": {},
    }

    def run_prompt(name, prompt):
        path = run_dir / (name + ".txt")
        path.write_text(prompt)
        reference = None
        times = {w: [] for w in (0, 3, 5, 7)}
        costs = []

        def measure(width, index, *, profile=False, exact=True):
            nonlocal reference
            cmd = [
                str(binary),
                "--model",
                str(model),
                "--text-file",
                str(path),
                "--runs",
                "1",
                "--repeats",
                "1",
                "--ignore-eos",
                "--max-tokens",
                str(args.tokens),
                "--emit-ids",
            ]
            if exact:
                cmd.append("--row-exact")
            if width:
                cmd += [
                    "--draft",
                    str(width),
                    "--drafter",
                    "dflash",
                    "--policy",
                    "fixed",
                    "--dflash-dir",
                    str(draft),
                    "--dflash-bits",
                    "4",
                ]
            if profile and width:
                cmd.append("--time-stages")
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
            (row,) = [
                json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
            ]
            if len(row["ids"]) != args.tokens:
                raise ValueError(f"{name}: generation stopped after {len(row['ids'])} tokens")
            if reference is None:
                if width or not exact:
                    raise ValueError("serial reference must be measured first")
                reference = row["ids"]
            row.update(
                width=width,
                pass_index=index,
                stage_pair=profile,
                row_exact=exact,
                matches_serial=row["ids"] == reference,
                prompt=name,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            )
            report["samples"].append(row)
            try:
                check_rounds(row, reference if exact else row["ids"], width)
            except ValueError as error:
                report.update(
                    status="failed", error=f"{name}, pass {index}, width {width}: {error}"
                )
                write_atomic(args.out, report)
                raise
            write_atomic(args.out, report)
            print(name, index, width, row["decode"], flush=True)
            return row

        widths = list(times)
        for i in range(args.runs):
            shift = i % len(widths)
            for width in widths[shift:] + widths[:shift]:
                times[width].append(measure(width, i)["decode"])
            # A full-width fenced target round versus one plain decode step in the paired run.
            pair = {}
            for width in (0, 7) if i % 2 == 0 else (7, 0):
                pair[width] = measure(width, i, profile=True)
            full = [r["target_s"] for r in pair[7]["rounds"] if len(r["draft"]) == 7]
            costs.append(statistics.mean(full) * (pair[0]["tokens"] - 1) / pair[0]["decode_s"])
        fast = measure(7, 0, exact=False)
        medians = {k: statistics.median(rates) for k, rates in times.items()}
        report["summary"][name] = {
            "median_decode_tok_s": medians,
            "seven_over_three": medians[7] / medians[3],
            "eight_row_target_over_plain": costs,
            "fast_seven_matches_serial": fast["matches_serial"],
        }

    for name, prompt in PROMPTS.items():
        run_prompt(name, prompt)
    ratios = [s["seven_over_three"] for s in report["summary"].values()]
    report["geomean"] = math.prod(ratios) ** (1 / len(ratios))
    max_cost = max(
        statistics.median(s["eight_row_target_over_plain"]) for s in report["summary"].values()
    )
    report["promote_seven"] = report["geomean"] >= 1.05 and min(ratios) >= 0.97 and max_cost <= 1.5
    report["status"] = "complete"
    write_atomic(args.out, report)
    print(args.out, "promote seven" if report["promote_seven"] else "retain three")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as run_dir:
        main(Path(run_dir))
