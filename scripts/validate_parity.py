# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema==4.23.0"]
# ///
"""Validates benchmark records, or compares two parity records.

`uv run scripts/validate_parity.py [paths…]`: every record (default: every JSON file under
site/static/benchmarks) against its schema (`schema` field),
and every path in index.json. Exit 1 on any error.

`uv run scripts/validate_parity.py --compare BASE CANDIDATE [--max-drop 0.02]`: the
nuvec / mlx-lm ratio of every decode and legacy prefill cell present in both; exit 1 if
any fell by more than `max-drop`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jsonschema
from bench_parity import (
    CONC_PROMPT,
    CONC_TOKENS,
    LOAD_SAMPLES,
    MAX_TOKENS,
    PASSES,
    QUICK_MODELS,
    REGRESSION,
    REPEATS,
    RUNS,
    baseline_cells,
)

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "site/static/benchmarks"
SCHEMAS = {"nuvec-parity-record": "parity-record-v1.schema.json"}


def validate(paths: list[Path]) -> int:
    errors = 0
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"✗ {path}: {error}", file=sys.stderr)
            errors += 1
            continue
        kind = data.get("schema") if isinstance(data, dict) else None
        if kind == "nuvec-parity-index":
            for release in data["releases"]:
                for rec in release["records"]:
                    if not (path.parent / rec["path"]).exists():
                        print(f"✗ {path}: {rec['path']} does not exist")
                        errors += 1
            continue
        if kind not in SCHEMAS:
            if path.name.endswith(".schema.json"):
                continue
            print(f"✗ {path}: unknown schema {kind!r}")
            errors += 1
            continue
        schema = json.loads((BENCH / SCHEMAS[kind]).read_text())
        failures = list(jsonschema.Draft202012Validator(schema).iter_errors(data))
        for e in failures:
            print(f"✗ {path}: {'/'.join(map(str, e.absolute_path))}: {e.message[:200]}")
        errors += len(failures)
        if not failures:
            print(f"✓ {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 1 if errors else 0


def cells(record: dict) -> dict[tuple, float]:
    out = {}
    for model in record["models"]:
        for section, rows, metric in (
            ("decode", model.get("decode", []), "tok_s"),
            ("prefill", model.get("ttft", {}).get("legacy", []), "prefill_tok_s"),
        ):
            for row in rows:
                key = model["id"], section, row["repeats"]
                ratio = row[metric]["ratio"]
                if key in out or ratio is None or not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"duplicate or invalid comparison cell {key}")
                out[key] = ratio
    return out


class CandidateRegression(ValueError):
    """Complete measurements crossed the release regression limit."""


def require_quick(record: dict, baseline: str) -> None:
    """A release decision needs each measured cell and its previous-release control."""
    if record["settings"].get("tier") != "quick" or record["settings"].get(
        "regression_baseline", {}).get("ref") != baseline:
        raise ValueError(f"expected quick tier and baseline {baseline}")
    if sorted(m["id"] for m in record["models"]) != sorted(QUICK_MODELS):
        raise ValueError("quick roster is incomplete or duplicated")
    required = {*(f"decode after {r}x" for r in REPEATS), "ttft 2048", "memory load",
                "memory decode", "memory prefill_2048", "warm model_ready_s",
                "warm first_token_s", "concurrency 4"}
    threshold = record["settings"]["regression_baseline"]["threshold"]
    if not REGRESSION <= threshold <= 1:
        raise ValueError(f"invalid quick regression threshold {threshold}; minimum is {REGRESSION}")
    settings = record["settings"]
    passes, runs = settings["passes"], settings["runs_per_pass"]
    if passes < PASSES or runs < RUNS:
        raise ValueError(f"qualification requires at least {PASSES} passes of {RUNS} runs")
    concurrency = settings.get("concurrency", {})
    if (settings.get("max_tokens") != MAX_TOKENS
            or concurrency.get("prompt_tokens") != CONC_PROMPT
            or concurrency.get("max_tokens") != CONC_TOKENS
            or concurrency.get("eos") != "ignored"):
        raise ValueError("quick record uses a different generation workload")
    minimum = {"decode": passes * runs, "ttft": passes * runs, "memory": passes,
               "load": settings["load"]["samples"], "concurrency": 1}
    if minimum["load"] < LOAD_SAMPLES:
        raise ValueError(f"qualification requires at least {LOAD_SAMPLES} warm load samples")
    regressions = []
    for model in record["models"]:
        if model.get("status") != "ok" or not model.get("snapshot"):
            raise ValueError(f"{model['id']}: missing checkpoint or incomplete run")
        if not all(isinstance(r.get("nuvec_matches_solo"), bool) for r in model.get("concurrency", [])):
            raise ValueError(f"{model['id']}: missing concurrent versus solo observation")
        found = set()
        for section, key, cell in baseline_cells(model):
            if key in found:
                raise ValueError(f"duplicate {key}")
            found.add(key)
            for side in ("nuvec", "mlx_lm", "baseline"):
                values = (cell.get(side) or {}).get("samples", [])
                if len(values) < minimum[section] or any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError(f"{model['id']}: missing or invalid {side} {key}")
            ratio = cell.get("vs_baseline")
            higher = section in ("decode", "concurrency")
            best = max if higher else min
            ours = best(cell["nuvec"]["samples"])
            previous = best(cell["baseline"]["samples"])
            measured = ours / previous if higher else previous / ours
            if ratio != round(measured, 4):
                raise ValueError(f"{model['id']}: inconsistent or missing ratio for {key}")
            if measured < threshold:
                regressions.append(f"{model['id']} {key} ({measured:.4f})")
        if required - found:
            raise ValueError(f"{model['id']}: missing {sorted(required - found)}")

    if regressions:
        raise CandidateRegression("below the previous-release gate: " + ", ".join(regressions))


def compare(base: Path, candidate: Path, max_drop: float) -> int:
    try:
        b, c = cells(json.loads(base.read_text())), cells(json.loads(candidate.read_text()))
    except (OSError, TypeError, KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    if not b or set(b) != set(c):
        print("records have missing, disjoint or invalid comparison cells", file=sys.stderr)
        return 1
    print("| model | metric | repeats | base | candidate | change |")
    print("|---|---|---:|---:|---:|---:|")
    dropped = []
    for key in sorted(set(b) & set(c)):
        change = c[key] / b[key] - 1
        flag = " ✗" if change < -max_drop else ""
        if flag:
            dropped.append(key)
        print(f"| {key[0].split('/')[-1]} | {key[1]} | {key[2]} | {b[key]:.2f}× | "
              f"{c[key]:.2f}× | {change:+.1%}{flag} |")
    if dropped:
        print(f"\n✗ {len(dropped)} cells fell by more than {max_drop:.0%}")
        return 1
    print(f"\n✓ no cell fell by more than {max_drop:.0%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CANDIDATE"))
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--require-quick", metavar="BASELINE", help="require a complete quick release gate")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare, args.max_drop)
    paths = args.paths or sorted(BENCH.rglob("*.json"))
    if validate(paths):
        return 1
    if args.require_quick:
        if not args.paths:
            ap.error("--require-quick needs a record path")
        for path in paths:
            try:
                require_quick(json.loads(path.read_text()), args.require_quick)
            except (ValueError, KeyError, TypeError) as error:
                print(f"{path}: {error}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
