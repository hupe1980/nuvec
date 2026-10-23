# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema==4.23.0"]
# ///
"""Validates benchmark records, or compares two parity records.

`uv run scripts/validate_parity.py [paths…]`: every record (default: every JSON file under
site/static/benchmarks) against its schema (`schema` field: parity record or mrun canary),
and every path in index.json. Exit 1 on any error.

`uv run scripts/validate_parity.py --compare BASE CANDIDATE [--max-drop 0.02]`: the
nuvec / mlx-lm ratio of every decode and legacy prefill cell present in both; exit 1 if
any fell by more than `max-drop`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "site/static/benchmarks"
SCHEMAS = {"nuvec-parity-record": "parity-record-v1.schema.json",
           "nuvec-mrun-canary": "mrun-canary-v1.schema.json"}


def validate(paths: list[Path]) -> int:
    errors = 0
    for path in paths:
        data = json.loads(path.read_text())
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
    for m in record["models"]:
        for row in m.get("decode", []):
            if row["tok_s"]["ratio"] is not None:
                out[(m["id"], "decode", row["repeats"])] = row["tok_s"]["ratio"]
        for row in m.get("ttft", {}).get("legacy", []):
            if row["prefill_tok_s"]["ratio"] is not None:
                out[(m["id"], "prefill", row["repeats"])] = row["prefill_tok_s"]["ratio"]
    return out


def compare(base: Path, candidate: Path, max_drop: float) -> int:
    b, c = cells(json.loads(base.read_text())), cells(json.loads(candidate.read_text()))
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
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare, args.max_drop)
    paths = args.paths or sorted(BENCH.rglob("*.json"))
    return validate(paths)


if __name__ == "__main__":
    sys.exit(main())
