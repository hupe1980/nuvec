# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema==4.23.0"]
# ///
"""Check quick-tier benchmark coverage, ratios, and comparison failures."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from bench_parity import BENCH_DIR, QUICK_MODELS, REPEATS, markdown, summary
from validate_parity import CandidateRegression, compare, require_quick, validate


def quick():
    metric = {"samples": [1.0] * 4}
    pair = {"nuvec": metric, "mlx_lm": metric, "baseline": metric, "vs_baseline": 1.0}
    sample = {
        "status": "ok",
        "snapshot": "revision",
        "decode": [{"repeats": r, "tok_s": pair} for r in REPEATS],
        "ttft": {"lengths": [{"prompt_tokens": 2048, "ttft_s": pair}]},
        "memory": {key: {"peak_bytes": pair} for key in ("load", "decode", "prefill_2048")},
        "load": {"warm": {k: pair for k in ("first_token_s", "model_ready_s")}},
        "concurrency": [{"level": 4, "nuvec_matches_solo": True, "aggregate_tok_s": pair}],
    }
    return {
        "settings": {
            "tier": "quick",
            "passes": 2,
            "runs_per_pass": 2,
            "max_tokens": 128,
            "concurrency": {"prompt_tokens": 512, "max_tokens": 128, "eos": "ignored"},
            "load": {"samples": 3},
            "regression_baseline": {"ref": "v0.15.0", "threshold": 0.97},
        },
        "models": [dict(copy.deepcopy(sample), id=name) for name in QUICK_MODELS],
    }


class QualificationTests(unittest.TestCase):
    def test_invalid_record_input(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()) as messages:
            path = Path(tmp) / "record.json"
            self.assertEqual(validate([path]), 1)
            path.write_text("{")
            self.assertEqual(validate([path]), 1)
            self.assertIn(str(path), messages.getvalue())

    def test_render_quick_summary_with_warm_loads(self):
        record = json.loads((BENCH_DIR / "0.8.0/m4-pro-48gb-quick.json").read_text())
        record["summary"] = summary(record["models"])
        self.assertIn("load_warm_geomean", record["summary"])
        self.assertIn("first token nuvec / mlx-lm", markdown(record))

    def test_quick_gate(self):
        require_quick(quick(), "v0.15.0")
        for change in (
            lambda r: r["models"].pop(),
            lambda r: r["settings"].update(runs_per_pass=3),
            lambda r: r["settings"].update(passes=1),
            lambda r: r["settings"]["load"].update(samples=1),
            lambda r: r["settings"]["concurrency"].update(prompt_tokens=128),
            lambda r: r["models"].append(r["models"][0]),
            lambda r: r["models"][0]["decode"].pop(),
            lambda r: r["models"][0]["decode"].append(r["models"][0]["decode"][0]),
            lambda r: r["models"][0]["load"]["warm"].pop("first_token_s"),
            lambda r: r["models"][0]["decode"][0]["tok_s"].pop("baseline"),
            lambda r: r["models"][0]["memory"]["load"]["peak_bytes"].update(vs_baseline=0.5),
            lambda r: r["settings"]["regression_baseline"].update(threshold=0.5),
            lambda r: r["models"][0]["decode"][0]["tok_s"].update(vs_baseline=1.1),
            lambda r: r["models"][0]["concurrency"][0].update(nuvec_matches_solo=None),
        ):
            record = quick()
            change(record)
            with self.assertRaises(ValueError):
                require_quick(record, "v0.15.0")

    def test_gate_recomputes_direction_from_samples(self):
        for section, key, baseline in (
            ("decode", "tok_s", 2.0),
            ("concurrency", "aggregate_tok_s", 2.0),
        ):
            record = quick()
            cell = record["models"][0][section][0][key]
            record["models"][0][section][0][key] = {
                **cell,
                "baseline": {"samples": [baseline] * 4},
                "vs_baseline": 0.5,
            }
            with self.assertRaises(CandidateRegression):
                require_quick(record, "v0.15.0")
        record = quick()
        warm = record["models"][0]["load"]["warm"]
        warm["first_token_s"] = {
            **warm["first_token_s"],
            "nuvec": {"samples": [2.0] * 3},
            "vs_baseline": 0.5,
        }
        with self.assertRaises(CandidateRegression):
            require_quick(record, "v0.15.0")

    def test_disjoint_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            p, q = Path(d) / "a", Path(d) / "b"
            for name, path in (("a", p), ("b", q)):
                path.write_text(
                    json.dumps(
                        {
                            "models": [
                                {"id": name, "decode": [{"repeats": 1, "tok_s": {"ratio": 1}}]}
                            ]
                        }
                    )
                )
            with redirect_stderr(io.StringIO()) as messages:
                self.assertEqual(compare(p, q, 0.02), 1)
            self.assertIn("disjoint", messages.getvalue())


if __name__ == "__main__":
    unittest.main()
