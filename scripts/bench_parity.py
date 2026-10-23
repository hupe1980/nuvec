# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2", "jsonschema==4.23.0"]
# ///
"""Parity record: nuvec-lm vs mlx-lm over a roster of models, on the same machine and day.

Models come from the local Hugging Face cache (nothing is downloaded; missing models are
recorded as skipped). Both engines get the same token ids and are timed the same way, from
outside the generation loop. Every measurement runs in a fresh child process; passes
alternate between the engines and the record keeps medians and samples.

Sections (`--sections`, default all):

- `decode`: greedy generation after prompts of about 25, 960 and 3,840 tokens (128 tokens,
  end-of-sequence ignored); also the legacy prefill rows (`ttft.legacy`).
- `ttft`: time to the first token after exactly 2,048 and 32,768 prompt tokens (plus the
  per-model extras of the roster).
- `memory`: peak memory per phase (load, decode, prefill at each length): nuvec's
  allocator high-water plus mapped files against mlx-lm's `get_peak_memory`, and the
  process's peak resident size for both.
- `load`: seconds to a loaded model and to its first token, warm (after a priming run) and
  cold (after `sudo -n purge`, when available without a password).
- `concurrency`: aggregate tokens per second of 1, 4 and 8 requests served together
  (nuvec `Batcher` vs mlx-lm `BatchGenerator`), and whether nuvec's batched tokens equal
  each request generated alone.

Ratios are oriented so that above 1.00 favours nuvec (speeds nuvec / mlx-lm, times and
memory mlx-lm / nuvec). Speed and time cells are marked against the gate (0.98) and the
goal (1.0).

Usage: `just bench-parity [models…] [--sections decode,ttft] [--out PATH] [--release]
[--resume] [--budget SECONDS] [--nuvec-ref REF] [--no-cold] [--no-max-context]
[--prompt-tokens 2048,32768] [--concurrency 1,4,8]`; `--mrun-canary-report DIR` converts
a canary run directory into its result file.

Each child's output is cached under `target/bench-parity/<model>/jobs/`; `--resume` reuses
it, and `--budget` stops starting children after that many seconds (exit 3), so a long run
can be continued.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "site/static/benchmarks"
WORK = ROOT / "target/bench-parity"
# (id, kind, extra TTFT prompt lengths)
ROSTER = [
    ("mlx-community/Qwen3-0.6B-4bit", "dense", []),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "dense", []),
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "dense", []),
    ("mlx-community/Qwen3-8B-4bit", "dense", []),
    ("mlx-community/Qwen3.5-9B-4bit", "hybrid", []),
    ("mlx-community/Qwen3.8-27B-4bit", "hybrid", []),
    ("mlx-community/Qwen3-30B-A3B-4bit", "moe", []),
    ("mlx-community/Qwen3.6-35B-A3B-4bit", "hybrid-moe", [32768, 65536, 261952]),
]
MAX_CONTEXT = 261952
TEXT = ("The lighthouse keeper climbed the spiral stairs every evening, counting the steps "
        "and listening to the sea against the rocks below. ")
REPEATS = [1, 40, 160]  # about 25, 960 and 3,840 prompt tokens
RUNS = 2  # per pass
PASSES = 2
MAX_TOKENS = 128
GATE, GOAL = 0.98, 1.0
# A decode cell needs at least this many tokens over at least this many seconds.
MIN_DECODE_TOKENS, MIN_DECODE_S = 8, 0.010
CONC_PROMPT, CONC_TOKENS = 512, 128
LOAD_SAMPLES = 3
SECTIONS = ["decode", "ttft", "memory", "load", "concurrency"]
GIB = 1 << 30


class OutOfBudget(Exception):
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────────────

def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def snapshot(model: str) -> Path | None:
    """The cached snapshot directory: the one `refs/main` names, else the newest."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    repo = hub / f"models--{model.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if ref.exists():
        path = repo / "snapshots" / ref.read_text().strip()
        return path if (path / "config.json").exists() else None
    found = sorted((p.parent for p in (repo / "snapshots").glob("*/config.json")),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def slug(model: str) -> str:
    return model.split("/")[-1]


def max_positions(snap: Path) -> int | None:
    config = json.loads((snap / "config.json").read_text())
    for c in (config, config.get("text_config") or {}):
        if "max_position_embeddings" in c:
            return int(c["max_position_embeddings"])
    return None


def weight_bytes(snap: Path) -> int:
    return sum(p.stat().st_size for p in snap.glob("*.safetensors"))


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def workspace_version() -> str:
    text = (ROOT / "Cargo.toml").read_text()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def git(*args: str) -> str:
    return sh("git", "-C", str(ROOT), *args)


# ── Environment ────────────────────────────────────────────────────────────────────────

def environment(nuvec_ref: str | None) -> dict:
    from importlib.metadata import version

    swap = sh("sysctl", "-n", "vm.swapusage")
    m = re.search(r"used = ([\d.]+)M", swap)
    gpu = re.search(r"Total Number of Cores: (\d+)", sh("system_profiler", "SPDisplaysDataType"))
    batt = sh("pmset", "-g", "batt")
    low_power = re.search(r"lowpowermode\s+(\d)", sh("pmset", "-g"))
    if nuvec_ref:
        commit, dirty = git("rev-parse", nuvec_ref), False
        ver = re.search(r'^version = "([^"]+)"', git("show", f"{nuvec_ref}:Cargo.toml"),
                        re.M).group(1)
    else:
        commit, dirty, ver = git("rev-parse", "HEAD"), bool(git("status", "--porcelain")), \
            workspace_version()
    return {
        "machine": sh("sysctl", "-n", "hw.model"),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_cores": {"performance": int(sh("sysctl", "-n", "hw.perflevel0.physicalcpu") or 0),
                      "efficiency": int(sh("sysctl", "-n", "hw.perflevel1.physicalcpu") or 0)},
        "gpu_cores": int(gpu.group(1)) if gpu else None,
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize")),
        "macos": {"version": sh("sw_vers", "-productVersion"),
                  "build": sh("sw_vers", "-buildVersion")},
        "power": {"source": "ac" if "AC Power" in batt else
                  "battery" if "Battery Power" in batt else "unknown",
                  "low_power_mode": bool(low_power and low_power.group(1) == "1")},
        "swap_used_bytes_at_start": int(float(m.group(1)) * (1 << 20)) if m else 0,
        "nuvec": {"version": ver, "commit": commit, "dirty": dirty},
        "rust": sh("rustc", "-V"),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "python": platform.python_version(),
    }


def machine_slug(env: dict) -> str:
    chip = re.sub(r"[^a-z0-9]+", "-", env["chip"].lower().removeprefix("apple ")).strip("-")
    return f"{chip}-{env['memory_bytes'] // GIB}gb"


# ── The mlx-lm worker (runs in a child process) ──────────────────────────────────────────

def read_ids(path: str) -> list[list[int]]:
    return [[int(t) for t in line.split(",")] for line in Path(path).read_text().splitlines()
            if line.strip()]


def mlx_worker(mode: str, snap: str, rest: list[str]) -> None:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.utils import load_tokenizer

    if mode == "ids":
        out = Path(rest[0])
        out.mkdir(parents=True, exist_ok=True)
        tokenizer = load_tokenizer(Path(snap))
        limit = max_positions(Path(snap))
        lengths = [int(n) for n in rest[1].split(",")]
        base = tokenizer.encode(TEXT * 64)
        reps = max(lengths) // len(base) + 2

        def write(name: str, prompts: list[list[int]]) -> None:
            tmp = out / f".{name}"
            tmp.write_text("".join(",".join(map(str, p)) + "\n" for p in prompts))
            tmp.replace(out / name)

        if max(lengths) > len(base):
            long = tokenizer.encode(TEXT * 64 * reps)
        else:
            long = base
        for n in lengths:
            if limit is None or n <= limit:
                write(f"ttft-{n}.txt", [long[:n]])
        write("first-token.txt", [base[:16]])
        write("concurrency.txt",
              [tokenizer.encode(f"{i + 1}. " + TEXT * 8)[:CONC_PROMPT] for i in range(8)])
        print(json.dumps({"limit": limit}))
        return

    load_start = time.perf_counter()
    model, tokenizer = load(snap)
    model_ready = time.perf_counter() - load_start
    tokenizer.eos_token_ids = []  # generate to max_tokens

    def run(prompt: list[int], max_tokens: int) -> tuple[float, float, int]:
        start = time.perf_counter()
        first = last = None
        count = 0
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            now = time.perf_counter()
            first = first or now
            last = now
            count += 1
        return first - start, last - first, count

    def warm_up() -> None:
        run(tokenizer.encode(TEXT), MAX_TOKENS)

    if mode == "load":
        prompt = read_ids(rest[0])[0]
        first, _, _ = run(prompt, 1)
        print(json.dumps({"phase": "load", "model_ready_s": model_ready,
                          "first_token_s": model_ready + first}), flush=True)
    elif mode.startswith("memory-"):
        if mode != "memory-load":
            warm_up()
            mx.reset_peak_memory()
            run(read_ids(rest[0])[0], MAX_TOKENS if mode == "memory-decode" else 1)
        print(json.dumps({"phase": mode, "peak_bytes": mx.get_peak_memory()}), flush=True)
    elif mode == "decode":
        repeats = [int(r) for r in rest[0].split(",")]
        runs = int(rest[1])
        warm_up()
        for r in repeats:
            ids = tokenizer.encode(TEXT * r)
            for _ in range(runs):
                first, decode_s, tokens = run(ids, MAX_TOKENS)
                print(json.dumps({"repeats": r, "prompt_tokens": len(ids),
                                  "prefill": len(ids) / first, "ttft_s": first,
                                  "decode": (max(tokens, 2) - 1) / decode_s if decode_s > 0
                                  else 0.0, "decode_s": decode_s, "tokens": tokens}),
                      flush=True)
    elif mode == "ttft":
        files, runs = rest[0].split(","), int(rest[1])
        warm_up()
        for path in files:
            ids = read_ids(path)[0]
            for _ in range(runs):
                first, _, _ = run(ids, 1)
                print(json.dumps({"ids_file": Path(path).name, "prompt_tokens": len(ids),
                                  "ttft_s": first, "prefill": len(ids) / first}), flush=True)
    elif mode == "concurrency":
        from mlx_lm.generate import BatchGenerator

        n = int(rest[1])
        prompts = read_ids(rest[0])[:n]
        warm_up()
        gen = BatchGenerator(model, stop_tokens=[], completion_batch_size=n,
                             prefill_batch_size=n, max_tokens=CONC_TOKENS)
        start = time.perf_counter()
        uids = gen.insert(prompts, [CONC_TOKENS] * n)
        counts = {u: 0 for u in uids}
        decode_start = None
        finished = 0
        while finished < n:
            for r in gen.next_generated():
                counts[r.uid] += 1
                if decode_start is None and all(counts.values()):
                    decode_start = time.perf_counter()
                if r.finish_reason is not None:
                    finished += 1
        wall = time.perf_counter() - start
        decode_wall = time.perf_counter() - (decode_start or start)
        gen.close()
        total = sum(counts.values())
        print(json.dumps({"level": n, "total_tokens": total, "wall_s": wall,
                          "decode_wall_s": decode_wall, "decode_tokens": total - n}),
              flush=True)
    else:
        raise SystemExit(f"unknown worker mode {mode}")


# ── Children and the job cache ───────────────────────────────────────────────────────────

class Runner:
    def __init__(self, args, bench: Path, ref_tag: str):
        self.args = args
        self.bench = bench
        self.ref_tag = ref_tag  # "" for the working tree, "@<sha>" for --nuvec-ref
        self.deadline = time.monotonic() + args.budget if args.budget else None

    def child(self, cmd: list[str]) -> tuple[list[dict], int, str | None]:
        """Runs one child; returns its JSON lines, its peak resident bytes (`ru_maxrss`,
        bytes on macOS) and an error."""
        with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True,
                                    env={**os.environ, "HF_HUB_OFFLINE": "1"})
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(), err.read()
        if proc.returncode:
            tail = [l for l in stderr.strip().splitlines() if l.strip()][-1:] or ["no output"]
            return lines(stdout), usage.ru_maxrss, f"exit {proc.returncode}: {tail[0][:300]}"
        return lines(stdout), usage.ru_maxrss, None

    def job(self, model: str, key: str, engine: str, cmd: list[str]) -> dict:
        """A cached child run: {"lines", "maxrss", "error"}."""
        tag = self.ref_tag if engine == "nuvec" else ""
        path = WORK / slug(model) / "jobs" / f"{engine}{tag}-{key}.json"
        if self.args.resume and path.exists():
            return json.loads(path.read_text())
        if self.deadline and time.monotonic() > self.deadline:
            raise OutOfBudget
        if not self.args.quiet:
            print(f"  {slug(model)}: {engine}{tag} {key}", file=sys.stderr, flush=True)
        if engine == "nuvec":
            out, maxrss, error = self.child([str(self.bench), *cmd])
        else:
            out, maxrss, error = self.child([sys.executable, __file__, "--mlx-worker", *cmd])
        result = {"lines": out, "maxrss": maxrss, "error": error}
        write_atomic(path, result)
        return result


# ── Record pieces ───────────────────────────────────────────────────────────────────────

def metric(samples: list[float], unit: str) -> dict | None:
    if not samples:
        return None
    return {"median": statistics.median(samples), "samples": samples, "unit": unit}


def pair(ours: list[float] | None, theirs: list[float] | None, unit: str,
         higher_is_better: bool, gated: bool = True, reason: str | None = None) -> dict:
    n, m = metric(ours or [], unit), metric(theirs or [], unit)
    p: dict = {"nuvec": n, "mlx_lm": m, "ratio": None}
    if n is None or m is None:
        p["reason"] = reason or ("nuvec failed" if n is None else "mlx-lm failed")
        return p
    if n["median"] <= 0 or m["median"] <= 0:
        p.update(nuvec=None, mlx_lm=None, reason="non-positive median")
        return p
    ratio = n["median"] / m["median"] if higher_is_better else m["median"] / n["median"]
    p["ratio"] = round(ratio, 4)
    if gated:
        p["gate_pass"] = ratio >= GATE
        p["goal_pass"] = ratio >= GOAL
    return p


def decode_check(row: dict) -> str | None:
    """Why a decode row is not a valid measurement, or None."""
    if row.get("tokens", 0) < MIN_DECODE_TOKENS:
        return f"generation ended after {row.get('tokens', 0)} tokens"
    if row.get("decode_s", 1.0) < MIN_DECODE_S:
        return f"decode took {row.get('decode_s')} s"
    return None


def side(rows: list[dict], key: str, check=None) -> tuple[list[float] | None, str | None]:
    if not rows:
        return None, None
    for row in rows:
        why = check(row) if check else None
        if why:
            return None, why
    return [row[key] for row in rows], None


def run_model(r: Runner, model: str, kind: str, extras: list[int], sections: list[str],
              prompt_tokens: list[int], levels: list[int]) -> dict:
    snap = snapshot(model)
    entry: dict = {"id": model, "kind": kind}
    if snap is None:
        entry.update(status="skipped", reason="not in the local Hugging Face cache")
        return entry
    entry["snapshot"] = snap.name
    s = str(snap)
    problems: list[str] = []
    limit = max_positions(snap)
    lengths = sorted(set(prompt_tokens + extras))
    if r.args.no_max_context:
        lengths = [n for n in lengths if n < MAX_CONTEXT]
    prompts = WORK / slug(model) / "prompts"
    ids = r.job(model, "ids-" + "-".join(map(str, lengths)), "mlx",
                ["ids", s, str(prompts), ",".join(map(str, lengths))])
    if ids["error"]:
        entry.update(status="failed", reason=f"prompt ids: {ids['error']}")
        return entry
    passes = range(r.args.passes)
    text = ["--model", s, "--text", TEXT]

    def both(key: str, nuvec_cmd: list[str], mlx_cmd: list[str], p: int):
        a = r.job(model, f"{key}-p{p}", "nuvec", nuvec_cmd)
        b = r.job(model, f"{key}-p{p}", "mlx", mlx_cmd)
        for e, res in (("nuvec", a), ("mlx-lm", b)):
            if res["error"]:
                problems.append(f"{key} {e}: {res['error']}")
        return a, b

    if "decode" in sections:
        ours, theirs = [], []
        for p in passes:
            a, b = both("decode",
                        [*text, "--repeats", ",".join(map(str, REPEATS)), "--runs",
                         str(r.args.runs), "--max-tokens", str(MAX_TOKENS), "--ignore-eos"],
                        ["decode", s, ",".join(map(str, REPEATS)), str(r.args.runs)], p)
            ours += a["lines"]
            theirs += b["lines"]
        decode, legacy = [], []
        for rep in REPEATS:
            o = [x for x in ours if x.get("repeats") == rep]
            t = [x for x in theirs if x.get("repeats") == rep]
            on, owhy = side(o, "decode", decode_check)
            tn, twhy = side(t, "decode", decode_check)
            why = "; ".join(f"{e}: {w}" for e, w in (("nuvec", owhy), ("mlx-lm", twhy)) if w)
            tokens = {k: v[0]["prompt_tokens"] for k, v in (("nuvec", o), ("mlx_lm", t)) if v}
            decode.append({"repeats": rep, "prompt_tokens": tokens,
                           "tok_s": pair(on, tn, "tok/s", True, reason=why or None)})
            legacy.append({"repeats": rep, "prompt_tokens": tokens,
                           "prefill_tok_s": pair(side(o, "prefill")[0], side(t, "prefill")[0],
                                                 "tok/s", True)})
        entry["decode"] = decode
        entry.setdefault("ttft", {})["legacy"] = legacy

    if "ttft" in sections:
        cells = []
        for n in lengths:
            f = prompts / f"ttft-{n}.txt"
            if not f.exists():
                cells.append({"prompt_tokens": n,
                              "ttft_s": pair(None, None, "s", False, reason="context limit"),
                              "prefill_tok_s": pair(None, None, "tok/s", True,
                                                    reason="context limit")})
                continue
            if n >= 65536 and weight_bytes(snap) + 2 * GIB > os.sysconf("SC_PHYS_PAGES") * \
                    os.sysconf("SC_PAGE_SIZE"):
                reason = "not enough memory"
                cells.append({"prompt_tokens": n, "ttft_s": pair(None, None, "s", False,
                              reason=reason), "prefill_tok_s": pair(None, None, "tok/s",
                              True, reason=reason)})
                continue
            runs = 1 if n >= 65536 else r.args.runs
            o, t, errors = [], [], []
            # One sample per engine at the longest context (about 20 minutes each).
            for p in range(1) if n >= MAX_CONTEXT else passes:
                a, b = both(f"ttft-{n}", [*text, "--ids-file", str(f), "--runs", str(runs)],
                            ["ttft", s, str(f), str(runs)], p)
                o += a["lines"]
                t += b["lines"]
                errors += [f"{e}: {x['error']}" for e, x in (("nuvec", a), ("mlx-lm", b))
                           if x["error"]]
            why = "; ".join(errors)[:300] or None
            cells.append({"prompt_tokens": n,
                          "ttft_s": pair(side(o, "ttft_s")[0], side(t, "ttft_s")[0], "s",
                                         False, reason=why),
                          "prefill_tok_s": pair(side(o, "prefill")[0], side(t, "prefill")[0],
                                                "tok/s", True, reason=why)})
        entry.setdefault("ttft", {})["lengths"] = cells

    if "memory" in sections:
        mem = {}
        phases = [("load", "memory-load", None), ("decode", "memory-decode",
                                                  prompts / "first-token.txt")]
        phases += [(f"prefill_{n}", f"memory-prefill:{n}", prompts / f"ttft-{n}.txt")
                   for n in lengths if n <= 65536 and (prompts / f"ttft-{n}.txt").exists()]
        for name, phase, f in phases:
            extra = ["--ids-file", str(f)] if f else []
            o, t = [], []
            for p in passes:
                a, b = both(f"mem-{name}", [*text, "--phase", phase, *extra],
                            [phase.split(":")[0], s, *([str(f)] if f else [])], p)
                o.append(a)
                t.append(b)
            ol = [x["lines"][0] for x in o if x["lines"] and not x["error"]]
            tl = [x["lines"][0] for x in t if x["lines"] and not x["error"]]
            cell = {
                "peak_bytes": pair([x["peak_with_mapped_bytes"] for x in ol],
                                   [x["peak_bytes"] for x in tl], "bytes", False, False),
                "os_peak_rss_bytes": pair([x["maxrss"] for x in o if not x["error"]],
                                          [x["maxrss"] for x in t if not x["error"]],
                                          "bytes", False, False),
            }
            if ol:
                last = ol[-1]
                cell["nuvec_raw"] = {"peak_bytes": last["peak_bytes"],
                                     "active_bytes": last["active_bytes"],
                                     "mapped_bytes": last["mapped_bytes"]}
            mem[name] = cell
        entry["memory"] = mem

    if "load" in sections:
        first = prompts / "first-token.txt"
        res = {}
        # Priming run (pages in the files), then warm samples.
        r.job(model, "load-prime", "nuvec", [*text, "--phase", "load", "--first-token-ids",
                                             str(first)])
        r.job(model, "load-prime", "mlx", ["load", s, str(first)])
        modes = [("warm", None)]
        if not r.args.no_cold:
            modes.append(("cold", ["sudo", "-n", "/usr/sbin/purge"]))
        for mode, prep in modes:
            o, t = [], []
            purge_failed = False
            for i in range(LOAD_SAMPLES):
                for engine, cmd in (("nuvec", [*text, "--phase", "load", "--first-token-ids",
                                               str(first)]),
                                    ("mlx", ["load", s, str(first)])):
                    if prep and subprocess.run(prep, capture_output=True).returncode != 0:
                        purge_failed = True
                        break
                    res_ = r.job(model, f"load-{mode}-{i}", engine, cmd)
                    if res_["error"]:
                        problems.append(f"load {engine}: {res_['error']}")
                    (o if engine == "nuvec" else t).extend(res_["lines"])
                if purge_failed:
                    break
            if purge_failed:
                why = "purge unavailable (needs passwordless sudo)"
                res[mode] = {"model_ready_s": pair(None, None, "s", False, reason=why),
                             "first_token_s": pair(None, None, "s", False, reason=why)}
            else:
                res[mode] = {"model_ready_s": pair(side(o, "model_ready_s")[0],
                                                   side(t, "model_ready_s")[0], "s", False),
                             "first_token_s": pair(side(o, "first_token_s")[0],
                                                   side(t, "first_token_s")[0], "s", False)}
        if r.args.no_cold:
            why = "not run (--no-cold)"
            res["cold"] = {"model_ready_s": pair(None, None, "s", False, reason=why),
                           "first_token_s": pair(None, None, "s", False, reason=why)}
        entry["load"] = res

    if "concurrency" in sections:
        f = prompts / "concurrency.txt"
        rows = []
        for n in levels:
            a = r.job(model, f"conc-{n}", "nuvec", [*text, "--batch", str(n), "--ids-file",
                                                    str(f), "--max-tokens", str(CONC_TOKENS)])
            b = r.job(model, f"conc-{n}", "mlx", ["concurrency", s, str(f), str(n)])
            for e, res_ in (("nuvec", a), ("mlx-lm", b)):
                if res_["error"]:
                    problems.append(f"concurrency {n} {e}: {res_['error']}")
            o = a["lines"][:1] if not a["error"] else []
            t = b["lines"][:1] if not b["error"] else []
            rows.append({
                "level": n,
                "total_tokens": n * CONC_TOKENS,
                "wall_s": pair(side(o, "wall_s")[0], side(t, "wall_s")[0], "s", False),
                "aggregate_tok_s": pair([x["total_tokens"] / x["wall_s"] for x in o],
                                        [x["total_tokens"] / x["wall_s"] for x in t],
                                        "tok/s", True),
                "decode_aggregate_tok_s": pair(
                    [x["decode_tokens"] / x["decode_wall_s"] for x in o],
                    [x["decode_tokens"] / x["decode_wall_s"] for x in t], "tok/s", True),
                "nuvec_matches_solo": o[0]["matches_solo"] if o else None,
            })
        entry["concurrency"] = rows

    if problems:
        entry["status"] = "partial"
        entry["reason"] = "; ".join(problems)[:2000]
    else:
        entry["status"] = "ok"
    return entry


def pairs_of(entry: dict):
    """(metric key, Pair) for every gated or summarised cell of a model."""
    for row in entry.get("decode", []):
        yield "decode", row["tok_s"]
    for row in entry.get("ttft", {}).get("legacy", []):
        if row["repeats"] != 1:  # the ≈25-token row is not a prefill measurement
            yield "prefill_legacy", row["prefill_tok_s"]
    for row in entry.get("ttft", {}).get("lengths", []):
        yield f"ttft_{row['prompt_tokens']}", row["ttft_s"]
    for name, cell in entry.get("memory", {}).items():
        yield "memory", cell["peak_bytes"]
    for mode, cell in entry.get("load", {}).items():
        yield f"load_{mode}", cell["first_token_s"]
    for row in entry.get("concurrency", []):
        yield f"concurrency_{row['level']}", row["aggregate_tok_s"]


def geomean(values: list[float]) -> dict | None:
    return {"value": round(statistics.geometric_mean(values), 4), "n": len(values)} \
        if values else None


def summary(models: list[dict]) -> dict:
    groups: dict[str, list[float]] = {}
    below_gate: dict[str, list] = {}
    below_goal: dict[str, list] = {}
    flags = []
    for entry in models:
        if entry["status"] in ("skipped", "failed"):
            continue
        for key, p in pairs_of(entry):
            if p["ratio"] is None:
                continue
            groups.setdefault(key, []).append(p["ratio"])
            if "gate_pass" in p:
                if not p["gate_pass"]:
                    below_gate.setdefault(key, []).append({"model": entry["id"],
                                                           "ratio": p["ratio"]})
                if not p["goal_pass"]:
                    below_goal.setdefault(key, []).append({"model": entry["id"],
                                                           "ratio": p["ratio"]})
        for row in entry.get("concurrency", []):
            if row["nuvec_matches_solo"] is False:
                flags.append({"model": entry["id"], "level": row["level"],
                              "check": "matches_solo"})
    out: dict = {"below_gate": below_gate, "below_goal": below_goal, "correctness_flags": flags}
    for name, keys in (("decode_geomean", ["decode"]), ("memory_geomean", ["memory"]),
                       ("load_warm_geomean", ["load_warm"]),
                       ("load_cold_geomean", ["load_cold"])):
        g = geomean([v for k in keys for v in groups.get(k, [])])
        if g:
            out[name] = g
    ttft = {k.removeprefix("ttft_"): geomean(v) for k, v in groups.items()
            if k.startswith("ttft_")}
    if "prefill_legacy" in groups:
        ttft["legacy_prefill"] = geomean(groups["prefill_legacy"])
    if ttft:
        out["ttft_geomean"] = ttft
    conc = {k.removeprefix("concurrency_"): geomean(v) for k, v in groups.items()
            if k.startswith("concurrency_")}
    if conc:
        out["concurrency_geomean"] = conc
    return out


def validate(record: dict, schema_name: str) -> None:
    import jsonschema

    schema = json.loads((BENCH_DIR / schema_name).read_text())
    jsonschema.validate(record, schema)


# ── Markdown ──────────────────────────────────────────────────────────────────────────

def fmt_ratio(p: dict) -> str:
    if p["ratio"] is None:
        return "—"
    mark = " ⚠" if p.get("gate_pass") is False else ""
    return f"{p['ratio']:.2f}×{mark}"


def med(p: dict, k: str, digits: int = 1) -> str:
    return f"{p[k]['median']:.{digits}f}" if p.get(k) else "—"


def markdown(record: dict) -> str:
    env = record["environment"]
    out = [f"{env['chip']}, {env['memory_bytes'] // GIB} GB, macOS {env['macos']['version']}; "
           f"nuvec {env['nuvec']['version']}, mlx {env['mlx']}, mlx-lm {env['mlx_lm']}; "
           f"{record['created'][:10]}. Ratios above 1.00 favour nuvec; ⚠ marks a cell below "
           f"{GATE}×.", ""]
    rows = [m for m in record["models"] if "decode" in m]
    if rows:
        out += ["| model | kind | prompt | prefill nuvec / mlx-lm (tok/s) | ratio "
                "| decode nuvec / mlx-lm (tok/s) | ratio |", "|---|---|---:|---:|---:|---:|---:|"]
        for m in rows:
            for d, lg in zip(m["decode"], m["ttft"]["legacy"]):
                t = d["prompt_tokens"]
                tokens = str(t.get("nuvec", t.get("mlx_lm", "?")))
                p = lg["prefill_tok_s"]
                out.append(f"| {slug(m['id'])} | {m['kind']} | {tokens} | {med(p, 'nuvec')} / "
                           f"{med(p, 'mlx_lm')} | {fmt_ratio(p)} | {med(d['tok_s'], 'nuvec')} / "
                           f"{med(d['tok_s'], 'mlx_lm')} | {fmt_ratio(d['tok_s'])} |")
        out.append("")
    rows = [m for m in record["models"] if m.get("ttft", {}).get("lengths")]
    if rows:
        out += ["| model | prompt | TTFT nuvec / mlx-lm (s) | ratio |", "|---|---:|---:|---:|"]
        for m in rows:
            for c in m["ttft"]["lengths"]:
                p = c["ttft_s"]
                out.append(f"| {slug(m['id'])} | {c['prompt_tokens']} | {med(p, 'nuvec', 2)} / "
                           f"{med(p, 'mlx_lm', 2)} | {fmt_ratio(p)} |")
        out.append("")
    rows = [m for m in record["models"] if m.get("memory")]
    if rows:
        out += ["| model | phase | peak nuvec / mlx-lm (GB) | ratio |", "|---|---|---:|---:|"]
        for m in rows:
            for name, c in m["memory"].items():
                p = c["peak_bytes"]
                g = lambda k: f"{p[k]['median'] / 1e9:.2f}" if p.get(k) else "—"  # noqa: E731
                out.append(f"| {slug(m['id'])} | {name} | {g('nuvec')} / {g('mlx_lm')} "
                           f"| {fmt_ratio(p)} |")
        out.append("")
    rows = [m for m in record["models"] if m.get("load")]
    if rows:
        out += ["| model | load | first token nuvec / mlx-lm (s) | ratio |",
                "|---|---|---:|---:|"]
        for m in rows:
            for mode, c in m["load"].items():
                p = c["first_token_s"]
                out.append(f"| {slug(m['id'])} | {mode} | {med(p, 'nuvec', 2)} / "
                           f"{med(p, 'mlx_lm', 2)} | {fmt_ratio(p)} |")
        out.append("")
    rows = [m for m in record["models"] if m.get("concurrency")]
    if rows:
        out += ["| model | requests | aggregate nuvec / mlx-lm (tok/s) | ratio | same tokens |",
                "|---|---:|---:|---:|---|"]
        for m in rows:
            for c in m["concurrency"]:
                p = c["aggregate_tok_s"]
                out.append(f"| {slug(m['id'])} | {c['level']} | {med(p, 'nuvec')} / "
                           f"{med(p, 'mlx_lm')} | {fmt_ratio(p)} | {c['nuvec_matches_solo']} |")
        out.append("")
    s = record["summary"]
    parts = [f"{k}: {v['value']:.2f}× (n={v['n']})" for k, v in s.items()
             if isinstance(v, dict) and "value" in v]
    for k in ("ttft_geomean", "concurrency_geomean"):
        parts += [f"{k} {n}: {v['value']:.2f}×" for n, v in s.get(k, {}).items() if v]
    out.append("Geometric means: " + "; ".join(parts))
    for m in record["models"]:
        if m["status"] != "ok":
            out.append(f"- {m['id']}: {m['status']} ({m.get('reason', '')[:300]})")
    return "\n".join(out)


# ── Release index and the mrun canary ─────────────────────────────────────────────────

def add_to_index(version: str, machine: str, path: str, kind: str) -> None:
    index_path = BENCH_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else \
        {"schema": "nuvec-parity-index", "releases": []}
    release = next((r for r in index["releases"] if r["version"] == version), None)
    if release is None:
        release = {"version": version, "records": []}
        index["releases"].insert(0, release)
    release["records"] = [r for r in release["records"] if r["path"] != path]
    release["records"].append({"machine": machine, "path": path, "kind": kind})
    write_atomic(index_path, index)


def canary_gate(alias: str, cell: str, phase: str) -> tuple[float, float]:
    if alias == "qwen3.6" and phase == "prefill" and cell in ("8K", "32K", "max"):
        return 0.97, {"8K": 0.98, "32K": 0.98, "max": 1.0}[cell]
    return GATE, GOAL


def canary_cells(result: dict) -> dict[tuple[str, str], float]:
    """{(cell, phase): median tok/s} from an mrun `result.json` (synthetic suite)."""
    out = {}
    for c in result.get("cells", []):
        name = c.get("cell", "").lower()
        size = re.search(r"\b(1k|8k|32k|max)\b", name)
        phase = re.search(r"\b(prefill|decode)\b", name)
        if size and phase and c.get("unit") == "tok/s" and c.get("median"):
            out[(size.group(1).upper().replace("MAX", "max"), phase.group(1))] = c["median"]
    return out


def canary_report(run_dir: Path) -> Path:
    env = environment(None)
    meta = json.loads((run_dir / "canary.json").read_text())
    record: dict = {
        "schema": "nuvec-mrun-canary", "schema_version": 1,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": env,
        "mrun": {"commit": meta["mrun_commit"], "ref_file": "ci/mrun.ref"},
        "nuvec": env["nuvec"],
        "status": meta["status"],
    }
    if meta["status"] != "ok":
        record["reason"] = meta.get("reason") or meta["status"]
    else:
        cells: list[dict] = []
        for alias in ("qwen3", "qwen3.6"):
            sides = [json.loads((run_dir / f"{b}-{alias}" / "result.json").read_text())
                     for b in ("mlx", "nuvec")]
            medians = [canary_cells(x) for x in sides]
            for key, mlx_tok_s in medians[0].items():
                if key not in medians[1]:
                    continue
                gate, goal = canary_gate(alias, *key)
                ratio = medians[1][key] / mlx_tok_s
                cells.append({"alias": alias, "cell": key[0], "phase": key[1],
                              "mlx_tok_s": mlx_tok_s, "nuvec_tok_s": medians[1][key],
                              "ratio": round(ratio, 4), "gate": gate, "goal": goal,
                              "gate_pass": ratio >= gate, "goal_pass": ratio >= goal})
        record["cells"] = cells
        mem = meta.get("memory_64k")
        record["memory_64k"] = mem
        if mem is None:
            record["memory_64k_reason"] = meta.get("memory_64k_reason", "not measured")
    validate(record, "mrun-canary-v1.schema.json")
    out = BENCH_DIR / env["nuvec"]["version"] / f"mrun-canary-{machine_slug(env)}.json"
    write_atomic(out, record)
    add_to_index(env["nuvec"]["version"], machine_slug(env),
                 str(out.relative_to(BENCH_DIR)), "mrun-canary")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────────────

def build(ref: str | None, skip: bool = False) -> tuple[Path, str]:
    if not ref and skip:
        return ROOT / "target/release/examples/bench", ""
    if not ref:
        subprocess.run(["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example",
                        "bench"], cwd=ROOT, check=True)
        return ROOT / "target/release/examples/bench", ""
    sha = git("rev-parse", ref)
    tree = WORK / f"ref-{sha[:12]}"
    binary = WORK / f"bench-{sha[:12]}"
    if not binary.exists():
        if tree.exists():
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)])
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(tree), sha],
                       check=True, capture_output=True)
        try:
            subprocess.run(["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example",
                            "bench", "--target-dir", str(WORK / "ref-target")], cwd=tree,
                           check=True)
            shutil.copy2(WORK / "ref-target/release/examples/bench", binary)
        finally:
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)])
    return binary, f"@{sha[:12]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--prompt-tokens", default="2048,32768")
    ap.add_argument("--no-max-context", action="store_true")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--out", type=Path, default=WORK / "record.json")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="with --release: accept an uncommitted tree (recorded as dirty)")
    ap.add_argument("--no-cold", action="store_true")
    ap.add_argument("--nuvec-ref")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=float, default=0)
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="use the built bench example")
    ap.add_argument("--mrun-canary-report", type=Path)
    args = ap.parse_args()

    if args.mrun_canary_report:
        print(canary_report(args.mrun_canary_report))
        return 0
    sections = [s for s in args.sections.split(",") if s]
    if unknown := set(sections) - set(SECTIONS):
        ap.error(f"unknown sections {sorted(unknown)}")
    if args.nuvec_ref:
        sections = [s for s in sections if s == "decode"]
    if args.release and not args.allow_dirty and git("status", "--porcelain"):
        ap.error("--release needs a clean working tree (or --allow-dirty)")
    bench, tag = build(args.nuvec_ref, args.no_build)
    runner = Runner(args, bench, tag)
    env = environment(args.nuvec_ref)
    roster = {m: (k, e) for m, k, e in ROSTER}
    models = args.models or [m for m, _, _ in ROSTER]
    prompt_tokens = [int(n) for n in args.prompt_tokens.split(",") if n]
    levels = [int(n) for n in args.concurrency.split(",") if n]
    results = []
    try:
        for model in models:
            kind, extras = roster.get(model, ("dense", []))
            results.append(run_model(runner, model, kind, extras, sections, prompt_tokens,
                                     levels))
    except OutOfBudget:
        print(f"budget spent after {len(results)} of {len(models)} models; run again with "
              "--resume", file=sys.stderr)
        return 3
    settings = {
        "passes": args.passes, "runs_per_pass": args.runs, "max_tokens": MAX_TOKENS,
        "prompt_tokens": prompt_tokens, "legacy_repeats": REPEATS,
        "extra_prompt_tokens": {m: e for m, _, e in ROSTER if e and m in models},
        "concurrency": {"levels": levels, "prompt_tokens": CONC_PROMPT,
                        "max_tokens": CONC_TOKENS, "eos": "ignored"},
        "load": {"samples": LOAD_SAMPLES, "first_token_prompt_tokens": 16,
                 "cold_method": None if args.no_cold else "sudo -n purge"},
        "memory": {"method": "peak_with_mapped-vs-get_peak_memory+ru_maxrss",
                   "note": "nuvec counts mapped files whole; both engines copy unaligned "
                           "(mlx-community) checkpoints once"},
        "gates": {"gate": GATE, "goal": GOAL},
    }
    if args.nuvec_ref:
        settings["baseline"] = {"ref": args.nuvec_ref, "commit": env["nuvec"]["commit"]}
    record = {
        "schema": "nuvec-parity-record", "schema_version": 1,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": env, "settings": settings, "sections": sections,
        "models": results, "summary": summary(results),
    }
    validate(record, "parity-record-v1.schema.json")
    out = args.out
    if args.release:
        if (unexpected := [line for line in git("status", "--porcelain").splitlines()
                           if "site/static/benchmarks/" not in line]) and not args.allow_dirty:
            print(f"unexpected changes during the run: {unexpected}", file=sys.stderr)
            return 1
        out = BENCH_DIR / env["nuvec"]["version"] / f"{machine_slug(env)}.json"
    write_atomic(out, record)
    if args.release:
        add_to_index(env["nuvec"]["version"], machine_slug(env),
                     str(out.relative_to(BENCH_DIR)), "parity")
    print(markdown(record))
    print(f"\nrecord: {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--mlx-worker":
        mlx_worker(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(main())
