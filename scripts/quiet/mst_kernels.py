"""Per-kernel GPU time from an `xctrace export --xpath` table (Metal System Trace).

usage: python3 -I mst_kernels.py TABLE.xml [--by name|pso-label|label|channel-name|event-label]
       [--process SUBSTR] [--from-ns T0] [--to-ns T1] [--top 40] [--csv OUT]
       [--family] [--segments GAP_MS]
--family groups names into kernel classes (FAMILIES) so nuvec and MLX names line up.
--segments prints GPU-busy segments split by idle gaps >= GAP_MS (prefill chunks: both
engines evaluate the cache after each chunk) as start/end ns for --from-ns/--to-ns.

TABLE.xml is one exported table, e.g. schema metal-shader-profiler-intervals (per shader,
sampled) or metal-gpu-intervals (per encoder / command buffer). Rows are resolved through
xctrace's id/ref dedup. Prints count, summed duration, union (wall) time and mean per key.
Diff two runs (e.g. 129 vs 1 generated tokens) with --csv and diff_kernels.py.
"""
import argparse, csv, sys
import re
import xml.etree.ElementTree as ET

FAMILIES = [  # first match wins; check unmatched names with --top and extend
    ("attention", r"sdpa|attention|paged"), ("gather-q", r"gather.*q(mv|mm)|q(mv|mm).*gather"),
    ("gather-mm", r"gather_mm|gather"), ("quant-mv", r"qmv|qvm|qgemv"), ("quant-mm", r"qmm"),
    ("gemv", r"gemv"), ("gemm", r"gemm|steel|matmul"), ("softmax", r"softmax"),
    ("norm", r"rms|layer_norm|norm"), ("rope", r"rope"), ("gated-delta", r"gated_delta|gdn"),
    ("sort/arg", r"sort|partition|arg_?reduce|argmax|topk"), ("conv", r"conv"),
    ("copy", r"copy|concat|slice|scatter|update|pad"), ("select", r"where|ternary|select"),
    ("reduce", r"reduce|sum|max|min|logsumexp"), ("fused", r"compiled|fused|nuvec_f_"),
    ("elementwise", r"binary|unary|add|mul|sub|div|exp|sigmoid|silu|tanh|cast|astype"),
]

def family(name):
    for fam, rx in FAMILIES:
        if re.search(rx, name, re.I):
            return fam
    return "other:" + name

def rows(path):
    ids, cols = {}, []
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == "mnemonic":
            cols.append(el.text)
        elif el.tag == "row":
            out = {}
            for col, c in zip(cols, list(el)):
                ref = c.get("ref")
                out[col] = ids.get(ref, (None, None)) if ref else (c.text, c.get("fmt"))
            yield out
            el.clear()
        if el.get("id") is not None:
            ids[el.get("id")] = (el.text, el.get("fmt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("--by", default="name")
    ap.add_argument("--process"); ap.add_argument("--from-ns", type=int, default=0)
    ap.add_argument("--to-ns", type=int, default=1 << 62)
    ap.add_argument("--top", type=int, default=40); ap.add_argument("--csv")
    ap.add_argument("--family", action="store_true")
    ap.add_argument("--segments", type=float)
    a = ap.parse_args()
    agg = {}
    for r in rows(a.table):
        if a.process and a.process not in (r.get("process", (None, ""))[1] or ""):
            continue
        try:
            t0, d = int(r["start"][0]), int(r["duration"][0])
        except (KeyError, TypeError, ValueError):
            continue
        if not (a.from_ns <= t0 < a.to_ns):
            continue
        key = (r.get(a.by) or (None, "?"))[1] or (r.get(a.by) or ("?",))[0] or "?"
        if a.family:
            key = family(key)
        if a.segments is not None:
            key = "all"
        agg.setdefault(key, []).append((t0, t0 + d))
    if a.segments is not None:
        iv = sorted(agg.get("all", [])); seg = []
        for s0, e0 in iv:
            if seg and s0 - seg[-1][1] < a.segments * 1e6:
                seg[-1][1] = max(seg[-1][1], e0); seg[-1][2] += 1
            else:
                seg.append([s0, e0, 1])
        for i, (s0, e0, n) in enumerate(seg):
            print(f"segment {i:3d}  --from-ns {s0} --to-ns {e0}  {(e0 - s0) / 1e6:9.2f} ms  {n} intervals")
        return
    table = []
    for k, iv in agg.items():
        iv.sort(); union, end = 0, -1
        for s, e in iv:
            if e > end:
                union += e - max(s, end); end = e
        total = sum(e - s for s, e in iv)
        table.append((k, len(iv), total, union))
    table.sort(key=lambda x: -x[3])
    tu = sum(x[3] for x in table) or 1
    print(f"{'union ms':>10} {'sum ms':>10} {'%':>6} {'count':>7} {'mean us':>8}  key")
    for k, n, s, u in table[: a.top]:
        print(f"{u/1e6:10.3f} {s/1e6:10.3f} {100*u/tu:6.2f} {n:7d} {s/n/1e3:8.2f}  {k}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            csv.writer(f).writerows([("key", "count", "sum_ns", "union_ns"), *table])

if __name__ == "__main__":
    sys.exit(main())
