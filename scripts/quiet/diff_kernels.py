"""Per-kernel difference of two mst_kernels.py CSVs: (long - short) / steps, per step.

usage: python3 -I diff_kernels.py LONG.csv SHORT.csv STEPS [--top 40]
LONG: e.g. 129 generated tokens, SHORT: 1 token, same prompt; STEPS = 128.
"""
import csv, sys

def load(p):
    return {r["key"]: (int(r["count"]), int(r["union_ns"])) for r in csv.DictReader(open(p))}

long, short, steps = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
rows = []
for k in long.keys() | short.keys():
    (n1, u1), (n0, u0) = long.get(k, (0, 0)), short.get(k, (0, 0))
    rows.append((k, (n1 - n0) / steps, (u1 - u0) / steps / 1e3))
rows.sort(key=lambda r: -r[2])
print(f"{'us/step':>9} {'n/step':>7}  key  (total {sum(r[2] for r in rows):.1f} us/step)")
for k, n, u in rows[:top]:
    print(f"{u:9.2f} {n:7.2f}  {k}")
