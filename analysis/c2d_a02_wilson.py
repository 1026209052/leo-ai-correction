# -*- coding: utf-8 -*-
"""α=0.2 配置下的 Wilson 区间 与 尾部比例分位（纯 CSV，本机可跑）。"""
import csv
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "data")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)


rows = list(csv.DictReader(open(D + r"\fig2_cdf.csv", encoding="utf-8-sig")))
eta = [(float(r["eta_mse"]), float(r["eta_soft"])) for r in rows]
n = len(eta)


def frac(vals, thr):
    return sum(1 for v in vals if v < thr) / len(vals) * 100


print(f"n = {n}（大机动组）")
print("\n=== Wilson 95% CI（优于SGP4比例；按样本独立假设）===")
for name, idx, p in (("Transformer", 0, 65.48), ("Transformer+软掩码", 1, 82.74),
                     ("LSTM", None, 80.05), ("LSTM+软掩码", None, 85.80)):
    if idx is not None:
        k = sum(1 for a, b in eta if (a if idx == 0 else b) > 0)
    else:
        k = round(p / 100 * n)
    lo, hi = wilson(k, n)
    print(f"  {name:20s} {k}/{n} = {k/n*100:5.2f}%  →  [{lo:.1f}%, {hi:.1f}%]")

print("\n=== 尾部比例（L105 用）===")
for thr in (10, 20, 30, 50, 100):
    m = frac([a for a, _ in eta], -thr)
    s = frac([b for _, b in eta], -thr)
    print(f"  P(η<-{thr:>3}%): 标准MSE {m:5.2f}%  →  软掩码 {s:5.2f}%")
