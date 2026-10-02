"""A1 -- identify how Table 6's two extra columns were actually computed.

Inputs:
  figures/data/alpha_ablation_ensemble_results.csv   (source of Table 6)
  figures/data/fig2_cdf.csv                          (main-experiment per-sample eta, 817)
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


abl = load("alpha_ablation_ensemble_results.csv")
print("=" * 100)
print("1. alpha_ablation_ensemble_results.csv  (11 rows -> source of Table 6)")
print("=" * 100)
cols = ["alpha", "big_impr", "big_better_pct", "better_impr_median",
        "worse_median", "worse_worst10", "worse_worst5", "worse_worst1"]
head = ["alpha", "big_impr", "big_better", "impr_median", "worse_med",
        "worst10", "worst5", "worst1"]
print("".join("%13s" % h for h in head))
for r in abl:
    print("".join("%13s" % (r[c] if c == "alpha" else f"{float(r[c]):.3f}") for c in cols))

eta = {r["sample_idx"]: (float(r["eta_mse"]), float(r["eta_soft"]))
       for r in load("fig2_cdf.csv")}
print(f"\nfig2_cdf.csv (main experiment): {len(eta)} samples")
print("  -> its alpha=1.0 model should equal the ablation's alpha=1.0 row")

a10 = [r for r in abl if float(r["alpha"]) == 1.0][0]
a03 = [r for r in abl if float(r["alpha"]) == 0.3][0]
MSE = [v[0] for v in eta.values()]
SOFT = [v[1] for v in eta.values()]


def pctl(x, q):
    s = sorted(x)
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


print("\n" + "=" * 100)
print("2. compare the four shared metrics (must match if it is the same model)")
print("=" * 100)
print(f"{'metric':>28} | {'ablation csv':>12} | {'from fig2 eta':>13}")
rows = [("RMSE improvement (%)", a10["big_impr"], None),
        ("superior-to-SGP4 (%)", a10["big_better_pct"], 100 * sum(1 for v in MSE if v > 0) / len(MSE)),
        ("degradation median (%)", a10["worse_median"], st.median([v for v in MSE if v < 0])),
        ("worst single sample (%)", a10["worse_worst1"], min(MSE)),
        ("improvement median (%)", a10["better_impr_median"], st.median([v for v in MSE if v > 0])),
        ("worst-10% mean (%)", a10["worse_worst10"], None)]
for n, a, b in rows:
    bs = "-" if b is None else f"{float(b):12.3f}"
    flag = ""
    if b is not None:
        flag = "  OK" if abs(float(a) - float(b)) < 0.15 else "  <<< DIFFERS"
    print(f"{n:>28} | {float(a):12.3f} | {bs}{flag}")

print("\n" + "=" * 100)
print("3. find k such that mean of the k worst samples reproduces the table value")
print("=" * 100)
for tag, xs, target in (("alpha=1.0 (MSE)", MSE, float(a10["worse_worst10"])),
                        ("alpha=0.3 (Soft)", SOFT, float(a03["worse_worst10"]))):
    s = sorted(xs)
    best = min(range(1, 260), key=lambda k: abs(st.mean(s[:k]) - target))
    print(f"{tag:>18}: target {target:8.2f} -> k={best:4d} (q={best/len(s):.3f}, "
          f"{100*best/len(s):5.2f}%)", end="")
    neg = [v for v in xs if v < 0]
    for name, kk in (("worst 10% of degraded", max(1, round(0.10 * len(neg)))),
                     ("worst 20% of all", max(1, round(0.20 * len(s)))),
                     ("worst 5% of all", max(1, round(0.05 * len(s))))):
        print(f"   [{name}={st.mean(sorted(xs)[:kk]):.2f}]", end="")
    print()

print("\nreference: n_degraded  MSE=%d  Soft=%d  (of 817)"
      % (sum(1 for v in MSE if v < 0), sum(1 for v in SOFT if v < 0)))
