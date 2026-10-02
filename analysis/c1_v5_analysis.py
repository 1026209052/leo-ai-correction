"""C1 (v5) -- analysis of the corrected independent-source run.

Input : figures/data/spacex_v5_verification.csv  (all geometry-valid windows, with
        passes_peak / has_maneuver flags)

Reports, for four nested subsets:
  A. all valid windows        (the "no-harm / premise check" population)
  B. peak >= 20 km            (the manuscript's previous large-maneuver filter)
  C. maneuver detected        (the physically meaningful maneuver population)
  D. both                     (peak >= 20 km AND maneuver detected)
"""
import csv
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
B = 10000
SEED = 20260919


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


rows = load("spacex_v5_verification.csv")
print(f"rows = {len(rows)}")
print("columns:", ", ".join(rows[0].keys()))


def istrue(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def f(v, k):
    try:
        return float(v[k])
    except (KeyError, ValueError, TypeError):
        return np.nan


sets = {
    "A. 全部有效窗口 (未筛选)": rows,
    "B. peak>=20 km": [r for r in rows if istrue(r.get("passes_peak", "False"))],
    "C. 机动检测为真": [r for r in rows if istrue(r.get("has_maneuver", "False"))],
    "D. peak>=20 且含机动": [r for r in rows if istrue(r.get("passes_peak", "False"))
                             and istrue(r.get("has_maneuver", "False"))],
}


def pctl(x, q):
    x = np.sort(np.asarray(x, float))
    if len(x) == 0:
        return np.nan
    k = q * (len(x) - 1)
    lo, hi = int(k), min(int(k) + 1, len(x) - 1)
    return float(x[lo] + (x[hi] - x[lo]) * (k - lo))


def cvar10(x):
    x = np.sort(np.asarray(x, float))
    k = max(1, int(round(0.10 * len(x))))
    return float(x[:k].mean())


def deg_med(x):
    n = [v for v in x if v < 0]
    return float(np.median(n)) if n else np.nan


def boot(x, fn, B=B, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        v = fn([x[rng.randrange(len(x))] for _ in range(len(x))])
        if v == v:
            out.append(v)
    out.sort()
    return float(out[int(0.025 * len(out))]), float(out[int(0.975 * len(out))])


print("\n" + "=" * 118)
print("subset sizes")
print("=" * 118)
for k, v in sets.items():
    print(f"  {k:>26}: n = {len(v)}")

print("\n" + "=" * 118)
print("same-unit error comparison (mean RMSE, km) and correction magnitude")
print("=" * 118)
print(f"{'subset':>26} | {'n':>4} | {'SGP4':>8} | {'trans+soft':>11} | {'lstm+soft':>10} | "
      f"{'corr mean(T/L)':>16}")
for k, v in sets.items():
    if not v:
        print(f"{k:>26} | {0:>4} |        - |           - |          - |                -")
        continue
    s = np.array([f(r, "sgp4_rmse_alt") for r in v], float)
    t = np.array([f(r, "ai_rmse_alt") for r in v], float)
    l = np.array([f(r, "lstm_ai_rmse_alt") for r in v], float)
    ct = np.nanmedian([f(r, "trans_residual_mean") for r in v])
    cl = np.nanmedian([f(r, "lstm_residual_mean") for r in v])
    print(f"{k:>26} | {len(v):>4} | {np.nanmean(s):8.2f} | {np.nanmean(t):11.2f} | "
          f"{np.nanmean(l):10.2f} | {ct:7.2f} / {cl:6.2f}")

print("\n" + "=" * 118)
print("eta_i distribution  (positive = AI better than SGP4)")
print("=" * 118)
hdr = ["n", "superior%", "median", "mean", "P5", "CVaR10", "worst",
       "CI median", "CI CVaR10", "CI worst"]
out_rows = []
for k, v in sets.items():
    for label, col in (("Transformer+软掩码", "eta_trans_soft"), ("LSTM+软掩码", "eta_lstm_soft")):
        e = np.array([f(r, col) for r in v], float)
        e = e[~np.isnan(e)]
        if len(e) == 0:
            continue
        sup = 100.0 * np.mean(e > 0)
        ci_m, ci_c, ci_w = boot(e, lambda z: float(np.median(z))), boot(e, cvar10), \
            boot(e, lambda z: float(np.min(z)))
        print(f"\n  [{k}]  {label}")
        print(f"     n={len(e)}  优于SGP4={sup:.1f}%  中位={np.median(e):+.2f}%  "
              f"均值={np.mean(e):+.2f}%  P5={pctl(e,0.05):+.2f}%  "
              f"CVaR10={cvar10(e):+.2f}%  最差={np.min(e):+.2f}%")
        print(f"     95%CI: 中位[{ci_m[0]:+.2f},{ci_m[1]:+.2f}]  "
              f"CVaR10[{ci_c[0]:+.2f},{ci_c[1]:+.2f}]  最差[{ci_w[0]:+.2f},{ci_w[1]:+.2f}]")
        print("     P(η<-5/-10/-20): " +
              "  ".join(f"{x}%:{100*np.mean(e < -x):.2f}" for x in (5, 10, 20)))
        out_rows.append(dict(subset=k, method=label, n=len(e), superior=round(sup, 2),
                             median=round(float(np.median(e)), 3),
                             mean=round(float(np.mean(e)), 3),
                             p5=round(pctl(e, 0.05), 3), cvar10=round(cvar10(e), 3),
                             worst=round(float(np.min(e)), 3),
                             ci_median=f"[{ci_m[0]:.2f},{ci_m[1]:.2f}]",
                             ci_cvar10=f"[{ci_c[0]:.2f},{ci_c[1]:.2f}]",
                             ci_worst=f"[{ci_w[0]:.2f},{ci_w[1]:.2f}]"))

print("\n" + "=" * 118)
print("p_S -- non-maneuver windows only, UNFILTERED (the valid estimate)")
print("=" * 118)
nm = [r for r in rows if not istrue(r.get("has_maneuver", "False"))]
s = np.array([f(r, "sgp4_rmse_alt") for r in nm], float)
print(f"  n = {len(nm)}")
for thr in (5, 10, 15, 20):
    k = int((s > thr).sum())
    lo, hi = boot(s, lambda z, t=thr: 100.0 * np.mean(np.asarray(z) > t))
    print(f"  P(SGP4 RMSE > {thr:>2} km | 非机动) = {100*k/len(nm):5.2f}%  ({k}/{len(nm)})  "
          f"bootstrap 95%CI [{lo:.2f}%, {hi:.2f}%]")

dst = os.path.join(OUT, "c1_v5_tails.csv")
with open(dst, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nwrote {dst}")

print("\n参考（主实验 TLE 验证集 n=817）: Transformer+软掩码 75.0% / 中位 +14.43 / "
      "CVaR10 -21.81 / 最差 -57.37")
