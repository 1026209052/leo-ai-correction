"""C1 -- tail metrics on the independent (SpaceX operational ephemeris) sample.

Input : figures/data/spacex_68p4_verification.csv
        (one row per validated 72 h window: norad_id, sgp4_peak, sgp4_rmse_alt,
         eta_trans_soft, eta_lstm_soft, ai_rmse_alt, lstm_ai_rmse_alt, ...)

Also re-expresses the "residual mean" claim in a like-for-like way
(mean SGP4 RMSE vs mean corrected RMSE) and reports the mean CORRECTION magnitude
separately, since the two are different quantities.
"""
import csv
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
B = 10000
SEED = 20260919


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


rows = load("spacex_68p4_verification.csv")
print(f"rows = {len(rows)}")
print("columns:", list(rows[0].keys()))


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


sats = [r["norad_id"] for r in rows]
dup = len(sats) - len(set(sats))
print(f"distinct norad_id = {len(set(sats))}   duplicates = {dup}")
pk = [num(r, "sgp4_peak") for r in rows]
pk = [v for v in pk if v is not None]
print(f"sgp4_peak: min={min(pk):.1f}  median={st.median(pk):.1f}  max={max(pk):.1f} km")


def pctl(x, q):
    s = sorted(x)
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def deg_med(v):
    n = [x for x in v if x < 0]
    return st.median(n) if n else float("nan")


def cvar10(v):
    s = sorted(v)
    k = max(1, int(round(0.10 * len(s))))
    return st.mean(s[:k])


def boot(v, f, groups=None):
    rng = random.Random(SEED)
    if groups is None:
        n = len(v)
        d = sorted(f([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    else:
        by = {}
        for a, g in zip(v, groups):
            by.setdefault(g, []).append(a)
        keys = list(by)
        d = []
        for _ in range(B):
            pool = []
            for _ in range(len(keys)):
                pool.extend(by[keys[rng.randrange(len(keys))]])
            d.append(f(pool))
        d.sort()
    return d[int(0.025 * B)], d[int(0.975 * B)]


STAT = {
    "superior-to-SGP4 (%)": lambda v: 100 * sum(1 for x in v if x > 0) / len(v),
    "improvement median (%)": lambda v: st.median([x for x in v if x > 0]),
    "degradation median (%)": deg_med,
    "CVaR10 (worst-10% mean, %)": cvar10,
    "worst single sample (%)": min,
}

hdr = ["method", "n"] + list(STAT) + ["ci_degmedian", "ci_cvar10", "ci_worst"]
out_rows = []
print("\n" + "=" * 112)
print("C1 -- tail metrics on the independent sample")
print("=" * 112)
for label, col in (("Transformer+Soft (independent)", "eta_trans_soft"),
                   ("LSTM+Soft (independent)", "eta_lstm_soft")):
    v = [num(r, col) for r in rows]
    v = [x for x in v if x is not None]
    print(f"\n{label}   n={len(v)}")
    vals = {}
    for k, f in STAT.items():
        vals[k] = f(v)
        print(f"  {k:>28} = {vals[k]:9.2f}")
    ci_dm = boot(v, deg_med)
    ci_cv = boot(v, cvar10)
    ci_w = boot(v, min)
    print(f"  {'95% CI degradation median':>28} = [{ci_dm[0]:.2f}, {ci_dm[1]:.2f}]")
    print(f"  {'95% CI CVaR10':>28} = [{ci_cv[0]:.2f}, {ci_cv[1]:.2f}]")
    print(f"  {'95% CI worst sample':>28} = [{ci_w[0]:.2f}, {ci_w[1]:.2f}]")
    for x in (0, 10, 20, 30, 50):
        print(f"  P(eta < -{x:>3}%)            = {100*sum(1 for t in v if t < -x)/len(v):6.2f}%")
    out_rows.append(dict(method=label, n=len(v), **vals,
                         ci_degmedian=f"[{ci_dm[0]:.2f}, {ci_dm[1]:.2f}]",
                         ci_cvar10=f"[{ci_cv[0]:.2f}, {ci_cv[1]:.2f}]",
                         ci_worst=f"[{ci_w[0]:.2f}, {ci_w[1]:.2f}]"))

print("\n" + "=" * 112)
print("like-for-like error comparison (same units: mean RMSE in km)")
print("=" * 112)
for label, base, ai, corr in (
        ("Transformer+Soft", "sgp4_rmse_alt", "ai_rmse_alt", "trans_residual_mean"),
        ("LSTM+Soft", "sgp4_rmse_alt", "lstm_ai_rmse_alt", "lstm_residual_mean")):
    b = [num(r, base) for r in rows]
    b = [x for x in b if x is not None]
    a = [num(r, ai) for r in rows]
    a = [x for x in a if x is not None]
    if not a:
        print(f"{label:>18}: column '{ai}' missing (script reproduced with the un-patched run)")
        continue
    c = [num(r, corr) for r in rows]
    c = [x for x in c if x is not None]
    print(f"{label:>18}: mean SGP4 RMSE = {st.mean(b):7.2f} km  ->  "
          f"mean corrected RMSE = {st.mean(a):7.2f} km  "
          f"({100*(1-st.mean(a)/st.mean(b)):5.1f}% reduction)"
          + (f"   [mean correction {st.mean(c):.2f} km]" if c else ""))

dst = os.path.join(OUT, "c1_independent_tail.csv")
with open(dst, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr)
    w.writeheader()
    w.writerows(out_rows)
print(f"\nwrote {dst}")

print("\n" + "=" * 112)
print("for comparison -- main TLE validation set (n=817)")
print("=" * 112)
print("  Transformer+Soft: 75.0% / +14.43 / -4.27 / CVaR10 -21.81 / worst -57.37")
print("  LSTM+Soft       : 80.3% / +15.80 / -13.62 / CVaR10 -36.28 / worst -213.07")
