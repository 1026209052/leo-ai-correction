"""C2 -- cluster (satellite-level) bootstrap for the tail metrics, plus LSTM tails.

Pairs: per-sample eta_mse / eta_soft from fig2_cdf.csv  (817 large-maneuver samples)
       satellite id from sample_metadata.csv (norad_id)
       LSTM eta from per_sample_eta_lstm.csv

Validation target: the clustered CI for the degradation-median difference must
reproduce the values already published in the manuscript
(+19.8 pp [6.8, 114.3] p=0.007 for Transformer; +58.4 pp [26.0, 73.8] p=0.001 for LSTM).
"""
import csv
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")
B = 10000
SEED = 20260919


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


meta = {r["sample_idx"]: r for r in load("sample_metadata.csv")}
fig2 = load("fig2_cdf.csv")
lstm = {r["sample_idx"]: r for r in load("per_sample_eta_lstm.csv")}

idx = [r["sample_idx"] for r in fig2]
missing = [i for i in idx if i not in meta]
print(f"fig2 samples={len(idx)}  matched in metadata={len(idx)-len(missing)}  missing={len(missing)}")
sat = [int(meta[i]["norad_id"]) for i in idx]
nonman = sum(1 for i in idx if meta[i]["is_non_maneuver"].strip().lower() == "true")
print(f"distinct satellites={len(set(sat))}   non-maneuver among them={nonman}")

MSE = [float(r["eta_mse"]) for r in fig2]
SOFT = [float(r["eta_soft"]) for r in fig2]
lidx = [i for i in idx if i in lstm]
LSE = [float(lstm[i]["eta_lstm"]) for i in lidx]
LSO = [float(lstm[i]["eta_lstm_softmask"]) for i in lidx]
lsat = [int(meta[i]["norad_id"]) for i in lidx]
print(f"LSTM rows matched={len(lidx)}  satellites={len(set(lsat))}\n")


def pctl(x, q):
    s = sorted(x)
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def deg_med(x):
    n = [v for v in x if v < 0]
    return st.median(n) if n else float("nan")


def cvar10(x):
    s = sorted(x)
    k = max(1, int(-(-0.10 * len(s) // 1)))  # ceil(0.10 n)
    return st.mean(s[:k])


STATS = {
    "superior-to-SGP4 (pp)": lambda v: 100 * sum(1 for x in v if x > 0) / len(v),
    "degradation median (pp)": deg_med,
    "improvement median (pp)": lambda v: st.median([x for x in v if x > 0]),
    "CVaR10 (worst-10% mean, pp)": cvar10,
    "P5 (pp)": lambda v: pctl(v, 0.05),
    "P10 (pp)": lambda v: pctl(v, 0.10),
    "worst single sample (pp)": min,
}


def cluster_paired(xa, xb, groups):
    """Paired cluster bootstrap: resample satellites, keep the pair together."""
    by_g = {}
    for a, b, g in zip(xa, xb, groups):
        by_g.setdefault(g, []).append((a, b))
    keys = list(by_g)
    rng = random.Random(SEED)
    out = {}
    for name, f in STATS.items():
        d = []
        for _ in range(B):
            A, Bv = [], []
            for _ in range(len(keys)):
                for a, b in by_g[keys[rng.randrange(len(keys))]]:
                    A.append(a)
                    Bv.append(b)
            d.append(f(Bv) - f(A))
        d.sort()
        lo, hi = d[int(0.025 * B)], d[int(0.975 * B)]
        p = 2 * min(sum(1 for v in d if v <= 0), sum(1 for v in d if v >= 0)) / B
        out[name] = (f(xb) - f(xa), lo, hi, p)
    return out


def report(title, res):
    print("=" * 92)
    print(title)
    print("=" * 92)
    print(f"{'statistic':>30} | {'diff':>9} | {'95% CI (cluster)':>24} | {'p':>7}")
    for k, (d, lo, hi, p) in res.items():
        print(f"{k:>30} | {d:9.2f} | [{lo:9.2f}, {hi:9.2f}]      | {p:7.3f}")
    print()


res_t = cluster_paired(MSE, SOFT, sat)
report("Transformer+Soft-Mask  minus  Transformer (clustered, paired)", res_t)
res_l = cluster_paired(LSE, LSO, lsat)
report("LSTM+Soft-Mask  minus  LSTM (clustered, paired)", res_l)

print("=" * 92)
print("published values to compare against")
print("=" * 92)
print("  Transformer: superior +18.2 pp [-1.8, 32.9] p=0.081 | deg median +19.8 pp [6.8, 114.3] p=0.007")
print("  LSTM       : superior  +1.0 pp [-3.5,  7.3] p=0.760 | deg median +58.4 pp [26.0, 73.8] p=0.001")
print()
print("=" * 92)
print("absolute tail levels (soft-mask deployable config, for the paper tables)")
print("=" * 92)
print(f"{'method':>22} | {'n':>4} | {'superior':>9} | {'imp med':>8} | {'deg med':>8} | "
      f"{'CVaR10':>9} | {'P5':>8} | {'worst':>9}")
for nm, x in (("Transformer", MSE), ("Transformer+Soft", SOFT),
              ("LSTM", LSE), ("LSTM+Soft", LSO)):
    print(f"{nm:>22} | {len(x):>4} | "
          f"{100*sum(1 for v in x if v>0)/len(x):8.1f}% | "
          f"{st.median([v for v in x if v>0]):8.2f} | {deg_med(x):8.2f} | "
          f"{cvar10(x):9.2f} | {pctl(x,0.05):8.2f} | {min(x):9.2f}")
