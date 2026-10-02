"""C3 -- satellite-level (block) Conformal Risk Control vs the i.i.d. Wilson version.

Inputs : figures/data/pareto_samples.csv   (sample_idx, peak, sgp4_rmse, eta)
         figures/data/sample_metadata.csv  (sample_idx, norad_id)

wilson  : one-sided Wilson upper bound on the conditional harm rate (manuscript).
cluster : 95th percentile of a *satellite* bootstrap of the same rate
          (resample the 42 satellites with replacement and pool their samples).

The coverage experiment splits the 42 satellites 50/50 at random, picks tau* on the
calibration half, then measures the harm rate on the held-out half -- i.e. it tests
the guarantee under the deployment condition (a satellite never seen in calibration).
"""
import csv
import os
import random
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")
Z = 1.6449
SPLITS = 200
BOOT = 500
SEED = 20260919


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


meta = {r["sample_idx"]: int(r["norad_id"]) for r in load("sample_metadata.csv")}
rows = [dict(sat=meta[r["sample_idx"]], peak=float(r["peak"]), eta=float(r["eta"]))
        for r in load("pareto_samples.csv") if r["sample_idx"] in meta]
big = [r for r in rows if r["peak"] >= 20.0]
sats = sorted({r["sat"] for r in big})
code = {s: i for i, s in enumerate(sats)}


def pack(sample):
    """descending-peak order -> prefix arrays (n_i, k_i) per satellite."""
    s = sorted(sample, key=lambda r: -r["peak"])
    peak = np.array([r["peak"] for r in s])
    harm = np.array([1 if r["eta"] < 0 else 0 for r in s])
    g = np.array([code[r["sat"]] for r in s])
    N = np.zeros((len(sats), len(s)), dtype=np.int32)
    K = np.zeros((len(sats), len(s)), dtype=np.int32)
    N[g, np.arange(len(s))] = 1
    K[g, np.arange(len(s))] = harm
    return peak, harm, N.cumsum(1), K.cumsum(1), len(sats)


def wilson(k, n):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    c = p + Z * Z / (2 * n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return np.minimum(1.0, (c + h) / (1 + Z * Z / n))


def cluster_bound(Nc, Kc, ncl, seed=SEED, sat_mask=None):
    """95th percentile of the satellite bootstrap of the pooled harm rate."""
    if sat_mask is not None:
        Nc, Kc, ncl = Nc[sat_mask], Kc[sat_mask], int(sat_mask.sum())
    if ncl == 0:
        return np.ones(Nc.shape[1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ncl, size=(BOOT, ncl))
    Nsum = Nc[idx, :].sum(axis=1)
    Ksum = Kc[idx, :].sum(axis=1)
    rate = np.divide(Ksum, Nsum, out=np.zeros_like(Ksum, float), where=Nsum > 0)
    return np.percentile(rate, 95, axis=0)


def recommend(peak, bound, delta):
    """Smallest tau (ascending) whose bound <= delta."""
    for i in range(len(peak) - 1, -1, -1):
        if bound[i] <= delta:
            return peak[i], i + 1
    return None


peak_all, harm_all, Nc_all, Kc_all, ncl_all = pack(big)
ub_w = wilson(Kc_all.sum(0), Nc_all.sum(0))
ub_c = cluster_bound(Nc_all, Kc_all, ncl_all)

groups = {}
for r in big:
    groups.setdefault(r["sat"], []).append(1.0 if r["eta"] < 0 else 0.0)
g = [v for v in groups.values() if v]
n = sum(len(v) for v in g)
pbar = sum(sum(v) for v in g) / n
mbar = n / len(g)
msb = sum(len(v) * (sum(v) / len(v) - pbar) ** 2 for v in g) / (len(g) - 1)
msw = sum(len(v) * (sum(v) / len(v)) * (1 - sum(v) / len(v)) for v in g) / (n - len(g))
icc = max(0.0, min(1.0, (msb - msw) / (msb + (mbar - 1) * msw)))
de = 1 + (mbar - 1) * icc
print(f"large-maneuver samples={len(big)}  satellites={ncl_all}  harm={int(harm_all.sum())}")
print(f"cluster size mbar={mbar:.1f}  ICC(harm)={icc:.3f}  design effect={de:.2f}  "
      f"-> effective n = {len(big)/de:.0f} (not {len(big)})\n")

print("=" * 100)
print("recommended tau* on the full 817-sample set")
print("=" * 100)
print(f"{'delta':>6} | {'wilson (manuscript)':>38} | {'cluster bootstrap (satellite-level)':>44}")
for delta in (0.05, 0.10, 0.15, 0.20):
    a = recommend(peak_all, ub_w, delta)
    b = recommend(peak_all, ub_c, delta)
    fa = f"tau={a[0]:6.1f} km  int={100*a[1]/len(big):5.1f}%  ub={ub_w[a[1]-1]*100:5.1f}%" if a else "no feasible tau"
    fb = f"tau={b[0]:6.1f} km  int={100*b[1]/len(big):5.1f}%  ub={ub_c[b[1]-1]*100:5.1f}%" if b else "no feasible tau"
    print(f"{delta:6.2f} | {fa:>38} | {fb:>44}")

print("\n" + "=" * 100)
print(f"empirical coverage on UNSEEN satellites ({SPLITS} random 50/50 satellite splits)")
print("=" * 100)
print(f"{'delta':>6} | {'bound':>8} | {'certified':>12} | {'mean tau*':>10} | "
      f"{'coverage|certified':>19} | {'overall':>8}")
rng = random.Random(SEED)
for delta in (0.10, 0.15, 0.20):
    for kind in ("wilson", "cluster"):
        cert = ok = 0
        tl = []
        for k_ in range(SPLITS):
            perm = sats[:]
            rng.shuffle(perm)
            h = len(perm) // 2
            cal_s, test_s = set(perm[:h]), set(perm[h:])
            m = np.array([s in cal_s for s in sats])
            pk = np.array([r["peak"] for r in big if r["sat"] in cal_s])
            if len(pk) == 0:
                continue
            ub = (wilson(Kc_all.sum(0), Nc_all.sum(0)) if kind == "wilson"
                  else cluster_bound(Nc_all, Kc_all, ncl_all, seed=SEED + k_, sat_mask=m))
            rec = recommend(pk, ub, delta)
            if rec is None:
                continue
            cert += 1
            tau = rec[0]
            ti = [r for r in big if r["sat"] in test_s and r["peak"] >= tau]
            rate = (sum(1 for r in ti if r["eta"] < 0) / len(ti)) if ti else 0.0
            ok += 1 if rate <= delta else 0
            tl.append(tau)
        cc = f"{100*ok/cert:17.1f}%" if cert else "n/a"
        print(f"{delta:6.2f} | {kind:>8} | {cert:>4}/{SPLITS:<7} | "
              f"{(st.mean(tl) if tl else float('nan')):10.1f} | {cc:>19} | {100*ok/SPLITS:7.1f}%")
