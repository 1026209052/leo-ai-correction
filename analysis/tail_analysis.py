"""Priority items 2/3/4/6/7 - analyses computable from the data already on disk.

Input : figures/fig2_cdf.csv  (817 samples x {eta_mse, eta_soft})
Output: analysis/out/*.csv + console report
"""
import csv
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "figures", "fig2_cdf.csv")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(RAW, encoding="utf-8-sig")))
ETA = {"mse": [float(r["eta_mse"]) for r in rows],
       "soft": [float(r["eta_soft"]) for r in rows]}
N = len(ETA["mse"])
EV = 12.80  # maneuver events per satellite-year


def pct(xs, q):
    s = sorted(xs)
    k = q * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def deg_median(xs):
    neg = [x for x in xs if x < 0]
    return st.median(neg)


print("=" * 78)
print("A. ACHIEVED FLOOR  (item 7: what the safety shell actually delivers)")
print("=" * 78)
print(f"{'x(%)':>6} | {'P(eta<-x) MSE':>14} | {'P(eta<-x) Soft':>15} | {'ratio':>7}")
hdr = []
for x in (0, 5, 10, 20, 30, 50, 100, 160):
    a = sum(1 for v in ETA["mse"] if v < -x) / N
    b = sum(1 for v in ETA["soft"] if v < -x) / N
    r = (b / a) if a else float("nan")
    hdr.append((x, a, b, r))
    print(f"{x:>6} | {a*100:13.2f}% | {b*100:14.2f}% | {r:7.2f}")
print("\ntail quantiles of eta (%)  [negative = degradation]:")
print(f"{'method':>6} | {'P1':>8} | {'P5':>8} | {'P10':>8} | {'P25':>8} | "
      f"{'deg med':>8} | {'min':>9}")
for k in ("mse", "soft"):
    xs = ETA[k]
    print(f"{k:>6} | {pct(xs,0.01):8.2f} | {pct(xs,0.05):8.2f} | {pct(xs,0.10):8.2f} | "
          f"{pct(xs,0.25):8.2f} | {deg_median(xs):8.2f} | {min(xs):9.2f}")

# ---- leave-out sensitivity of the worst case (no satellite ids needed) ----
print("\nworst-case sensitivity to removing the k most extreme samples:")
print(f"{'k':>3} | {'min(MSE)':>9} | {'min(Soft)':>10}")
for k in range(0, 6):
    a = sorted(ETA["mse"])[k:]
    b = sorted(ETA["soft"])[k:]
    print(f"{k:>3} | {min(a):9.2f} | {min(b):10.2f}")

print("\n" + "=" * 78)
print("B. BOOTSTRAP CI  (item 3: i.i.d. = LOWER BOUND on the true clustered CI)")
print("=" * 78)
rng = random.Random(20260919)


def boot(xs, ys, stat, B=10000):
    d = []
    n = len(xs)
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(stat([ys[i] for i in idx]) - stat([xs[i] for i in idx]))
    d.sort()
    return d[int(0.025 * B)], d[int(0.975 * B)]


mse, soft = ETA["mse"], ETA["soft"]
stats = {
    "superior-to-SGP4 (pp)": lambda v: 100 * sum(1 for x in v if x > 0) / len(v),
    "degradation median (pp)": deg_median,
    "P5 of eta / P95 degradation (pp)": lambda v: pct(v, 0.05),
    "P10 of eta / P90 degradation (pp)": lambda v: pct(v, 0.10),
    "worst single sample (pp)": min,
}
print(f"{'statistic':>34} | {'diff':>9} | {'i.i.d. 95% CI':>22} | p~0?")
boot_rows = []
for name, f in stats.items():
    d = f(soft) - f(mse)
    lo, hi = boot(mse, soft, f)
    print(f"{name:>34} | {d:9.2f} | [{lo:8.2f}, {hi:8.2f}] | {'yes' if (lo>0 or hi<0) else 'NO'}")
    boot_rows.append((name, d, lo, hi))
with open(os.path.join(OUT, "bootstrap_ci.csv"), "w", newline="", encoding="ascii") as fh:
    w = csv.writer(fh)
    w.writerow(["statistic", "diff", "ci_lo", "ci_hi", "excludes_zero"])
    for n_, d, lo, hi in boot_rows:
        w.writerow([n_, f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}", int(lo > 0 or hi < 0)])
print("note: 817 samples come from 42 satellites -> the true CI is WIDER;")
print("      the manuscript's clustered bootstrap for the degradation median gave")
print("      +19.8 pp [6.8, 114.3] (p=0.007) vs the i.i.d. value printed above.")

print("\n" + "=" * 78)
print("C. R(tau) WITH THE GATE-INDEPENDENT p_S TERM  (item 4)")
print("=" * 78)
p_G1, p_nonman, p_S, beta = 0.0235, 0.4181, 0.0275, 1.5
X = p_nonman * p_S
T1 = p_G1 + X - p_G1 * X + (beta - 1) * min(p_G1, X)
c_miss = (T1 - p_G1) * EV          # p_S-driven, tau-independent
c_g1 = p_G1 * EV                   # proxy missed-trigger part
print(f"T1 (fault tree, beta=1.5, p_S=2.75%) = {T1*EV:.4f} events/sat-yr")
print(f"   proxy missed-trigger part  p_G1  = {c_g1:.4f}  ({100*c_g1/(T1*EV):.1f}%)")
print(f"   gate-INDEPENDENT p_S part  c     = {c_miss:.4f}  ({100*c_miss/(T1*EV):.1f}%)")
tab7 = [(15, 0.626, 0.0660, 0.0), (20, 0.582, 0.0477, 0.0),
        (30, 0.373, 0.0115, 0.0351), (50, 0.159, 0.0, 0.316)]
print(f"\n{'tau':>4} | {'r(tau)*EV':>9} | {'N_miss(obs)':>11} | {'N_miss(incl c)':>14} | "
      f"{'N_harm':>7} | {'R as published':>14} | {'R incl. c':>11}")
R_pub, R_new = {}, {}
for tau, pi, ph, r in tab7:
    nm_obs, nh = r * EV, pi * ph * EV
    nm = nm_obs + c_miss
    Rp, Rn = 1e4 * nm_obs + nh, 1e4 * nm + nh
    R_pub[tau], R_new[tau] = Rp, Rn
    print(f"{tau:>4} | {nm_obs:9.3f} | {nm_obs:11.3f} | {nm:14.3f} | {nh:7.3f} | "
          f"{Rp:14.2f} | {Rn:11.2f}")
print(f"\nargmin as published : tau = {min(R_pub, key=R_pub.get)} km")
print(f"argmin incl. term c : tau = {min(R_new, key=R_new.get)} km   <-- ranking preserved? "
      f"{min(R_pub, key=R_pub.get) == min(R_new, key=R_new.get)}")
print(f"published claim 'tau=20 wastes 49% vs tau=15' -> with c included the gap is "
      f"{100*(R_new[15]-R_new[20])/R_new[20]:.3f}%")

print("\n" + "=" * 78)
print("D. FMEA COMPONENT TRANSFERABILITY ACROSS tau  (item 2)")
print("=" * 78)
pH20 = 0.0477
print(f"{'tau':>4} | {'P(H|I) measured':>16} | {'predicted from tau=20':>21} | {'measured N_harm':>15} | {'ratio':>6}")
cross = []
for tau, pi, ph, r in tab7:
    pred = pi * pH20 * EV
    meas = pi * ph * EV
    ratio = (pred / meas) if meas else float("inf")
    cross.append((tau, ph, pred, meas, ratio))
    print(f"{tau:>4} | {ph*100:15.2f}% | {pred:21.3f} | {meas:15.3f} | {ratio:6.2f}")
print("-> the conditional harm rate falls with tau (6.60/4.77/1.15/0.00%), so the")
print("   tau=20-calibrated component does NOT extrapolate; must be re-measured per tau.")
with open(os.path.join(OUT, "fmea_cross_tau.csv"), "w", newline="", encoding="ascii") as fh:
    w = csv.writer(fh)
    w.writerow(["tau_km", "P_H_given_I_measured", "N_harm_pred_from_tau20", "N_harm_measured", "ratio"])
    for t, ph, p, m, r in cross:
        w.writerow([t, f"{ph:.4f}", f"{p:.4f}", f"{m:.4f}", f"{r:.2f}"])

print("\n" + "=" * 78)
print("E. TABLE 6 EXTRA COLUMNS - candidate definitions  (item 6)")
print("=" * 78)
print(f"{'definition':>42} | {'MSE':>9} | {'Soft':>9}")
variants = {
    "median of positive eta (improvement median)": lambda v: st.median([x for x in v if x > 0]),
    "mean of positive eta": lambda v: st.mean([x for x in v if x > 0]),
    "mean of worst 10% of ALL samples": lambda v: st.mean(sorted(v)[:max(1, round(0.10*len(v)))]),
    "mean of worst 10% of DEGRADED samples": lambda v: st.mean(
        sorted([x for x in v if x < 0])[:max(1, round(0.10*len([x for x in v if x < 0])))]),
    "mean of worst 5% of ALL samples": lambda v: st.mean(sorted(v)[:max(1, round(0.05*len(v)))]),
    "median of worst 10% of ALL samples": lambda v: st.median(sorted(v)[:max(1, round(0.10*len(v)))]),
    "P10 of eta": lambda v: pct(v, 0.10),
    "mean of eta below deg-median": lambda v: st.mean([x for x in v if x < deg_median(v)]),
}
for name, f in variants.items():
    print(f"{name:>42} | {f(ETA['mse']):9.2f} | {f(ETA['soft']):9.2f}")
print("\npaper (Table 6, alpha=1.0 / 0.3):  improvement median 26.2 / 15.1 ;")
print("                                   worst-10% mean degradation -68.2 / -38.0")

with open(os.path.join(OUT, "achieved_floor.csv"), "w", newline="", encoding="ascii") as fh:
    w = csv.writer(fh)
    w.writerow(["x_pct", "P_eta_lt_-x_mse", "P_eta_lt_-x_soft"])
    for x, a, b, _ in hdr:
        w.writerow([x, f"{a:.5f}", f"{b:.5f}"])
print(f"\nwrote: {OUT}")
