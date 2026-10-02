"""C5 -- uncertainty of the maneuver event rate (events / satellite-year).

Inputs : figures/data/maneuver_summary_starlink.csv  (NORADID, data points, maneuvers, ...)
         figures/data/satellite_observation_spans.csv   (norad_id, duration_years)
         figures/data/sample_metadata.csv            (the 879 satellites of the dataset)

Exposure = the calendar span of ALL 879 satellites of the dataset (not only those in
which a maneuver was detected) -- that is the denominator behind the manuscript's
12.80 events/satellite-year.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "figures", "data")
SEED = 20260919
B = 10000


def load(name):
    with open(os.path.join(D, name), newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


man = load("maneuver_summary_starlink.csv")
dur = load("satellite_observation_spans.csv")
meta = load("sample_metadata.csv")

norad_key = list(man[0])[0]                 # 'NORADID'
man_key = list(man[0])[2]                   # 3rd column = maneuver count
events = {int(float(r[norad_key])): float(r[man_key]) for r in man}
expo = {int(float(r["norad_id"])): float(r["duration_years"]) for r in dur}
sats = sorted({int(float(r["norad_id"])) for r in meta})

pair = [(events.get(s, 0.0), expo.get(s, 0.0)) for s in sats]
missing = sum(1 for s in sats if s not in expo)
n_ev = sum(e for e, _ in pair)
n_ex = sum(d for _, d in pair)
n_pos = sum(1 for e, _ in pair if e > 0)
print(f"satellites = {len(sats)}   missing exposure = {missing}")
print(f"satellites with >=1 detected maneuver = {n_pos} ({100*n_pos/len(sats):.1f}%)")
print(f"total events = {n_ev:.0f}   total exposure = {n_ex:.2f} satellite-years")
print(f"EVENT RATE = {n_ev/n_ex:.3f} events/satellite-year   (manuscript: 12.80)\n")

rng = random.Random(SEED)
n = len(pair)
rates = []
for _ in range(B):
    e = d = 0.0
    for _ in range(n):
        a, b = pair[rng.randrange(n)]
        e += a
        d += b
    rates.append(e / d if d else float("nan"))
rates.sort()
pt = n_ev / n_ex
lo, hi = rates[int(0.025 * B)], rates[int(0.975 * B)]
print("=" * 84)
print("1. sampling uncertainty of the event rate (bootstrap over the 879 satellites)")
print("=" * 84)
print(f"  point estimate = {pt:.3f}")
print(f"  95% CI         = [{lo:.3f}, {hi:.3f}]  (width {100*(hi-lo)/pt:.1f}% of the point estimate)")
print(f"  P5 - P95       = [{rates[int(0.05*B)]:.3f}, {rates[int(0.95*B)]:.3f}]")

print("\n" + "=" * 84)
print("2. propagate the uncertainty into the risk currency  N = p x rate")
print("=" * 84)
CASES = [("N_miss (tau=20, proxy)", 0.518),
         ("N_harm (tau=20, proxy)", 0.355),
         ("N_harm (tau=15, proxy)", 0.529),
         ("N_harm (tau=30, proxy)", 0.055)]
print(f"{'quantity':>24} | {'p':>6} | {'N (9.35)':>9} | {'N (12.80)':>10} | "
      f"{'N at 95% CI of the rate':>26}")
for name, p in CASES:
    print(f"{name:>24} | {p:6.3f} | {p*9.35/12.80:9.3f} | {p:10.3f} | "
          f"{p*lo/12.80:11.3f} - {p*hi/12.80:.3f}")
print("\n  (N scales linearly in the event rate, so every N inherits the same")
print("   relative uncertainty as the rate itself.)")
