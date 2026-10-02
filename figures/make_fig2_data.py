"""Build the empirical CDF for Figure 2 from the real per-sample data.

Input : figures/data/fig2_cdf.csv   (raw samples: sample_idx, eta_mse, eta_soft)
Output: figures/data/fig2_cdf_ecdf.csv   (empirical CDF on a uniform grid, plot-ready)

Also cross-checks the sample statistics against the values reported in the
manuscript (Table 4), so any mismatch is visible immediately.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Raw per-sample file uploaded by the author (sample_idx, eta_mse, eta_soft).
RAW = os.path.join(HERE, "data", "fig2_cdf.csv")
# Plot-ready ECDF grid consumed by fig2_cdf.tex (eta, F_mse, F_soft).
OUT = os.path.join(HERE, "data", "fig2_cdf_ecdf.csv")


def read_raw(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    mse = [float(r["eta_mse"]) for r in rows]
    soft = [float(r["eta_soft"]) for r in rows]
    return mse, soft


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def report(name, xs, ref):
    pos = [x for x in xs if x > 0]
    neg = [x for x in xs if x < 0]
    got = {
        "n": len(xs),
        "superior": 100.0 * len(pos) / len(xs),
        "imp_median": median(pos),
        "deg_median": median(neg),
        "worst": min(xs),
    }
    print(f"--- {name} ---")
    for k, v in got.items():
        r = ref.get(k)
        flag = ""
        if r is not None:
            ok = abs(v - r) <= max(0.06 * abs(r), 0.06)
            flag = f"   paper={r}   {'OK' if ok else '<<< MISMATCH'}"
        print(f"  {k:11s} = {v:12.4f}{flag}")
    return got


# 论文表5 的当前值（重训 + 内层验证最优口径；部署配置 α=0.2）
MSE_REF = dict(n=817, superior=65.5, imp_median=23.0, deg_median=-5.3, worst=-199.8)
SOFT_REF = dict(n=817, superior=82.7, imp_median=11.0, deg_median=-5.3, worst=-77.6)

mse, soft = read_raw(RAW)
report("eta_mse (standard MSE, alpha=1.0)", mse, MSE_REF)
report("eta_soft (soft mask, alpha=0.2)", soft, SOFT_REF)

# ---- empirical CDF on a uniform grid -------------------------------------
lo = min(min(mse), min(soft))
hi = max(max(mse), max(soft))
lo, hi = lo - 2.0, hi + 2.0
STEP = 0.25
n = len(mse)
smse, ssoft = sorted(mse), sorted(soft)
grid, i, j = [], 0, 0
x = lo
while x <= hi + 1e-9:
    while i < n and smse[i] <= x:
        i += 1
    while j < n and ssoft[j] <= x:
        j += 1
    grid.append((x, i / n, j / n))
    x += STEP

with open(OUT, "w", newline="", encoding="ascii") as fh:
    w = csv.writer(fh)
    w.writerow(["eta", "F_mse", "F_soft"])
    for x, fm, fs in grid:
        w.writerow([f"{x:.2f}", f"{fm:.6f}", f"{fs:.6f}"])

print(f"\nrange = [{lo:.2f}, {hi:.2f}]   grid points = {len(grid)}")
print("data source:", RAW)
print("wrote      :", OUT)


# ---- anchor points actually used to annotate the figure ------------------
def F_at(sorted_vals, x):
    c = 0
    for v in sorted_vals:
        if v <= x:
            c += 1
        else:
            break
    return c / len(sorted_vals)


print("\nanchor (x, F) for the figure markers:")
for tag, sv in (("mse", smse), ("soft", ssoft)):
    pos = [v for v in sv if v > 0]
    neg = [v for v in sv if v < 0]
    dm, w, im = median(neg), min(sv), median(pos)
    print(f"  {tag}: deg_median ({dm:.1f}, {F_at(sv, dm):.4f})   "
          f"worst ({w:.1f}, {F_at(sv, w):.4f})   imp_median ({im:.1f}, {F_at(sv, im):.4f})   "
          f"F(0) = {F_at(sv, 0.0):.4f}")
