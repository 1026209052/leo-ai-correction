"""图5 数据：τ 侧（模型无关）与 α 侧（来自 α 扫描）。

输出：
  figures/data/fig5_tau.csv    tau, miss_rate, intervention_rate
  figures/data/fig5_alpha.csv  alpha, rmse_drop, severe20, better_pct, worst1
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'figures', 'data')
NAP = 912


def load(p):
    with open(p, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


allr = load(os.path.join(D, 'fig2_cdf_all.csv'))
par = {r['sample_idx']: r for r in load(os.path.join(D, 'pareto_samples.csv'))}
danger = [r for r in allr if float(par[r['sample_idx']]['sgp4_rmse']) > 15]

rows = []
tau = 12.0
while tau <= 36.01:
    miss = sum(1 for r in danger if float(r['peak']) < tau)
    iv = sum(1 for r in allr if float(r['peak']) >= tau)
    rows.append(dict(tau=round(tau, 2), miss_rate=round(100 * miss / NAP, 4),
                     intervention_rate=round(100 * iv / len(allr), 4)))
    tau += 0.5
fn = os.path.join(D, 'fig5_tau.csv')
with open(fn, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'写出 {fn}（{len(rows)} 行）')
print(f"  零漏报边界 τ_max = {min(float(r['peak']) for r in danger if float(r['peak'])>=20):.2f} km")

A = load(os.path.join(HERE, 'out', 'alpha_sweep_metrics_inner_val_ckpt.csv'))
arows = [dict(alpha=float(r['alpha']), rmse_drop=float(r['rmse_drop']),
              severe20=float(r['severe20']), better_pct=float(r['better_pct']),
              worst1=float(r['worst1'])) for r in A]
fn2 = os.path.join(D, 'fig5_alpha.csv')
with open(fn2, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(arows[0]))
    w.writeheader()
    w.writerows(arows)
print(f'写出 {fn2}（{len(arows)} 行）')
