"""核查论文表9（碰撞预警失效概率）的 SGP4 列口径。

SGP4 失效概率 = P(SGP4误差 > T) 与模型无关，因此可以用手上的
pareto_samples.csv 复算，与旧稿的 62.5%@15km 对照，判断旧稿用的是
哪个样本集 / 哪个误差量。
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'figures', 'data')

rows = list(csv.DictReader(open(os.path.join(DATA, 'pareto_samples.csv'),
                                newline='', encoding='utf-8-sig')))
print(f'pareto_samples.csv: {len(rows)} 行；列 = {list(rows[0])}')

rmse = {r['sample_idx']: float(r['sgp4_rmse']) for r in rows}
peak = {r['sample_idx']: float(r['peak']) for r in rows}

for name, sel in [('全部 1404', lambda k: True),
                  ('peak>=20 (817)', lambda k: peak[k] >= 20.0),
                  ('peak<20 (95... 机动小样本)', lambda k: peak[k] < 20.0)]:
    ks = [k for k in rmse if sel(k)]
    print(f'\n--- {name}：n={len(ks)} ---')
    print('  sgp4_rmse 分位: P50=%.1f P90=%.1f P95=%.1f max=%.1f'
          % (sorted(rmse[k] for k in ks)[len(ks)//2],
             sorted(rmse[k] for k in ks)[int(0.9*len(ks))],
             sorted(rmse[k] for k in ks)[int(0.95*len(ks))],
             max(rmse[k] for k in ks)))
    for T in (10, 15, 20, 30):
        p = sum(rmse[k] > T for k in ks) / len(ks) * 100
        print(f'  P(sgp4_rmse > {T:>2}) = {p:5.1f}%   ({sum(rmse[k] > T for k in ks)}/{len(ks)})')

print('\n旧稿表9 的 SGP4 列：10km 75.0% / 15km 62.5% / 20km 38.6% / 30km 27.3%')
