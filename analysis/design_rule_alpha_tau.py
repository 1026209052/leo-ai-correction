"""② (alpha, tau) 联合风险约束设计规则 —— 数值准备。

结构洞察：**危险漏报率只依赖 τ**（门控 + SGP4 误差分布，模型无关），
因此设计问题可分离：
  第一步 τ：由"零危险漏报"约束定出可用区间 [τ_min, τ_max]，并在该区间内取最小介入面；
  第二步 α：给定 δ（严重退化比例上限）或 W（最差单样本退化上界），在可行 α 集内最大化平均精度。

数据：figures/data/fig2_cdf_all.csv（τ 侧，模型无关）、
      eval_out/alpha_sweep_metrics_inner_val_ckpt.csv（α 侧，11 个取值）
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'figures', 'data')
NAP = 912
R = 12.80


def load(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


allr = load(os.path.join(D, 'fig2_cdf_all.csv'))
par = {r['sample_idx']: r for r in load(os.path.join(D, 'pareto_samples.csv'))}

# ---------------- 第一步：τ 侧（模型无关） ----------------
print('=' * 84)
print('第一步  τ 侧：危险漏报率与介入率（模型无关）')
print('=' * 84)
danger = [r for r in allr
          if float(par[r['sample_idx']]['sgp4_rmse']) > 15]
pk_danger_big = [float(r['peak']) for r in danger if float(r['peak']) >= 20]
print(f'  sgp4_rmse>15 的样本数 = {len(danger)}（其中 peak≥20 的 {len(pk_danger_big)} 个）')
tau_max = min(pk_danger_big)
print(f'  → 零危险漏报可保持的最大阈值 τ_max = {tau_max:.2f} km')
print(f'  → 部署阈值 τ=20 km 距该边界的安全裕度 = {tau_max-20:.2f} km')
print()
print(f"{'τ (km)':>8} | {'介入率(1404)':>12} | {'介入率(817)':>11} | {'危险漏报率':>10} | {'n_miss':>7}")
for tau in (15, 18, 20, 22, 24, 26, 28, 30, 35):
    iv_all = sum(1 for r in allr if float(r['peak']) >= tau)
    iv_big = sum(1 for r in allr if float(r['peak']) >= tau and float(r['peak']) >= 20)
    ms = sum(1 for r in danger if float(r['peak']) < tau)
    print(f'{tau:>8} | {100*iv_all/len(allr):11.1f}% | {100*iv_big/817:10.1f}% | '
          f'{100*ms/NAP:9.2f}% | {ms:>7}')

# ---------------- 第二步：α 侧 ----------------
cand = [os.path.join(HERE, 'out', 'alpha_sweep_metrics_inner_val_ckpt.csv'),
        os.path.join(HERE, 'out', 'alpha_sweep_metrics.csv'),
        os.path.join(HERE, '..', 'eval_out', 'alpha_sweep_metrics_inner_val_ckpt.csv')]
for c in cand:
    if os.path.exists(c):
        A = load(c)
        print(f'\n[α 侧数据] {c}')
        break
else:
    raise SystemExit('未找到 alpha_sweep_metrics*.csv')

rows = []
for r in A:
    rows.append(dict(alpha=float(r['alpha']), rmse=float(r['rmse_drop']),
                     better=float(r['better_pct']), severe=float(r['severe20']),
                     worst=float(r['worst1'])))
print('=' * 84)
print('第二步  α 侧：严重退化比例 / 最差单样本 / RMSE降幅（τ=20 km 门控）')
print('=' * 84)
print(f"{'α':>5} | {'RMSE降幅':>8} | {'优于SGP4':>8} | {'严重退化%':>9} | {'最差%':>9} | "
      f"{'支配者?':>8}")
for r in rows:
    dom = [q['alpha'] for q in rows
           if q is not r and q['rmse'] >= r['rmse'] and q['severe'] <= r['severe']
           and q['worst'] >= r['worst'] and (q['rmse'], -q['severe'], q['worst']) !=
           (r['rmse'], -r['severe'], r['worst'])]
    print(f"{r['alpha']:>5.1f} | {r['rmse']:7.2f}% | {r['better']:7.2f}% | "
          f"{r['severe']:8.2f}% | {r['worst']:8.2f}% | {('是('+','.join('%g'%d for d in dom)+')') if dom else '-':>8}")

# ---------------- 规则的解：两个约束族 ----------------
print('=' * 84)
print('设计规则的解')
print('=' * 84)
print('  (a) 约束：严重退化比例 ≤ δ，目标：最大化 RMSE 降幅')
print(f"{'δ':>5} | {'可行 α':>34} | {'α*':>5} | {'RMSE降幅':>9} | {'严重退化':>8} | {'最差':>9}")
for d in (3, 4, 5, 6, 7, 8, 10, 13, 15):
    fe = [r for r in rows if r['severe'] <= d]
    if not fe:
        print(f'{d:>4}% | {"—":>34} | {"—":>5}')
        continue
    best = max(fe, key=lambda r: r['rmse'])
    print(f"{d:>4}% | {','.join('%g'%r['alpha'] for r in fe):>34} | {best['alpha']:>5.1f} | "
          f"{best['rmse']:8.2f}% | {best['severe']:7.2f}% | {best['worst']:8.2f}%")
print()
print('  (b) 约束：最差单样本退化 ≥ −W，目标：最大化 RMSE 降幅')
print(f"{'W':>5} | {'可行 α':>34} | {'α*':>5} | {'RMSE降幅':>9} | {'严重退化':>8} | {'最差':>9}")
for w in (60, 70, 80, 90, 100, 120, 150, 200):
    fe = [r for r in rows if r['worst'] >= -w]
    if not fe:
        print(f'{w:>4}% | {"—":>34} | {"—":>5}')
        continue
    best = max(fe, key=lambda r: r['rmse'])
    print(f"{w:>4}% | {','.join('%g'%r['alpha'] for r in fe):>34} | {best['alpha']:>5.1f} | "
          f"{best['rmse']:8.2f}% | {best['severe']:7.2f}% | {best['worst']:8.2f}%")

# ---------------- 风险货币下的 (α,τ) 目标函数 ----------------
print()
print('  (c) 目标函数 R(α,τ) = r·Ṅ_miss(τ) + Ṅ_harm(α,τ)，r=1e4，硬约束 零危险漏报')
print(f"{'α':>5} | {'Ṅ_harm':>8} | {'R(α,20km)':>10}")
for r in rows:
    nh = 0.582 * r['severe'] / 100 * R
    print(f"{r['alpha']:>5.1f} | {nh:8.3f} | {1e4*0 + nh:10.3f}")
