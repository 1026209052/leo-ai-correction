"""表8（CRC 阈值选择）与表2(b)（期望代价）的新数值。

表8 口径（由旧值反推并验证）：
  候选阈值 τ 的介入集合 = {peak≥τ} ∩ 817 大机动组（故 介入率 = |set|/817）
  风险 = P(η<0 | peak≥τ) = #{η<0}/|set|
  上界 = Wilson **单边** 95% 上界（z=1.645）
  验证：旧数据 τ=20 → 25.0%(27.5% 上界)；τ=105 → 1.1%(4.8%) ✓ 与稿中一致
表2(b)：E[代价] = (C_miss/C_false)·N_miss + N_harm，N 取 MC 的均值
"""
import csv
import math
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data')
Z = 1.645                      # 单边 95%


def load(n):
    with open(os.path.join(D, n), newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def wilson_upper(k, n, z=Z):
    if n == 0:
        return float('nan')
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + h) / d


allr = load('fig2_cdf_all.csv')
BIG = [r for r in allr if float(r['peak']) >= 20.0]
print('大机动组 n =', len(BIG))


def at(tau, rows=BIG):
    s = [r for r in rows if float(r['peak']) >= tau]
    n = len(s)
    k = sum(1 for r in s if float(r['eta_soft']) < 0)
    return n, k, (k / n if n else float('nan')), wilson_upper(k, n)


print('\n=== 表8：CRC 阈值选择 ===')
taus = list(range(20, 301, 5))
tab = {t: at(t) for t in taus}
for d in (0.05, 0.10, 0.15, 0.20):
    cand = [t for t in taus if tab[t][3] <= d]
    if not cand:
        print(f'  δ={d*100:>2.0f}%: 网格内无满足项（需扩大 τ 范围）')
        continue
    t = min(cand)
    n, k, rate, ub = tab[t]
    print(f'  δ={d*100:>2.0f}% → τ*={t:>5.1f} km  介入率={100*n/len(BIG):5.1f}%  '
          f'有害修正率={100*rate:5.2f}% ({k}/{n})  上界={100*ub:5.2f}%')
n, k, rate, ub = tab[20]
print(f'  默认 τ=20 km（无拒绝）：介入率={100*n/len(BIG):5.1f}%  '
      f'有害修正率={100*rate:5.2f}% ({k}/{n})  上界={100*ub:5.2f}%')
print('\n  参考：旧稿 δ=5/10% →105 km(11.1%,1.1%,4.8%)；δ=15%→30 km(64.1%,11.6%,14.1%)；'
      'δ=20%→25 km(83.8%,15.9%,18.3%)；默认20 km(100%,25.0%,27.5%)')
print('  中间网格（供核对）:')
for t in (20, 25, 30, 40, 50, 80, 105, 120, 150, 200, 250, 300):
    if t in tab:
        n, k, rate, ub = tab[t]
        print(f'    τ={t:>3} km  n={n:>3}  介入率={100*n/len(BIG):5.1f}%  '
              f'rate={100*rate:5.2f}%  上界={100*ub:5.2f}%')

# ---------------- 表2(b) 期望代价 ----------------
mc = load('monte_carlo_results_v3.csv')
t1 = [float(r['N_T1']) for r in mc]
old_t2 = sorted({round(float(r['N_T2']), 4) for r in mc}, reverse=True)
new_vals = [1.422, 0.272, 0.171]
mp = {o: v for o, v in zip(old_t2, new_vals)}
t2 = [mp[round(float(r['N_T2']), 4)] for r in mc]
m1 = sum(t1) / len(t1)
m2 = sum(t2) / len(t2)
print(f'\n=== 表2(b)：期望代价（代理口径，C_false=1）===')
print(f'  MC: N_T1 均值={m1:.6f}（不变）  N_T2 均值={m2:.6f}（重映射后）')
print(f"  {'C_miss/C_false':>16} | {'E[代价]':>10} | {'漏报占比':>8}")
for r in (1e2, 1e3, 1e4, 1e5):
    E = r * m1 + m2
    print(f'  {r:>16.0e} | {E:>10.2f} | {100*r*m1/E:>7.1f}%')
