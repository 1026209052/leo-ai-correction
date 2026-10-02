"""第2层（概率化FMEA）新数：p_H(theta) 及其 CI 与 T2。

数据：figures/data/fig2_cdf.csv（软掩码逐样本 eta，817 行）+ sample_metadata.csv
"""
import csv
import math
import os
import random

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data')
B = 10000
R = 12.80


def load(name):
    with open(os.path.join(D, name), newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


rows = load('fig2_cdf.csv')
meta = {r['sample_idx']: r for r in load('sample_metadata.csv')}
eta = [float(r['eta_soft']) for r in rows]
sat = [int(meta[r['sample_idx']]['norad_id']) for r in rows]
n = len(eta)
print('软掩码大机动组 n=%d  卫星数=%d' % (n, len(set(sat))))


def wilson(k, m, z=1.96):
    p = k / m
    d = 1 + z * z / m
    c = p + z * z / (2 * m)
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return (c - h) / d * 100, (c + h) / d * 100


def cluster_ci(vals, groups, seed=20260919):
    by = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    keys = list(by)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        s = []
        for _ in range(len(keys)):
            s.extend(by[keys[rng.randrange(len(keys))]])
        out.append(100.0 * sum(1 for x in s if x < 0) / len(s))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


print('')
print('表1 的 p_H（新值）:')
for th, kind in ((0, 'cluster'), (20, 'wilson'), (50, 'wilson')):
    k = sum(1 for v in eta if v < -th)
    p = 100.0 * k / n
    if kind == 'wilson':
        lo, hi = wilson(k, n)
        print('  p_H(theta=%2d%%) = %6.2f%%  (%d/%d)  Wilson 95%% CI [%.2f%%, %.2f%%]'
              % (th, p, k, n, lo, hi))
    else:
        lo, hi = cluster_ci(eta, sat)
        print('  p_H(theta=%2d%%) = %6.2f%%  (%d/%d)  聚簇 95%% CI [%.2f%%, %.2f%%]'
              % (th, p, k, n, lo, hi))

pI_proxy, pI_oracle = 0.6439, 0.582
print('')
print('T2 = p_I_AI x p_H(theta) x 12.80:')
vals = []
for th in (0, 20, 50):
    pH = sum(1 for v in eta if v < -th) / n
    a = pI_oracle * pH * R
    b = pI_proxy * pH * R
    vals.append(b)
    print('  theta=%2d%%  p_H=%5.2f%%  Oracle %.3f   代理 %.3f' % (th, pH * 100, a, b))
print('  表2(a) Oracle(theta=20%%) 预测=实测 = %.3f' % (pI_oracle * vals[1] / pI_proxy))
print('  三者等权平均（代理口径）= %.3f' % (sum(vals) / 3))
