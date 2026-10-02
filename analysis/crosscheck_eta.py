"""独立复算 η 派生的六个指标，用于和 verify_all_tables.py 的输出对照。

输入：figures/data/fig2_cdf.csv  →  sample_idx, eta_mse, eta_soft
      （列名可换成任意两列，脚本按第 2、3 列当两种方法）

用途：
  1) 现在：用**旧**数据跑，应复现 README 记录的
        eta_mse : 优于SGP4 56.79%  退化中位 -24.07%  最差 -162.65%
        eta_soft: 优于SGP4 75.03%  退化中位  -4.27%  最差  -57.37%
     —— 复现成功即说明本脚本与 verify_all_tables.py 的口径一致，
        于是它可以在**新**数据上充当独立核对器。
  2) 换上新 fig2_cdf.csv 后再跑一次，直接与 verify_all_tables 的表4 数字对撞。

退化中位数给出两种约定，用于确认 `>=` 与 `<` 的差异（论文 §4.2 定义为 η<0）：
  deg_med_strict : median(η | η < 0)
  deg_med_ge     : median(η | η ≤ 0)   ← verify_all_tables.compute_metrics 的写法
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'figures', 'data')


def report(name, xs):
    n = len(xs)
    better = 100.0 * sum(x > 0 for x in xs) / n
    imp = [x for x in xs if x > 0]
    neg_strict = [x for x in xs if x < 0]
    neg_ge = [x for x in xs if x <= 0]
    s = sorted(xs)
    k = max(1, int(0.10 * n))
    print(f'\n--- {name}  (n={n}) ---')
    print(f'  优于SGP4比例          : {better:8.2f}%')
    print(f'  改善中位数 median(>0) : {st.median(imp):+8.2f}%   (n={len(imp)})')
    print(f'  退化中位数 median(<0) : {st.median(neg_strict):+8.2f}%   (n={len(neg_strict)})  ← 论文定义')
    print(f'  退化中位数 median(≤0) : {st.median(neg_ge):+8.2f}%   (n={len(neg_ge)})  ← verify_all_tables 写法')
    print(f'  最差10%平均退化       : {sum(s[:k])/k:+8.2f}%   (k={k})')
    print(f'  最差单样本            : {s[0]:+8.2f}%')
    print(f'  η 分位 P1/P5/P10/P25  : {s[int(0.01*n)]:+.2f} / {s[int(0.05*n)]:+.2f} /'
          f' {s[int(0.10*n)]:+.2f} / {s[int(0.25*n)]:+.2f}')


def main():
    fn = os.path.join(DATA, 'fig2_cdf.csv')
    rows = list(csv.DictReader(open(fn, newline='', encoding='utf-8-sig')))
    cols = list(rows[0])
    print(f'{fn}: {len(rows)} 行，列 = {cols}')
    for c in cols[1:]:
        report(c, [float(r[c]) for r in rows])


if __name__ == '__main__':
    main()
