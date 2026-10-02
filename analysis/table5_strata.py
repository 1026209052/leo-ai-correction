"""表5：按 SGP4 基线 RMSE 分层的性能对比（在 817 个大机动样本上）。

输入（跑完重训后把这些文件拷到 figures/data/）：
  figures/data/fig2_cdf.csv       sample_idx, eta_mse, eta_soft      (817 行)
  figures/data/pareto_samples.csv sample_idx, peak, sgp4_rmse, ...   (1404 行)

输出：打印可直接填入论文表5 的表体（markdown/LaTeX 均可），
      并写出 analysis/out/table5_strata.csv。

口径（与 §4.2 一致）：
  优于SGP4比例 = mean(eta > 0)；退化中位数 = median(eta | eta < 0)（该层内）。
  eta_mse  ↔ 表5 的 "Transformer"（α=1.0，标准MSE）
  eta_soft ↔ 表5 的 "Transformer+软掩码"（α=0.3）
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'figures', 'data')
OUT = os.path.join(HERE, 'out')


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def layer_of(r):
    if r < 10.0:
        return '<10 km'
    if r < 20.0:
        return '10–20 km'
    return '≥20 km'


def stats(etas):
    n = len(etas)
    better = 100.0 * sum(e > 0 for e in etas) / n
    neg = [e for e in etas if e < 0]
    deg = st.median(neg) if neg else float('nan')
    return better, deg


def severe(etas, t=20.0):
    """严重退化比例 P(η < -t)：无条件、同一分母（论文新主指标）。"""
    return 100.0 * sum(e < -t for e in etas) / len(etas)


def main():
    etas = {r['sample_idx']: r for r in read_csv(os.path.join(DATA, 'fig2_cdf.csv'))}
    pareto = {r['sample_idx']: float(r['sgp4_rmse'])
              for r in read_csv(os.path.join(DATA, 'pareto_samples.csv'))}

    joined = [(float(pareto[k]), float(v['eta_mse']), float(v['eta_soft']))
              for k, v in etas.items() if k in pareto]
    print(f'对齐样本数 = {len(joined)}（期望 817）')
    if len(joined) != 817:
        print('  [!] 与 817 不一致：请确认 fig2_cdf.csv 是新的 817 行大机动组文件')

    order = ['<10 km', '10–20 km', '≥20 km']
    buckets = {k: {'mse': [], 'soft': []} for k in order}
    for r, em, es in joined:
        b = buckets[layer_of(r)]
        b['mse'].append(em)
        b['soft'].append(es)

    rows = []
    print()
    print('| 层级 | 样本数 | 指标 | Transformer | Transformer+软掩码 |')
    print('|---|---|---|---|---|')
    for k in order:
        n = len(buckets[k]['mse'])
        bm, dm = stats(buckets[k]['mse'])
        bs, ds = stats(buckets[k]['soft'])
        sm, ss = severe(buckets[k]['mse']), severe(buckets[k]['soft'])
        print(f'| {k} | {n} | 优于SGP4比例 | {bm:.1f}% | {"**" if bs > bm else ""}{bs:.1f}%{"**" if bs > bm else ""} |')
        print(f'| | | 严重退化比例(η<-20%) | {sm:.1f}% | **{ss:.1f}%** |')
        print(f'| | | _退化中位数（仅描述性，分母随方法变）_ | _{dm:.1f}%_ | _{ds:.1f}%_ |')
        rows.append(dict(layer=k, n=n, better_mse=round(bm, 1), better_soft=round(bs, 1),
                         severe20_mse=round(sm, 2), severe20_soft=round(ss, 2),
                         deg_mse=round(dm, 1), deg_soft=round(ds, 1)))
    tot = sum(r['n'] for r in rows)
    print(f'\n合计样本数 = {tot}（应等于 817）')

    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, 'table5_strata.csv')
    with open(fn, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'写出 {fn}')


if __name__ == '__main__':
    main()
