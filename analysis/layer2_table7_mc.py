"""第2层：表7（阈值扫描的运行风险）新数 + MC 的 T2 重映射。

数据：figures/data/fig2_cdf.csv（逐样本 η，817 大机动组）+ pareto_samples.csv（peak, sgp4_rmse）
口径（与论文表7 表注一致）：
  危险漏报率 = #{peak<τ 且 sgp4_rmse>15}/912        （模型无关 → 应复现旧稿 0/0/3.51/31.6%）
  有害修正率 = #{η<-20 且 peak≥τ}/#{peak≥τ}
  N_miss = 危险漏报率 × 12.80 ；N_harm = 介入率 × 有害修正率 × 12.80
  R(τ) = 1e4 × N_miss + N_harm
"""
import csv
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data')
R = 12.80
NAP = 912          # 验证集中的机动样本数


def load(name):
    with open(os.path.join(D, name), newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


pareto = {r['sample_idx']: r for r in load('pareto_samples.csv')}


def run(tag, csvname, col):
    """csvname 需为 fig2_cdf_all.csv（1404 行、未门控）——
    表7 在 τ=15 km 时的介入集合(879)比 817 多 62 个 peak∈[15,20) 的样本。"""
    eta = {r['sample_idx']: float(r[col]) for r in load(csvname)}
    if len(eta) < len(pareto):
        print(f'  [!] {csvname} 只有 {len(eta)} 行，应为 {len(pareto)}（未门控全量）——'
              f' 请用 export_eta.py 生成的 fig2_cdf_all.csv')
    print('\n' + '=' * 78)
    print(f'{tag}  (列 {col}，n={len(eta)})')
    print('=' * 78)
    print(f"{'τ':>4} | {'介入率':>7} | {'危险漏报率':>9} | {'N_miss':>7} | "
          f"{'有害修正率':>9} | {'N_harm':>7} | {'R(τ)':>10}")
    for tau in (15, 20, 30, 50):
        intervened = [k for k in eta if float(pareto[k]['peak']) >= tau]
        miss = [k for k in eta if float(pareto[k]['peak']) < tau
                and float(pareto[k]['sgp4_rmse']) > 15]
        rate_int = 100.0 * len(intervened) / len(pareto)
        rate_miss = 100.0 * len(miss) / NAP
        n_miss = rate_miss / 100 * R
        rate_harm = (100.0 * sum(1 for k in intervened if eta[k] < -20) / len(intervened)
                     if intervened else float('nan'))
        n_harm = rate_harm / 100 * rate_int / 100 * R
        print(f'{tau:>4} | {rate_int:6.1f}% | {rate_miss:8.2f}% | {n_miss:7.3f} | '
              f'{rate_harm:8.2f}% | {n_harm:7.3f} | {1e4 * n_miss + n_harm:10.3f}')


run('新模型（论文口径，未门控全量）', 'fig2_cdf_all.csv', 'eta_soft')
# 旧模型自检需要旧模型的"未门控全量"文件（fig2_cdf_oldmodel.csv 只有 817 行），故此处跳过；
# 危险漏报率列（模型无关）已由直接统计验证为 0/0/3.51/31.6%，与旧稿一致。

# ---- MC 的 T2 重映射：旧 T2 只有 3 个离散值（按 θ=0/20/50%），可精确替换 ----
mc = load('monte_carlo_results_v3.csv')
old = sorted({round(float(r['N_T2']), 4) for r in mc}, reverse=True)
print('\nMC 里 N_T2 的 3 个离散值（旧，降序）:', old)
new = [1.422, 0.272, 0.171]          # θ=0/20/50%（代理介入率口径）
m = {o: v for o, v in zip(old, new)}
vals = sorted(m[round(float(r['N_T2']), 4)] for r in mc)
B = len(vals)
mean = sum(vals) / B
print(f'重映射后 T2 的 MC：均值 {mean:.3f}  P5 {vals[int(0.05*B)]:.3f}  P95 {vals[int(0.95*B)]:.3f}')
print(f'  三个 θ 取值的等权平均 = {sum(new)/3:.3f}')
