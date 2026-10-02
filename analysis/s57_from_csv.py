"""从 spacex_v5_verification.csv 复算 §5.7 表10 的全部量。

口径：n=721 窗口；门控 = sgp4_peak >= 20 km（= passes_peak 列）；
      逐样本 η 已在 CSV 里（eta_trans_soft / eta_lstm_soft）。
"""
import csv
import os
import statistics as st

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data',
                 'spacex_v5_verification.csv')
rows = list(csv.DictReader(open(P, newline='', encoding='utf-8-sig')))
n = len(rows)
print('窗口数 =', n)


def deg_med(x):
    v = [y for y in x if y < 0]
    return st.median(v) if v else float('nan')


def cvar10(x):
    s = sorted(x)
    k = max(1, int(0.10 * len(s)))
    return st.mean(s[:k])


for tag, eta_c, ai_c in (('Transformer+软掩码', 'eta_trans_soft', 'ai_rmse_alt'),
                         ('LSTM+软掩码', 'eta_lstm_soft', 'lstm_ai_rmse_alt')):
    eta = [float(r[eta_c]) for r in rows]
    sgp4 = [float(r['sgp4_rmse_alt']) for r in rows]
    ai = [float(r[ai_c]) for r in rows]
    big = [i for i, r in enumerate(rows) if float(r['sgp4_peak']) >= 20.0]
    print('=' * 78)
    print(f'{tag}')
    print('=' * 78)
    m_s, m_a = st.mean(sgp4), st.mean(ai)
    print(f'  [无条件应用] 平均RMSE  SGP4 {m_s:.3f} → AI {m_a:.3f} km  '
          f'（相对纯SGP4 {100*(m_a/m_s-1):+.2f}%）')
    print(f'               改善比例 {100*sum(1 for y in eta if y>0)/n:.1f}%   '
          f'改善中位 {st.median([y for y in eta if y>0]):+.2f}%   '
          f'退化中位 {deg_med(eta):+.2f}%   CVaR10 {cvar10(eta):+.2f}%   '
          f'最差 {min(eta):+.2f}%')
    # 门控：只在大机动窗口用 AI，其余回退纯 SGP4
    gated = [ai[i] if i in big else sgp4[i] for i in range(n)]
    g_eta = [float(rows[i][eta_c]) if i in big else 0.0 for i in range(n)]
    m_g = st.mean(gated)
    print(f'  [Fail-Safe门控] 介入 {len(big)}/{n} 窗口；平均RMSE '
          f'SGP4 {m_s:.3f} → 门控后 {m_g:.3f} km（相对纯SGP4 {100*(m_g/m_s-1):+.3f}%）')
    print(f'               门控后最差单样本退化 {min(g_eta):+.2f}%   '
          f'（= 大机动子集的最差）')
    print(f'  → 平均RMSE 相对变化: 无条件 {100*(m_a/m_s-1):+.2f}%  vs  '
          f'门控 {100*(m_g/m_s-1):+.3f}%')
