"""
eval_loss_baselines.py  --  表3 的实证评估：MSE / 软掩码 vs Huber / Focal / 自适应加权

复用 verify_all_tables.py 的权重加载与评估口径（完全相同，避免"表3 与表5 对不上"）：
  - 权重：论文口径 = 内层验证最优（inner_val_ckpt/*_innerval.pth），6 种子集成
  - 反归一化：masked res scaler；cond 用 truth scaler
  - 决策：Oracle 门控，tau=20 km
  - 逐样本 eta：1 - RMSE_ours/RMSE_SGP4（高度+沿轨双通道联合），大机动组 n=817

对照家族
  MSE（Transformer）   : error_predictor_alpha1.0_seed*   （已有）
  软掩码（本文, a=0.2） : error_predictor_alpha0.2_seed*   （已有）
  Huber                : error_predictor_huber_seed*      （train_loss_baselines.py 产出）
  Focal                : error_predictor_focal_seed*      （同上）
  自适应加权 exp(-lam r^2): error_predictor_adaptive_seed* （同上）

用法
    python eval_loss_baselines.py                 # 论文口径（内层验证最优）
    python eval_loss_baselines.py --ckpt-dir=     # 对照口径（训练损失最优）

输出
    eval_out/loss_baselines_metrics.csv
        列: method, models, n, better_pct, imp_median, severe20, worst10, worst1, rmse_drop,
            d_better, ci_lo_better, ci_hi_better, p_better,
            d_severe20, ci_lo_severe20, ci_hi_severe20, p_severe20
        （Delta 与 p 值均为「相对软掩码」的按卫星聚簇配对 bootstrap，与 5.1 节口径一致）
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

B_BOOT = int(os.environ.get('B_BOOT', '10000'))
BOOT_SEED = 20240926

FAMILIES = [
    ('MSE', 'transformer', lambda a: a.PREFIX['Transformer']),
    ('软掩码(本文)', 'transformer', lambda a: a.PREFIX['Transformer+SoftMask']),
    ('Huber', 'transformer', lambda a: 'error_predictor_huber_seed'),
    ('Focal', 'transformer', lambda a: 'error_predictor_focal_seed'),
    ('自适应加权', 'transformer', lambda a: 'error_predictor_adaptive_seed'),
]
REF_NAME = '软掩码(本文)'


def per_sample_rmse(traj, truth):
    return np.sqrt(np.mean((traj - truth) ** 2, axis=(1, 2)))


def metrics(eta):
    n = len(eta)
    if n == 0:
        return dict(n=0, better_pct=np.nan, imp_median=np.nan, severe20=np.nan,
                    worst10=np.nan, worst1=np.nan)
    neg = eta[eta < 0]
    s = np.sort(eta)
    k = max(1, int(0.10 * n))
    return dict(
        n=n,
        better_pct=100.0 * float(np.mean(eta > 0)),
        imp_median=float(np.median(eta[eta > 0])) if (eta > 0).any() else np.nan,
        severe20=100.0 * float(np.mean(eta < -20.0)),
        worst10=float(np.mean(s[:k])),
        worst1=float(s[0]),
    )


def cluster_paired_bootstrap(sat_of, eta_a, eta_b, stat, B=B_BOOT, seed=BOOT_SEED):
    """按卫星聚簇的配对 bootstrap：返回 (Delta, ci_lo, ci_hi, p)。"""
    rng = np.random.RandomState(seed)
    sats = np.unique(sat_of)
    groups = {s: np.where(sat_of == s)[0] for s in sats}
    point = stat(eta_a) - stat(eta_b)
    diffs = np.empty(B)
    for b in range(B):
        pick = rng.randint(0, len(sats), len(sats))
        idx = np.concatenate([groups[sats[i]] for i in pick])
        diffs[b] = stat(eta_a[idx]) - stat(eta_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return point, float(lo), float(hi), min(p, 1.0)


def main():
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            vt.CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            vt.CKPT_DIR = sys.argv[i + 1]
    tag = '论文口径：内层验证最优' if vt.CKPT_DIR else '对照口径：训练损失最优'
    print(f"[口径] {tag}" + (f"（{vt.CKPT_DIR}）" if vt.CKPT_DIR else ""))

    device = torch.device(vt.DEVICE)
    cond_mean = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    m_rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    m_rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    idx_all = np.asarray(vt.load_val_indices())
    meta = pd.read_csv(f'{vt.MASKED_DIR}/sample_metadata.csv')
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sat_map = dict(zip(meta['sample_idx'].values, meta['norad_id'].values))
    print(f'验证样本数: {len(idx_all)}')

    rows, etas = [], {}
    for name, mtype, pfx_fn in FAMILIES:
        pfx = pfx_fn(vt)
        models = vt.load_ensemble(pfx, mtype, device)
        if not models:
            print(f'  [!] 跳过 {name}：未找到 {pfx}*（先跑 train_loss_baselines.py）')
            continue
        sgps, trues, preds = vt.evaluate_method(
            models, idx_all, cond_mean, cond_std, m_rm, m_rs, device)[:3]
        peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
        big = peaks >= vt.THRESHOLD
        eta = (1.0 - per_sample_rmse(preds, trues) /
               (per_sample_rmse(sgps, trues) + 1e-8)) * 100.0
        rmse_drop = (1.0 - np.mean(per_sample_rmse(preds, trues)[big]) /
                     (np.mean(per_sample_rmse(sgps, trues)[big]) + 1e-8)) * 100.0
        m = metrics(eta[big])
        m.update(method=name, models=len(models), rmse_drop=rmse_drop)
        rows.append(m)
        etas[name] = (np.asarray([sat_map.get(int(i), '?') for i in idx_all])[big], eta[big])
        print(f'  {name:14s} n={m["n"]} 优于SGP4={m["better_pct"]:6.2f}% '
              f'严重退化={m["severe20"]:5.2f}% 最差10%={m["worst10"]:8.2f}% '
              f'最差={m["worst1"]:9.2f}% RMSE降幅={rmse_drop:5.2f}%')

    df = pd.DataFrame(rows)[['method', 'models', 'n', 'better_pct', 'imp_median',
                             'severe20', 'worst10', 'worst1', 'rmse_drop']]

    if REF_NAME in etas:
        sat_ref, eta_ref = etas[REF_NAME]
        dcols = {}
        for name, (sat_a, eta_a) in etas.items():
            if name == REF_NAME:
                continue
            d_b, lo_b, hi_b, p_b = cluster_paired_bootstrap(
                sat_ref, eta_a, eta_ref, lambda e: 100.0 * np.mean(e > 0))
            d_s, lo_s, hi_s, p_s = cluster_paired_bootstrap(
                sat_ref, eta_a, eta_ref, lambda e: 100.0 * np.mean(e < -20.0))
            dcols[name] = dict(d_better=d_b, ci_lo_better=lo_b, ci_hi_better=hi_b, p_better=p_b,
                               d_severe20=d_s, ci_lo_severe20=lo_s, ci_hi_severe20=hi_s,
                               p_severe20=p_s)
            print(f'  vs {name:14s} D优于SGP4={d_b:+6.2f} pp [{lo_b:+6.2f},{hi_b:+6.2f}] p={p_b:.3f}'
                  f' | D严重退化={d_s:+6.2f} pp [{lo_s:+6.2f},{hi_s:+6.2f}] p={p_s:.3f}')
        for col in ('d_better', 'ci_lo_better', 'ci_hi_better', 'p_better',
                    'd_severe20', 'ci_lo_severe20', 'ci_hi_severe20', 'p_severe20'):
            df[col] = [dcols.get(r, {}).get(col, np.nan) for r in df['method']]

    os.makedirs('eval_out', exist_ok=True)
    out = 'eval_out/loss_baselines_metrics.csv'
    df.to_csv(out, index=False)
    print(f'\n写出 {out}')
    print('  -> 直接填表3 的实证列（Huber/Focal/自适应加权 vs 软掩码）')


if __name__ == '__main__':
    main()