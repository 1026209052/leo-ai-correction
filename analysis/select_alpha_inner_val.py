"""select_alpha_inner_val.py -- 在【内层验证集】上为 α 做选择

用途（对应审稿意见："α=0.3 的辩护牵强；你有内层验证集，完全可以在上面选 α"）：
  · 主实验的 α=0.3 是**训练前预设**的（避免在评估集上选择，见 §5.9 与 §4.3）；
  · 但内层验证集（118 颗卫星 / 1832 样本，训练时仅用于早停）本就可以承担"超参数选择"，
    且**不需要重训**：α 消融的 11 个取值各有 6 个种子，其权重已存在。
  · 本脚本把 11 个 α 的 6 项指标在【内层验证样本】上重算，并按 §5.9 的规则
    "在风险约束内最大化 RMSE 降幅"给出 α*，与预设 0.3 对照。

口径与 eval_alpha_sweep.py 完全一致（逐样本 MSE / 池化 RMS 比 / Oracle 门控 / 6 种子先平均预测）。
注意：内层验证集参与了早停，故其指标对**被选中的检查点**偏乐观；但该偏置对 11 个 α 的作用方式
相同（同一训练与早停协议），因此用于 α 之间的**排序**是合理的——这正是验证集的用途。

用法（在 starlink_fixed/ 下，与 eval_alpha_sweep.py 同环境）：
    python select_alpha_inner_val.py                 # 论文口径权重 inner_val_ckpt/_innerval.pth
    python select_alpha_inner_val.py --boot          # 追加 α* vs 0.3 的聚簇配对 bootstrap
    python select_alpha_inner_val.py --ckpt-dir=     # 对照：训练损失最优权重
输出：eval_out/alpha_inner_val_metrics.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_alpha_sweep as eas   # 复用 ErrorPredictor / load_ensemble / evaluate / compute_metrics / boot_cluster

DATA_DIR = eas.DATA_DIR
MASKED_DIR = eas.MASKED_DIR
OUTDIR = eas.OUTDIR
THRESHOLD = eas.THRESHOLD
RISK_DELTAS = [3, 5, 8, 10, 13, 15, 20]      # 严重退化比例约束 δ (%)
RISK_W = [80, 100, 150]                       # 最差单样本退化上界 W (%)


def inner_val_indices():
    """复现 train_alpha_inner_val.build_splits 的内层验证划分。"""
    meta = pd.read_csv(f'{DATA_DIR}/sample_metadata.csv')
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sats = meta['norad_id'].values
    uniq = np.unique(sats)
    rng_eval = np.random.RandomState(42)          # 论文评估划分种子（勿改）
    rng_eval.shuffle(uniq)
    n_eval = max(1, int(len(uniq) * 0.10))
    eval_sats = set(uniq[:n_eval])
    train_sats = np.array([s for s in uniq if s not in eval_sats])
    rng_inner = np.random.RandomState(2024)       # 内层划分种子（勿改）
    rng_inner.shuffle(train_sats)
    n_inner = max(1, int(len(train_sats) * 0.15))
    inner_sats = set(train_sats[:n_inner])
    assert not (inner_sats & eval_sats), '内层验证集与评估集重叠！'
    sel = np.isin(sats, list(inner_sats))
    return meta.loc[sel, 'sample_idx'].values, meta.loc[sel, 'norad_id'].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=None,
                    help="权重目录；默认沿用 eval_alpha_sweep.CKPT_DIR；传空串回到训练损失最优")
    ap.add_argument('--alphas', default=','.join(f'{a:.1f}' for a in eas.ALPHAS))
    ap.add_argument('--boot', action='store_true', help='追加 α* vs 预设 0.3 的聚簇配对 bootstrap')
    args = ap.parse_args()
    if args.ckpt_dir is not None:
        eas.CKPT_DIR = args.ckpt_dir
    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    device = torch.device(eas.DEVICE)
    cond_mean = np.load(f'{DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    res_mean = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    res_std = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    idx, sat = inner_val_indices()
    print(f'[内层验证集] {len(idx)} 样本 / {len(set(sat))} 颗卫星'
          f'（权重: {eas.CKPT_DIR or "训练损失最优"}）')
    print('  注：该集合参与了早停，故指标对选中的检查点偏乐观；用于 α 之间的排序仍成立\n')

    rows, store = [], {}
    for a in alphas:
        models = eas.load_ensemble(eas.FAMILIES['alpha'], a, device)
        if not models:
            continue
        mse_s, mse_a, pk = eas.evaluate(models, idx, cond_mean, cond_std, res_mean, res_std, device)
        m = eas.compute_metrics(mse_s, mse_a, pk)
        m['alpha'] = a
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        store[a] = (mse_s, mse_a, pk)
        print(f"  α={a:.1f}: RMSE降幅={m['rmse_drop']:6.2f}%  优于SGP4={m['better_pct']:6.2f}%  "
              f"改善中位={m['imp_median']:+7.2f}%  严重退化(η<-20%)={m['severe20']:5.2f}%  "
              f"最差10%={m['worst10']:+8.2f}%  最差={m['worst1']:+9.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    tag = ('_' + os.path.basename(eas.CKPT_DIR.rstrip('/'))) if eas.CKPT_DIR else ''
    fn = os.path.join(OUTDIR, f'alpha_inner_val_metrics{tag}.csv')
    df.to_csv(fn, index=False)
    print(f'\n写出 {fn}')

    # ---- §5.9 选择规则：在风险约束内最大化 RMSE 降幅 ----
    print('\n=== 按 §5.9 规则在内层验证集上选 α* ===')
    print('约束类型        阈值      可行集                     α*     RMSE降幅')
    for d in RISK_DELTAS:
        ok = df[df['severe20'] <= d]
        if len(ok):
            b = ok.loc[ok['rmse_drop'].idxmax()]
            print(f'  严重退化 δ≤{d:>2}%   {d:>2}%   {sorted(ok["alpha"].tolist())}   '
                  f'{b["alpha"]:.1f}   {b["rmse_drop"]:.2f}%')
        else:
            print(f'  严重退化 δ≤{d:>2}%   {d:>2}%   空集')
    for w in RISK_W:
        ok = df[df['worst1'] >= -w]
        if len(ok):
            b = ok.loc[ok['rmse_drop'].idxmax()]
            print(f'  最差退化 W≤{w:>3}%  {w:>3}%   {sorted(ok["alpha"].tolist())}   '
                  f'{b["alpha"]:.1f}   {b["rmse_drop"]:.2f}%')
        else:
            print(f'  最差退化 W≤{w:>3}%  {w:>3}%   空集')

    # ---- 预设 0.3 vs 选定 α* 的逐项对照 ----
    if 0.3 in store:
        for cand in (0.2, 0.4, 0.6):
            if cand in store:
                r3 = df[df['alpha'] == 0.3].iloc[0]
                rc = df[df['alpha'] == cand].iloc[0]
                print(f'\n  预设 α=0.3 vs α={cand}: '
                      f"RMSE降幅 {r3['rmse_drop']:.2f}→{rc['rmse_drop']:.2f} | "
                      f"优于SGP4 {r3['better_pct']:.2f}→{rc['better_pct']:.2f} | "
                      f"严重退化 {r3['severe20']:.2f}→{rc['severe20']:.2f} | "
                      f"最差 {r3['worst1']:.2f}→{rc['worst1']:.2f}")

    if args.boot and 0.3 in store:
        best = df.loc[df['severe20'] <= 5, 'rmse_drop'].idxmax() if (df['severe20'] <= 5).any() else None
        cand = float(df.loc[best, 'alpha']) if best is not None else 0.2
        if cand in store and cand != 0.3:
            print(f'\n聚簇配对 bootstrap（α={cand} vs 预设 0.3，以内层验证卫星为单位）')
            pk = store[0.3][2]
            g = np.array([str(s) for s in sat])[pk >= THRESHOLD]

            def eta(x):
                return (1 - np.sqrt(x[1]) / (np.sqrt(x[0]) + 1e-8)) * 100
            ea = eta(store[cand])[pk >= THRESHOLD]
            eb = eta(store[0.3])[pk >= THRESHOLD]
            for name, f in (('优于SGP4比例(pp)', lambda v: 100 * np.mean(v > 0)),
                            ('严重退化比例P(η<-20%)(pp)', lambda v: 100 * np.mean(v < -20)),
                            ('CVaR10(pp)', lambda v: np.sort(v)[:max(1, int(0.10 * len(v)))].mean()),
                            ('最差单样本(pp)', lambda v: v.min())):
                d, lo, hi, p = eas.boot_cluster(ea, eb, g, f)
                print(f'  {name:>16}: {d:+8.2f}  95%CI [{lo:8.2f}, {hi:8.2f}]  p={p:.3f}')
    print('\n完成。把 eval_out/alpha_inner_val_metrics*.csv 发我，我把结论写进 §5.9 / §4.3。')


if __name__ == '__main__':
    main()
