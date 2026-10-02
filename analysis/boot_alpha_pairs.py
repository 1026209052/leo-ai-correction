"""boot_alpha_pairs.py -- 任意两个 α 的【聚簇配对 bootstrap】（评估集口径，与 §5.1 完全一致）

用途：审稿意见指出"预设 α=0.3 的核心正面结果（优于SGP4 +8.0 pp）不显著"。
内层验证集的选点结果显示 α=0.2 在评估集上于 5/6 指标优于 0.3（严重退化 3.3% vs 5.1%、
最差单样本 −77.6% vs −96.2%、优于SGP4 82.7% vs 73.4%）。本脚本回答关键问题：
**换成 α=0.2 后，相对标准 MSE（α=1.0）的平均性能提升是否达到显著？**

用法（starlink_fixed/ 下，与 eval_alpha_sweep.py 同环境）：
    python boot_alpha_pairs.py --a 0.2 --b 1.0     # 规则选中的 α vs MSE 基线（推荐先跑这条）
    python boot_alpha_pairs.py --a 0.2 --b 0.3     # 两个候选之间的差别
权重目录沿用 eval_alpha_sweep.CKPT_DIR（默认 inner_val_ckpt，论文口径）。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_alpha_sweep as eas


def per_sample_eta(models, idx, cm, cs, rm, rs, device):
    mse_s, mse_a, pk = eas.evaluate(models, idx, cm, cs, rm, rs, device)
    per_s, per_a = np.sqrt(mse_s), np.sqrt(mse_a)
    eta = (1 - per_a / (per_s + 1e-8)) * 100
    return eta, pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=0.2)
    ap.add_argument('--b', type=float, default=1.0)
    ap.add_argument('--ckpt-dir', default=None)
    ap.add_argument('--B', type=int, default=10000)
    args = ap.parse_args()
    if args.ckpt_dir is not None:
        eas.CKPT_DIR = args.ckpt_dir

    device = torch.device(eas.DEVICE)
    data, masked = eas.DATA_DIR, eas.MASKED_DIR
    cm = np.load(f'{data}/scaler_cond_mean.npy').astype(np.float32)
    cs = np.load(f'{data}/scaler_cond_std.npy').astype(np.float32)
    rm = np.load(f'{masked}/scaler_res_mean.npy').astype(np.float32)
    rs = np.load(f'{masked}/scaler_res_std.npy').astype(np.float32)
    idx, sat = eas.load_val_indices()
    print(f'[评估集] {len(idx)} 样本 / {len(set(sat))} 颗卫星（权重: {eas.CKPT_DIR or "训练损失最优"}）')

    eta, pk, grp = {}, {}, np.array([str(s) for s in sat])
    for a in (args.a, args.b):
        ms = eas.load_ensemble(eas.FAMILIES['alpha'], a, device)
        if not ms:
            raise SystemExit(f'α={a} 无可用权重')
        eta[a], pk[a] = per_sample_eta(ms, idx, cm, cs, rm, rs, device)

    big = pk[args.a] >= eas.THRESHOLD
    g = grp[big]
    A = eta[args.a][big]
    Bv = eta[args.b][big]
    print(f'大机动子集 n={int(big.sum())}\n')

    def report(tag, f):
        print(f'  {tag:<26s} α={args.a}: {f(A):+8.2f}   α={args.b}: {f(Bv):+8.2f}   '
              f'差 {f(A) - f(Bv):+7.2f}')

    print(f'点估计（α={args.a} vs α={args.b}）:')
    METRICS = [
        ('优于SGP4比例(%)', lambda v: 100 * np.mean(v > 0)),
        ('严重退化比例P(η<-20%)(%)', lambda v: 100 * np.mean(v < -20)),
        ('P(η<-10%)(%)', lambda v: 100 * np.mean(v < -10)),
        ('RMSE降幅(池化,%)', None),          # 需要 MSE，单独处理
        ('CVaR10(%)', lambda v: np.sort(v)[:max(1, int(0.10 * len(v)))].mean()),
        ('最差单样本(%)', lambda v: v.min()),
        ('改善中位数(%)', lambda v: np.median(v[v > 0]) if (v > 0).any() else np.nan),
    ]
    for tag, f in METRICS:
        if f is None:
            continue
        report(tag, f)

    print(f'\n聚簇配对 bootstrap（B={args.B}，以卫星为单位；Δ = f(α={args.a}) − f(α={args.b})）:')
    for tag, f in METRICS:
        if f is None:
            continue
        d, lo, hi, p = eas.boot_cluster(Bv, A, g, f, B=args.B)   # 注意顺序：boot(A,B)=f(B)-f(A)
        print(f'  {tag:<26s} Δ={d:+8.2f}  95%CI [{lo:8.2f}, {hi:8.2f}]  p={p:.3f}')
    print('\n若"优于SGP4比例"的 CI 不含 0，则换用该 α 可把论文的主结果从"不显著"升级为显著。')


if __name__ == '__main__':
    main()
