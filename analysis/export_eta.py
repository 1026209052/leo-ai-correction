"""导出逐样本 η → figures/data/fig2_cdf.csv（817 行大机动组）。

为什么需要它：老的 fig2_data_export.py 指向旧模型（error_predictor_v4_seed* /
error_predictor_seed*），口径已过期。本脚本 **直接复用 ../verify_all_tables.py**
的加载与评估逻辑，因此口径与之完全一致，不会出现"图2 和表4 对不上"的问题：

  - 权重：论文口径 = 内层验证最优（inner_val_ckpt/*_innerval.pth）；用 --ckpt-dir= 切回训练损失最优
  - 模型：Transformer(α=1.0) 与 Transformer+软掩码(α=0.3)，6 种子集成
  - 反归一化：masked res scaler（α 家族全在 masked 管线训练）
  - 决策：Oracle 门控，阈值 τ=20 km
  - 逐样本 RMSE：高度+沿轨 **双通道联合** 均方根（§4.2 定义）
  - 样本集：peak≥20 km 的 817 个大机动样本

用法（放在 verify_all_tables.py 同一目录）：
    python export_eta.py                     # 论文口径
    python export_eta.py --ckpt-dir=         # 对照口径（训练损失最优）

输出：
    fig2_cdf.csv        列 = sample_idx, eta_mse, eta_soft   （817 行）
    终端同时打印六项指标，可直接与 verify_all_tables 的表4 对撞。
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt          # 复用同一套加载/评估逻辑

THRESHOLD = vt.THRESHOLD


def per_sample_rmse(traj, truth):
    """§4.2 定义：高度与沿轨两通道的联合均方根。"""
    return np.sqrt(np.mean((traj - truth) ** 2, axis=(1, 2)))


def metrics(eta):
    n = len(eta)
    neg = eta[eta < 0]
    s = np.sort(eta)
    k = max(1, int(0.10 * n))
    return dict(n=n,
                better_pct=100.0 * float(np.mean(eta > 0)),
                imp_median=float(np.median(eta[eta > 0])) if (eta > 0).any() else np.nan,
                deg_median=float(np.median(neg)) if len(neg) else np.nan,
                worst10=float(np.mean(s[:k])),
                worst1=float(s[0]))


def main():
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            vt.CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            vt.CKPT_DIR = sys.argv[i + 1]
    print(f"[口径] {'论文口径：内层验证最优' if vt.CKPT_DIR else '对照口径：训练损失最优'}"
          + (f"（{vt.CKPT_DIR}）" if vt.CKPT_DIR else ""))

    device = torch.device(vt.DEVICE)
    cond_mean = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    m_rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    m_rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    idx_all = vt.load_val_indices()
    print(f'验证样本数: {len(idx_all)}')

    # ---- 同时导出**未门控**的全体验证集 η（1404 行），供表7 的 τ 扫描使用 ----
    # 表7 在 τ=15 km 时的介入集合（peak≥15）比 817 多出 62 个样本，
    # 这些样本的 η 必须用未门控的预测计算。
    wide = None

    out = None
    for tag, pfx in (('eta_mse', vt.PREFIX['Transformer']),
                     ('eta_soft', vt.PREFIX['Transformer+SoftMask'])):
        print(f'\n评估 {tag}  ←  {pfx}*')
        models = vt.load_ensemble(pfx, 'transformer', device)
        if not models:
            raise SystemExit(f'  没有可用模型：{pfx}*（检查 --ckpt-dir 目录）')
        sgps, trues, preds, _ = vt.evaluate_method(
            models, idx_all, cond_mean, cond_std, m_rm, m_rs, device)

        peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
        big = peaks >= THRESHOLD
        eta = (1.0 - per_sample_rmse(preds, trues) /
               (per_sample_rmse(sgps, trues) + 1e-8)) * 100.0
        m = metrics(eta[big])
        print(f'  大机动组 n={m["n"]}  优于SGP4={m["better_pct"]:.2f}%  '
              f'改善中位={m["imp_median"]:+.2f}%  退化中位={m["deg_median"]:+.2f}%  '
              f'最差10%={m["worst10"]:+.2f}%  最差={m["worst1"]:+.2f}%')

        sub = pd.DataFrame({'sample_idx': np.asarray(idx_all)[big], tag: eta[big]})
        out = sub if out is None else out.merge(sub, on='sample_idx', how='inner')

        # 未门控版本：全部 1404 个样本。做法是把模块级 THRESHOLD 暂时设为极负值，
        # 使 evaluate_method 内部的
        # `if peak < THRESHOLD: avg_res = 0` 永不成立 —— 这样不依赖任何版本新增的
        # gate 参数（服务器的 verify_all_tables.py 可能是旧版）。
        # 注意：本脚本里两个方法都在 masked 管线上（α 家族），故统一用 m_rm/m_rs。
        _saved_th = vt.THRESHOLD
        vt.THRESHOLD = -1e9
        try:
            sgps_u, trues_u, preds_u, _ = vt.evaluate_method(
                models, idx_all, cond_mean, cond_std, m_rm, m_rs, device)
        finally:
            vt.THRESHOLD = _saved_th
        eta_u = (1.0 - per_sample_rmse(preds_u, trues_u) /
                 (per_sample_rmse(sgps_u, trues_u) + 1e-8)) * 100.0
        peak_u = np.max(np.abs(trues_u - sgps_u)[:, :, 0], axis=1)
        w = pd.DataFrame({'sample_idx': np.asarray(idx_all), 'peak': peak_u,
                          'sgp4_rmse': per_sample_rmse(sgps_u, trues_u), tag: eta_u})
        wide = w if wide is None else wide.merge(w.drop(columns=['peak', 'sgp4_rmse']),
                                                 on='sample_idx', how='inner')

    fn = 'fig2_cdf.csv'
    out = out.sort_values('sample_idx')
    out.to_csv(fn, index=False)
    print(f'\n写出 {fn}（{len(out)} 行，列 {list(out)}）')
    fn2 = 'fig2_cdf_all.csv'
    wide.sort_values('sample_idx').to_csv(fn2, index=False)
    print(f'写出 {fn2}（{len(wide)} 行，列 {list(wide)}）← 未门控全量，供表7 的 τ 扫描')
    print('  → 拷到 figures/data/fig2_cdf.csv，再跑 figures/make_fig2_data.py 重绘 图2')
    print('  → 把同一份文件给我，我做独立复算 + 配对诊断 + 聚簇 bootstrap')


if __name__ == '__main__':
    main()
