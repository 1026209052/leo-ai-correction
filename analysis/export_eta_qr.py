"""导出分位数回归（QR）四行的逐样本 η 与新主指标（表4 的 QR 行）。

QR 是**单模型**（seed=42），且训练时为标准 MES 之外的 pinball 损失；
按 train_quantile_regression.py 的配置：
  权重文件  error_predictor_qr_seed42_q{50,75,90,95}.pth
  数据/归一化  preprocessed_samples（truth 管线，非 masked）
其余口径与 export_eta.py 一致（Oracle 门控 τ=20 km、双通道联合 RMSE、817 大机动组）。

用法（放在 verify_all_tables.py 同一目录）：
    python export_eta_qr.py                     # 论文口径：内层验证最优
    python export_eta_qr.py --ckpt-dir=          # 该家族无诊断权重，等价于训练损失最优
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

THRESHOLD = vt.THRESHOLD
QUANTILES = [50, 75, 90, 95]
SEED = 42


def load_qr(q, device):
    """QR 单模型：train_quantile_regression.py 的落盘规则。"""
    stem = f'{vt.CKPT_DIR}/error_predictor_qr_seed{SEED}_q{q:02d}' if vt.CKPT_DIR \
        else f'error_predictor_qr_seed{SEED}_q{q:02d}'
    for suf in ('.pth', '_best.pth'):
        if os.path.exists(stem + suf):
            m = vt.ErrorPredictor().to(device)
            m.load_state_dict(torch.load(stem + suf, map_location=device))
            m.eval()
            print(f'  载入 {stem + suf}')
            return [m]
    print(f'  [!] 未找到 {stem}.pth')
    return []


def metrics(eta):
    s = np.sort(eta)
    neg = eta[eta < 0]
    k = max(1, int(0.10 * len(eta)))
    return dict(n=len(eta),
                better_pct=100.0 * float(np.mean(eta > 0)),
                imp_median=float(np.median(eta[eta > 0])) if (eta > 0).any() else np.nan,
                deg_median=float(np.median(neg)) if len(neg) else np.nan,
                worst10=float(np.mean(s[:k])), worst1=float(s[0]),
                severe10=100.0 * float(np.mean(eta < -10)),
                severe20=100.0 * float(np.mean(eta < -20)),
                severe30=100.0 * float(np.mean(eta < -30)))


def main():
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            vt.CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            vt.CKPT_DIR = sys.argv[i + 1]
    print(f"[口径] CKPT_DIR = {vt.CKPT_DIR!r}（QR 家族通常只有训练损失最优的权重）")

    device = torch.device(vt.DEVICE)
    cond_mean = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    rm = np.load(f'{vt.DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    rs = np.load(f'{vt.DATA_DIR}/scaler_res_std.npy').astype(np.float32)

    idx_all = vt.load_val_indices()
    print(f'验证样本数: {len(idx_all)}')

    out = None
    print('\n表4 的 QR 行（单模型 seed=42）：')
    print(f"{'τ':>6} | {'优于SGP4':>9} | {'改善中位':>9} | {'退化中位':>9} | "
          f"{'最差10%':>9} | {'最差':>9} | {'P(η<-20%)':>10}")
    for q in QUANTILES:
        models = load_qr(q, device)
        if not models:
            continue
        sgps, trues, preds, _ = vt.evaluate_method(
            models, idx_all, cond_mean, cond_std, rm, rs, device)
        peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
        big = peaks >= THRESHOLD
        r_sgp4 = np.sqrt(np.mean((sgps - trues) ** 2, axis=(1, 2)))
        r_ai = np.sqrt(np.mean((preds - trues) ** 2, axis=(1, 2)))
        eta = (1.0 - r_ai / (r_sgp4 + 1e-8)) * 100.0
        m = metrics(eta[big])
        print(f"{q/100:>6.2f} | {m['better_pct']:8.2f}% | {m['imp_median']:+8.2f}% | "
              f"{m['deg_median']:+8.2f}% | {m['worst10']:+8.2f}% | {m['worst1']:+8.2f}% | "
              f"{m['severe20']:9.2f}%")
        sub = pd.DataFrame({'sample_idx': np.asarray(idx_all)[big], f'eta_q{q}': eta[big]})
        out = sub if out is None else out.merge(sub, on='sample_idx', how='inner')

    if out is not None:
        fn = 'qr_eta.csv'
        out.sort_values('sample_idx').to_csv(fn, index=False)
        print(f'\n写出 {fn}（{len(out)} 行，列 {list(out)}）')


if __name__ == '__main__':
    main()
