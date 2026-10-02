"""导出 LSTM 的逐样本 η → per_sample_eta_lstm.csv（817 行大机动组）。

与 export_eta.py 同一套口径（复用 verify_all_tables 的加载与评估逻辑）：
  权重：论文口径 = inner_val_ckpt/*_innerval.pth（--ckpt-dir= 可切回训练损失最优）
  模型：LSTM（标准MSE，truth 管线）与 LSTM+软掩码（α=0.3，masked 管线）
  决策：Oracle 门控 τ=20 km；逐样本 RMSE 为双通道联合均方根（§4.2）

输出列与 c2_cluster_bootstrap.py 期望的一致：sample_idx, norad_id, eta_lstm, eta_lstm_softmask
用法（放在 verify_all_tables.py 同一目录）：
    python export_eta_lstm.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

THRESHOLD = vt.THRESHOLD


def metrics(eta):
    neg = eta[eta < 0]
    s = np.sort(eta)
    k = max(1, int(0.10 * len(eta)))
    return (100.0 * float(np.mean(eta > 0)),
            float(np.median(eta[eta > 0])) if (eta > 0).any() else np.nan,
            float(np.median(neg)) if len(neg) else np.nan,
            float(np.mean(s[:k])), float(s[0]), int(len(neg)))


def main():
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            vt.CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            vt.CKPT_DIR = sys.argv[i + 1]
    print(f"[口径] {'论文口径：内层验证最优' if vt.CKPT_DIR else '对照口径：训练损失最优'}")

    device = torch.device(vt.DEVICE)
    cond_mean = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    o_rm = np.load(f'{vt.DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    o_rs = np.load(f'{vt.DATA_DIR}/scaler_res_std.npy').astype(np.float32)
    m_rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    m_rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    idx_all = vt.load_val_indices()
    print(f'验证样本数: {len(idx_all)}')

    out = None
    # (标签, prefix, res_mean, res_std) —— LSTM 基线与软掩码各自的反归一化口径
    for tag, pfx, rm, rs in (('eta_lstm', vt.PREFIX['LSTM'], o_rm, o_rs),
                             ('eta_lstm_softmask', vt.PREFIX['LSTM+SoftMask'], m_rm, m_rs)):
        print(f'\n评估 {tag}  ←  {pfx}*')
        models = vt.load_ensemble(pfx, 'lstm', device)
        if not models:
            raise SystemExit(f'  没有可用模型：{pfx}*')
        sgps, trues, preds, _ = vt.evaluate_method(
            models, idx_all, cond_mean, cond_std, rm, rs, device)

        peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
        big = peaks >= THRESHOLD
        r_sgp4 = np.sqrt(np.mean((sgps - trues) ** 2, axis=(1, 2)))
        r_ai = np.sqrt(np.mean((preds - trues) ** 2, axis=(1, 2)))
        eta = (1.0 - r_ai / (r_sgp4 + 1e-8)) * 100.0
        bt, im, dm, w10, w1, ndeg = metrics(eta[big])
        print(f'  大机动组 n={int(big.sum())}  优于SGP4={bt:.2f}%  改善中位={im:+.2f}%  '
              f'退化中位={dm:+.2f}%  最差10%={w10:+.2f}%  最差={w1:+.2f}%  退化样本数={ndeg}')

        sub = pd.DataFrame({'sample_idx': np.asarray(idx_all)[big], tag: eta[big]})
        out = sub if out is None else out.merge(sub, on='sample_idx', how='inner')

    meta = pd.read_csv(f'{vt.DATA_DIR}/sample_metadata.csv')
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    out = out.merge(meta[['sample_idx', 'norad_id']], on='sample_idx', how='left')
    out = out[['sample_idx', 'norad_id', 'eta_lstm', 'eta_lstm_softmask']] \
        .sort_values('sample_idx')
    fn = 'per_sample_eta_lstm.csv'
    out.to_csv(fn, index=False)
    print(f'\n写出 {fn}（{len(out)} 行，列 {list(out)}）')
    print('  → 拷到 figures/data/per_sample_eta_lstm.csv')
    print('  → 再跑 analysis/c2_cluster_bootstrap.py 得到 LSTM 的聚簇 CI 与 p 值')


if __name__ == '__main__':
    main()
