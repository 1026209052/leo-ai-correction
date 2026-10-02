"""逐种子（单模型）指标 → 附录B 表10 的"均值±标准差"。

4 个方法 × 6 种子，每个种子单独评估（不集成），指标含：
  优于SGP4比例 / 严重退化比例 P(η<-20%) / 最差单样本退化
并同时打印集成值（应等于表4），便于对照。

用法（放在 verify_all_tables.py 同一目录）：
    python export_eta_perseed.py
输出：per_seed_metrics.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

THRESHOLD = vt.THRESHOLD


def load_one(prefix, seed, model_type, device):
    """只加载指定种子的单模型。"""
    stem = f'{vt.CKPT_DIR}/{prefix}{seed}' if vt.CKPT_DIR else f'{prefix}{seed}'
    cands = ([f'{stem}_innerval.pth'] if vt.CKPT_DIR else [f'{stem}.pth']) + [f'{stem}_best.pth']
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        return None
    m = (vt.ErrorPredictor() if model_type == 'transformer' else vt.LSTMPredictor()).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def metrics(eta):
    s = np.sort(eta)
    return (100.0 * float(np.mean(eta > 0)),
            100.0 * float(np.mean(eta < -20)),
            float(s[0]))


def main():
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            vt.CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            vt.CKPT_DIR = sys.argv[i + 1]
    print(f"[口径] {'论文口径：内层验证最优' if vt.CKPT_DIR else '对照口径：训练损失最优'}")

    device = torch.device(vt.DEVICE)
    cm = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cs = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    o_rm = np.load(f'{vt.DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    o_rs = np.load(f'{vt.DATA_DIR}/scaler_res_std.npy').astype(np.float32)
    m_rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    m_rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)
    idx_all = vt.load_val_indices()

    # 反归一化 scaler 必须与各方法的训练管线一致：
    #   Transformer(α=1.0) 与 Transformer+软掩码 都在 masked 管线上 → masked scaler
    #   LSTM 基线用 truth 管线、LSTM+软掩码用 masked 管线
    METHODS = [('Transformer', vt.PREFIX['Transformer'], 'transformer', m_rm, m_rs),
               ('Transformer+软掩码', vt.PREFIX['Transformer+SoftMask'], 'transformer', m_rm, m_rs),
               ('LSTM', vt.PREFIX['LSTM'], 'lstm', o_rm, o_rs),
               ('LSTM+软掩码', vt.PREFIX['LSTM+SoftMask'], 'lstm', m_rm, m_rs)]

    # 参考值：由 verify_all_tables.py（表4）得到的 6 种子**集成**结果，用于自检
    REF_ENS = {'Transformer': (65.5, 9.3, -199.8),
               'Transformer+软掩码': (73.4, 5.1, -96.2),
               'LSTM': (80.0, 12.6, -602.6),
               'LSTM+软掩码': (78.8, 5.1, -208.5)}

    rows, ens = [], {}
    for name, pfx, mtype, rm, rs in METHODS:
        per = {'better': [], 'severe20': [], 'worst1': []}
        res_list, sgp4_ref, true_ref = [], None, None
        for seed in vt.SEEDS:
            m = load_one(pfx, seed, mtype, device)
            if m is None:
                print(f'  [!] 缺少 {pfx}{seed}')
                continue
            sgps, trues, preds, resids = vt.evaluate_method([m], idx_all, cm, cs, rm, rs, device)
            peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
            big = peaks >= THRESHOLD
            r_s = np.sqrt(np.mean((sgps - trues) ** 2, axis=(1, 2)))
            r_a = np.sqrt(np.mean((preds - trues) ** 2, axis=(1, 2)))
            eta = (1 - r_a / (r_s + 1e-8)) * 100.0
            b, s20, w1 = metrics(eta[big])
            per['better'].append(b)
            per['severe20'].append(s20)
            per['worst1'].append(w1)
            res_list.append(resids)                    # 保存各模型（已反归一化）的预测残差
            sgp4_ref, true_ref = sgps, trues
        if not per['better']:
            continue
        # ★ 正确的集成：先对各模型的**预测残差**取算术平均，再算 η（与 verify_all_tables 一致）；
        #   绝不能对 η 求平均（那不是集成）
        mean_res = np.mean(np.array(res_list), axis=0)
        preds_ens = sgp4_ref + mean_res
        peaks = np.max(np.abs(true_ref - sgp4_ref)[:, :, 0], axis=1)
        big = peaks >= THRESHOLD
        r_s = np.sqrt(np.mean((sgp4_ref - true_ref) ** 2, axis=(1, 2)))
        r_a = np.sqrt(np.mean((preds_ens - true_ref) ** 2, axis=(1, 2)))
        # ★ 必须与逐种子一样只在大机动组上统计 [big]！
        #   漏掉 [big] 会把 587 个被门控置零的样本也算进去：由于 (r_s+1e-8) 的守卫，
        #   这些样本的 η 是极小的正数（≈+1e-6/r_s），会被计成“优于SGP4”，
        #   从而虚高胜率、并让严重退化比例的**分母**从 817 变成 1404。
        ens[name] = metrics(((1 - r_a / (r_s + 1e-8)) * 100.0)[big])
        rows.append(dict(method=name, n_seeds=len(per['better']),
                         better_mean=np.mean(per['better']), better_std=np.std(per['better']),
                         severe20_mean=np.mean(per['severe20']), severe20_std=np.std(per['severe20']),
                         worst1_mean=np.mean(per['worst1']), worst1_std=np.std(per['worst1']),
                         ens_better=ens[name][0], ens_severe20=ens[name][1], ens_worst1=ens[name][2]))
        ref = REF_ENS.get(name, (float('nan'),) * 3)
        ok = all(abs(a - b) < 0.6 for a, b in zip(ens[name], ref))
        print(f'\n{name}   [自检 vs 表4: {"MATCH" if ok else "MISMATCH"}]')
        print(f'  优于SGP4比例   单模型 {np.mean(per["better"]):5.1f} ± {np.std(per["better"]):4.1f}%   集成 {ens[name][0]:5.2f}%')
        print(f'  严重退化比例   单模型 {np.mean(per["severe20"]):5.2f} ± {np.std(per["severe20"]):4.2f}%   集成 {ens[name][1]:5.2f}%')
        print(f'  最差单样本退化 单模型 {np.mean(per["worst1"]):6.1f} ± {np.std(per["worst1"]):5.1f}%   集成 {ens[name][2]:6.2f}%')
        print(f'  （表4 参考: {ref[0]}% / {ref[1]}% / {ref[2]}%）')

    fn = 'per_seed_metrics.csv'
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f'\n写出 {fn}')
    print('  → 表10：方法 | 优于SGP4比例（均值±标准差） | 严重退化比例（均值±标准差） | 最差单样本退化（均值±标准差）')


if __name__ == '__main__':
    main()
