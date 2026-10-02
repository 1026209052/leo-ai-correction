"""诊断：为什么"6模型一次调用"与"6次单模型调用再平均残差"给出不同的集成指标。

假定两者数学等价，因此若不一致，差异必然出在模型加载或数组对齐上。
放在 verify_all_tables.py 同一目录运行：
    python debug_ensemble.py
"""
import os

import numpy as np
import torch

import verify_all_tables as vt

device = torch.device(vt.DEVICE)
cm = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
cs = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)
idx = vt.load_val_indices()
print(f'[口径] CKPT_DIR={vt.CKPT_DIR!r}  验证样本={len(idx)}  SEEDS={vt.SEEDS}')


def eta_of(preds, sgps, trues):
    r_s = np.sqrt(np.mean((sgps - trues) ** 2, axis=(1, 2)))
    r_a = np.sqrt(np.mean((preds - trues) ** 2, axis=(1, 2)))
    return (1 - r_a / (r_s + 1e-8)) * 100.0


for tag, pfx in (('Transformer', vt.PREFIX['Transformer']),
                 ('Transformer+软掩码', vt.PREFIX['Transformer+SoftMask'])):
    print('\n' + '=' * 80)
    print(tag, ' <-', pfx + '*')
    models = vt.load_ensemble(pfx, 'transformer', device)
    print(f'  加载模型数 = {len(models)} （应为 {len(vt.SEEDS)}）')
    if not models:
        continue

    # 路径 A：一次调用，6 个模型（= verify_all_tables / export_eta 的路径）
    sgps, trues, predsA, resA = vt.evaluate_method(models, idx, cm, cs, rm, rs, device)

    # 路径 B：逐种子单独调用，再把**预测残差**平均（= export_eta_perseed 的路径）
    singles = []
    for k, m in enumerate(models):
        s, t, p, r = vt.evaluate_method([m], idx, cm, cs, rm, rs, device)
        singles.append(r)
        if k == 0:
            sgps0, trues0 = s, t
    resB = np.mean(np.array(singles), axis=0)

    print(f'  max|resA - resB|        = {np.abs(resA - resB).max():.6g}   （应为 0）')
    print(f'  max|sgpsA - sgps0|      = {np.abs(sgps - sgps0).max():.6g}   （应为 0）')
    peaks = np.max(np.abs(trues - sgps)[:, :, 0], axis=1)
    big = peaks >= vt.THRESHOLD
    print(f'  n_big                   = {int(big.sum())}   （应为 817）')
    for nm, rr in (('A 一次调用6模型', resA), ('B 单模型残差平均', resB)):
        e = eta_of(sgps + rr, sgps, trues)[big]
        print(f'  {nm}: 优于SGP4={100*np.mean(e>0):6.2f}%  '
              f'P(η<-20%)={100*np.mean(e<-20):5.2f}%  ({int(np.sum(e<-20))}/{len(e)})  '
              f'min={e.min():9.2f}%')
