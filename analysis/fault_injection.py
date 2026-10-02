"""
fault_injection.py -- Fail-Safe 门控的故障注入测试（回应 RESS 审稿意见 #2 / #4）

目的
----
回答："当输入偏离训练分布（TLE 质量下降、观测噪声、基线偏差）时，三层架构是否仍然
无害？" 做法：在**已预处理样本**上注入三类故障，比较 Oracle 门控 / 代理门控下的指标，
检验门控的"拒绝"行为是否稳定。**不需要重新训练**（只改推理输入）。

三种注入
  1. cond_noise : 条件窗口加高斯噪声    cond' = cond + N(0, level * cond_std)
                  （观测/拟合噪声变大；会改变代理门控看到的高度极差）
  2. cond_bias  : 条件窗口高度通道加常数偏置    cond'[:,0] += level (km)
                  （定轨系统偏差；**不改变**高度极差，故代理门控决定不变）
  3. sgp4_bias  : SGP4 基线加常数偏差    sgp4' = sgp4 + level (km)
                  （TLE 质量下降 → 基线本身变差、AI 修正量"过期"；会改变 Oracle 判据）

注入水平
  cond_noise: {0, 0.05, 0.1, 0.2, 0.5}   （× 通道标准差；0 = 自检基线）
  cond_bias : {0, 5, 10, 20, 50}         （km，加在高度通道）
  sgp4_bias : {0, 0.5, 1, 2, 5}          （km，加在高度通道）

指标（评估集固定为**原始**大机动组 n=817，保证样本集可比；eta 相对**注入后**的 SGP4）
  介入率、优于SGP4比例、严重退化比例 P(eta<-20%)、最差10%、最差单样本
  关键判据：门控是否仍能拒绝有害修正 —— 最差单样本与严重退化是否保持有界。

用法
  python fault_injection.py                    # 三种注入全套
  python fault_injection.py --mode cond_noise
  python fault_injection.py --check            # level=0 须复现 表5（MSE 65.5/9.3；软掩码 82.7/3.3）

输出
  figures/data/fault_injection.csv + 终端 markdown 表
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

PROXY_THR = float(os.environ.get('PROXY_THR', '15.0'))
LEVELS = {
    'cond_noise': [0.0, 0.05, 0.1, 0.2, 0.5],
    'cond_bias': [0.0, 5.0, 10.0, 20.0, 50.0],
    'sgp4_bias': [0.0, 0.5, 1.0, 2.0, 5.0],
}
FAMILIES = [('MSE', lambda a: a.PREFIX['Transformer']),
            ('SoftMask', lambda a: a.PREFIX['Transformer+SoftMask'])]
REF_TABLE5 = {'MSE': (65.5, 9.3), 'SoftMask': (82.7, 3.3)}
NOISE_SEED = 20260926


def per_sample_rmse(traj, truth):
    """归约最后两个轴（时间 x 通道）。

    单样本 (T,2) -> 标量；批量 (N,T,2) -> (N,)。本脚本是逐样本调用（传 (T,2)），
    而 verify_all_tables 是批量调用，用 axis=range(ndim-2, ndim) 同时兼容两种形状。
    """
    diff = np.asarray(traj, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    axes = tuple(range(diff.ndim - 2, diff.ndim))
    return np.sqrt(np.mean(diff ** 2, axis=axes))


def out_dir():
    """把 CSV 写到与脚本同级的 figures/data（兼容从项目根目录或 analysis/ 运行）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'figures', 'data'),
                 os.path.join(here, '..', 'figures', 'data')):
        d = os.path.normpath(cand)
        if os.path.isdir(os.path.dirname(d)):
            os.makedirs(d, exist_ok=True)
            return d
    d = os.path.join(here, 'figures', 'data')
    os.makedirs(d, exist_ok=True)
    return d

def metrics(eta):
    s = np.sort(eta)
    k = max(1, int(0.10 * len(eta)))
    return dict(n=len(eta),
                better_pct=100.0 * float(np.mean(eta > 0)),
                severe20=100.0 * float(np.mean(eta < -20.0)),
                worst10=float(np.mean(s[:k])),
                worst1=float(s[0]))


def inject(mode, level, cond, sgp4, cond_std, rng):
    """返回 (cond', sgp4')。"""
    cond2, sgp42 = cond.copy(), sgp4.copy()
    if mode == 'cond_noise' and level > 0:
        cond2 = cond + rng.normal(0.0, level, size=cond.shape) * cond_std
    elif mode == 'cond_bias' and level > 0:
        cond2[:, 0] = cond[:, 0] + level
    elif mode == 'sgp4_bias' and level > 0:
        sgp42[:, 0] = sgp4[:, 0] + level
    return cond2, sgp42


def main():
    only = None
    for i, a in enumerate(sys.argv):
        if a.startswith('--mode='):
            only = a.split('=', 1)[1]
        elif a == '--mode' and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
    check = '--no-check' not in sys.argv      # 默认自检：防止用错 alpha 权重
    device = torch.device(vt.DEVICE)

    cond_mean = np.load(f'{vt.DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{vt.DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    m_rm = np.load(f'{vt.MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    m_rs = np.load(f'{vt.MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    idx_all = np.asarray(vt.load_val_indices())
    meta = pd.read_csv(f'{vt.MASKED_DIR}/sample_metadata.csv')
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sat_map = dict(zip(meta['sample_idx'].values, meta['norad_id'].values))
    sat_of = np.array([sat_map.get(int(i), '?') for i in idx_all])

    raw = {}
    for j, idx in enumerate(idx_all):
        raw[idx] = (np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_cond.npy'),
                    np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_true.npy'),
                    np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_sgp4.npy'))
    peak_orig = np.array([np.max(np.abs(raw[i][1] - raw[i][2])[:, 0]) for i in idx_all])
    big = peak_orig >= vt.THRESHOLD
    print(f'验证样本数 {len(idx_all)} | 原始大机动组 n={int(big.sum())} | 代理阈值 {PROXY_THR} km')
    print(f'[口径] CKPT_DIR={vt.CKPT_DIR!r} | 软掩码前缀={vt.PREFIX["Transformer+SoftMask"]}*  '
          f'(SM_ALPHA={os.environ.get("SM_ALPHA", "0.3")})')
    print('[口径] 论文表5 期望（SM_ALPHA=0.2）= 82.7% / 3.3%；'
          '若 level=0 给出 73.4% / 5.1% 说明用错 alpha 权重 -> 加 SM_ALPHA=0.2 重跑')

    models = {}
    for name, pfx_fn in FAMILIES:
        ms = vt.load_ensemble(pfx_fn(vt), 'transformer', device)
        if not ms:
            raise SystemExit(f'没有可用模型: {pfx_fn(vt)}*')
        models[name] = ms
        print(f'  {name}: {len(ms)} 个模型')

    def predict(cond_norm, name):
        t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return np.mean([m(t).detach().cpu().numpy().squeeze() * m_rs + m_rm
                            for m in models[name]], axis=0)

    rows = []
    modes = [only] if only else list(LEVELS)
    for mode in modes:
        for level in LEVELS[mode]:
            rng = np.random.RandomState(NOISE_SEED)
            r_sgp4 = np.empty(len(idx_all)); peak_inj = np.empty(len(idx_all))
            proxy_inj = np.empty(len(idx_all)); res = {k: np.empty((len(idx_all), 864, 2))
                                                       for k in models}
            sgp4_inj = np.empty((len(idx_all), 864, 2))
            for j, idx in enumerate(idx_all):
                cond, true_tgt, sgp4_tgt = raw[idx]
                cond2, sgp42 = inject(mode, level, cond, sgp4_tgt, cond_std, rng)
                sgp4_inj[j] = sgp42
                cond_norm = (cond2 - cond_mean) / (cond_std + 1e-8)
                for name in models:
                    res[name][j] = predict(cond_norm, name)
                r_sgp4[j] = per_sample_rmse(sgp42, true_tgt)
                peak_inj[j] = np.max(np.abs(true_tgt - sgp42)[:, 0])
                proxy_inj[j] = float(np.ptp(cond2[:, 0]))

            oracle_lab = peak_inj >= vt.THRESHOLD
            proxy_lab = proxy_inj >= PROXY_THR
            tp = int(np.sum(proxy_lab & oracle_lab)); fp = int(np.sum(proxy_lab & ~oracle_lab))
            fn = int(np.sum(~proxy_lab & oracle_lab))
            print(f'\n[{mode} level={level}] 注入后大机动 {int(oracle_lab.sum())} | '
                  f'代理 TP={tp} FP={fp} FN={fn} 介入率={100.0*proxy_lab.mean():.2f}%')

            for name in models:
                r_full = np.array([per_sample_rmse(sgp4_inj[j] + res[name][j],
                                                   raw[idx_all[j]][1])
                                   for j in range(len(idx_all))])
                for tag, mask in (('Oracle', oracle_lab), ('Proxy', proxy_lab)):
                    pred = np.where(mask, r_full, r_sgp4)
                    eta = (1.0 - pred / (r_sgp4 + 1e-8)) * 100.0
                    m = metrics(eta[big])
                    rows.append(dict(mode=mode, level=level, family=name, gate=tag,
                                     intervened_pct=100.0 * float(mask[big].mean()), **m))
                    print(f'   {name:9s} {tag:6s} 介入率={100.0*mask[big].mean():5.1f}% '
                          f'优于SGP4={m["better_pct"]:5.1f}% 严重退化={m["severe20"]:5.2f}% '
                          f'最差10%={m["worst10"]:7.2f}% 最差={m["worst1"]:8.2f}%')

    if check:
        ok = True
        for name, (b_exp, s_exp) in REF_TABLE5.items():
            r = [x for x in rows if x['mode'] == 'cond_noise' and x['level'] == 0.0
                 and x['family'] == name and x['gate'] == 'Oracle'][0]
            good = abs(r['better_pct'] - b_exp) < 0.2 and abs(r['severe20'] - s_exp) < 0.2
            ok &= good
            print(f'  [自检 {name}] level=0 优于SGP4={r["better_pct"]:.2f}%（表5 {b_exp}）'
                  f' 严重退化={r["severe20"]:.2f}%（表5 {s_exp}）-> {"OK" if good else "MISMATCH"}')
        print('  [自检] 结论:', 'OK（口径与表5 一致）' if ok else
          'MISMATCH —— 若为 73.4%/5.1% 说明加载的是 α=0.3 权重，请用 SM_ALPHA=0.2 重跑；'
          '否则检查数据目录与归一化口径')

    df = pd.DataFrame(rows)
    _dir = out_dir()
    df.to_csv(os.path.join(_dir, 'fault_injection.csv'), index=False)
    print('\n写出 ' + os.path.join(out_dir(), 'fault_injection.csv'))
    sm = df[(df.family == 'SoftMask') & (df.gate == 'Oracle')]
    print('\n软掩码 / Oracle 门控（大机动组 n=817）：')
    print('| 注入 | 水平 | 介入率 | 优于SGP4 | 严重退化 | 最差10% | 最差单样本 |')
    print('|---|---|---|---|---|---|---|')
    for _, r in sm.iterrows():
        print(f'| {r["mode"]} | {r["level"]} | {r.intervened_pct:.1f}% | {r.better_pct:.1f}% | '
              f'{r.severe20:.2f}% | {r.worst10:.1f}% | {r.worst1:.1f}% |')


if __name__ == '__main__':
    main()
