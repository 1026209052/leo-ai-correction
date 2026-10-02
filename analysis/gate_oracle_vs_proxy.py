"""
gate_oracle_vs_proxy.py -- Oracle 门控 vs 代理门控的性能对比（回应 RESS 审稿意见 #3）

背景
----
论文 5.5.3 节已给出代理门控（条件窗口高度极差 >= 15 km）相对 Oracle 门控的判别质量
（精确率 86.7%、召回率 96.0%、F1 91.1%）与介入率（64.39% vs 58.2%），但没有给出
**两种门控下的性能对比**。审稿人可能质疑："安全阀是不是在知道答案之后才启动的？"
本脚本补齐这一对比，**不需要重新训练**（只在决策层重算）。

口径（与 verify_all_tables.py 完全一致）
  - 权重：内层验证最优（inner_val_ckpt/*_innerval.pth），6 种子集成
  - cond 用 truth scaler 归一化；预测残差用 masked res scaler 反归一化
  - 逐样本 eta = 1 - RMSE(pred)/RMSE(SGP4)（高度 + 沿轨双通道联合）
  - Oracle 门控：max|true_res[:,0]| >= 20 km 才介入（事后真值）
  - 代理门控：条件窗口(288 步)高度极差 >= PROXY_THR 才介入（可观测）

输出
  - 代理阈值扫描表（TP/FP/FN/TN、precision/recall/F1）→ 复核 91.1%
  - 两种门控在 (a) 大机动组 n=817、(b) 全体验证集 n=1404 上的指标
  - 按卫星聚簇配对 bootstrap：Delta = f(proxy) - f(oracle) 的 95% CI 与 p
  - figures/data/gate_oracle_vs_proxy.csv + 可直接粘贴的 markdown 表

用法
  python gate_oracle_vs_proxy.py             # 代理阈值默认 15 km
  python gate_oracle_vs_proxy.py --sweep     # 扫描阈值，定位 F1=91.1% 的取值
  python gate_oracle_vs_proxy.py --check     # 自检：应复现 表5 的大机动组指标
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

import verify_all_tables as vt

PROXY_THR = float(os.environ.get('PROXY_THR', '15.0'))
B_BOOT = int(os.environ.get('B_BOOT', '10000'))
# 全体验证集里，非机动样本的 sgp4 与 true 按设计相等（目标残差设为零），r_sgp4 会趋于 0，
# 使 eta=(1-r_full/(r_sgp4+1e-8)) 爆成 1e8 量级。故全集口径只在 r_sgp4>=RMSE_FLOOR 的
# 样本上统计（大机动组不受影响：peak>=20 保证分母远离零）。
RMSE_FLOOR = float(os.environ.get('RMSE_FLOOR', '1.0'))
BOOT_SEED = 20260926
FAMILIES = [('MSE', lambda a: a.PREFIX['Transformer']),
            ('SoftMask', lambda a: a.PREFIX['Transformer+SoftMask'])]
REF_TABLE5 = {'MSE': (65.5, 9.3), 'SoftMask': (82.7, 3.3)}   # 表5：优于SGP4 / 严重退化


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
    n = len(eta)
    s = np.sort(eta)
    k = max(1, int(0.10 * n))
    return dict(n=n,
                better_pct=100.0 * float(np.mean(eta > 0)),
                severe20=100.0 * float(np.mean(eta < -20.0)),
                worst10=float(np.mean(s[:k])),
                worst1=float(s[0]))


def cluster_paired_bootstrap(sat_of, v_a, v_b, stat, B=B_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    sats = np.unique(sat_of)
    groups = {s: np.where(sat_of == s)[0] for s in sats}
    point = stat(v_a) - stat(v_b)
    diffs = np.empty(B)
    for b in range(B):
        pick = rng.randint(0, len(sats), len(sats))
        idx = np.concatenate([groups[sats[i]] for i in pick])
        diffs[b] = stat(v_a[idx]) - stat(v_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return point, float(lo), float(hi), min(p, 1.0)


def eta_from(rmse_sgp4, rmse_full, mask):
    """给门控决定 mask（True=介入）后，逐样本 eta（%）。"""
    pred = np.where(mask, rmse_full, rmse_sgp4)
    return (1.0 - pred / (rmse_sgp4 + 1e-8)) * 100.0


def main():
    sweep = '--sweep' in sys.argv
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
    sat_of_all = np.array([sat_map.get(int(i), '?') for i in idx_all])
    print(f'验证样本数: {len(idx_all)} | Oracle 阈值 = {vt.THRESHOLD} km | '
          f'代理阈值 = {PROXY_THR} km')
    print(f'[口径] DATA={vt.DATA_DIR} | MASKED={vt.MASKED_DIR} | CKPT_DIR={vt.CKPT_DIR!r} | '
          f'SEEDS={vt.SEEDS}')
    print(f'[口径] 软掩码前缀 = {vt.PREFIX["Transformer+SoftMask"]}*  '
          f'(由 SM_ALPHA={os.environ.get("SM_ALPHA", "0.3")} 决定)')
    print('[口径] 论文表5 期望（SM_ALPHA=0.2）= 82.7% / 3.3%；'
          '若得到 73.4% / 5.1% 说明加载的是 α=0.3 权重 —— 请加 SM_ALPHA=0.2 重跑')

    peak = np.empty(len(idx_all))
    proxy = np.empty(len(idx_all))
    store = {}
    for name, pfx_fn in FAMILIES:
        models = vt.load_ensemble(pfx_fn(vt), 'transformer', device)
        if not models:
            raise SystemExit(f'没有可用模型: {pfx_fn(vt)}*')
        r_sgp4, r_full = np.empty(len(idx_all)), np.empty(len(idx_all))
        for j, idx in enumerate(idx_all):
            cond = np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{vt.DATA_DIR}/sample_{idx:05d}_sgp4.npy')
            cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
            t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                avg_res = np.mean([m(t).detach().cpu().numpy().squeeze() * m_rs + m_rm
                                   for m in models], axis=0)
            true_res = true_tgt - sgp4_tgt
            peak[j] = np.max(np.abs(true_res[:, 0]))
            proxy[j] = float(np.ptp(cond[:, 0]))          # 条件窗口高度极差（km）
            r_sgp4[j] = per_sample_rmse(sgp4_tgt, true_tgt)
            r_full[j] = per_sample_rmse(sgp4_tgt + avg_res, true_tgt)
        store[name] = (r_sgp4, r_full)
        print(f'  {name}: 集成模型数={len(models)}')

    oracle_lab = peak >= vt.THRESHOLD
    big = oracle_lab

    # ---- (1) 代理门控判别质量 + 阈值扫描 ----
    def confusion(thr):
        pl = proxy >= thr
        tp = int(np.sum(pl & oracle_lab)); fp = int(np.sum(pl & ~oracle_lab))
        fn = int(np.sum(~pl & oracle_lab)); tn = int(np.sum(~pl & ~oracle_lab))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return tp, fp, fn, tn, prec, rec, f1

    tp, fp, fn, tn, prec, rec, f1 = confusion(PROXY_THR)
    print(f'\n[代理门控 @ {PROXY_THR} km] TP={tp} FP={fp} FN={fn} TN={tn}  '
          f'precision={prec*100:.2f}%  recall={rec*100:.2f}%  F1={f1*100:.2f}%  '
          f'介入率={100.0*(tp+fp)/len(idx_all):.2f}%')
    if sweep:
        print('\n[阈值扫描]  thr |   TP   FP  FN   TN |  prec%  rec%    F1%   介入率%')
        for thr in np.arange(2.0, 40.5, 0.5):
            a, b, c_, d, p_, r_, f_ = confusion(thr)
            mark = '  <-- F1≈91.1%' if abs(f_ * 100 - 91.1) < 0.35 else ''
            print(f'        {thr:5.1f} | {a:5d} {b:4d} {c_:4d} {d:5d} | '
                  f'{p_*100:6.2f} {r_*100:6.2f} {f_*100:6.2f}  {100.0*(a+b)/len(idx_all):6.2f}{mark}')

    proxy_lab = proxy >= PROXY_THR
    r_sgp4_ref = store[list(store)[0]][0]          # r_sgp4 与家族无关
    valid_all = r_sgp4_ref >= RMSE_FLOOR
    print(f'\n[全集口径] r_sgp4>={RMSE_FLOOR} km 的样本 {int(valid_all.sum())}/{len(idx_all)}'
          f'（剔除 {int((~valid_all).sum())} 个 sgp4≈true 的非机动构造样本，'
          f'其 eta 分母趋零）；尾部分位仅在 big 口径给出')

    rows = []
    for name, (r_sgp4, r_full) in store.items():
        eta_o = eta_from(r_sgp4, r_full, oracle_lab)
        eta_p = eta_from(r_sgp4, r_full, proxy_lab)
        for tag, eta, mask in (('Oracle', eta_o, oracle_lab), ('Proxy', eta_p, proxy_lab)):
            for scope, sub in (('big', big), ('all', valid_all)):
                m = metrics(eta[sub])
                if scope == 'all':                 # 分母趋零样本已剔除，尾部分位不再有意义
                    m['worst10'] = float('nan')
                    m['worst1'] = float('nan')
                harm = 100.0 * float(np.mean(eta[sub & mask] < 0)) if np.any(sub & mask) else np.nan
                rows.append(dict(family=name, gate=tag, scope=scope,
                                 intervened_pct=100.0 * float(np.mean(mask[sub])),
                                 harmful_pct=harm, **m))
                tail = (f'最差10%={m["worst10"]:.2f}% 最差={m["worst1"]:.2f}%'
                        if scope == 'big' else '尾部=—（口径不适用）')
                print(f'  [{name}/{tag}/{scope}] n={m["n"]} 优于SGP4={m["better_pct"]:.2f}% '
                      f'严重退化={m["severe20"]:.2f}% {tail} '
                      f'介入率={100.0*float(np.mean(mask[sub])):.2f}%')

    # ---- (2) 聚簇配对 bootstrap：Delta = proxy - oracle ----
    print('\n[聚簇配对 bootstrap] Delta = f(Proxy) - f(Oracle)（大机动组 n=%d）' % big.sum())
    for name, (r_sgp4, r_full) in store.items():
        eta_o = eta_from(r_sgp4, r_full, oracle_lab)
        eta_p = eta_from(r_sgp4, r_full, proxy_lab)
        so = sat_of_all[big]
        for label, fn_stat in (('优于SGP4比例(pp)', lambda e: 100.0 * np.mean(e > 0)),
                               ('严重退化比例(pp)', lambda e: 100.0 * np.mean(e < -20.0)),
                               ('最差单样本(pp)', lambda e: float(np.min(e)))):
            d, lo, hi, p = cluster_paired_bootstrap(so, eta_p[big], eta_o[big], fn_stat)
            print(f'  [{name}] {label}: Delta={d:+7.2f} 95%CI [{lo:+7.2f}, {hi:+7.2f}] p={p:.3f}')

    # ---- (3) 自检 ----
    if check:
        ok = True
        for name, (b_exp, s_exp) in REF_TABLE5.items():
            r_sgp4, r_full = store[name]
            m = metrics(eta_from(r_sgp4, r_full, oracle_lab)[big])
            good = abs(m['better_pct'] - b_exp) < 0.2 and abs(m['severe20'] - s_exp) < 0.2
            ok &= good
            print(f'  [自检] {name} Oracle/大机动组: 优于SGP4={m["better_pct"]:.2f}%（表5 {b_exp}）、'
                  f'严重退化={m["severe20"]:.2f}%（表5 {s_exp}） -> {"OK" if good else "MISMATCH"}')
        print('  [自检] 结论:', 'OK（口径与表5 一致）' if ok else
              'MISMATCH —— 检查 CKPT_DIR / 数据目录 / 归一化口径')

    df = pd.DataFrame(rows)
    _dir = out_dir()
    df.to_csv(os.path.join(_dir, 'gate_oracle_vs_proxy.csv'), index=False)
    print('\n写出 ' + os.path.join(out_dir(), 'gate_oracle_vs_proxy.csv'))
    sm = df[(df.family == 'SoftMask') & (df.scope == 'big')]
    print('\n可直接粘贴的表（软掩码，大机动组 n=817）：')
    print('| 门控 | 介入率 | 优于SGP4比例 | 严重退化 P(eta<-20%) | 最差10% | 最差单样本 |')
    print('|---|---|---|---|---|---|')
    for _, r in sm.iterrows():
        print(f'| {r.gate} | {r.intervened_pct:.1f}% | {r.better_pct:.1f}% | '
              f'{r.severe20:.1f}% | {r.worst10:.1f}% | {r.worst1:.1f}% |')


if __name__ == '__main__':
    main()
