"""
eval_alpha_sweep.py -- 评估 alpha 家族模型（口径与 verify_all_tables.py 完全一致）

口径来源：/verify_all_tables.py 的 load_val_indices() / evaluate_method() / compute_metrics()
  · 验证划分：np.unique(norad_id) -> RandomState(42) 洗牌 -> 前 10%（=87 颗/1404 样本）
  · 逐样本 RMSE = sqrt(mean((pred-true)**2, axis=(1,2)))（双通道联合，T x 2 全体元素）
  · 集成 = 6 个种子【先平均预测】再算指标
  · Oracle 门控：peak<20 km 的样本 AI 修正强制置零（对 big 子集无影响）
  · RMSE降幅 = 池化 RMS 比：100*(1 - sqrt(mean(se_AI[big]))/sqrt(mean(se_sgp4[big])))
  · 优于SGP4比例 = mean(per_ai[big] < per_sgp4[big])*100
  · 改善中位数 = median(eta[big][per_ai < per_sgp4])；退化中位数 = median(eta[big][per_ai >= per_sgp4])
  · 最差10%平均退化 = mean(sort(eta[big])[:int(0.1*n_big)])；最差单样本 = min(eta[big])
  · res scaler：alpha 家族用 masked 目录（与训练一致）

用法
----
  python eval_alpha_sweep.py --check-old              # 用旧 alpha 模型复现旧消融表（期望 6/6）
  python eval_alpha_sweep.py                          # 重训后：产出新 表6 / 图3 数值
  python eval_alpha_sweep.py --boot                   # 追加 alpha=0.3 vs 1.0 聚簇 bootstrap
  python eval_alpha_sweep.py --family v4              # 顺手评估表4 的 Transformer+软掩码（v4 家族）
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------- 配置（与 verify_all_tables.py 对齐） ----------------
DATA_DIR = 'preprocessed_samples'
MASKED_DIR = 'preprocessed_samples_masked'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 20.0
SEEDS = [42, 123, 789, 101, 202, 303]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OUTDIR = 'eval_out'

# 其它家族（表4 用），供交叉对照
FAMILIES = {
    'alpha': 'error_predictor_alpha{alpha:.1f}_seed{seed}',   # 本次重训的 α 家族
    'v4': 'error_predictor_v4_seed{seed}',                    # 表4 的 Transformer+软掩码
    'std': 'error_predictor_seed{seed}',                      # 表4 的 Transformer（标准MSE）
    'lstm': 'lstm_seed{seed}',
    'lstm_soft': 'lstm_softmask_seed{seed}',
}
# 各家族的 res scaler（与 verify_all_tables.py 的 configs 一致）
FAMILY_SCALER = {'alpha': MASKED_DIR, 'v4': MASKED_DIR, 'std': DATA_DIR,
                 'lstm': DATA_DIR, 'lstm_soft': MASKED_DIR}

OLD_REF = {   # alpha_ablation_ensemble_results.csv 的旧值（含 big_impr -> rmse）
    1.0: dict(rmse=21.455, better=55.324, imp_med=26.194, deg_med=-9.614,
              worst10=-68.195, worst=-128.313),
    0.3: dict(rmse=10.607, better=75.887, imp_med=15.061, deg_med=-3.622,
              worst10=-38.006, worst=-80.974),
}

# 权重口径开关：默认 ''（读 error_predictor_alpha*_seed*.pth = 训练损失最优的论文权重）；
# 传 --ckpt-dir=inner_val_ckpt 则改读 *_innerval.pth（内层验证最优的诊断权重），
# 用于在不重训的前提下比较两种检查点规则。
CKPT_DIR = 'inner_val_ckpt'   # 论文口径：内层验证最优；用 --ckpt-dir= 清空即回到训练损失最优（仅供对照）


class ErrorPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, d_model=128, nhead=8,
                 num_layers=6, target_len=864):
        super().__init__()
        self.target_len = target_len
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=512, dropout=0.1,
                                           activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.expand = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(),
                                    nn.Linear(d_model * 2, target_len * output_dim))

    def forward(self, cond):
        B = cond.shape[0]
        x = self.encoder(self.input_proj(cond)).mean(dim=1)
        return self.expand(x).view(B, self.target_len, 2)


def load_ensemble(template, alpha, device):
    """权重加载顺序（避免旧文件抢先）：

    - CKPT_DIR 非空（默认 `inner_val_ckpt`，**论文口径**）：`_innerval.pth`（内层验证最优）→ `.pth` → `_best.pth`
    - CKPT_DIR 为空（对照口径，加 `--ckpt-dir=`）：`.pth`（训练损失最优）→ `_best.pth`
    """
    models = []
    for seed in SEEDS:
        base = template.format(alpha=alpha, seed=seed)
        if CKPT_DIR:
            base = os.path.join(CKPT_DIR, base)
        path = None
        sufs = ('_innerval.pth', '.pth', '_best.pth') if CKPT_DIR else ('.pth', '_best.pth')
        for suf in sufs:
            if os.path.exists(base + suf):
                path = base + suf
                break
        if path is None:
            print(f'   [!] 缺少 {base}[_best].pth')
            continue
        m = ErrorPredictor().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    if len(models) != len(SEEDS):
        print(f'   [!] 只有 {len(models)}/{len(SEEDS)} 个种子，集成结果不可比')
    return models


def load_val_indices():
    meta = pd.read_csv(f'{DATA_DIR}/sample_metadata.csv')
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sats = meta['norad_id'].values
    uniq = np.unique(sats)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * 0.1))
    val_sats = set(uniq[:n_val])
    return meta.loc[np.isin(sats, list(val_sats)), 'sample_idx'].values, \
        meta.loc[np.isin(sats, list(val_sats)), 'norad_id'].values


def evaluate(models, val_indices, cond_mean, cond_std, res_mean, res_std, device):
    """返回逐样本 MSE 数组（双通道联合）与 peak，口径与 verify_all_tables 一致。"""
    mse_sgp4, mse_ai, peaks = [], [], []
    for idx in val_indices:
        cond = np.load(f'{DATA_DIR}/sample_{idx:05d}_cond.npy')
        true_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_true.npy')
        sgp4_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_sgp4.npy')
        cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
        t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_list = [m(t).detach().cpu().numpy().squeeze() * res_std + res_mean
                          for m in models]
        avg_res = np.mean(preds_list, axis=0)
        true_res = true_tgt - sgp4_tgt
        if np.max(np.abs(true_res[:, 0])) < THRESHOLD:      # Oracle 门控
            avg_res = np.zeros_like(avg_res)
        se_sgp4 = (sgp4_tgt - true_tgt) ** 2
        se_ai = ((sgp4_tgt + avg_res) - true_tgt) ** 2
        mse_sgp4.append(float(np.mean(se_sgp4)))            # 双通道联合 MSE
        mse_ai.append(float(np.mean(se_ai)))
        peaks.append(float(np.max(np.abs(true_res[:, 0]))))
    return np.array(mse_sgp4), np.array(mse_ai), np.array(peaks)


def compute_metrics(mse_sgp4, mse_ai, peaks, threshold=THRESHOLD):
    per_sgp4, per_ai = np.sqrt(mse_sgp4), np.sqrt(mse_ai)
    big = peaks >= threshold
    n_big = int(big.sum())
    rmse_sgp4_big = np.sqrt(np.mean(mse_sgp4[big]))
    rmse_ai_big = np.sqrt(np.mean(mse_ai[big]))
    impr_big = (1 - rmse_ai_big / rmse_sgp4_big) * 100          # ★ 表4 口径
    eta = (1 - per_ai / (per_sgp4 + 1e-8)) * 100
    eta_big = eta[big]
    worse = per_ai[big] >= per_sgp4[big]
    return dict(big_n=n_big,
                rmse_drop=float(impr_big),
                better_pct=float(np.mean(per_ai[big] < per_sgp4[big]) * 100),
                imp_median=float(np.median(eta_big[~worse])) if (~worse).any() else float('nan'),
                deg_median=float(np.median(eta_big[worse])) if worse.any() else float('nan'),
                worst10=float(np.mean(np.sort(eta_big)[:max(1, int(len(eta_big) * 0.1))])),
                worst1=float(np.min(eta_big)),
                mean_eta=float(np.mean(eta_big)),
                # ★ 论文新的主指标：严重退化比例（无条件，全 n_big 同一分母）
                severe10=float(np.mean(eta_big < -10) * 100),
                severe20=float(np.mean(eta_big < -20) * 100),
                severe30=float(np.mean(eta_big < -30) * 100))


def boot_cluster(xa, xb, groups, fn, B=10000, seed=20260919):
    by = {}
    for a, b, g in zip(xa, xb, groups):
        by.setdefault(g, []).append((a, b))
    keys = list(by)
    rng = random.Random(seed)
    d = []
    for _ in range(B):
        A, Bv = [], []
        for _ in range(len(keys)):
            for a, b in by[keys[rng.randrange(len(keys))]]:
                A.append(a)
                Bv.append(b)
        v = fn(np.array(Bv)) - fn(np.array(A))
        if v == v:
            d.append(v)
    d.sort()
    lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d))]
    p = 2 * min(float(np.mean(np.array(d) <= 0)), float(np.mean(np.array(d) >= 0)))
    return fn(np.array(xb)) - fn(np.array(xa)), lo, hi, p


def main():
    global CKPT_DIR
    os.makedirs(OUTDIR, exist_ok=True)
    check_old = '--check-old' in sys.argv
    do_boot = '--boot' in sys.argv
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            CKPT_DIR = sys.argv[i + 1]
    print(f'[口径] {"论文口径：内层验证最优" if CKPT_DIR else "对照口径：训练损失最优"}'
          + (f'（{CKPT_DIR}）' if CKPT_DIR else ''))
    device = torch.device(DEVICE)

    cond_mean = np.load(f'{DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    idx_all, sat_all = load_val_indices()
    print(f'验证集: {len(idx_all)} 样本 / {len(set(sat_all))} 颗卫星')

    rows, store = [], {}
    alphas = sorted(OLD_REF) if check_old else ALPHAS
    for a in alphas:
        rm = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
        rs = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)
        models = load_ensemble(FAMILIES['alpha'], a, device)
        if not models:
            continue
        mse_s, mse_a, pk = evaluate(models, idx_all, cond_mean, cond_std, rm, rs, device)
        m = compute_metrics(mse_s, mse_a, pk)
        m['alpha'] = a
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        store[a] = (mse_s, mse_a, pk)
        print(f"  alpha={a:.1f}: RMSE降幅={m['rmse_drop']:6.2f}%  优于SGP4={m['better_pct']:6.2f}%  "
              f"改善中位={m['imp_median']:+7.2f}%  退化中位={m['deg_median']:+7.2f}%  "
              f"最差10%={m['worst10']:+8.2f}%  最差={m['worst1']:+9.2f}%")
        print(f"           严重退化比例 P(η<-10/20/30%) = "
              f"{m['severe10']:.2f}% / {m['severe20']:.2f}% / {m['severe30']:.2f}%   ← 论文新主指标")
        if check_old:
            r = OLD_REF[a]
            hit = (abs(m['rmse_drop'] - r['rmse']) < 0.05) \
                + (abs(m['better_pct'] - r['better']) < 0.05) + (abs(m['imp_median'] - r['imp_med']) < 0.05) \
                + (abs(m['deg_median'] - r['deg_med']) < 0.05) + (abs(m['worst10'] - r['worst10']) < 0.05) \
                + (abs(m['worst1'] - r['worst']) < 0.05)
            print(f'          逐项对照旧值(含 RMSE降幅) → 命中 {hit}/6')

    df = pd.DataFrame(rows)
    tag = ('_' + os.path.basename(CKPT_DIR.rstrip('/'))) if CKPT_DIR else ''
    fn = os.path.join(OUTDIR, f'alpha_sweep_metrics{tag}.csv')
    df.to_csv(fn, index=False)
    print(f'\n写出 {fn}（列: alpha, big_n, rmse_drop, better_pct, imp_median, deg_median, '
          f'worst10, worst1, mean_eta, severe10, severe20, severe30）')
    print('  -> 表6 主指标用 severe20；deg_median 仅供描述性参考（条件中位数，分母随方法变）')

    if do_boot and 0.3 in store and 1.0 in store:
        print('\n聚簇配对 bootstrap（alpha=0.3 vs 1.0，以卫星为单位；口径同 §5.1）')
        pk = store[0.3][2]
        ga = np.array([f'{s}' for s in sat_all])[pk >= THRESHOLD]
        ea = (1 - np.sqrt(store[0.3][1]) / (np.sqrt(store[0.3][0]) + 1e-8)) * 100
        ea = ea[pk >= THRESHOLD]
        eb = (1 - np.sqrt(store[1.0][1]) / (np.sqrt(store[1.0][0]) + 1e-8)) * 100
        eb = eb[pk >= THRESHOLD]
        for name, f in (('优于SGP4比例(pp)', lambda v: 100 * np.mean(v > 0)),
                        ('严重退化比例P(η<-20%)(pp)', lambda v: 100 * np.mean(v < -20)),
                        ('P(η<-10%)(pp)', lambda v: 100 * np.mean(v < -10)),
                        ('退化中位数(pp)', lambda v: np.median(v[v < 0]) if (v < 0).any() else np.nan),
                        ('CVaR10(pp)', lambda v: np.sort(v)[:max(1, int(0.10 * len(v)))].mean()),
                        ('最差单样本(pp)', lambda v: v.min())):
            d, lo, hi, p = boot_cluster(eb, ea, ga, f)
            print(f'  {name:>16}: {d:+8.2f}  95%CI [{lo:8.2f}, {hi:8.2f}]  p={p:.3f}')


if __name__ == '__main__':
    main()
