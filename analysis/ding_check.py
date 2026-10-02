"""
ding_check.py -- 方案丁：判定 α 消融的 α=1.0/0.3 两行与表4 的差异来自哪里

用法
----
    python ding_check.py "<checkpoint glob>" [target]

    <checkpoint glob>  指向该 α 的 6 个模型，例如
                          "error_predictor_alpha1.0_seed*_best.pth"
                          "error_predictor_alpha1.0_seed*_best.pth"  # 换成你的真实命名
    target             'alpha1.0'（默认，对比表4 的 Transformer 行 19.9/56.8/-24.1/-162.6）
                       或 'alpha0.3'（对比表4 的 Transformer+软掩码行 11.1/75.0/-4.3/-57.4）

它会做的事
----------
对给定集成，枚举  {res scaler: 未掩码目录, 掩码目录}  x  {通道口径: 高度单通道, 双通道}
共 4 种组合，算出表4 的四个指标，并与目标值逐项比对，标出哪一种组合与表4 一致。
这样"丁"就从一个猜测变成一次可判定的小实验（只跑 6x817 次前向，几十秒）。
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------- 路径（按你的工程改这三行即可） ----------------
DATA_UNMASKED = 'preprocessed_samples'
DATA_MASKED = 'preprocessed_samples_masked'
PARETO_CSV = 'pareto_samples.csv'        # 提供 sample_idx 与 peak（用于选 817 个样本）
SEEDS = [42, 123, 789, 101, 202, 303]
PEAK_THRESHOLD = 20.0
EXPECTED_MODELS = 6

TARGETS = {
    'alpha1.0': dict(name='表4 的 Transformer（标准MSE）',
                     rmse=19.9, better=56.8, deg_med=-24.1, worst=-162.6),
    'alpha0.3': dict(name='表4 的 Transformer+软掩码',
                     rmse=11.1, better=75.0, deg_med=-4.3, worst=-57.4),
}

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ErrorPredictor(nn.Module):
    """与训练脚本保持一致"""
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


def load_scalers(d):
    return (np.load(f'{d}/scaler_cond_mean.npy').astype(np.float32),
            np.load(f'{d}/scaler_cond_std.npy').astype(np.float32),
            np.load(f'{d}/scaler_res_mean.npy').astype(np.float32),
            np.load(f'{d}/scaler_res_std.npy').astype(np.float32))


def load_samples(pareto_csv):
    df = pd.read_csv(pareto_csv)
    df = df[df['peak'] >= PEAK_THRESHOLD]
    return df['sample_idx'].tolist()


def build_ensemble(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[!] 没有匹配到 checkpoint：{pattern}\n"
                         f"    请先在目录下 ls 一下真实命名，再作为参数传入。")
    print(f"匹配到 {len(files)} 个 checkpoint:")
    for f in files:
        print('   ', f)
    if len(files) != EXPECTED_MODELS:
        print(f"   [!] 期望 {EXPECTED_MODELS} 个（6 种子）。数量不符会让集成指标对不上。")
    models = []
    for f in files:
        m = ErrorPredictor().to(DEVICE)
        m.load_state_dict(torch.load(f, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models


def predict_ens(models, cond_norm):
    t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = np.mean([m(t).cpu().numpy().squeeze() for m in models], axis=0)
    return preds          # (864, 2) 归一化空间


def metrics_from_eta(eta):
    eta = np.asarray(eta, float)
    pos, neg = eta[eta > 0], eta[eta < 0]
    return dict(better=100.0 * len(pos) / len(eta),
                deg_med=(np.median(neg) if len(neg) else np.nan),
                worst=eta.min())


def evaluate(models, idxs, cond_scalers, res_scalers, channel):
    cm, cs, rm, rs = cond_scalers
    _, _, rm2, rs2 = res_scalers
    sgp4_r, ai_r = [], []
    for idx in idxs:
        cond = np.load(f'{DATA_UNMASKED}/sample_{idx:05d}_cond.npy').astype(np.float32)
        res = np.load(f'{DATA_UNMASKED}/sample_{idx:05d}_res.npy').astype(np.float32)
        if res.ndim == 1:
            res = res[:, None]
        cond_n = (cond - cm) / (cs + 1e-8)
        pred = predict_ens(models, cond_n) * rs2 + rm2      # 反归一化
        n = min(len(res), len(pred))
        resid_true = res[:n]
        resid_pred = pred[:n]
        corr = resid_true - resid_pred                      # 修正后残留（= SGP4 + R̂ - 真值）
        if channel == 'height':
            sgp4_r.append(np.sqrt(np.mean(resid_true[:n, 0] ** 2)))
            ai_r.append(np.sqrt(np.mean(corr[:n, 0] ** 2)))
        else:
            sgp4_r.append(np.sqrt(np.mean(np.sum(resid_true[:n] ** 2, axis=1))))
            ai_r.append(np.sqrt(np.mean(np.sum(corr[:n] ** 2, axis=1))))
    sgp4_r, ai_r = np.array(sgp4_r), np.array(ai_r)
    eta = (1 - ai_r / (sgp4_r + 1e-8)) * 100
    m = metrics_from_eta(eta)
    m['rmse_agg'] = 100.0 * (1 - ai_r.mean() / sgp4_r.mean())      # 聚合降幅
    m['rmse_mean_eta'] = float(np.mean(eta))                       # 逐样本均值
    return m


def check(m, tgt, label):
    ok = lambda a, b, tol: abs(a - b) <= tol
    flags = [ok(m['rmse_agg'], tgt['rmse'], 0.35),
             ok(m['better'], tgt['better'], 1.0),
             ok(m['deg_med'], tgt['deg_med'], 1.0),
             ok(m['worst'], tgt['worst'], 5.0)]
    hit = sum(flags)
    print(f"   {label:>38} | RMSE降幅 {m['rmse_agg']:6.2f} | 优于SGP4 {m['better']:6.2f}% | "
          f"退化中位 {m['deg_med']:8.2f} | 最差 {m['worst']:9.2f} | 命中 {hit}/4"
          + ("   <<< 复现表4" if hit == 4 else ""))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pattern = sys.argv[1]
    tkey = sys.argv[2] if len(sys.argv) > 2 else 'alpha1.0'
    tgt = TARGETS[tkey]

    print('=' * 112)
    print(f'方案丁 判定实验   目标 = {tkey}  ({tgt["name"]})')
    print(f'目标值: RMSE降幅 {tgt["rmse"]}% | 优于SGP4 {tgt["better"]}% | '
          f'退化中位数 {tgt["deg_med"]}% | 最差单样本 {tgt["worst"]}%')
    print('=' * 112)

    models = build_ensemble(pattern)
    idxs = load_samples(PARETO_CSV)
    print(f'样本集: {len(idxs)} 个（peak >= {PEAK_THRESHOLD}）')

    sc_u = load_scalers(DATA_UNMASKED)
    sc_m = load_scalers(DATA_MASKED)
    print(f'\nscaler 对照（若两个目录的 res scaler 相同，则这条不是差异来源）:')
    for i, nm in enumerate(['cond_mean', 'cond_std', 'res_mean', 'res_std']):
        print(f'   {nm:>10}: unmasked={np.ravel(sc_u[i])}   masked={np.ravel(sc_m[i])}')
    print(f'\n逐组合结果（4 种评估口径 x 2 种通道口径 -> 共 8 行）:')
    for sname, sc in (('unmasked scaler', sc_u), ('masked scaler', sc_m)):
        for ch in ('height', 'both'):
            m = evaluate(models, idxs, sc_u, sc, ch)
            check(m, tgt, f'{sname} / {ch}通道')

    print('\n判读：')
    print('  · 有任一组合命中 4/4 -> 差异纯粹来自评估口径，改 alpha_ablation_ensemble.py 对齐即可，')
    print('    表6 那两行无需拼接，方案丁成立，图3 也不用加标注。')
    print('  · 全部不命中 -> 说明两个 α 的 checkpoint 与主实验不是同一批训练，')
    print('    请检查：训练轮次上限 / 早停耐心值 / 是否用了 masked 预处理的 target / 学习率。')
    print('    （此时再退回方案乙。）')


if __name__ == '__main__':
    main()
