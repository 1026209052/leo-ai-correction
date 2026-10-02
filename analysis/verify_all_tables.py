# !!! 本文件由 _refcheck/port_verify_all_tables.js 从桌面版移植而来，勿手工编辑 !!!
# 相对桌面版的差异（仅两处，即"丁"方案）：
#   1) PREFIX[Transformer]          = error_predictor_alpha1.0_seed
#      PREFIX[Transformer+SoftMask] = error_predictor_alpha0.3_seed
#      -> 表4 与表6 同源（都来自重训的 alpha 家族），不再引用旧的 v4/std 模型
#   2) table4 的 Transformer 行改用 masked res scaler（alpha=1.0 也在 masked 管线上训练）
# 注意：本文件内部的"表N"编号沿用旧稿编号，与当前 0915.txt 的表号不完全对应。
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

# ==================== 配置 ====================
DATA_DIR = 'preprocessed_samples'
MASKED_DIR = 'preprocessed_samples_masked'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 20.0
SEEDS = [42, 123, 789, 101, 202, 303]

# 部署软掩码的 α（环境变量 SM_ALPHA，默认 0.3 = 原预设配置）。
#   切换为规则选出的 α=0.2：  SM_ALPHA=0.2 python verify_all_tables.py
#   注意：LSTM+软掩码家族的文件名不含 α，若改了 SM_ALPHA，
#        必须已用同一 α 重训 lstm_softmask_seed*（train_lstm_inner_val.py VARIANTS=soft02）。
SM_ALPHA = os.environ.get('SM_ALPHA', '0.3').strip()

# 模型前缀（根据您的实际文件命名调整）
PREFIX = {
    'Transformer': 'error_predictor_alpha1.0_seed',                      # 标准 MSE（α=1.0）
    'Transformer+SoftMask': f'error_predictor_alpha{SM_ALPHA}_seed',     # 部署软掩码
    'LSTM': 'lstm_seed',
    'LSTM+SoftMask': 'lstm_softmask_seed',
}
if SM_ALPHA != '0.3':
    print(f"[口径] 软掩码部署 α = {SM_ALPHA}（Transformer 家族 {PREFIX['Transformer+SoftMask']}）；"
          f"请确认 LSTM+软掩码家族已用 α={SM_ALPHA} 重训，否则两架构 α 不一致！")

# 权重口径开关：'' = 训练损失最优（论文口径）；'inner_val_ckpt' = 内层验证最优（诊断口径）
CKPT_DIR = 'inner_val_ckpt'   # 论文口径：内层验证最优；用 --ckpt-dir= 清空即回到训练损失最优（仅供对照）

# ==================== 模型定义 ====================
class ErrorPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, d_model=128, nhead=8, num_layers=6, target_len=864):
        super().__init__()
        self.target_len = target_len
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.expand = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, target_len * output_dim))

    def forward(self, cond):
        B = cond.shape[0]
        x = self.input_proj(cond)
        x = self.encoder(x)
        x = x.mean(dim=1)
        out = self.expand(x)
        return out.view(B, self.target_len, 2)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=256, num_layers=3, target_len=864, dropout=0.1):
        super().__init__()
        self.target_len = target_len
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        lstm_out_dim = hidden_dim * 2
        self.expand = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim * 2), nn.SiLU(),
            nn.Linear(lstm_out_dim * 2, target_len * output_dim))

    def forward(self, cond):
        B = cond.shape[0]
        out, _ = self.lstm(cond)
        out = out[:, -1, :]
        out = self.expand(out)
        return out.view(B, self.target_len, 2)


def load_ensemble(prefix, model_type, device):
    """权重口径：

    - CKPT_DIR == 'inner_val_ckpt'（默认，**论文口径**）：读 `*_innerval.pth` = 内层验证最优权重；
    - CKPT_DIR == ''（对照口径，加 `--ckpt-dir=`）：读 `{prefix}{seed}.pth` = 训练损失最优权重。
    两种权重都已由 train_*_inner_val.py 落盘，切换无需重训。
    """
    models = []
    for seed in SEEDS:
        stem = f'{CKPT_DIR}/{prefix}{seed}' if CKPT_DIR else f'{prefix}{seed}'
        cands = ([f'{stem}_innerval.pth'] if CKPT_DIR else [f'{stem}.pth']) + \
                [f'{stem}_best.pth']
        path = next((c for c in cands if os.path.exists(c)), None)
        if path and path.endswith('.pth') and not path.endswith(('_best.pth', '_innerval.pth')) \
                and os.path.exists(f'{stem}_best.pth'):
            print(f"  [注意] 同时存在 {stem}_best.pth（旧文件），已忽略")
        if path is None:
            print(f"  警告: 未找到 {stem}[.pth]")
            continue
        model = (ErrorPredictor() if model_type == 'transformer' else LSTMPredictor()).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


# ==================== 数据加载 ====================
def load_val_indices():
    meta_df = pd.read_csv(f'{DATA_DIR}/sample_metadata.csv')
    meta_df['norad_id'] = meta_df['norad_id'].astype(str).str.strip()
    all_norad = meta_df['norad_id'].values
    unique_sats = np.unique(all_norad)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_sats)
    n_val = max(1, int(len(unique_sats) * 0.1))
    val_sats = set(unique_sats[:n_val])
    val_mask = np.array([n in val_sats for n in all_norad])
    return meta_df['sample_idx'].values[val_mask]


def evaluate_method(models, val_indices, cond_mean, cond_std, res_mean, res_std, device, gate=True):
    """返回每个样本的 (sgp4_traj, true_traj, pred_traj, pred_residual)"""
    sgps, trues, preds, residuals = [], [], [], []
    with torch.no_grad():
        for idx in val_indices:
            cond = np.load(f'{DATA_DIR}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_sgp4.npy')

            cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)

            preds_list = []
            for m in models:
                pred_norm = m(cond_tensor).detach().cpu().numpy().squeeze()
                preds_list.append(pred_norm * res_std + res_mean)
            avg_res = np.mean(preds_list, axis=0)

            # Oracle决策
            true_res = true_tgt - sgp4_tgt
            if gate and np.max(np.abs(true_res[:, 0])) < THRESHOLD:
                avg_res = np.zeros_like(avg_res)

            sgps.append(sgp4_tgt)
            trues.append(true_tgt)
            preds.append(sgp4_tgt + avg_res)
            residuals.append(avg_res)
    return (np.array(sgps), np.array(trues), np.array(preds), np.array(residuals))


def compute_metrics(sgps, trues, preds, threshold=20.0):
    se_sgp4 = (sgps - trues) ** 2
    se_ai = (preds - trues) ** 2
    per_sgp4 = np.sqrt(np.mean(se_sgp4, axis=(1, 2)))
    per_ai = np.sqrt(np.mean(se_ai, axis=(1, 2)))

    true_peaks = np.array([np.max(np.abs(trues[i] - sgps[i])[:, 0]) for i in range(len(sgps))])
    big = true_peaks >= threshold
    n_big = np.sum(big)

    rmse_sgp4_big = np.sqrt(np.mean(se_sgp4[big]))
    rmse_ai_big = np.sqrt(np.mean(se_ai[big]))
    impr_big = (1 - rmse_ai_big / rmse_sgp4_big) * 100

    better_big = np.mean(per_ai[big] < per_sgp4[big]) * 100
    impr_vals = (1 - per_ai / (per_sgp4 + 1e-8)) * 100
    big_impr = impr_vals[big]
    worse_mask = per_ai[big] >= per_sgp4[big]

    return {
        'big_n': n_big,
        'rmse_drop': impr_big,
        'better_pct': better_big,
        'worse_median': np.median(big_impr[worse_mask]) if worse_mask.sum() > 0 else 0,
        'worst1': np.min(big_impr),
        'best_median': np.median(big_impr[~worse_mask]) if (~worse_mask).sum() > 0 else 0,
        'worst10': np.mean(np.sort(big_impr)[:max(1, int(len(big_impr)*0.1))]),
    }


# ==================== 表4：整体性能对比 ====================
def table4(val_indices, cond_mean, cond_std, device):
    print("\n" + "="*70)
    print("表4：大机动组综合性能对比（6种子集成，20 km阈值）")
    print("="*70)

    # 标准化参数
    orig_res_mean = np.load(f'{DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    orig_res_std = np.load(f'{DATA_DIR}/scaler_res_std.npy').astype(np.float32)
    masked_res_mean = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    masked_res_std = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    configs = [
        ('Transformer', 'transformer', masked_res_mean, masked_res_std),  # alpha=1.0 亦走 masked 管线
        ('Transformer+SoftMask', 'transformer', masked_res_mean, masked_res_std),
        ('LSTM', 'lstm', orig_res_mean, orig_res_std),
        ('LSTM+SoftMask', 'lstm', masked_res_mean, masked_res_std),
    ]

    results = []
    for name, mtype, rmean, rstd in configs:
        print(f"\n评估 {name}...")
        models = load_ensemble(PREFIX[name], mtype, device)
        if not models:
            print(f"  跳过（无模型）")
            continue
        sgps, trues, preds, _ = evaluate_method(models, val_indices, cond_mean, cond_std, rmean, rstd, device)
        m = compute_metrics(sgps, trues, preds, THRESHOLD)
        m['method'] = name
        results.append(m)
        print(f"  RMSE降幅={m['rmse_drop']:.1f}%, 优于SGP4={m['better_pct']:.1f}%, "
              f"退化中位数={m['worse_median']:.1f}%, 最差={m['worst1']:.1f}%")

    df = pd.DataFrame(results)
    tag = ('_' + os.path.basename(CKPT_DIR.rstrip('/'))) if CKPT_DIR else ''
    fn = f'verify_table4{tag}.csv'
    df.to_csv(fn, index=False)
    print(f"\n已保存: {fn}")
    return df


# ==================== 表6：边界样本非峰值理论验证 ====================
def table6(val_indices, cond_mean, cond_std, device):
    print("\n" + "="*70)
    print("表6：边界样本非峰值区域的理论验证（N=204，α=0.3）")
    print("="*70)

    orig_res_mean = np.load(f'{DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    orig_res_std = np.load(f'{DATA_DIR}/scaler_res_std.npy').astype(np.float32)
    masked_res_mean = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    masked_res_std = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    for name, mtype, rmean, rstd in [
        ('Transformer', 'transformer', masked_res_mean, masked_res_std),  # alpha=1.0 亦走 masked 管线
        ('Transformer+SoftMask', 'transformer', masked_res_mean, masked_res_std),
    ]:
        models = load_ensemble(PREFIX[name], mtype, device)
        if not models:
            continue

        var_list, noise_list = [], []
        with torch.no_grad():
            for idx in val_indices:
                cond = np.load(f'{DATA_DIR}/sample_{idx:05d}_cond.npy')
                true_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_true.npy')
                sgp4_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_sgp4.npy')

                true_res = true_tgt - sgp4_tgt
                sgp4_rmse = np.sqrt(np.mean((sgp4_tgt - true_tgt)**2))
                if sgp4_rmse >= 10.0:  # 边界样本
                    continue

                # 峰值位置
                peak_idx = np.argmax(np.abs(true_res[:, 0]))
                mask = np.ones(864, dtype=bool)
                mask[max(0, peak_idx-50):min(864, peak_idx+51)] = False
                non_peak = mask

                cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
                cond_tensor = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)

                preds_list = []
                for m in models:
                    pred_norm = m(cond_tensor).detach().cpu().numpy().squeeze()
                    preds_list.append(pred_norm * rstd + rmean)
                preds_arr = np.array(preds_list)  # (6, 864, 2)

                # 非峰值区域AI噪声
                noise = np.sum(np.linalg.norm(preds_arr[:, non_peak, :], axis=2)**2)
                noise_list.append(noise)
                var_list.append(np.var(preds_arr[:, non_peak, 0]))

        print(f"\n{name}:")
        print(f"  边界样本数: {len(noise_list)}")
        print(f"  非峰值集成方差均值: {np.mean(var_list):.2f} km²")
        print(f"  非峰值AI噪声总和: {np.sum(noise_list):.0f} km²")


# ==================== 表8：阈值敏感性 ====================
def table8(val_indices, cond_mean, cond_std, device):
    print("\n" + "="*70)
    print("表8：阈值敏感性分析（Transformer+软掩码，α=0.3）")
    print("="*70)

    masked_res_mean = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    masked_res_std = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)
    models = load_ensemble(PREFIX['Transformer+SoftMask'], 'transformer', device)

    # 保存所有样本的预测
    all_sgps, all_trues, all_preds, all_peaks = [], [], [], []
    with torch.no_grad():
        for idx in val_indices:
            cond = np.load(f'{DATA_DIR}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_sgp4.npy')

            true_res = true_tgt - sgp4_tgt
            peak = np.max(np.abs(true_res[:, 0]))

            cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
            preds_list = []
            for m in models:
                pred_norm = m(cond_tensor).detach().cpu().numpy().squeeze()
                preds_list.append(pred_norm * masked_res_std + masked_res_mean)
            avg_res = np.mean(preds_list, axis=0)

            all_sgps.append(sgp4_tgt)
            all_trues.append(true_tgt)
            all_preds.append(sgp4_tgt + avg_res)
            all_peaks.append(peak)

    all_sgps = np.array(all_sgps)
    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    all_peaks = np.array(all_peaks)

    print(f"\n{'阈值':<8} {'介入率':<10} {'RMSE降幅':<12} {'优于SGP4':<12}")
    for tau in [15.0, 20.0, 30.0, 50.0]:
        ai_mask = all_peaks >= tau
        if ai_mask.sum() == 0:
            continue
        # AI介入样本用修正，其余用SGP4
        final_preds = all_sgps.copy()
        final_preds[ai_mask] = all_preds[ai_mask]

        se_sgp4 = (all_sgps - all_trues) ** 2
        se_ai = (final_preds - all_trues) ** 2
        per_sgp4 = np.sqrt(np.mean(se_sgp4, axis=(1, 2)))
        per_ai = np.sqrt(np.mean(se_ai, axis=(1, 2)))

        rmse_sgp4 = np.sqrt(np.mean(se_sgp4[ai_mask]))
        rmse_ai = np.sqrt(np.mean(se_ai[ai_mask]))
        impr = (1 - rmse_ai / rmse_sgp4) * 100
        better = np.mean(per_ai[ai_mask] < per_sgp4[ai_mask]) * 100
        rate = ai_mask.mean() * 100

        print(f"{tau:<8.0f} {rate:<10.1f}% {impr:<12.1f}% {better:<12.1f}%")


# ==================== 表10：应力-强度分析 ====================
def table10(val_indices, cond_mean, cond_std, device):
    print("\n" + "="*70)
    print("表10：碰撞预警失效概率与可靠度指数β")
    print("="*70)

    orig_res_mean = np.load(f'{DATA_DIR}/scaler_res_mean.npy').astype(np.float32)
    orig_res_std = np.load(f'{DATA_DIR}/scaler_res_std.npy').astype(np.float32)
    masked_res_mean = np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32)
    masked_res_std = np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32)

    # 标准MSE
    models_sm = load_ensemble(PREFIX['Transformer+SoftMask'], 'transformer', device)

    sgp4_rmse_list, ai_rmse_list = [], []      # 大机动组（论文表9 口径）
    sgp4_rmse_all, ai_rmse_all = [], []        # 全体验证集（对照）
    with torch.no_grad():
        for idx in val_indices:
            cond = np.load(f'{DATA_DIR}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{DATA_DIR}/sample_{idx:05d}_sgp4.npy')

            cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)

            preds = []
            for m in models_sm:
                pred_norm = m(cond_tensor).detach().cpu().numpy().squeeze()
                preds.append(pred_norm * masked_res_std + masked_res_mean)
            avg_res = np.mean(preds, axis=0)
            ai_traj = sgp4_tgt + avg_res

            # 论文表9/表5 的口径 = **单通道高度 RMSE**（见表5 表注"分层依据为SGP4
            # 高度RMSE（单通道）"）。注意：不是双通道联合 RMS —— 用双通道会得到
            # 明显偏小的失效概率（旧稿 15 km 处 62.5% vs 双通道 38.3%）。
            sgp4_rmse = np.sqrt(np.mean((sgp4_tgt[:, 0] - true_tgt[:, 0])**2))
            ai_rmse = np.sqrt(np.mean((ai_traj[:, 0] - true_tgt[:, 0])**2))
            peak = np.max(np.abs((true_tgt - sgp4_tgt)[:, 0]))
            sgp4_rmse_all.append(sgp4_rmse)
            ai_rmse_all.append(ai_rmse)
            if peak >= THRESHOLD:
                sgp4_rmse_list.append(sgp4_rmse)
                ai_rmse_list.append(ai_rmse)

    sgp4_rmse = np.array(sgp4_rmse_list)
    ai_rmse = np.array(ai_rmse_list)
    sgp4_rmse_all = np.array(sgp4_rmse_all)
    ai_rmse_all = np.array(ai_rmse_all)

    print(f"\n[口径] 大机动组 n={len(sgp4_rmse)}（论文表9 口径，peak>={THRESHOLD:.0f} km）；"
          f"全体验证集 n={len(sgp4_rmse_all)}（对照）")
    print(f"SGP4误差均值: {sgp4_rmse.mean():.1f} km, 标准差: {sgp4_rmse.std():.1f} km")
    print(f"AI误差均值: {ai_rmse.mean():.1f} km, 标准差: {ai_rmse.std():.1f} km")
    print(f"\n{'阈值':<8} {'SGP4失效':<12} {'AI失效':<12} {'改善':<10} {'β_SGP4':<10} {'β_AI':<10}")

    for T in [10, 15, 20, 30]:
        p_sgp4 = np.mean(sgp4_rmse > T) * 100
        p_ai = np.mean(ai_rmse > T) * 100
        p_sgp4_all = np.mean(sgp4_rmse_all > T) * 100
        p_ai_all = np.mean(ai_rmse_all > T) * 100
        beta_sgp4 = (T - sgp4_rmse.mean()) / sgp4_rmse.std()
        beta_ai = (T - ai_rmse.mean()) / ai_rmse.std()
        # 改善度 = **相对降幅**（与论文表9 表头一致）：(p_sgp4 - p_ai)/p_sgp4
        impr_rel = (p_sgp4 - p_ai) / p_sgp4 * 100 if p_sgp4 > 0 else 0.0
        print(f"{T:<8} {p_sgp4:<12.1f}% {p_ai:<12.1f}% {impr_rel:<10.1f}% "
              f"{beta_sgp4:<10.2f} {beta_ai:<10.2f}")
        print(f"{'':<8} [绝对降幅 {-1*(p_ai - p_sgp4):.1f} pp]")
        print(f"{'':<8} └ 全体验证集对照: SGP4 {p_sgp4_all:.1f}% → AI {p_ai_all:.1f}%")


# ==================== 主函数 ====================
def main():
    global CKPT_DIR
    for i, a in enumerate(sys.argv):
        if a.startswith('--ckpt-dir='):
            CKPT_DIR = a.split('=', 1)[1]
        elif a == '--ckpt-dir' and i + 1 < len(sys.argv):
            CKPT_DIR = sys.argv[i + 1]
    print(f"[口径] {'论文口径：内层验证最优' if CKPT_DIR else '对照口径：训练损失最优'}"
          + (f"（{CKPT_DIR}）" if CKPT_DIR else ""))
    device = torch.device(DEVICE)
    cond_mean = np.load(f'{DATA_DIR}/scaler_cond_mean.npy').astype(np.float32)
    cond_std = np.load(f'{DATA_DIR}/scaler_cond_std.npy').astype(np.float32)
    val_indices = load_val_indices()
    print(f"验证样本数: {len(val_indices)}")

    table4(val_indices, cond_mean, cond_std, device)
    table6(val_indices, cond_mean, cond_std, device)
    table8(val_indices, cond_mean, cond_std, device)
    table10(val_indices, cond_mean, cond_std, device)

    print("\n" + "="*70)
    print("验证完成。请对比以下论文当前值：")
    print("="*70)
    print("表4: Transformer 19.9/56.8/-24.1/-162.6 | SoftMask 11.1/75.0/-4.3/-57.4")
    print("表6: 19.89→3.70 km², 41582→4522 km²")
    print("表8: 15km 62.6/19.9/54.9 | 20km 58.2/11.1/75.0 | 30km 37.3/14.2/84.5 | 50km 15.9/16.5/92.4")
    print("表10: 15km 62.5→40.4 | 30km β: -0.04→+0.06")


if __name__ == '__main__':
    main()
