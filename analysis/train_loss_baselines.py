"""
train_loss_baselines.py  --  表3 的实证对比：Huber / Focal / 自适应加权 vs 软掩码

动机
----
论文 §3.3.4 表3 目前只给出软掩码与 Huber 损失、Focal Loss、自适应加权
（w(r)=exp(-λr²)）的**理论**权重函数对比，没有实验。审稿人几乎一定会要求补实证。
本脚本用与主实验**完全相同**的训练协议，只替换"损失/目标构造"这一处，
其余（模型、划分、早停、检查点规则、归一化）一律不动，从而得到可比的对照。

与主实验的对齐（逐条）
----------------------
1. 划分：复用 train_alpha_inner_val.py 的 build_splits ——
   879 颗卫星 → 87 颗评估（全程隔离）/ 118 颗内层验证（早停）/ 674 颗拟合。
2. 模型：同一 ErrorPredictor（d_model=128, nhead=8, 6 层, 前馈 512, dropout 0.1）。
3. 归一化：cond 用 truth scaler；res 用 masked 管线 scaler（与主实验一致；
   归一化/反归一化用同一 scaler，仿射偏移自动抵消）。
4. 检查点：内层验证最优 → `inner_val_ckpt/error_predictor_{loss}_seed{seed}_innerval.pth`
   （论文口径）。同时把训练损失最优另存为 `error_predictor_{loss}_seed{seed}.pth`
   （供 `verify_all_tables.py --ckpt-dir=` 的对照口径）。
5. 命名与主实验前缀兼容：下游 `verify_all_tables.py` 只要把
   PREFIX['Huber'] = 'error_predictor_huber_seed' 即可直接评估（见 eval_loss_baselines.py）。

五种"损失/目标"定义（表3 的四类 + MSE 参照）
---------------------------------------------
  mse       : 目标=原始残差 R，损失=MSE            → 等价 α=1.0 的标准训练（已有权重，无需重训）
  softmask  : 目标= w_t·R，损失=MSE（本文，α=0.2） → 已有权重，无需重训
  huber     : 目标= R，损失=SmoothL1(beta=δ)       → w(r)=I(|r|≤δ)+δ/|r|·I(|r|>δ)
  focal     : 目标= R，损失=加权 MSE，w=(|e|/mean|e|)^γ（依赖模型误差，属"被动"权重）
  adaptive  : 目标= R，损失=加权 MSE，w=exp(-λ r²)（依赖目标残差，表3 的自适应加权）

用法
----
    # 校验划分（不训练）
    python train_loss_baselines.py --check-split

    # 全部三种基线 x 6 种子（约 18 次训练）
    TRAIN_DEVICE=cuda:0 python train_loss_baselines.py

    # 分片并行
    LOSSES=huber SEEDS=42,123,789 CUDA_VISIBLE_DEVICES=0 python train_loss_baselines.py
    LOSSES=focal,adaptive SEEDS=42,123,789 CUDA_VISIBLE_DEVICES=1 python train_loss_baselines.py

    # 快速试跑（3 种子）
    LOSSES=huber SEEDS=42,123,789 python train_loss_baselines.py

超参（可用环境变量覆盖）
    HUBER_DELTA=1.0   归一化尺度上的 Huber 拐点 δ
    FOCAL_GAMMA=2.0   Focal 指数 γ
    ADAPT_LAMBDA=0.1  自适应加权的 λ
    SM_ALPHA=0.2      软掩码 α（仅当 LOSSES 含 softmask 时用于命名，通常无需重训）

输出
    inner_val_ckpt/error_predictor_{loss}_seed{seed}_innerval.pth   （论文口径）
    error_predictor_{loss}_seed{seed}.pth                           （对照口径）
    loss_baselines_training_log.csv
"""
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 配置（与 train_alpha_inner_val.py 保持一致） ====================
BASE_CONFIG = {
    'data_dir': 'preprocessed_samples_masked',
    'truth_data_dir': 'preprocessed_samples',
    'diag_ckpt_dir': 'inner_val_ckpt',
    'cond_len': 288,
    'tgt_len': 864,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'batch_size': 16,
    'epochs': 300,
    'lr': 1e-4,
    'eval_split_seed': 42,
    'eval_fraction': 0.10,
    'inner_val_seed': 2024,
    'inner_val_fraction': 0.15,
    'eval_start_epoch': 100,
    'eval_interval': 10,
    'val_patience': 5,
}

DEVICE = os.environ.get('TRAIN_DEVICE', 'cuda:0')
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,123,789,101,202,303').split(',')]
LOSSES = [x.strip() for x in os.environ.get('LOSSES', 'huber,focal,adaptive').split(',') if x.strip()]
HUBER_DELTA = float(os.environ.get('HUBER_DELTA', '1.0'))
FOCAL_GAMMA = float(os.environ.get('FOCAL_GAMMA', '2.0'))
ADAPT_LAMBDA = float(os.environ.get('ADAPT_LAMBDA', '0.1'))
SM_ALPHA = float(os.environ.get('SM_ALPHA', '0.2'))

KNOWN_LOSSES = {'mse', 'softmask', 'huber', 'focal', 'adaptive'}


def resolve_device():
    want = os.environ.get('TRAIN_DEVICE', 'cuda:0')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_available():
        print(f'[设备] CUDA 不可用，使用 CPU（TRAIN_DEVICE={want}, CUDA_VISIBLE_DEVICES={cvd}）', flush=True)
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if want.startswith('cuda'):
        try:
            idx = int(want.split(':')[1]) if ':' in want else 0
        except ValueError:
            idx = 0
        if idx >= n:
            print(f'[设备] TRAIN_DEVICE={want} 越界（可见 GPU 数={n}，CUDA_VISIBLE_DEVICES={cvd}）'
                  f'-> 自动回退到 cuda:0', flush=True)
            idx = 0
        dev = torch.device(f'cuda:{idx}')
        print(f'[设备] 使用 {dev}（可见 GPU 数={n}，CUDA_VISIBLE_DEVICES={cvd}，'
              f'name={torch.cuda.get_device_name(idx)}）', flush=True)
        return dev
    return torch.device(want)


# ==================== 模型（与论文一致，未改） ====================
class ErrorPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, d_model=128, nhead=8,
                 num_layers=6, target_len=864):
        super().__init__()
        self.target_len = target_len
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1,
            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.expand = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, target_len * output_dim))

    def forward(self, cond):
        B = cond.shape[0]
        x = self.encoder(self.input_proj(cond)).mean(dim=1)
        return self.expand(x).view(B, self.target_len, 2)


# ==================== 数据集：唯一差异在"目标构造" ====================
class LossBaselineDataset(Dataset):
    """返回 (cond, target, w_adapt)。

    target    : 训练目标（归一化后）。softmask 用 w_t·R；其余用原始 R。
    w_adapt   : 自适应加权的逐元素权重 exp(-λ·target²)（仅 adaptive 用；其余为 1）。
    """

    def __init__(self, sample_indices, data_dir, cond_mean, cond_std,
                 res_mean, res_std, loss='huber', alpha=0.2, lam=0.1):
        self.sample_indices = sample_indices
        self.data_dir = data_dir
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = cond_std.astype(np.float32)
        self.res_mean = res_mean.astype(np.float32)
        self.res_std = res_std.astype(np.float32)
        self.loss = loss
        self.alpha = alpha
        self.lam = lam

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s = self.sample_indices[idx]
        cond = np.load(f'{self.data_dir}/sample_{s:05d}_cond.npy')
        true_tgt = np.load(f'{self.data_dir}/sample_{s:05d}_true.npy')
        sgp4_tgt = np.load(f'{self.data_dir}/sample_{s:05d}_sgp4.npy')
        original_res = true_tgt - sgp4_tgt

        if self.loss == 'softmask':
            mask = np.load(f'{self.data_dir}/sample_{s:05d}_mask.npy')
            w = self.alpha + (1.0 - self.alpha) * mask
            target = original_res.copy()
            target[:, 0] *= w
            target[:, 1] *= w
        else:
            target = original_res.copy()

        cond = (cond - self.cond_mean) / (self.cond_std + 1e-8)
        target = (target - self.res_mean) / (self.res_std + 1e-8)
        if self.loss == 'adaptive':
            w_adapt = np.exp(-self.lam * target ** 2).astype(np.float32)
        else:
            w_adapt = np.ones_like(target, dtype=np.float32)
        return (torch.tensor(cond, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(w_adapt, dtype=torch.float32))


def compute_loss(loss, pred, target, w_adapt):
    """表3 四类权重 / 目标的统一实现（均在归一化尺度上）。"""
    if loss in ('mse', 'softmask'):
        return F.mse_loss(pred, target)
    if loss == 'huber':
        return F.smooth_l1_loss(pred, target, beta=HUBER_DELTA)
    if loss == 'focal':
        e = pred - target
        a = e.detach().abs()
        w = (a / (a.mean() + 1e-12)).pow(FOCAL_GAMMA)
        return (w * e.pow(2)).mean()
    if loss == 'adaptive':
        return (w_adapt * (pred - target).pow(2)).mean()
    raise ValueError('unknown loss: ' + loss)


# ==================== 划分（与主实验完全一致） ====================
def _count_big(indices, truth_dir, threshold=20.0):
    c = 0
    for s in indices:
        try:
            t = np.load(f'{truth_dir}/sample_{s:05d}_true.npy')
            g = np.load(f'{truth_dir}/sample_{s:05d}_sgp4.npy')
            if np.max(np.abs(t - g)[:, 0]) >= threshold:
                c += 1
        except Exception:
            continue
    return c


def build_splits(config, verbose=True):
    meta = pd.read_csv(f"{config['data_dir']}/sample_metadata.csv")
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sats_all = meta['norad_id'].values
    uniq = np.unique(sats_all)

    rng_eval = np.random.RandomState(config['eval_split_seed'])
    rng_eval.shuffle(uniq)
    n_eval = max(1, int(len(uniq) * config['eval_fraction']))
    eval_sats = set(uniq[:n_eval])

    train_sats = np.array([s for s in uniq if s not in eval_sats])
    rng_inner = np.random.RandomState(config['inner_val_seed'])
    rng_inner.shuffle(train_sats)
    n_inner = max(1, int(len(train_sats) * config['inner_val_fraction']))
    inner_sats = set(train_sats[:n_inner])
    fit_sats = set(train_sats[n_inner:])

    assert not (inner_sats & eval_sats), '内层验证集与评估集重叠！'
    assert not (fit_sats & eval_sats), '训练集与评估集重叠！'

    in_eval = np.isin(sats_all, list(eval_sats))
    in_inner = np.isin(sats_all, list(inner_sats))
    eval_idx = meta.loc[in_eval, 'sample_idx'].values
    inner_idx = meta.loc[in_inner, 'sample_idx'].values
    fit_idx = meta.loc[~in_eval & ~in_inner, 'sample_idx'].values

    if verbose:
        print('=' * 88)
        print('划分（评估卫星全程隔离；内层验证集仅用于早停）')
        print('=' * 88)
        print(f"  卫星: 全部 {len(uniq)} | 评估 {len(eval_sats)} | 内层 {len(inner_sats)} | 拟合 {len(fit_sats)}")
        print(f"  样本: 拟合 {len(fit_idx)} | 内层 {len(inner_idx)} | 评估 {len(eval_idx)}")
        print(f"  交集: 内层∩评估 = {len(inner_sats & eval_sats)}（必须 0）| 拟合∩评估 = {len(fit_sats & eval_sats)}（必须 0）")
        print(f"  内层验证中 peak>=20 km 的样本数 = {_count_big(inner_idx, config['truth_data_dir'])}")
        print('=' * 88)
    return fit_idx, inner_idx, eval_idx


# ==================== 内层验证指标（与主实验一致） ====================
def evaluate_on_inner_val(model, inner_indices, truth_dir, device,
                          orig_cond_mean, orig_cond_std, masked_res_mean,
                          masked_res_std, threshold=20.0):
    model.eval()
    sgps, trues, preds = [], [], []
    with torch.no_grad():
        for idx in inner_indices:
            cond = np.load(f'{truth_dir}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{truth_dir}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{truth_dir}/sample_{idx:05d}_sgp4.npy')
            cond_norm = (cond - orig_cond_mean) / (orig_cond_std + 1e-8)
            t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(t).detach().cpu().numpy().squeeze() * masked_res_std + masked_res_mean
            sgps.append(sgp4_tgt)
            trues.append(true_tgt)
            preds.append(sgp4_tgt + pred)
    sgps, trues, preds = np.array(sgps), np.array(trues), np.array(preds)
    per_sgp4 = np.sqrt(np.mean((sgps - trues) ** 2, axis=(1, 2)))
    per_ours = np.sqrt(np.mean((preds - trues) ** 2, axis=(1, 2)))
    peaks = np.array([np.max(np.abs(trues[i] - sgps[i])[:, 0]) for i in range(len(sgps))])
    big = peaks >= threshold
    better = float(np.mean(per_ours[big] < per_sgp4[big]) * 100) if big.sum() else 0.0
    model.train()
    return better


def train_one(loss, seed, config, fit_idx, inner_idx, device):
    set_seed(seed)
    t0 = time.time()

    m_cm = np.load(f"{config['data_dir']}/scaler_cond_mean.npy").astype(np.float32)
    m_cs = np.load(f"{config['data_dir']}/scaler_cond_std.npy").astype(np.float32)
    m_rm = np.load(f"{config['data_dir']}/scaler_res_mean.npy").astype(np.float32)
    m_rs = np.load(f"{config['data_dir']}/scaler_res_std.npy").astype(np.float32)
    o_cm = np.load(f"{config['truth_data_dir']}/scaler_cond_mean.npy").astype(np.float32)
    o_cs = np.load(f"{config['truth_data_dir']}/scaler_cond_std.npy").astype(np.float32)

    ds = LossBaselineDataset(fit_idx, config['data_dir'], m_cm, m_cs, m_rm, m_rs,
                             loss=loss, alpha=SM_ALPHA, lam=ADAPT_LAMBDA)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    model = ErrorPredictor(d_model=config['d_model'], nhead=config['nhead'],
                           num_layers=config['num_layers'],
                           target_len=config['tgt_len']).to(device)
    opt = optim.Adam(model.parameters(), lr=config['lr'])

    final_path = f"error_predictor_{loss}_seed{seed}.pth"
    os.makedirs(config['diag_ckpt_dir'], exist_ok=True)
    inner_best_path = os.path.join(
        config['diag_ckpt_dir'], f"error_predictor_{loss}_seed{seed}_innerval.pth")

    best_inner, best_epoch, patience, final_epoch, avg = -1.0, 0, 0, 0, float('nan')
    best_train, best_train_epoch = float('inf'), 0
    for epoch in range(config['epochs']):
        model.train()
        tot = 0.0
        for cond, tgt, w_adapt in dl:
            cond, tgt, w_adapt = cond.to(device), tgt.to(device), w_adapt.to(device)
            loss_val = compute_loss(loss, model(cond), tgt, w_adapt)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            tot += loss_val.item()
        avg = tot / len(dl)
        final_epoch = epoch + 1

        if avg < best_train:
            best_train, best_train_epoch = avg, epoch + 1
            torch.save(model.state_dict(), final_path)

        if epoch >= config['eval_start_epoch'] and (epoch + 1) % config['eval_interval'] == 0:
            v = evaluate_on_inner_val(model, inner_idx, config['truth_data_dir'], device,
                                      o_cm, o_cs, m_rm, m_rs)
            print(f"[{loss} s={seed}] Epoch {epoch+1}: Loss={avg:.6f}, "
                  f"内层验证优于SGP4={v:.1f}%, 耗时={time.time()-t0:.0f}s", flush=True)
            if v > best_inner:
                best_inner, best_epoch, patience = v, epoch + 1, 0
                torch.save(model.state_dict(), inner_best_path)
            else:
                patience += 1
            if patience >= config['val_patience']:
                print(f"[{loss} s={seed}] 早停于 epoch {epoch+1}, 内层最优={best_inner:.1f}% "
                      f"@epoch {best_epoch}", flush=True)
                break
        elif (epoch + 1) % 30 == 0 and epoch < config['eval_start_epoch']:
            print(f"[{loss} s={seed}] Epoch {epoch+1}/{config['epochs']}: Loss={avg:.6f}, "
                  f"耗时={time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(final_path):
        torch.save(model.state_dict(), final_path)
    print(f"[{loss} s={seed}] 完成 | 末轮={final_epoch} | 训练损失最优 {best_train:.6f} "
          f"@epoch {best_train_epoch} | 内层最优={best_inner:.1f}% @epoch {best_epoch} "
          f"| 输出 {inner_best_path}", flush=True)
    return dict(loss=loss, seed=seed, final_epoch=final_epoch, best_train=best_train,
                best_train_epoch=best_train_epoch, inner_best=best_inner,
                inner_best_epoch=best_epoch, final_loss=avg, seconds=time.time() - t0)


def main():
    bad = [l for l in LOSSES if l not in KNOWN_LOSSES]
    if bad:
        raise SystemExit(f'未知 loss：{bad}（可选 {sorted(KNOWN_LOSSES)}）')
    if '--check-split' in sys.argv:
        build_splits(BASE_CONFIG)
        return
    todo = [l for l in LOSSES if l not in ('mse', 'softmask')]
    if not todo:
        print('[提示] mse / softmask 的权重已有（error_predictor_alpha1.0_seed* 与 '
              'error_predictor_alpha0.2_seed*），无需重训。')
        return
    device = resolve_device()
    print(f"设备: {device} | 种子: {SEEDS} | loss: {todo}", flush=True)
    print(f"[超参] HUBER_DELTA={HUBER_DELTA} FOCAL_GAMMA={FOCAL_GAMMA} ADAPT_LAMBDA={ADAPT_LAMBDA}", flush=True)
    fit_idx, inner_idx, _ = build_splits(BASE_CONFIG)

    logs = []
    for loss in todo:
        for seed in SEEDS:
            logs.append(train_one(loss, seed, BASE_CONFIG, fit_idx, inner_idx, device))
            pd.DataFrame(logs).to_csv('loss_baselines_training_log.csv', index=False)
    print('\n全部完成。训练日志: loss_baselines_training_log.csv')
    print('下一步: python eval_loss_baselines.py   # 生成表3 的实证对照')


if __name__ == '__main__':
    main()
