"""
train_alpha_inner_val.py  --  alpha 消融训练（内层验证早停版）

相对你原训练脚本的改动（只改训练协议，模型/损失/优化器完全不变）
--------------------------------------------------------------------------
1. 【核心】硬隔离评估验证集。
   先用与论文完全相同的口径复现评估划分：879 颗卫星（np.unique 字符串排序后）
   用 RandomState(42) 洗牌，取前 10% = 87 颗 = 论文的评估验证卫星。
   这些卫星的数据在训练中【一律不用】：不参与拟合、不参与早停、不参与任何监控。
   （原脚本的问题正在于此：它从全量元数据里切 10%，恰好切出了这 87 颗，
     于是早停指标算在了评估集上。）

2. 【早停改到训练集内部】在剩下的训练卫星里再切一个"内层验证集"（默认 15%），
   早停监控指标（大机动组优于 SGP4 比例）只在它上面计算。内层验证集与评估集
   无交集，脚本用 assert 强制保证。

3. 遍历 SEEDS x ALPHA_VALUES（默认 6 种子 x 11 个 alpha），输出命名保持
   error_predictor_alpha{alpha:.1f}_seed{seed}.pth，下游脚本无需修改。

4. 不再写出任何 *_best_val.pth（避免下游误用）；内层最优权重另存到
   diag_ckpt_dir（默认 inner_val_ckpt/），仅供诊断。

5. 训练前打印并断言划分；训练后写出 alpha_sweep_training_log.csv。

用法
----
    # 校验划分（只跑划分，不训练）：应与论文评估集完全一致
    python train_alpha_inner_val.py --check-split

    # 全量：11 个 alpha x 6 种子
    TRAIN_DEVICE=cuda:3 python train_alpha_inner_val.py

    # 分片并行（推荐）：66 个组合按 GPU 分片
    ALPHAS=0.3 SEEDS=42,123 CUDA_VISIBLE_DEVICES=0 python train_alpha_inner_val.py
    ALPHAS=0.3 SEEDS=789,101 CUDA_VISIBLE_DEVICES=1 python train_alpha_inner_val.py
"""
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# ==================== 配置 ====================
BASE_CONFIG = {
    'data_dir': 'preprocessed_samples_masked',      # 训练用（软掩码目标）
    'truth_data_dir': 'preprocessed_samples',       # 只用于读真实残差/条件归一化以算指标
    'output_model_prefix': 'error_predictor_alpha',
    'diag_ckpt_dir': 'inner_val_ckpt',              # 内层最优权重（不参与论文）
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
    # ---- 划分相关（核心：评估隔离） ----
    'eval_split_seed': 42,       # 复现论文评估划分（勿改）
    'eval_fraction': 0.10,       # 论文口径：评估卫星占 10%
    'inner_val_seed': 2024,      # 内层划分种子
    'inner_val_fraction': 0.15,  # 训练卫星内部用于早停的比例（越大越稳，训练数据越少）
    # ---- 早停相关（保留原逻辑，只换监控集） ----
    'eval_start_epoch': 100,
    'eval_interval': 10,
    'val_patience': 5,
}

DEVICE = os.environ.get('TRAIN_DEVICE', 'cuda:0')
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,123,789,101,202,303').split(',')]
ALPHA_VALUES = [float(x) for x in
                os.environ.get('ALPHAS', '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]


def resolve_device():
    """把 TRAIN_DEVICE 解析成可用的 torch.device。

    注意 CUDA_VISIBLE_DEVICES 会把可见设备重编号：例如 CUDA_VISIBLE_DEVICES=3 之后，
    进程内只有 cuda:0 合法。这里做一次校验并自动回退，避免 'invalid device ordinal'。
    """
    want = os.environ.get('TRAIN_DEVICE', 'cuda:0')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_available():
        print(f'[设备] CUDA 不可用，使用 CPU（TRAIN_DEVICE={want}, '
              f'CUDA_VISIBLE_DEVICES={cvd}）', flush=True)
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if want.startswith('cuda'):
        try:
            idx = int(want.split(':')[1]) if ':' in want else 0
        except ValueError:
            idx = 0
        if idx >= n:
            print(f'[设备] TRAIN_DEVICE={want} 越界（可见 GPU 数={n}，'
                  f'CUDA_VISIBLE_DEVICES={cvd}）-> 自动回退到 cuda:0', flush=True)
            idx = 0
        dev = torch.device(f'cuda:{idx}')
        print(f'[设备] 使用 {dev}（可见 GPU 数={n}，'
              f'CUDA_VISIBLE_DEVICES={cvd}，name={torch.cuda.get_device_name(idx)}）',
              flush=True)
        try:
            free, total = torch.cuda.mem_get_info(idx)
            print(f'[设备] 显存 free={free/2**30:.1f} GB / total={total/2**30:.1f} GB',
                  flush=True)
            if free / 2**30 < 2.0:
                print('[设备] [!] 该卡可用显存不足 2 GB —— 很可能被其他进程占用。'
                      '请用 nvidia-smi 查看，换一张空闲卡启动。', flush=True)
        except Exception as e:
            print(f'[设备] 显存查询失败: {e}', flush=True)
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


# ==================== 数据集（与论文一致，未改） ====================
class SoftMaskedDataset(Dataset):
    def __init__(self, sample_indices, data_dir, cond_mean, cond_std,
                 res_mean, res_std, alpha=0.3):
        self.sample_indices = sample_indices
        self.data_dir = data_dir
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = cond_std.astype(np.float32)
        self.res_mean = res_mean.astype(np.float32)
        self.res_std = res_std.astype(np.float32)
        self.alpha = alpha

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s = self.sample_indices[idx]
        cond = np.load(f'{self.data_dir}/sample_{s:05d}_cond.npy')
        true_tgt = np.load(f'{self.data_dir}/sample_{s:05d}_true.npy')
        sgp4_tgt = np.load(f'{self.data_dir}/sample_{s:05d}_sgp4.npy')
        mask = np.load(f'{self.data_dir}/sample_{s:05d}_mask.npy')
        original_res = true_tgt - sgp4_tgt
        soft_mask = self.alpha + (1.0 - self.alpha) * mask
        mixed_res = original_res.copy()
        mixed_res[:, 0] *= soft_mask
        mixed_res[:, 1] *= soft_mask
        cond = (cond - self.cond_mean) / (self.cond_std + 1e-8)
        mixed_res = (mixed_res - self.res_mean) / (self.res_std + 1e-8)
        return (torch.tensor(cond, dtype=torch.float32),
                torch.tensor(mixed_res, dtype=torch.float32))


# ==================== 划分（核心改动） ====================
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
    """返回 (fit_indices, inner_val_indices, eval_indices)。

    eval_indices 按论文口径复现，训练中【绝不使用】，仅用于断言隔离。
    """
    meta = pd.read_csv(f"{config['data_dir']}/sample_metadata.csv")
    meta['norad_id'] = meta['norad_id'].astype(str).str.strip()
    sats_all = meta['norad_id'].values
    uniq = np.unique(sats_all)

    # ---- (1) 复现论文评估划分并隔离 ----
    rng_eval = np.random.RandomState(config['eval_split_seed'])
    rng_eval.shuffle(uniq)
    n_eval = max(1, int(len(uniq) * config['eval_fraction']))
    eval_sats = set(uniq[:n_eval])

    # ---- (2) 训练卫星内部再切内层验证集（早停用） ----
    train_sats = np.array([s for s in uniq if s not in eval_sats])
    rng_inner = np.random.RandomState(config['inner_val_seed'])
    rng_inner.shuffle(train_sats)
    n_inner = max(1, int(len(train_sats) * config['inner_val_fraction']))
    inner_sats = set(train_sats[:n_inner])
    fit_sats = set(train_sats[n_inner:])

    # ---- (3) 硬隔离断言（本脚本的核心保护） ----
    assert not (inner_sats & eval_sats), '内层验证集与评估集重叠！'
    assert not (fit_sats & eval_sats), '训练集与评估集重叠！'

    in_eval = np.isin(sats_all, list(eval_sats))
    in_inner = np.isin(sats_all, list(inner_sats))
    eval_idx = meta.loc[in_eval, 'sample_idx'].values
    inner_idx = meta.loc[in_inner, 'sample_idx'].values
    fit_idx = meta.loc[~in_eval & ~in_inner, 'sample_idx'].values

    if verbose:
        print('=' * 96)
        print('划分结果（评估卫星按论文口径复现，训练全程隔离）')
        print('=' * 96)
        print(f"  卫星: 全部 {len(uniq)} | 评估(隔离) {len(eval_sats)} | "
              f"内层验证(早停用) {len(inner_sats)} | 拟合 {len(fit_sats)}")
        print(f"  样本: 拟合 {len(fit_idx)} | 内层验证 {len(inner_idx)} | 评估(隔离) {len(eval_idx)}")
        print(f"  交集: 内层∩评估 = {len(inner_sats & eval_sats)}（必须 0）| "
              f"拟合∩评估 = {len(fit_sats & eval_sats)}（必须 0）")
        for cand in ('pareto_samples.csv', '../figures/data/pareto_samples.csv'):
            if os.path.exists(cand):
                ref = set(pd.read_csv(cand)['sample_idx'])
                print(f"  交叉核验: 与 {cand} 交集 = {len(ref & set(eval_idx))}/{len(ref)}"
                      f"（应为 {len(ref)}/{len(ref)}）")
                break
        print(f"  内层验证中 peak>=20 km 的样本数 = "
              f"{_count_big(inner_idx, config['truth_data_dir'])}（早停指标的样本量）")
        print('=' * 96)
    return fit_idx, inner_idx, eval_idx


# ==================== 内层验证指标（只换数据，算法不变） ====================
def evaluate_on_inner_val(model, inner_indices, truth_dir, device,
                          orig_cond_mean, orig_cond_std,
                          masked_res_mean, masked_res_std, threshold=20.0):
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


# ==================== 训练 ====================
def train_one(alpha, seed, config, fit_idx, inner_idx, device):
    set_seed(seed)
    t0 = time.time()

    m_cm = np.load(f"{config['data_dir']}/scaler_cond_mean.npy").astype(np.float32)
    m_cs = np.load(f"{config['data_dir']}/scaler_cond_std.npy").astype(np.float32)
    m_rm = np.load(f"{config['data_dir']}/scaler_res_mean.npy").astype(np.float32)
    m_rs = np.load(f"{config['data_dir']}/scaler_res_std.npy").astype(np.float32)
    o_cm = np.load(f"{config['truth_data_dir']}/scaler_cond_mean.npy").astype(np.float32)
    o_cs = np.load(f"{config['truth_data_dir']}/scaler_cond_std.npy").astype(np.float32)

    ds = SoftMaskedDataset(fit_idx, config['data_dir'], m_cm, m_cs, m_rm, m_rs, alpha=alpha)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    model = ErrorPredictor(d_model=config['d_model'], nhead=config['nhead'],
                           num_layers=config['num_layers'],
                           target_len=config['tgt_len']).to(device)
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()

    final_path = f"{config['output_model_prefix']}{alpha:.1f}_seed{seed}.pth"
    os.makedirs(config['diag_ckpt_dir'], exist_ok=True)
    inner_best_path = os.path.join(
        config['diag_ckpt_dir'],
        f"{config['output_model_prefix']}{alpha:.1f}_seed{seed}_innerval.pth")

    best_inner, best_epoch, patience, final_epoch, avg = -1.0, 0, 0, 0, float('nan')
    best_train, best_train_epoch = float('inf'), 0
    for epoch in range(config['epochs']):
        model.train()
        tot = 0.0
        for cond, res in dl:
            cond, res = cond.to(device), res.to(device)
            loss = loss_fn(model(cond), res)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        avg = tot / len(dl)
        final_epoch = epoch + 1

        # ★ 检查点规则（与分位数回归一致）：保存【训练损失最优】的权重作为论文模型；
        #   早停仍由内层验证指标（训练集内部）触发。
        if avg < best_train:
            best_train, best_train_epoch = avg, epoch + 1
            torch.save(model.state_dict(), final_path)

        # ★ 唯一协议改动：监控集从"评估验证集"换成"训练集内部的内层验证集"
        if epoch >= config['eval_start_epoch'] and (epoch + 1) % config['eval_interval'] == 0:
            v = evaluate_on_inner_val(model, inner_idx, config['truth_data_dir'], device,
                                      o_cm, o_cs, m_rm, m_rs)
            print(f"[a={alpha} s={seed}] Epoch {epoch+1}: Loss={avg:.6f}, "
                  f"内层验证(训练集内)优于SGP4={v:.1f}%, 耗时={time.time()-t0:.0f}s", flush=True)
            if v > best_inner:
                best_inner, best_epoch, patience = v, epoch + 1, 0
                torch.save(model.state_dict(), inner_best_path)      # 仅供诊断
            else:
                patience += 1
            if patience >= config['val_patience']:
                print(f"[a={alpha} s={seed}] 内层验证早停于 epoch {epoch+1}, "
                      f"最优 epoch={best_epoch}, 最优={best_inner:.1f}%", flush=True)
                break
        elif (epoch + 1) % 30 == 0 and epoch < config['eval_start_epoch']:
            print(f"[a={alpha} s={seed}] Epoch {epoch+1}/{config['epochs']}, "
                  f"Loss={avg:.6f}, 耗时={time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(final_path):             # 兜底（正常情况下循环内已保存）
        torch.save(model.state_dict(), final_path)
    print(f"[a={alpha} s={seed}] 完成 | 末轮={final_epoch} | 论文权重=训练损失最优 "
          f"{best_train:.6f} @epoch {best_train_epoch} | 内层最优={best_inner:.1f}% "
          f"@epoch {best_epoch} | 输出 {final_path}", flush=True)
    return dict(alpha=alpha, seed=seed, final_epoch=final_epoch,
                best_train=best_train, best_train_epoch=best_train_epoch,
                inner_best=best_inner, inner_best_epoch=best_epoch,
                final_loss=avg, seconds=time.time() - t0)


def main():
    if '--check-split' in sys.argv:
        build_splits(BASE_CONFIG)
        return
    device = resolve_device()
    print(f"设备: {device} | 种子: {SEEDS} | alpha: {ALPHA_VALUES}", flush=True)
    print('[诊断] 两个目录的 scaler 对照（若不同，说明训练与指标的归一化口径不一致）')
    for nm in ('scaler_cond_mean', 'scaler_cond_std', 'scaler_res_mean', 'scaler_res_std'):
        a = np.ravel(np.load(f"{BASE_CONFIG['data_dir']}/{nm}.npy"))
        b = np.ravel(np.load(f"{BASE_CONFIG['truth_data_dir']}/{nm}.npy"))
        print(f'   {nm:>18}: masked={a}  truth={b}')
    fit_idx, inner_idx, _ = build_splits(BASE_CONFIG)

    logs = []
    for alpha in ALPHA_VALUES:
        for seed in SEEDS:
            logs.append(train_one(alpha, seed, BASE_CONFIG, fit_idx, inner_idx, device))
            pd.DataFrame(logs).to_csv('alpha_sweep_training_log.csv', index=False)
    print('\n全部完成。训练日志: alpha_sweep_training_log.csv')
    print('论文使用的权重: error_predictor_alpha{alpha}_seed{seed}.pth（末态，未按验证集选点）')


if __name__ == '__main__':
    main()
