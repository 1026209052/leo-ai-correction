"""
train_lstm_inner_val.py -- LSTM 训练（内层验证早停版，评估集全程隔离）

与原训练脚本（train_all_lstm_softmask-*.py）的差别只有一处协议：
  原脚本：在【评估验证集】上算指标并早停（RandomState(42) 从全量 879 颗里切 10% = 那 87 颗）
  本脚本：把 87 颗评估卫星【硬隔离】（不拟合/不早停/不监控），
          早停指标改在【训练卫星内部】再切出的内层验证集上计算（默认 15%）。

同时支持两条 LSTM 家族（表4 的两行）：
  --variant soft : data_dir=preprocessed_samples_masked, alpha=0.3 -> lstm_softmask_seed{seed}.pth
  --variant base : data_dir=preprocessed_samples,        alpha=1.0 -> lstm_seed{seed}.pth
（两者超参一致：hidden=256, layers=3, dropout=0.1, batch=16, lr=3e-4, epochs=200）

用法
----
    python train_lstm_inner_val.py --check-split          # 只验证划分与隔离

    # 单卡串行两个家族
    VARIANTS=soft,base TRAIN_DEVICE=cuda:0 nohup python train_lstm_inner_val.py > log_lstm.txt 2>&1 &

    # 分片（每卡一个种子，两个家族）
    for s in 42 123 789 101 202 303; do
      CUDA_VISIBLE_DEVICES=0 TRAIN_DEVICE=cuda:0 VARIANTS=soft,base SEEDS=$s \
        nohup python train_lstm_inner_val.py > log_lstm_$s.txt 2>&1 &
    done
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
DATA_TRUTH = 'preprocessed_samples'
DATA_MASKED = 'preprocessed_samples_masked'

VARIANTS = {
    # soft: 表4 的 "LSTM+软掩码"（α=0.3，掩码管线，masked res scaler）
    'soft': dict(data_dir=DATA_MASKED, alpha=0.3, prefix='lstm_softmask_seed'),
    # base: 表4 的 "LSTM"（标准MSE，等价 alpha=1.0，未掩码管线）
    'base': dict(data_dir=DATA_TRUTH, alpha=1.0, prefix='lstm_seed'),
    # soft02: 规则选出的 α=0.2（与 Transformer 部署配置一致）。
    #   刻意沿用 prefix='lstm_softmask_seed'，使 verify_all_tables.py / eval_alpha_sweep.py
    #   等下游脚本无需改动即可读到新家族（运行前请先备份旧的 α=0.3 权重）。
    'soft02': dict(data_dir=DATA_MASKED, alpha=0.2, prefix='lstm_softmask_seed'),
}

BASE_CONFIG = {
    'cond_len': 288,
    'tgt_len': 864,
    'lstm_hidden': 256,
    'lstm_layers': 3,
    'dropout': 0.1,
    'batch_size': 16,
    'epochs': 200,
    'lr': 3e-4,
    # ---- 划分（评估隔离） ----
    'eval_split_seed': 42,       # 复现论文评估划分（勿改）
    'eval_fraction': 0.10,
    'inner_val_seed': 2024,
    'inner_val_fraction': 0.15,
    # ---- 早停（保留原逻辑，只换监控集） ----
    'eval_start_epoch': 120,
    'eval_interval': 10,
    'val_patience': 5,
    'diag_ckpt_dir': 'inner_val_ckpt',
}

DEVICE = os.environ.get('TRAIN_DEVICE', 'cuda:0')
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,123,789,101,202,303').split(',')]
VARIANTS_TO_RUN = [v.strip() for v in os.environ.get('VARIANTS', 'soft,base').split(',')]


def resolve_device():
    """解析 TRAIN_DEVICE，自动兜底（CUDA_VISIBLE_DEVICES 会重编号，注意用 cuda:0）。"""
    want = os.environ.get('TRAIN_DEVICE', 'cuda:1')
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_available():
        print(f'[设备] CUDA 不可用，使用 CPU（TRAIN_DEVICE={want}, CVD={cvd}）', flush=True)
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if want.startswith('cuda'):
        try:
            idx = int(want.split(':')[1]) if ':' in want else 0
        except ValueError:
            idx = 0
        if idx >= n:
            print(f'[设备] TRAIN_DEVICE={want} 越界（可见 GPU 数={n}，CVD={cvd}）'
                  f'-> 自动回退到 cuda:0', flush=True)
            idx = 0
        dev = torch.device(f'cuda:{idx}')
        print(f'[设备] 使用 {dev}（可见 GPU 数={n}，CVD={cvd}，'
              f'name={torch.cuda.get_device_name(idx)}）', flush=True)
        try:
            free, total = torch.cuda.mem_get_info(idx)
            print(f'[设备] 显存 free={free/2**30:.1f} GB / total={total/2**30:.1f} GB', flush=True)
            if free / 2**30 < 2.0:
                print('[设备] [!] 可用显存不足 2 GB —— 很可能被其他进程占用，'
                      '请用 nvidia-smi 查看并换卡。', flush=True)
        except Exception as e:
            print(f'[设备] 显存查询失败: {e}', flush=True)
        return dev
    return torch.device(want)


# ==================== 模型（与论文一致，未改） ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=256, num_layers=3,
                 target_len=864, dropout=0.1):
        super().__init__()
        self.target_len = target_len
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        d = hidden_dim * 2
        self.expand = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(),
                                    nn.Linear(d * 2, target_len * output_dim))

    def forward(self, cond):
        B = cond.shape[0]
        lstm_out, _ = self.lstm(cond)
        return self.expand(lstm_out[:, -1, :]).view(B, self.target_len, 2)


# ==================== 数据集 ====================
class SoftMaskedDataset(Dataset):
    """data_dir 决定目标与归一化口径；mask 一律取自 masked 目录（alpha=1 时不用 mask）。"""

    def __init__(self, sample_indices, data_dir, mask_dir, cond_mean, cond_std,
                 res_mean, res_std, alpha=0.3):
        self.sample_indices = sample_indices
        self.data_dir = data_dir
        self.mask_dir = mask_dir
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
        mixed = (true_tgt - sgp4_tgt).copy()
        if self.alpha < 1.0:                       # alpha=1 时掩码无作用，跳过读取
            mask = np.load(f'{self.mask_dir}/sample_{s:05d}_mask.npy')
            w = self.alpha + (1.0 - self.alpha) * mask
            mixed[:, 0] *= w
            mixed[:, 1] *= w
        cond = (cond - self.cond_mean) / (self.cond_std + 1e-8)
        mixed = (mixed - self.res_mean) / (self.res_std + 1e-8)
        return (torch.tensor(cond, dtype=torch.float32),
                torch.tensor(mixed, dtype=torch.float32))


# ==================== 划分（评估硬隔离） ====================
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


def build_splits(config, meta_dir, verbose=True):
    meta = pd.read_csv(f'{meta_dir}/sample_metadata.csv')
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
        print(f"  内层验证中 peak>=20 km 的样本数 = {_count_big(inner_idx, DATA_TRUTH)}"
              f"（早停指标的样本量）")
        print('=' * 96)
    return fit_idx, inner_idx, eval_idx


# ==================== 内层验证指标 ====================
def evaluate_on_inner_val(model, inner_indices, device, cond_mean, cond_std,
                          res_mean, res_std, threshold=20.0):
    model.eval()
    sgps, trues, preds = [], [], []
    with torch.no_grad():
        for idx in inner_indices:
            cond = np.load(f'{DATA_TRUTH}/sample_{idx:05d}_cond.npy')
            true_tgt = np.load(f'{DATA_TRUTH}/sample_{idx:05d}_true.npy')
            sgp4_tgt = np.load(f'{DATA_TRUTH}/sample_{idx:05d}_sgp4.npy')
            cond_norm = (cond - cond_mean) / (cond_std + 1e-8)
            t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(t).detach().cpu().numpy().squeeze() * res_std + res_mean
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
def train_one(vname, seed, config, fit_idx, inner_idx, device, verbose_split=False):
    v = VARIANTS[vname]
    set_seed(seed)
    t0 = time.time()
    ddir = v['data_dir']
    cm = np.load(f'{ddir}/scaler_cond_mean.npy').astype(np.float32)
    cs = np.load(f'{ddir}/scaler_cond_std.npy').astype(np.float32)
    rm = np.load(f'{ddir}/scaler_res_mean.npy').astype(np.float32)
    rs = np.load(f'{ddir}/scaler_res_std.npy').astype(np.float32)
    o_cm = np.load(f'{DATA_TRUTH}/scaler_cond_mean.npy').astype(np.float32)
    o_cs = np.load(f'{DATA_TRUTH}/scaler_cond_std.npy').astype(np.float32)

    ds = SoftMaskedDataset(fit_idx, ddir, DATA_MASKED, cm, cs, rm, rs, alpha=v['alpha'])
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    model = LSTMPredictor(hidden_dim=config['lstm_hidden'],
                          num_layers=config['lstm_layers'],
                          target_len=config['tgt_len']).to(device)
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()

    final_path = f"{v['prefix']}{seed}.pth"
    os.makedirs(config['diag_ckpt_dir'], exist_ok=True)
    inner_best_path = os.path.join(config['diag_ckpt_dir'], f"{v['prefix']}{seed}_innerval.pth")

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

        # ★ 唯一协议改动：监控集 = 训练集内部的内层验证集（评估集从不参与）
        if epoch >= config['eval_start_epoch'] and (epoch + 1) % config['eval_interval'] == 0:
            val = evaluate_on_inner_val(model, inner_idx, device, o_cm, o_cs, rm, rs)
            print(f"[LSTM-{vname} s={seed}] Epoch {epoch+1}: Loss={avg:.6f}, "
                  f"内层验证(训练集内)优于SGP4={val:.1f}%, 耗时={time.time()-t0:.0f}s", flush=True)
            if val > best_inner:
                best_inner, best_epoch, patience = val, epoch + 1, 0
                torch.save(model.state_dict(), inner_best_path)     # 仅诊断
            else:
                patience += 1
            if patience >= config['val_patience']:
                print(f"[LSTM-{vname} s={seed}] 内层验证早停于 epoch {epoch+1}, "
                      f"最优 epoch={best_epoch}, 最优={best_inner:.1f}%", flush=True)
                break
        elif (epoch + 1) % 20 == 0 and epoch < config['eval_start_epoch']:
            print(f"[LSTM-{vname} s={seed}] Epoch {epoch+1}/{config['epochs']}, "
                  f"Loss={avg:.6f}, 耗时={time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(final_path):           # 兜底
        torch.save(model.state_dict(), final_path)
    print(f"[LSTM-{vname} s={seed}] 完成 | 末轮={final_epoch} | 论文权重=训练损失最优 "
          f"{best_train:.6f} @epoch {best_train_epoch} | 内层最优={best_inner:.1f}% "
          f"@epoch {best_epoch} | 输出 {final_path}", flush=True)
    return dict(variant=vname, seed=seed, final_epoch=final_epoch,
                best_train=best_train, best_train_epoch=best_train_epoch,
                inner_best=best_inner, inner_best_epoch=best_epoch,
                final_loss=avg, seconds=time.time() - t0)


def main():
    if '--check-split' in sys.argv:
        build_splits(BASE_CONFIG, VARIANTS['soft']['data_dir'])
        return
    device = resolve_device()
    print(f"设备: {device} | 变体: {VARIANTS_TO_RUN} | 种子: {SEEDS}", flush=True)
    print('[诊断] 两个目录的 res scaler 对照（base 用 truth，soft 用 masked）')
    for nm in ('scaler_res_mean', 'scaler_res_std'):
        a = np.ravel(np.load(f'{DATA_MASKED}/{nm}.npy'))
        b = np.ravel(np.load(f'{DATA_TRUTH}/{nm}.npy'))
        print(f'   {nm:>16}: masked={a}  truth={b}')
    fit_idx, inner_idx, _ = build_splits(BASE_CONFIG, VARIANTS['soft']['data_dir'])

    logs = []
    for vname in VARIANTS_TO_RUN:
        if vname not in VARIANTS:
            print(f'[!] 未知变体 {vname}，跳过'); continue
        for seed in SEEDS:
            logs.append(train_one(vname, seed, BASE_CONFIG, fit_idx, inner_idx, device))
            pd.DataFrame(logs).to_csv('lstm_training_log.csv', index=False)
    print('\n全部完成。日志: lstm_training_log.csv')
    print('论文权重: lstm_softmask_seed{seed}.pth（软掩码）与 lstm_seed{seed}.pth（基线）')


if __name__ == '__main__':
    main()
