"""
validate_with_space_v5.py  --  独立星历验证（修正版 v5.1）

相对 v4 的关键修正
------------------
1. 【致命】真值重构。v4 写的是
       true_r = r_sgp4 + U*r_hat + V*v_hat + W*w_hat
   把星历前三列当成"相对 SGP4 的位置偏移量"。实测该三列的模 |UVW| ≈ 6710 km
   （-> 高度 338-389 km，正是 Starlink 的真实高度），|VUVWVW| ≈ 7.69 km/s
   （LEO 惯性速度），说明它们是【完整的星历位置/速度矢量】，相加导致半径翻倍，
   于是 v4 报出 3755 km 的荒谬基线。
   v5 同时计算三种候选，并以"新鲜 TLE 的 SGP4 高度"为标尺自动判定：
       vector_km : |U r̂ + V v̂ + W ŵ| - R_E        （完整矢量, km）  <- 正确
       offset_km : |r_sgp4 + U r̂ + V v̂ + W ŵ| - R_E（v4 写法，偏移量）
       vector_m  : 同 vector_km 但按 m 处理
   正确者 |Δalt| 为公里量级，错误者为上千公里（实测 12.1 / 2974 / 6817 km）。

2. 【崩溃】v4 的 print 引用了 df['lstm_residual_mean']，而 LSTM 分支从未写入该列
   -> KeyError（已修）。

3. 【自检】恒常打印基线统计；基线荒谬时大声告警；直接给出尾部指标
   （退化中位数 / CVaR10 / 最差单样本 + bootstrap CI）。

4. 【重要】peak 过滤从"丢弃样本"改为"打标记"（passes_peak）：
   - 尾部统计仍只用大机动子集；
   - 但 【p_S（非机动 SGP4 失效概率）必须在未筛选样本上估计】，
     否则就是在"已知残差大的样本里统计残差大的比例"（选择偏差）。
   v5.1 同时输出"未筛选"与"仅大机动子集"两行，方便对照。

用法
----
    python validate_with_space_v5.py --diagnose 5   # 只打印星历列统计，不加载模型
    python validate_with_space_v5.py                # 正式跑
    POSITION_MODE = 'vector_km' | 'offset_km' | 'vector_m' | 'auto'   (见下)
"""
import os
import re
import sys
import glob
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sgp4.api import Satrec, jday

# ================== 配置 ==================
SPACEX_DIR = './starlink_ephemerides'
TLE_DIR = './tle_files'
# 论文口径：软掩码 Transformer（α 由环境变量 SM_ALPHA 决定，默认 0.3）与 LSTM+软掩码，
# 权重取**内层验证最优**（inner_val_ckpt/*_innerval.pth）。
#   切换为规则选出的 α=0.2：  SM_ALPHA=0.2 python validate_with_space_v5.py
_SM_ALPHA = os.environ.get('SM_ALPHA', '0.3').strip()
MODEL_PREFIX_TRANS = f'error_predictor_alpha{_SM_ALPHA}_seed'
MODEL_PREFIX_LSTM = 'lstm_softmask_seed'
CKPT_DIR = 'inner_val_ckpt'      # 论文口径=内层验证最优；置 '' 则回到训练损失最优
PREPROCESSED_DIR = 'preprocessed_samples'
MASKED_DIR = 'preprocessed_samples_masked'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

R_EARTH = 6371.0
GM = 3.986e5

POSITION_MODE = 'vector_km'     # 'vector_km' | 'offset_km' | 'vector_m' | 'auto'
APPLY_U_MEAN_FILTER = False     # v4 的 |mean(U)|<=100 过滤（错误约定的产物，默认关）
MIN_SGP4_PEAK_KM = 20.0         # 大机动筛选阈值（现在只打标记，不丢弃）
OUTPUT_CSV = 'spacex_v5_verification.csv'
CANDIDATE_MODES = ('vector_km', 'offset_km', 'vector_m')


# ================== 星历解析 ==================
def parse_spacex_ephemeris(filepath):
    """返回 (df, t_start, t_stop)。列名按文件表头（UVW）理解。"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    line2 = lines[1].strip()
    m_start = re.search(r'ephemeris_start:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', line2)
    m_stop = re.search(r'ephemeris_stop:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', line2)
    if not m_start or not m_stop:
        raise ValueError("无法解析星历头部")
    t_start = datetime.strptime(m_start.group(1), '%Y-%m-%d %H:%M:%S')
    t_stop = datetime.strptime(m_stop.group(1), '%Y-%m-%d %H:%M:%S')

    data = []
    for line in lines[4:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            t = float(parts[0])
            u, v, w = float(parts[1]), float(parts[2]), float(parts[3])
            vu, vv, vw = float(parts[4]), float(parts[5]), float(parts[6])
            ts = str(int(t)).zfill(13)
            dt = (datetime(int(ts[0:4]), 1, 1)
                  + timedelta(days=int(ts[4:7]) - 1, hours=int(ts[7:9]),
                              minutes=int(ts[9:11]), seconds=int(ts[11:13])))
            data.append([dt, u, v, w, vu, vv, vw])
        except (ValueError, IndexError):
            continue
    if not data:
        raise ValueError("星历数据为空")
    df = pd.DataFrame(data, columns=['pubtime', 'U', 'V', 'W', 'VU', 'VV', 'VW'])
    return df, t_start, t_stop


def diagnose_ephemeris_files(n_files=5):
    """打印星历各列的统计量 -> 一分钟内判定列的真实含义与单位。"""
    files = sorted(glob.glob(os.path.join(SPACEX_DIR, '*.txt')))[:n_files]
    if not files:
        print(f"  [!] {SPACEX_DIR} 下没有 .txt 星历文件")
        return
    for fp in files:
        print("=" * 96)
        print(f"文件: {os.path.basename(fp)}")
        try:
            with open(fp) as f:
                raw = [next(f) for _ in range(6)]
        except StopIteration:
            raw = []
        for ln in raw:
            print("   raw | " + ln.rstrip())
        try:
            df, t0, t1 = parse_spacex_ephemeris(fp)
        except Exception as e:
            print(f"   解析失败: {e}")
            continue
        print(f"  解析: {len(df)} 点, {t0} -> {t1}")
        print(f"  {'列':>4} {'min':>14} {'median':>14} {'max':>14} {'mean(|.|)':>14}")
        for c in ['U', 'V', 'W', 'VU', 'VV', 'VW']:
            v = df[c].values
            print(f"  {c:>4} {v.min():14.3f} {np.median(v):14.3f} {v.max():14.3f} "
                  f"{np.mean(np.abs(v)):14.3f}")
        norm = np.linalg.norm(df[['U', 'V', 'W']].values, axis=1)
        print(f"  |UVW| : median={np.median(norm):.1f}   "
              f"-> 若解释为高度: {np.median(norm) - R_EARTH:.1f} km   "
              f"（若解释为 m: {(np.median(norm)/1000) - R_EARTH:.1f} km）")
        vnorm = np.linalg.norm(df[['VU', 'VV', 'VW']].values, axis=1)
        print(f"  |VUVWVW|: median={np.median(vnorm):.4f}  （LEO 速度应为 ~7.6 km/s）")


# ================== TLE -> Satrec ==================
def load_tle_via_sgp4init(tle_file, norad_id):
    df = pd.read_csv(tle_file)
    id_col = 'NORAD_CAT_ID' if 'NORAD_CAT_ID' in df.columns else 'norad_id'
    df[id_col] = df[id_col].astype(str).str.strip()
    sub = df[df[id_col] == str(norad_id)]
    records = []
    for _, row in sub.iterrows():
        try:
            sat = Satrec()
            epoch_str = str(row['EPOCH'])
            if 'T' in epoch_str:
                date_part, time_part = epoch_str.split('T')
                dt = datetime.strptime(date_part, '%Y-%m-%d')
                hms = time_part.split('.')[0].split(':')
                h, m, s = int(hms[0]), int(hms[1]), int(hms[2])
                micro = float('0.' + time_part.split('.')[1]) if '.' in time_part else 0.0
            else:
                dt = datetime.strptime(epoch_str.strip(), '%Y-%m-%d %H:%M:%S')
                h, m, s, micro = dt.hour, dt.minute, dt.second, 0.0
            jd, fr = jday(dt.year, dt.month, dt.day, h, m, s)
            fr += micro / 86400.0
            sat.sgp4init(2, 'i', int(row[id_col]), jd + fr - 2433281.5,
                         float(row['BSTAR']),
                         float(row['MEAN_MOTION_DOT']) if pd.notna(row['MEAN_MOTION_DOT']) else 0.0,
                         float(row['MEAN_MOTION_DDOT']) if pd.notna(row['MEAN_MOTION_DDOT']) else 0.0,
                         float(row['ECCENTRICITY']), float(row['ARG_OF_PERICENTER']),
                         float(row['INCLINATION']), float(row['MEAN_ANOMALY']),
                         float(row['MEAN_MOTION']) * (2 * np.pi) / (24 * 60),
                         float(row['RA_OF_ASC_NODE']))
            if sat.error != 0:
                continue
            records.append((sat, dt))
        except Exception:
            continue
    return records


def get_tle_before(records, target_time):
    best_sat, best_epoch = None, None
    for sat, epoch in records:
        if epoch <= target_time and (best_epoch is None or epoch > best_epoch):
            best_sat, best_epoch = sat, epoch
    return best_sat, best_epoch


# ================== SGP4 传播 ==================
def propagate_sgp4_to_times(sat, times):
    rows = []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            rows.append(dict(pubtime=t, x=np.nan, y=np.nan, z=np.nan, vx=np.nan,
                             vy=np.nan, vz=np.nan, altitude_km=np.nan, velocity_kms=np.nan))
            continue
        x, y, z = r
        vx, vy, vz = v
        rows.append(dict(pubtime=t, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                         altitude_km=np.sqrt(x * x + y * y + z * z) - R_EARTH,
                         velocity_kms=np.sqrt(vx * vx + vy * vy + vz * vz)))
    return pd.DataFrame(rows)


def resample_to_times(df_uvw, times):
    df = df_uvw.set_index('pubtime')[['U', 'V', 'W', 'VU', 'VV', 'VW']]
    new_index = pd.DatetimeIndex(times)
    df = df.reindex(df.index.union(new_index)).interpolate(method='time').loc[new_index]
    return df.reset_index().rename(columns={'index': 'pubtime'})


# ================== 真值重构（三种候选） ==================
def reconstruct_candidates(r, v, uvw, vuvw):
    """r,v: SGP4 位置/速度 (N,3)；uvw/vuvw: 星历给出的 (N,3)。

    (r̂, v̂, ŵ) 由 SGP4 瞬时状态定义（正交基，保范数，故旋转不改变 |·|）。
    返回 {mode: (alt, vel)}。
    """
    r_norm = np.linalg.norm(r, axis=1)
    ok = r_norm >= R_EARTH
    r_hat = np.zeros_like(r)
    r_hat[ok] = r[ok] / r_norm[ok, None]
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)
    ok &= h_norm > 1e-6
    w_hat = np.zeros_like(h)
    w_hat[ok] = h[ok] / h_norm[ok, None]
    v_hat = np.cross(w_hat, r_hat)

    out = {}
    for mode in CANDIDATE_MODES:
        scale = 1e-3 if mode == 'vector_m' else 1.0
        p = uvw * scale
        q = vuvw * scale
        p_rsw = p[:, 0:1] * r_hat + p[:, 1:2] * v_hat + p[:, 2:3] * w_hat
        q_rsw = q[:, 0:1] * r_hat + q[:, 1:2] * v_hat + q[:, 2:3] * w_hat
        if mode.startswith('vector'):
            true_r, true_v = p_rsw, q_rsw           # 星历 = 完整矢量
        else:
            true_r, true_v = r + p_rsw, v + q_rsw   # 星历 = 相对偏移（v4 写法）
        alt = np.linalg.norm(true_r, axis=1) - R_EARTH
        vel = np.linalg.norm(true_v, axis=1)
        alt[~ok] = np.nan
        vel[~ok] = np.nan
        out[mode] = (alt, vel)
    return out


def detect_maneuver_from_altitude(alt, vel, sigma=5.0):
    r = R_EARTH + alt
    eps = 0.5 * vel ** 2 - GM / r
    deps = np.diff(eps)
    if len(deps) < 10:
        return []
    mad = np.median(np.abs(deps - np.median(deps)))
    s = 1.4826 * mad
    if s <= 0:
        return []
    return list(np.where(np.abs(deps) > sigma * s)[0])


def extract_norad_from_filename(filename):
    for p in filename.replace('.txt', '').split('_'):
        if p.isdigit() and len(p) == 5:
            return p
    return None


# ================== 模型 ==================
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
        out, _ = self.lstm(cond)
        return self.expand(out[:, -1, :]).view(B, self.target_len, 2)


def load_ensemble(prefix, model_type, device):
    models = []
    for seed in [42, 123, 789, 101, 202, 303]:
        path = None
        stem = f'{CKPT_DIR}/{prefix}{seed}' if CKPT_DIR else f'{prefix}{seed}'
        cands = ([f'{stem}_innerval.pth'] if CKPT_DIR else [f'{stem}.pth']) + [f'{stem}_best.pth']
        for cand in cands:
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            print(f'  [!] 未找到 {stem}[_innerval].pth')
            continue
        model = (ErrorPredictor() if model_type == 'transformer' else LSTMPredictor()).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


# ================== 单星验证 ==================
def validate_single_satellite(spacex_file, tle_file, norad_id, trans_models,
                              lstm_models, scalers):
    try:
        df_spx_raw, t_start, _ = parse_spacex_ephemeris(spacex_file)

        if APPLY_U_MEAN_FILTER:
            u_mean = df_spx_raw['U'].mean()
            if u_mean < -100 or u_mean > 100:
                return None

        records = load_tle_via_sgp4init(tle_file, norad_id)
        if not records:
            return None
        tle_sat, tle_epoch = get_tle_before(records, t_start)
        if tle_sat is None:
            return None

        times = [t_start + timedelta(minutes=5 * i) for i in range(288 + 864)]
        sgp4_df = propagate_sgp4_to_times(tle_sat, times)
        mask = ~sgp4_df['altitude_km'].isna()
        if mask.sum() < 400:
            return None
        times = [t for t, m in zip(times, mask) if m]
        sgp4_df = sgp4_df[mask].reset_index(drop=True)
        spx = resample_to_times(df_spx_raw, times)

        r = sgp4_df[['x', 'y', 'z']].values
        v = sgp4_df[['vx', 'vy', 'vz']].values
        uvw = spx[['U', 'V', 'W']].values.astype(float)
        vuvw = spx[['VU', 'VV', 'VW']].values.astype(float)
        if np.isnan(uvw).any():
            return None

        cands = reconstruct_candidates(r, v, uvw, vuvw)
        alt_sgp4_full = sgp4_df['altitude_km'].values

        # ---- 用"新鲜 TLE 的 SGP4 高度"作为标尺，给每种候选打分 ----
        scores = {}
        for mode, (alt, _) in cands.items():
            good = ~np.isnan(alt)
            scores[mode] = float(np.median(np.abs(alt[good] - alt_sgp4_full[good]))) \
                if good.sum() >= 400 else np.inf
        mode_used = min(scores, key=scores.get) if POSITION_MODE == 'auto' else POSITION_MODE
        true_alt_all, true_vel_all = cands[mode_used]

        valid = ~np.isnan(true_alt_all)
        if valid.sum() < 400:
            return None
        true_alt = true_alt_all[valid]
        true_vel = true_vel_all[valid]
        alt_sgp4 = alt_sgp4_full[valid]
        times_valid = np.array(times)[valid]

        maneuver_idx = detect_maneuver_from_altitude(true_alt, true_vel)
        has_maneuver = len(maneuver_idx) > 0
        cond_len = 288
        if has_maneuver:
            m = max(maneuver_idx[0], cond_len)
            cond_start, pred_start = m - cond_len, m
        else:
            mid = max(len(times_valid) // 2, cond_len)
            cond_start, pred_start = mid - cond_len, mid
        pred_len = min(864, len(times_valid) - pred_start)
        if pred_len < 100:
            return None

        sgp4_alt_pred = alt_sgp4[pred_start:pred_start + pred_len]
        true_alt_pred = true_alt[pred_start:pred_start + pred_len]
        sgp4_peak = float(np.max(np.abs(sgp4_alt_pred - true_alt_pred)))
        sgp4_rmse_alt = float(np.sqrt(np.mean((sgp4_alt_pred - true_alt_pred) ** 2)))
        # ★ 不再丢弃，只标记（p_S 需要未筛选样本）
        passes_peak = bool(sgp4_peak >= MIN_SGP4_PEAK_KM)

        cond_mean, cond_std, res_mean, res_std = scalers
        cond_input = np.stack([true_alt[cond_start:cond_start + cond_len],
                               true_vel[cond_start:cond_start + cond_len]], axis=1)
        cond_norm = (cond_input - cond_mean) / (cond_std + 1e-8)
        cond_t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        res = dict(norad_id=norad_id, has_maneuver=has_maneuver,
                   tle_age_days=(t_start - tle_epoch).total_seconds() / 86400.0,
                   pred_len=pred_len, sgp4_peak=sgp4_peak,
                   sgp4_rmse_alt=sgp4_rmse_alt,
                   ephemeris_alt_median=float(np.median(true_alt)),
                   mode_used=mode_used, passes_peak=passes_peak)
        for mode in CANDIDATE_MODES:
            res[f'score_{mode}'] = scores.get(mode, np.nan)

        with torch.no_grad():
            if trans_models:
                preds = np.mean([m(cond_t).cpu().numpy().squeeze() for m in trans_models], axis=0)
                corr = preds * res_std + res_mean
                ai_alt = sgp4_alt_pred + corr[:pred_len, 0]
                ai_rmse = float(np.sqrt(np.mean((ai_alt - true_alt_pred) ** 2)))
                res['eta_trans_soft'] = (1 - ai_rmse / (sgp4_rmse_alt + 1e-8)) * 100
                res['ai_rmse_alt'] = ai_rmse
                res['trans_better'] = ai_rmse < sgp4_rmse_alt
                res['trans_residual_mean'] = float(corr[:pred_len, 0].mean())
                res['trans_residual_std'] = float(corr[:pred_len, 0].std())
            if lstm_models:
                preds = np.mean([m(cond_t).cpu().numpy().squeeze() for m in lstm_models], axis=0)
                corr = preds * res_std + res_mean
                ai_alt = sgp4_alt_pred + corr[:pred_len, 0]
                ai_rmse = float(np.sqrt(np.mean((ai_alt - true_alt_pred) ** 2)))
                res['eta_lstm_soft'] = (1 - ai_rmse / (sgp4_rmse_alt + 1e-8)) * 100
                res['lstm_ai_rmse_alt'] = ai_rmse
                res['lstm_better'] = ai_rmse < sgp4_rmse_alt
                res['lstm_residual_mean'] = float(corr[:pred_len, 0].mean())
                res['lstm_residual_std'] = float(corr[:pred_len, 0].std())
        return res
    except Exception as e:
        print(f"  验证失败 {norad_id}: {e}")
        return None


# ================== 统计 ==================
def pctl(x, q):
    s = np.sort(np.asarray(x, float))
    if len(s) == 0:
        return float('nan')
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def tail_metrics(eta):
    eta = np.asarray(eta, float)
    eta = eta[~np.isnan(eta)]
    if len(eta) == 0:
        return None
    pos, neg = eta[eta > 0], eta[eta < 0]
    k = max(1, int(round(0.10 * len(eta))))
    return dict(n=len(eta),
                superior=100.0 * len(pos) / len(eta),
                imp_median=float(np.median(pos)) if len(pos) else np.nan,
                deg_median=float(np.median(neg)) if len(neg) else np.nan,
                cvar10=float(np.sort(eta)[:k].mean()),
                worst=float(eta.min()),
                p_crit={x: float((eta < -x).mean() * 100) for x in (10, 20, 30, 50)})


def bootstrap_ci(x, fn, B=10000, seed=20260919):
    """i.i.d. bootstrap；自动跳过 nan / 空切片（例如某次重采样里没有退化样本）。"""
    x = [v for v in np.asarray(x, float) if not np.isnan(v)]
    n = len(x)
    if n == 0:
        return float('nan'), float('nan')
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        try:
            v = fn([x[rng.randrange(n)] for _ in range(n)])
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if isinstance(v, float) and v != v:      # nan
            continue
        out.append(v)
    if not out:
        return float('nan'), float('nan')
    out.sort()
    return float(out[int(0.025 * len(out))]), float(out[int(0.975 * len(out))])


def _deg_median(v):
    n = [x for x in v if x < 0]
    return float(np.median(n)) if n else float('nan')


def _cvar10(v):
    v = np.sort(np.asarray(v, float))
    k = max(1, int(round(0.10 * len(v))))
    return float(v[:k].mean())


def report(df_all, df, skipped_no_tle):
    print(f"\n{'='*96}")
    print(f"全部通过几何检查的样本 {len(df_all)} 条；其中大机动"
          f"(peak≥{MIN_SGP4_PEAK_KM:.0f} km) {len(df)} 条；跳过无 TLE {skipped_no_tle} 颗")
    print(f"{'='*96}")

    # ---------- 0. peak 分布（看筛选怎么咬的） ----------
    if len(df_all):
        pk = df_all['sgp4_peak']
        print(f"\n[自检 0] SGP4 高度残差峰值分布（未筛选，n={len(df_all)}）: "
              f"median={pk.median():.2f}  P90={pctl(pk,0.90):.2f}  "
              f"P95={pctl(pk,0.95):.2f}  P99={pctl(pk,0.99):.2f}  max={pk.max():.1f} km")
        print(f"         机动检测判定为含机动的窗口: "
              f"{int(df_all['has_maneuver'].astype(bool).sum())}/{len(df_all)} 条")

    # ---------- 1. 坐标约定自检 ----------
    print("\n[自检 1] 三种真值重构相对 SGP4 高度的中位偏差（应只有一种是公里量级）")
    for mode in CANDIDATE_MODES:
        c = f'score_{mode}'
        if c in df_all.columns:
            print(f"   {mode:>10}: median |Δalt| = {df_all[c].median():12.1f} km   "
                  f"（被选中 {int((df_all['mode_used'] == mode).sum())}/{len(df_all)} 次）")

    # ---------- 2. 基线（未筛选） ----------
    if len(df_all):
        base = df_all['sgp4_rmse_alt']
        print(f"\n[自检 2] SGP4 基线高度 RMSE（未筛选，n={len(df_all)}）: "
              f"median={base.median():.2f} km  P90={pctl(base,0.90):.2f}  "
              f"P95={pctl(base,0.95):.2f}  P99={pctl(base,0.99):.2f}  max={base.max():.1f}")
        print(f"         星历高度中位数 = {df_all['ephemeris_alt_median'].median():.1f} km "
              f"（Starlink 应在 250-570 km）")
        print(f"         TLE 年龄 median = {df_all['tle_age_days'].median():.2f} 天")
        if base.median() > 100:
            print("\n  " + "!" * 90)
            print("  [!] 基线仍不合理（>100 km）。先跑 --diagnose 复核坐标约定与单位；")
            print("      若约定无误，则是 TLE 与星历不同期（见 spacex_ephemeris_diagnosis.py 的结论）。")
            print("  " + "!" * 90)
        else:
            print("  -> 基线在合理量级，可用于论文。")

    # ---------- 3. 尾部指标（仅大机动子集） ----------
    print(f"\n[自检 3] 大机动子集上的逐样本 η_i 尾部指标（这才是跨数据源泛化的证据）")
    if len(df) == 0:
        print("  无样本通过 peak 筛选 —— 本批星历不含可用的独立大机动事件。")
    for label, col in (("Transformer+软掩码", "eta_trans_soft"),
                       ("LSTM+软掩码", "eta_lstm_soft")):
        if col not in df.columns or len(df) == 0:
            continue
        e = df[col].dropna().values
        t = tail_metrics(e)
        if t is None:
            continue
        lo_dm, hi_dm = bootstrap_ci(e, _deg_median)
        lo_cv, hi_cv = bootstrap_ci(e, _cvar10)
        lo_w, hi_w = bootstrap_ci(e, lambda v: float(np.min(v)))
        print(f"\n  {label}  (n={t['n']})")
        print(f"    优于SGP4比例        : {t['superior']:.1f}%")
        _imp = t['imp_median']
        _deg = t['deg_median']
        print("    改善中位数          : " + ("n/a（该子集无改善样本）" if _imp != _imp else f"{_imp:+.2f}%"))
        print("    退化中位数          : " + ("n/a（该子集无退化样本）" if _deg != _deg
              else f"{_deg:+.2f}%   95%CI [{lo_dm:.2f}, {hi_dm:.2f}]"))
        print(f"    CVaR10（最差10%均值）: {t['cvar10']:+.2f}%   95%CI [{lo_cv:.2f}, {hi_cv:.2f}]")
        print(f"    最差单样本          : {t['worst']:+.2f}%   95%CI [{lo_w:.2f}, {hi_w:.2f}]")
        print("    P(η<-10/-20/-30/-50): " +
              "  ".join(f"{x}%:{t['p_crit'][x]:.2f}" for x in (10, 20, 30, 50)))
    print(f"\n  参考（主实验 TLE 验证集 n=817，论文口径=内层验证最优）: "
          f"Transformer+软掩码 优于SGP4 73.44% / 改善中位 +17.45 / 退化中位 -9.07 / "
          f"CVaR10 -35.84 / 最差 -96.22")

    # ---------- 4. 同量纲误差对比 ----------
    print(f"\n[自检 4] 同量纲误差对比（mean RMSE, km）")
    for tag, sub in (("大机动子集", df), ("未筛选全集", df_all)):
        if len(sub) == 0:
            continue
        for label, ai in (("Transformer+软掩码", "ai_rmse_alt"),
                          ("LSTM+软掩码", "lstm_ai_rmse_alt")):
            if ai not in sub.columns:
                continue
            print(f"  [{tag}] {label:>18}: SGP4 {sub['sgp4_rmse_alt'].mean():9.2f} km -> "
                  f"AI {sub[ai].mean():9.2f} km")
    print("\n  注意：'残差均值'(trans/lstm_residual_mean) 是【修正量的窗口均值】，"
          "不是剩余误差；\n       与 SGP4 误差比较时应使用上面的同量纲数值。")

    # ---------- 5. 本底风险 p_S（必须在未筛选样本上估计） ----------
    print(f"\n[自检 5] 非机动样本的 SGP4 失效概率 p_S（论文现用假设 U[0.5%, 5%]）")
    nm_all = df_all[~df_all['has_maneuver'].astype(bool)] if len(df_all) else df_all
    nm_big = df[~df['has_maneuver'].astype(bool)] if len(df) else df
    print(f"  未筛选非机动样本 n={len(nm_all)}；大机动子集内的'非机动' n={len(nm_big)}")
    for tag, sub in (("未筛选(用于 p_S)", nm_all), ("仅大机动子集(有选择偏差)", nm_big)):
        if len(sub) == 0:
            continue
        print(f"  [{tag}] " + "  ".join(
            f"P(RMSE>{d}km)={100*(sub['sgp4_rmse_alt'] > d).mean():5.2f}%" for d in (10, 15, 20)))
        print(f"  [{tag}] RMSE median={sub['sgp4_rmse_alt'].median():.2f} km  "
              f"P95={pctl(sub['sgp4_rmse_alt'], 0.95):.2f} km  "
              f"max={sub['sgp4_rmse_alt'].max():.2f} km")
    print("  提示：'仅大机动子集'一行几乎必然偏高——那是选择偏差，不要用它当 p_S。")
    print("  提示：p_S 的窗口为 3 天（星历跨度）；论文定义为 72 小时窗口内高度 RMSE > 15 km，")
    print("        与本估计一致，但样本期只有 2026-08-03 起的一段，需在论文中注明。")


# ================== 主函数 ==================
def main():
    if '--diagnose' in sys.argv:
        idx = sys.argv.index('--diagnose')
        n = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 5
        print("=" * 96)
        print("星历列诊断（只用前若干文件，不加载模型）")
        print("=" * 96)
        diagnose_ephemeris_files(n)
        return

    print("=" * 96)
    print("独立星历验证 v5.1（修正真值重构 + 自动判定坐标约定 + 未筛选 p_S）")
    print(f"POSITION_MODE = {POSITION_MODE}   APPLY_U_MEAN_FILTER = {APPLY_U_MEAN_FILTER}   "
          f"MIN_SGP4_PEAK_KM = {MIN_SGP4_PEAK_KM}")
    print("=" * 96)

    scalers = (np.load(f'{PREPROCESSED_DIR}/scaler_cond_mean.npy').astype(np.float32),
               np.load(f'{PREPROCESSED_DIR}/scaler_cond_std.npy').astype(np.float32),
               np.load(f'{MASKED_DIR}/scaler_res_mean.npy').astype(np.float32),
               np.load(f'{MASKED_DIR}/scaler_res_std.npy').astype(np.float32))
    print("加载 Transformer+软掩码 ...")
    trans_models = load_ensemble(MODEL_PREFIX_TRANS, 'transformer', DEVICE)
    print(f"  {len(trans_models)} 个模型")
    print("加载 LSTM+软掩码 ...")
    lstm_models = load_ensemble(MODEL_PREFIX_LSTM, 'lstm', DEVICE)
    print(f"  {len(lstm_models)} 个模型")

    files = sorted(f for f in os.listdir(SPACEX_DIR) if f.endswith('.txt'))
    print(f"\n星历文件 {len(files)} 个\n")

    results, skipped_no_tle = [], 0
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(files)}  已收 {len(results)} 条")
        nid = extract_norad_from_filename(f)
        if nid is None:
            continue
        tle_file = os.path.join(TLE_DIR, f'tle_{nid}.txt')
        if not os.path.exists(tle_file):
            skipped_no_tle += 1
            continue
        r = validate_single_satellite(os.path.join(SPACEX_DIR, f), tle_file, nid,
                                      trans_models or None, lstm_models or None, scalers)
        if r is not None:
            results.append(r)

    if not results:
        print("\n无有效样本。请先运行:  python validate_with_space_v5.py --diagnose 5")
        return

    df_all = pd.DataFrame(results)
    df_all.to_csv(OUTPUT_CSV, index=False)
    df = df_all[df_all['passes_peak']].reset_index(drop=True)

    report(df_all, df, skipped_no_tle)
    print(f"\n输出: {OUTPUT_CSV}  （含全部样本，含 passes_peak 标记）")


if __name__ == '__main__':
    main()
