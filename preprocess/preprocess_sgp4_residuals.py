import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import os

# ==================== config ====================
TRAJECTORY_FILE = 'starlink_trajectory_500.csv'
TLE_FILE = 'starlink_tle_500.csv'
PAIRS_FILE = 'training_pairs_500_final.csv'
OUTPUT_DIR = 'preprocessed_samples'

COND_LEN = 288
TGT_LEN = 864
R_EARTH = 6371.0
GM = 3.986e5

# residual guard rails (samples outside the range are skipped)
MAX_ALTITUDE_RESIDUAL = 500.0   # km
MAX_VELOCITY_RESIDUAL = 1.0     # km/s

# TLE-freshness threshold (older samples are skipped)
MAX_TLE_AGE_DAYS = 3   # justified by preprocess/analyze_tle_age.py (see ../README.md, Part B)

# ==================== SGP4 propagator ====================
class SGP4Propagator:
    def __init__(self, tle_df):
        self.sats = {}
        tle_df = tle_df.copy()
        tle_df['norad_id'] = tle_df['norad_id'].astype(str).str.strip()
        for _, r in tle_df.iterrows():
            nid = r['norad_id']
            if nid not in self.sats:
                self.sats[nid] = []
            try:
                lines = str(r['tle']).strip().split('\n')
                if len(lines) >= 2:
                    sat = Satrec.twoline2rv(lines[0].strip(), lines[1].strip())
                    self.sats[nid].append(sat)
            except:
                pass

    def get_tle_before(self, norad_id, target_time):
        records = self.sats.get(str(norad_id).strip(), [])
        best = None
        for sat in records:
            try:
                epoch = sat.epoch
            except AttributeError:
                year = int(sat.epochyr)
                doy = sat.epochdays
                full_year = 2000 + year if year < 57 else 1900 + year
                jan1 = datetime(full_year, 1, 1)
                epoch = jan1 + timedelta(days=doy - 1)
            if epoch <= target_time:
                if best is None or epoch > best.epoch if hasattr(best, 'epoch') else True:
                    best = sat
        return best

    def propagate(self, norad_id, start_time, duration_hours=72, step_minutes=5):
        sat = self.get_tle_before(str(norad_id).strip(), start_time)
        if sat is None:
            return None
        times = [start_time + timedelta(minutes=i * step_minutes)
                 for i in range(int(duration_hours * 60 / step_minutes) + 1)]
        data = []
        for t in times:
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                continue
            x, y, z = r
            vx, vy, vz = v
            alt = np.sqrt(x**2 + y**2 + z**2) - R_EARTH
            vel = np.sqrt(vx**2 + vy**2 + vz**2)
            data.append({'pubtime': t, 'ALTITUDEKM': alt, 'velocity_kms': vel})
        return pd.DataFrame(data)

# ==================== main ====================
def main():
    # 1. load
    print("Reading trajectory and TLE data ...")
    traj_df = pd.read_csv(TRAJECTORY_FILE)
    traj_df['pubtime'] = pd.to_datetime(traj_df['pubtime'])
    traj_df['NORADID'] = traj_df['NORADID'].astype(str).str.strip()
    
    tle_df = pd.read_csv(TLE_FILE)
    tle_df['norad_id'] = tle_df['norad_id'].astype(str).str.strip()
    
    pairs_df = pd.read_csv(PAIRS_FILE)
    pairs_df['norad_id'] = pairs_df['norad_id'].astype(str).str.strip()
    
    sgp4 = SGP4Propagator(tle_df)
    
    # 2. output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. cache the trajectory per satellite
    print("Caching trajectories ...")
    sat_traj_cache = {}
    for nid in pairs_df['norad_id'].unique():
        sat = traj_df[traj_df['NORADID'] == nid].sort_values('pubtime')
        if len(sat) >= COND_LEN + TGT_LEN:
            sat_traj_cache[nid] = sat
    
    # 4. build and save every sample
    print("Computing SGP4 residuals and saving samples ...")
    all_cond = []
    all_residual = []
    metadata = []
    skipped_anomaly = 0
    skipped_tle_age = 0

    for idx, (_, row) in enumerate(pairs_df.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(pairs_df)}")
        
        nid = str(row['norad_id']).strip()
        if nid not in sat_traj_cache:
            continue
        
        sat = sat_traj_cache[nid]
        m_time = pd.to_datetime(row['maneuver_time'])
        m_epoch = m_time.timestamp()
        times_arr = sat['pubtime'].astype(np.int64).values // 10**9
        
        idx_time = np.searchsorted(times_arr, m_epoch)
        if idx_time >= len(times_arr):
            idx_time = len(times_arr) - 1
        if idx_time > 0 and (m_epoch - times_arr[idx_time-1] < times_arr[idx_time] - m_epoch):
            idx_time -= 1
        
        cond_start = max(0, idx_time - COND_LEN)
        tgt_start = idx_time
        tgt_end = min(len(times_arr), idx_time + TGT_LEN)
        
        if (idx_time - cond_start) < COND_LEN or (tgt_end - tgt_start) < TGT_LEN:
            continue
        
        # conditioning window
        cond_abs = sat.iloc[cond_start:idx_time][['ALTITUDEKM', 'velocity_kms']].values.astype(np.float32)
        # reference (target) window
        true_tgt = sat.iloc[tgt_start:tgt_end][['ALTITUDEKM', 'velocity_kms']].values.astype(np.float32)
        # SGP4 forecast
        start_time = pd.Timestamp(sat['pubtime'].iloc[idx_time])
        sgp4_df = sgp4.propagate(nid, start_time)
        if sgp4_df is None or len(sgp4_df) < TGT_LEN:
            continue
        sgp4_tgt = sgp4_df[['ALTITUDEKM', 'velocity_kms']].values[:TGT_LEN].astype(np.float32)
        residual = true_tgt - sgp4_tgt

        # ========== TLE freshness ==========
        sat_tle = sgp4.get_tle_before(nid, start_time)
        if sat_tle is not None:
            try:
                epoch = sat_tle.epoch
            except AttributeError:
                year = int(sat_tle.epochyr)
                doy = sat_tle.epochdays
                full_year = 2000 + year if year < 57 else 1900 + year
                jan1 = datetime(full_year, 1, 1)
                epoch = jan1 + timedelta(days=doy - 1)
            
            time_diff = start_time - epoch
            max_age = timedelta(days=MAX_TLE_AGE_DAYS)
            
            if time_diff > max_age:
                skipped_tle_age += 1
                continue
        # ====================================
        
        # ========== residual guard rails ==========
        if np.max(np.abs(residual[:, 0])) > MAX_ALTITUDE_RESIDUAL:
            skipped_anomaly += 1
            continue
        if np.max(np.abs(residual[:, 1])) > MAX_VELOCITY_RESIDUAL:
            skipped_anomaly += 1
            continue
        # ================================
        
        # save
        np.save(f'{OUTPUT_DIR}/sample_{idx:05d}_cond.npy', cond_abs)
        np.save(f'{OUTPUT_DIR}/sample_{idx:05d}_res.npy', residual)
        np.save(f'{OUTPUT_DIR}/sample_{idx:05d}_true.npy', true_tgt)
        np.save(f'{OUTPUT_DIR}/sample_{idx:05d}_sgp4.npy', sgp4_tgt)
        
        all_cond.append(cond_abs)
        all_residual.append(residual)
        metadata.append({'sample_idx': idx, 'norad_id': nid})
    
    if not all_cond:
        raise RuntimeError("no valid sample produced")
    
    # 5. scaler statistics (from the filtered samples)
    total_skipped = skipped_anomaly + skipped_tle_age
    print(f"\nScaler statistics ({total_skipped} samples skipped: {skipped_anomaly} outliers, {skipped_tle_age} stale TLE) ...")
    all_cond = np.concatenate(all_cond, axis=0)
    all_residual = np.concatenate(all_residual, axis=0)
    
    scaler_cond_mean = all_cond.mean(axis=0)
    scaler_cond_std = all_cond.std(axis=0)
    scaler_res_mean = all_residual.mean(axis=0)
    scaler_res_std = all_residual.std(axis=0)
    
    np.save(f'{OUTPUT_DIR}/scaler_cond_mean.npy', scaler_cond_mean)
    np.save(f'{OUTPUT_DIR}/scaler_cond_std.npy', scaler_cond_std)
    np.save(f'{OUTPUT_DIR}/scaler_res_mean.npy', scaler_res_mean)
    np.save(f'{OUTPUT_DIR}/scaler_res_std.npy', scaler_res_std)
    
    # 6. sample metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(f'{OUTPUT_DIR}/sample_metadata.csv', index=False)
    
    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    print(f"valid samples: {len(metadata)}")
    print(f"skipped (outliers): {skipped_anomaly}")
    print(f"skipped (stale TLE): {skipped_tle_age}")
    print(f"output directory: {OUTPUT_DIR}")
    print("\nscaler parameters:")
    print(f"  cond mean:  {scaler_cond_mean}")
    print(f"  cond std:   {scaler_cond_std}")
    print(f"  res mean:   {scaler_res_mean}")
    print(f"  res std:    {scaler_res_std}")

if __name__ == '__main__':
    main()