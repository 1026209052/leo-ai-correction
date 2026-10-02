import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ==================== config ====================
MANEUVER_SUMMARY_FILE = 'maneuver_summary_starlink.csv'  # manoeuvre-detection summary
TRAJECTORY_FILE = 'starlink_trajectory_sgp4.csv'  # full stitched trajectory (all 8,430 satellites)
TLE_FILE = 'starlink_tle.csv'  # TLE history
OUTPUT_PAIRS_FILE = 'training_pairs_500.csv'  # output: window pairs

COND_LEN = 288  # 24 h before
TGT_LEN = 864   # 72 h after
STEP_MINUTES = 5
R_EARTH = 6371.0
GM = 3.986e5
MILE_TO_KM = 1.60934
THRESHOLD_FACTOR = 5  # detection threshold, in standard deviations

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

# ==================== stratified sampling ====================
def stratified_sample_satellites(summary_df, total_sample=500, seed=42):
    rng = np.random.RandomState(seed)
    
    high_freq = summary_df[summary_df['机动次数'] >= 100]
    mid_freq = summary_df[(summary_df['机动次数'] >= 10) & (summary_df['机动次数'] < 100)]
    low_freq = summary_df[(summary_df['机动次数'] >= 1) & (summary_df['机动次数'] < 10)]
    
    print(f"high-freq: {len(high_freq)}, mid: {len(mid_freq)}, low: {len(low_freq)}")
    
    n_high = min(50, len(high_freq))
    n_mid = min(200, len(mid_freq))
    n_low = min(250, len(low_freq))
    
    sampled_high = high_freq.sample(n=n_high, random_state=seed)
    sampled_mid = mid_freq.sample(n=n_mid, random_state=seed)
    sampled_low = low_freq.sample(n=n_low, random_state=seed)
    
    sampled = pd.concat([sampled_high, sampled_mid, sampled_low])
    print(f"sampled: high {n_high} + mid {n_mid} + low {n_low} = {len(sampled)}")
    print(f"manoeuvre time points expected: {sampled['机动次数'].sum():.0f}")
    
    return sampled['NORADID'].astype(str).str.strip().tolist()

# ==================== manoeuvre detection ====================
def detect_maneuver_times(sat_data, threshold_factor=5):
    data = sat_data.sort_values('pubtime').reset_index(drop=True)
    if len(data) < 3:
        return []
    # SPEEDMIS is in miles/s
    data['velocity_kms'] = data['SPEEDMIS'] * MILE_TO_KM
    data['r'] = R_EARTH + data['ALTITUDEKM']
    data['specific_energy'] = 0.5 * data['velocity_kms']**2 - GM / data['r']
    data['energy_diff'] = data['specific_energy'].diff()
    std = data['energy_diff'].std()
    if pd.isna(std) or std == 0:
        return []
    threshold = threshold_factor * std
    mask = abs(data['energy_diff']) > threshold
    mask.iloc[0] = False
    return data.loc[mask, 'pubtime'].tolist()

# ==================== window construction ====================
def build_pairs(sat_data, maneuver_times, norad_id):
    sat_data = sat_data.sort_values('pubtime').reset_index(drop=True)
    times_epoch = sat_data['pubtime'].astype(np.int64) // 10**9
    
    pairs = []
    for m_time in maneuver_times:
        m_epoch = m_time.timestamp()
        idx = np.searchsorted(times_epoch, m_epoch)
        if idx >= len(times_epoch):
            idx = len(times_epoch) - 1
        if idx > 0 and (m_epoch - times_epoch.iloc[idx-1] < times_epoch.iloc[idx] - m_epoch):
            idx = idx - 1
        
        cond_start = max(0, idx - COND_LEN)
        tgt_start = idx
        tgt_end = min(len(times_epoch), idx + TGT_LEN)
        
        if (idx - cond_start) >= COND_LEN and (tgt_end - tgt_start) >= TGT_LEN:
            pairs.append({
                'norad_id': norad_id,
                'maneuver_time': m_time.strftime('%Y-%m-%d %H:%M:%S'),
                'cond_start': sat_data['pubtime'].iloc[cond_start].strftime('%Y-%m-%d %H:%M:%S'),
                'cond_end': sat_data['pubtime'].iloc[idx-1].strftime('%Y-%m-%d %H:%M:%S'),
                'tgt_start': sat_data['pubtime'].iloc[tgt_start].strftime('%Y-%m-%d %H:%M:%S'),
                'tgt_end': sat_data['pubtime'].iloc[tgt_end-1].strftime('%Y-%m-%d %H:%M:%S'),
                'cond_points': COND_LEN,
                'tgt_points': TGT_LEN
            })
    return pairs

# ==================== main ====================
def main():
    # 1. manoeuvre summary
    print("Reading the manoeuvre summary ...")
    summary_df = pd.read_csv(MANEUVER_SUMMARY_FILE)
    summary_df['NORADID'] = summary_df['NORADID'].astype(str).str.strip()
    
    # 2. stratified sample of 500 satellites
    print("\nStratified sampling of 500 satellites ...")
    sample_sat_ids = stratified_sample_satellites(summary_df, total_sample=500, seed=42)
    
    # 3. full trajectory (chunked, because the file is large)
    print("\nReading the trajectory (sampled satellites only) ...")
    # chunked read keeps memory bounded
    chunk_size = 500000  # 500k rows per chunk
    traj_chunks = []
    for chunk in pd.read_csv(TRAJECTORY_FILE, chunksize=chunk_size):
        chunk['NORADID'] = chunk['NORADID'].astype(str).str.strip()
        filtered = chunk[chunk['NORADID'].isin(sample_sat_ids)]
        if len(filtered) > 0:
            traj_chunks.append(filtered)
    traj_df = pd.concat(traj_chunks, ignore_index=True)
    traj_df['pubtime'] = pd.to_datetime(traj_df['pubtime'], format='%Y/%m/%d %H:%M')
    print(f"trajectory rows kept: {len(traj_df)}")
    
    # 4. TLE history (sampled satellites only)
    print("\nReading the TLE history ...")
    tle_df = pd.read_csv(TLE_FILE)
    tle_df['norad_id'] = tle_df['norad_id'].astype(str).str.strip()
    tle_df = tle_df[tle_df['norad_id'].isin(sample_sat_ids)]
    print(f"TLE records kept: {len(tle_df)}")
    
    # 5. SGP4 propagator
    sgp4 = SGP4Propagator(tle_df)
    
    # 6. detect manoeuvres and build the window pairs
    print("\nDetecting manoeuvres and building window pairs ...")
    all_pairs = []
    
    for norad_id in sample_sat_ids:
        sat_data = traj_df[traj_df['NORADID'] == norad_id]
        if len(sat_data) < COND_LEN + TGT_LEN:
            continue
        
        # detect manoeuvre time points
        maneuver_times = detect_maneuver_times(sat_data)
        if not maneuver_times:
            continue
        
        # build windows
        pairs = build_pairs(sat_data, maneuver_times, norad_id)
        all_pairs.extend(pairs)
        
        if len(sample_sat_ids) <= 10 or len(pairs) > 0:
            print(f"satellite {norad_id}: {len(maneuver_times)} points, {len(pairs)} windows")
    
    # 7. save
    if not all_pairs:
        raise RuntimeError("no window pair produced; check the data.")
    
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(OUTPUT_PAIRS_FILE, index=False)
    
    print("\n===== done =====")
    print(f"sampled satellites: {len(sample_sat_ids)}")
    print(f"satellites with pairs: {pairs_df['norad_id'].nunique()}")
    print(f"window pairs: {len(pairs_df)}")
    print(f"output: {OUTPUT_PAIRS_FILE}")

if __name__ == '__main__':
    main()