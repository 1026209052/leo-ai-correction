import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import os
import sys

# ==================== config ====================
TRAJECTORY_FILE = 'starlink_trajectory_500.csv'
TLE_FILE = 'starlink_tle_500.csv'
PAIRS_FILE = 'training_pairs_500_final.csv'

R_EARTH = 6371.0
GM = 3.986e5
COND_LEN = 288
TGT_LEN = 864

# ==================== helpers ====================
def get_tle_epoch(sat_tle):
    """Epoch of a Satrec object."""
    try:
        return sat_tle.epoch
    except:
        year = int(sat_tle.epochyr)
        doy = sat_tle.epochdays
        full_year = 2000 + year if year < 57 else 1900 + year
        jan1 = datetime(full_year, 1, 1)
        return jan1 + timedelta(days=doy - 1)

class SGP4Propagator:
    """Minimal SGP4 propagator loaded straight from the TLE table."""
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
        """Newest TLE at or before target_time."""
        records = self.sats.get(str(norad_id).strip(), [])
        best = None
        for sat in records:
            epoch = get_tle_epoch(sat)
            if epoch <= target_time:
                if best is None or epoch > get_tle_epoch(best):
                    best = sat
        return best

    def propagate(self, norad_id, start_time, duration_hours=72, step_minutes=5):
        """Forecast trajectory starting at start_time."""
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
    print("Loading ...")
    # TLE history
    tle_df = pd.read_csv(TLE_FILE)
    tle_df['norad_id'] = tle_df['norad_id'].astype(str).str.strip()
    sgp4 = SGP4Propagator(tle_df)
    
    # window pairs and trajectory
    pairs_df = pd.read_csv(PAIRS_FILE)
    pairs_df['norad_id'] = pairs_df['norad_id'].astype(str).str.strip()
    
    traj_df = pd.read_csv(TRAJECTORY_FILE)
    traj_df['pubtime'] = pd.to_datetime(traj_df['pubtime'])
    traj_df['NORADID'] = traj_df['NORADID'].astype(str).str.strip()
    
    # cache trajectories
    print("Caching trajectories ...")
    sat_traj_cache = {}
    for nid in pairs_df['norad_id'].unique():
        sat = traj_df[traj_df['NORADID'] == nid].sort_values('pubtime')
        if len(sat) >= COND_LEN + TGT_LEN:
            sat_traj_cache[nid] = sat
    
    tle_ages_days = []
    sgp4_start_errors = []
    maneuver_magnitudes = []
    
    print("Analysing samples ...")
    valid_count = 0
    for idx, row in pairs_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(pairs_df)}")
        
        nid = str(row['norad_id']).strip()
        m_time = pd.to_datetime(row['maneuver_time'])
        
        # newest TLE before the manoeuvre
        best_sat = sgp4.get_tle_before(nid, m_time)
        if best_sat is None:
            continue
        
        tle_epoch = get_tle_epoch(best_sat)
        tle_age = (m_time - tle_epoch).total_seconds() / 86400.0
        
        # conditioning window from the stitched series
        if nid not in sat_traj_cache:
            continue
        
        sat = sat_traj_cache[nid]
        m_epoch = m_time.timestamp()
        times_arr = sat['pubtime'].astype(np.int64).values // 10**9
        
        idx_time = np.searchsorted(times_arr, m_epoch)
        if idx_time >= len(times_arr):
            idx_time = len(times_arr) - 1
        if idx_time > 0 and (m_epoch - times_arr[idx_time-1] < times_arr[idx_time] - m_epoch):
            idx_time -= 1
        
        tgt_start = idx_time
        tgt_end = min(len(times_arr), idx_time + TGT_LEN)
        if (tgt_end - tgt_start) < TGT_LEN:
            continue
        
        # reference altitude at the window start
        true_tgt = sat.iloc[tgt_start:tgt_end][['ALTITUDEKM', 'velocity_kms']].values.astype(np.float32)
        if len(true_tgt) < TGT_LEN:
            continue
        
        # SGP4 altitude at the window start
        start_time = pd.Timestamp(sat['pubtime'].iloc[idx_time])
        sgp4_df = sgp4.propagate(nid, start_time)
        if sgp4_df is None or len(sgp4_df) < TGT_LEN:
            continue
        sgp4_tgt = sgp4_df[['ALTITUDEKM', 'velocity_kms']].values[:TGT_LEN].astype(np.float32)
        
        # metrics
        start_error = np.abs(true_tgt[0, 0] - sgp4_tgt[0, 0])
        maneuver_mag = np.max(np.abs(true_tgt[:, 0] - sgp4_tgt[:, 0]))
        
        tle_ages_days.append(tle_age)
        sgp4_start_errors.append(start_error)
        maneuver_magnitudes.append(maneuver_mag)
        valid_count += 1
    
    tle_ages_days = np.array(tle_ages_days)
    sgp4_start_errors = np.array(sgp4_start_errors)
    maneuver_magnitudes = np.array(maneuver_magnitudes)
    
    print(f"\nsamples analysed: {valid_count}")
    
    # binning
    age_bins = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 180, 365]
    bin_labels = [f'{age_bins[i]}-{age_bins[i+1]}d' for i in range(len(age_bins)-1)]
    
    bin_medians = []
    bin_means = []
    bin_counts = []
    bin_p95s = []
    
    for i in range(len(age_bins)-1):
        mask = (tle_ages_days >= age_bins[i]) & (tle_ages_days < age_bins[i+1])
        if mask.sum() > 0:
            bin_medians.append(np.median(sgp4_start_errors[mask]))
            bin_means.append(np.mean(sgp4_start_errors[mask]))
            bin_counts.append(mask.sum())
            bin_p95s.append(np.percentile(sgp4_start_errors[mask], 95))
        else:
            bin_medians.append(0)
            bin_means.append(0)
            bin_counts.append(0)
            bin_p95s.append(0)
    
    # plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # fig 1: TLE age distribution
    ax1 = axes[0, 0]
    ax1.hist(tle_ages_days, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(x=7, color='red', linestyle='--', linewidth=2, label='7 days')
    ax1.axvline(x=14, color='orange', linestyle='--', linewidth=2, label='14 days')
    ax1.axvline(x=30, color='darkred', linestyle='--', linewidth=2, label='30 days')
    ax1.set_xlabel('TLE Age (days)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of TLE Ages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # fig 2: start error per age bin
    ax2 = axes[0, 1]
    x_pos = np.arange(len(bin_labels))
    width = 0.35
    ax2.bar(x_pos - width/2, bin_medians, width, label='Median Error', color='steelblue', alpha=0.7)
    ax2.bar(x_pos + width/2, bin_p95s, width, label='95th Percentile', color='darkorange', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.set_ylabel('SGP4 Start Error (km)')
    ax2.set_title('SGP4 Start Error by TLE Age Bin')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # fig 3: sample count per age bin
    ax3 = axes[0, 2]
    ax3.bar(x_pos, bin_counts, color='steelblue', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Sample Count by TLE Age Bin')
    ax3.grid(True, alpha=0.3)
    
    # fig 4: scatter
    ax4 = axes[1, 0]
    ax4.scatter(tle_ages_days, sgp4_start_errors, alpha=0.3, s=5, color='steelblue')
    ax4.axhline(y=10, color='orange', linestyle='--', label='10 km')
    ax4.axhline(y=20, color='red', linestyle='--', label='20 km')
    ax4.axvline(x=7, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=14, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('TLE Age (days)')
    ax4.set_ylabel('SGP4 Start Error (km)')
    ax4.set_title('TLE Age vs SGP4 Start Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # fig 5: mean +/- std
    ax5 = axes[1, 1]
    bin_stds = [np.std(sgp4_start_errors[(tle_ages_days >= age_bins[i]) & (tle_ages_days < age_bins[i+1])]) 
                if ((tle_ages_days >= age_bins[i]) & (tle_ages_days < age_bins[i+1])).sum() > 0 else 0 
                for i in range(len(age_bins)-1)]
    
    ax5.errorbar(x_pos, bin_means, yerr=bin_stds, fmt='o-', color='steelblue', 
                 capsize=5, markersize=8, linewidth=2, label='Mean ± Std')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax5.set_ylabel('SGP4 Start Error (km)')
    ax5.set_title('Mean SGP4 Start Error by TLE Age (with Std Dev)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # fig 6: manoeuvre magnitude vs TLE age
    ax6 = axes[1, 2]
    ax6.scatter(tle_ages_days, maneuver_magnitudes, alpha=0.3, s=5, color='steelblue')
    ax6.axvline(x=7, color='gray', linestyle=':', alpha=0.5)
    ax6.axvline(x=14, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('TLE Age (days)')
    ax6.set_ylabel('Maneuver Magnitude (km)')
    ax6.set_title('TLE Age vs Maneuver Magnitude')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tle_age_analysis.png', dpi=150)
    plt.show()
    
    # per-bin table
    print(f"\n{'='*80}")
    print("        SGP4 START ERROR BY TLE AGE")
    print(f"{'='*80}")
    print(f"{'age (d)':<12} {'n':<8} {'mean (km)':<12} {'median (km)':<12} {'p95 (km)':<15} {'std (km)':<12}")
    print("-" * 80)
    for i in range(len(bin_labels)):
        print(f"{bin_labels[i]:<12} {bin_counts[i]:<8} {bin_means[i]:<12.2f} {bin_medians[i]:<12.2f} {bin_p95s[i]:<15.2f} {bin_stds[i]:<12.2f}")
    
    # suggested threshold
    baseline_median = np.median(sgp4_start_errors[tle_ages_days < 2])
    print(f"\nbaseline median (TLE age < 2 d): {baseline_median:.2f} km")
    
    for i in range(len(bin_labels)):
        if bin_medians[i] > baseline_median * 3 and bin_counts[i] > 10:
            print(f"\nsuggested TLE-freshness threshold: {age_bins[i]} days")
            print(f"reason: in this bin the median start error ({bin_medians[i]:.2f} km) exceeds 3x the baseline ({baseline_median:.2f} km)")
            break

if __name__ == '__main__':
    main()