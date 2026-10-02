import pandas as pd
import numpy as np
from datetime import timedelta

# ==================== config ====================
TRAJECTORY_FILE = 'training_satellites_trajectory.csv'  # input trajectory
MANEUVER_FILE = 'maneuver_timepoints.csv'               # manoeuvre time points
OUTPUT_PAIRS_FILE = 'training_pairs.csv'                # output: window-pair index

WINDOW_BEFORE = timedelta(hours=24)                    # conditioning window: 24 h before
WINDOW_AFTER = timedelta(hours=72)                     # target window: 72 h after
SAMPLING_INTERVAL = timedelta(minutes=5)               # sampling interval
MIN_POINTS_BEFORE = 200                                # min points in the conditioning window
MIN_POINTS_AFTER = 600                                 # min points in the target window

# ==================== load ====================
print("Reading trajectory ...")
df = pd.read_csv(TRAJECTORY_FILE)
df['pubtime'] = pd.to_datetime(df['pubtime'], format='%Y/%m/%d %H:%M')
df = df.sort_values(['NORADID', 'pubtime'])

print("Reading manoeuvre time points ...")
maneuver_df = pd.read_csv(MANEUVER_FILE)
maneuver_df['maneuver_time'] = pd.to_datetime(maneuver_df['maneuver_time'])

# ==================== build window pairs per satellite ====================
training_pairs = []  # one entry per accepted window pair

for norad_id in maneuver_df['NORADID'].unique():
    print(f"satellite {norad_id} ...")
    
    # all trajectory rows of this satellite
    sat_traj = df[df['NORADID'] == norad_id].set_index('pubtime').sort_index()
    
    # all manoeuvre time points of this satellite
    sat_maneuvers = maneuver_df[maneuver_df['NORADID'] == norad_id]['maneuver_time']
    sat_maneuvers = sorted(sat_maneuvers)
    
    for idx, m_time in enumerate(sat_maneuvers):
        # window boundaries
        cond_start = m_time - WINDOW_BEFORE
        cond_end = m_time - SAMPLING_INTERVAL  # last sample before the manoeuvre
        tgt_start = m_time + SAMPLING_INTERVAL   # first sample after the manoeuvre
        tgt_end = m_time + WINDOW_AFTER
        
        # enough data points?
        cond_data = sat_traj.loc[cond_start:cond_end]
        tgt_data = sat_traj.loc[tgt_start:tgt_end]
        
        if len(cond_data) < MIN_POINTS_BEFORE or len(tgt_data) < MIN_POINTS_AFTER:
            continue
        
        # record the pair (indices are kept so the data can be extracted later)
        training_pairs.append({
            'norad_id': norad_id,
            'maneuver_time': m_time,
            'maneuver_idx': idx,
            'cond_start': cond_start,
            'cond_end': cond_end,
            'tgt_start': tgt_start,
            'tgt_end': tgt_end,
            'cond_points': len(cond_data),
            'tgt_points': len(tgt_data)
        })

# ==================== save ====================
if len(training_pairs) == 0:
    print("No window pairs produced; check the data or relax the point limits.")
else:
    pairs_df = pd.DataFrame(training_pairs)
    pairs_df.to_csv(OUTPUT_PAIRS_FILE, index=False)
    print(f"\nsaved to: {OUTPUT_PAIRS_FILE}")
    print(f"{len(pairs_df)} window pairs")
    print("\npairs per satellite:")
    print(pairs_df['norad_id'].value_counts().to_string())

# ==================== show one example ====================
print("\n===== example window pair =====")
sample = pairs_df.iloc[0]
sat_data = df[df['NORADID'] == sample['norad_id']].set_index('pubtime').sort_index()

cond_data = sat_data.loc[sample['cond_start']:sample['cond_end']]
tgt_data = sat_data.loc[sample['tgt_start']:sample['tgt_end']]

print(f"NORAD ID: {sample['norad_id']}")
print(f"manoeuvre time: {sample['maneuver_time']}")
print(f"conditioning window: {sample['cond_start']} .. {sample['cond_end']}  ({len(cond_data)} points)")
print(f"target window: {sample['tgt_start']} .. {sample['tgt_end']}  ({len(tgt_data)} points)")
print(f"\nfirst 5 rows of the conditioning window:")
print(cond_data[['ALTITUDEKM', 'velocity_kms']].head())
print(f"\nfirst 5 rows of the target window:")
print(tgt_data[['ALTITUDEKM', 'velocity_kms']].head())