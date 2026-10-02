import pandas as pd
import numpy as np
from datetime import datetime
import os
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

TRAJECTORY_FILE = 'starlink_trajectory_sgp4.csv'
MANEUVER_SUMMARY_FILE = 'maneuver_summary_starlink.csv'
OUTPUT_DIR = 'preprocessed_samples'
COND_LEN = 288
TGT_LEN = 864
NUM_NON_MANEUVER_SATS = 500   # raised from 200 to 500
SAMPLES_PER_SAT = 12           # raised from 10 to 12
R_EARTH = 6371.0
GM = 3.986e5

def main():
    set_seed(42)
    
    # 1. zero-manoeuvre satellites
    print("Reading the manoeuvre summary ...")
    summary_df = pd.read_csv(MANEUVER_SUMMARY_FILE)
    summary_df['NORADID'] = summary_df['NORADID'].astype(str).str.strip()
    zero_maneuver_sats = summary_df[summary_df['机动次数'] == 0]['NORADID'].tolist()
    print(f"zero-manoeuvre satellites: {len(zero_maneuver_sats)}")
    
    # 2. existing metadata
    meta_path = f'{OUTPUT_DIR}/sample_metadata.csv'
    existing_meta = pd.read_csv(meta_path)
    existing_meta['norad_id'] = existing_meta['norad_id'].astype(str).str.strip()
    existing_sats = set(existing_meta['norad_id'].unique())
    existing_max_idx = existing_meta['sample_idx'].max()
    
    # 3. only satellites that are not already in the dataset
    candidate_sats = [s for s in zero_maneuver_sats if s not in existing_sats]
    print(f"candidates: {len(candidate_sats)}")
    
    if len(candidate_sats) == 0:
        print("ERROR: no candidate satellite available")
        return
    
    # 4. random draw
    rng = np.random.RandomState(42)
    n_sample_sats = min(NUM_NON_MANEUVER_SATS, len(candidate_sats))
    selected_sats = rng.choice(candidate_sats, size=n_sample_sats, replace=False)
    selected_set = set(selected_sats)
    print(f"selected {n_sample_sats} satellites, {SAMPLES_PER_SAT} windows each")
    
    # 5. load their trajectories
    print("Reading trajectories ...")
    traj_chunks = []
    for chunk in pd.read_csv(TRAJECTORY_FILE, chunksize=500000):
        chunk['NORADID'] = chunk['NORADID'].astype(str).str.strip()
        filtered = chunk[chunk['NORADID'].isin(selected_set)]
        if len(filtered) > 0:
            traj_chunks.append(filtered)
    traj_df = pd.concat(traj_chunks, ignore_index=True)
    traj_df['pubtime'] = pd.to_datetime(traj_df['pubtime'])
    
    # 6. build non-manoeuvre windows
    print("Building non-manoeuvre samples ...")
    new_metadata = []
    sample_idx = existing_max_idx + 1
    total_generated = 0
    
    for sat_id in selected_sats:
        sat_data = traj_df[traj_df['NORADID'] == sat_id].sort_values('pubtime')
        if len(sat_data) < COND_LEN + TGT_LEN:
            continue
        
        # random start indices
        max_start = len(sat_data) - TGT_LEN - 1
        possible_starts = np.arange(COND_LEN, max_start)
        n_samples = min(SAMPLES_PER_SAT, len(possible_starts))
        chosen_starts = rng.choice(possible_starts, size=n_samples, replace=False)
        
        for start_idx in chosen_starts:
            cond_start = start_idx - COND_LEN
            cond_abs = sat_data.iloc[cond_start:start_idx][['ALTITUDEKM', 'velocity_kms']].values.astype(np.float32)
            if len(cond_abs) < COND_LEN:
                continue
            
            true_tgt = sat_data.iloc[start_idx:start_idx + TGT_LEN][['ALTITUDEKM', 'velocity_kms']].values.astype(np.float32)
            if len(true_tgt) < TGT_LEN:
                continue
            
            # key design choice: the training target residual is zero
            residual = np.zeros((TGT_LEN, 2), dtype=np.float32)
            sgp4_tgt = true_tgt.copy()
            
            np.save(f'{OUTPUT_DIR}/sample_{sample_idx:05d}_cond.npy', cond_abs)
            np.save(f'{OUTPUT_DIR}/sample_{sample_idx:05d}_res.npy', residual)
            np.save(f'{OUTPUT_DIR}/sample_{sample_idx:05d}_true.npy', true_tgt)
            np.save(f'{OUTPUT_DIR}/sample_{sample_idx:05d}_sgp4.npy', sgp4_tgt)
            
            new_metadata.append({
                'sample_idx': sample_idx,
                'norad_id': sat_id,
                'is_non_maneuver': True
            })
            sample_idx += 1
            total_generated += 1
    
    # 7. update the metadata
    existing_meta['is_non_maneuver'] = False
    combined_meta = pd.concat([existing_meta, pd.DataFrame(new_metadata)], ignore_index=True)
    combined_meta.to_csv(meta_path, index=False)
    
    # summary
    total_maneuver = len(existing_meta)
    total_non = len(new_metadata)
    total = total_maneuver + total_non
    
    print(f"\n{'='*60}")
    print("Non-manoeuvre samples built.")
    print(f"{'='*60}")
    print(f"existing (manoeuvre) samples: {total_maneuver}")
    print(f"new (non-manoeuvre) samples: {total_non}")
    print(f"total samples: {total}")
    print(f"non-manoeuvre share: {total_non/total*100:.1f}%")
    print(f"metadata updated: {meta_path}")

if __name__ == '__main__':
    main()