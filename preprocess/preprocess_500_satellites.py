import pandas as pd
import numpy as np

# ==================== config ====================
MANEUVER_SUMMARY_FILE = 'maneuver_summary_starlink.csv'      # manoeuvre-detection summary
TRAJECTORY_FILE = 'starlink_trajectory_sgp4.csv'    # full stitched trajectory
TLE_FILE = 'starlink_tle.csv'                       # full TLE history
PAIRS_FILE = 'training_pairs_500.csv'               # window pairs from training_data_extract_500_pair.py

OUTPUT_TRAJECTORY = 'starlink_trajectory_500.csv'   # output: trajectory of the 500 satellites
OUTPUT_TLE = 'starlink_tle_500.csv'                 # output: TLE of the 500 satellites
OUTPUT_PAIRS = 'training_pairs_500_final.csv'       # output: final window pairs

# ==================== stratified sampling (same seed as step 3) ====================
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

# ==================== main ====================
def main():
    # 1. stratified sample of 500 satellites
    print("=" * 60)
    print("step 1: stratified sample of 500 satellites")
    print("=" * 60)
    summary_df = pd.read_csv(MANEUVER_SUMMARY_FILE)
    summary_df['NORADID'] = summary_df['NORADID'].astype(str).str.strip()
    sample_sat_ids = stratified_sample_satellites(summary_df, total_sample=500, seed=42)
    sample_set = set(sample_sat_ids)  # set for fast filtering

    # 2. extract the sampled satellites from the full trajectory
    print(f"\n{'='*60}")
    print("step 2: chunked read of the full trajectory")
    print("=" * 60)
    chunk_size = 500000
    traj_chunks = []
    total_rows = 0
    for chunk in pd.read_csv(TRAJECTORY_FILE, chunksize=chunk_size):
        # normalise the id format
        chunk['NORADID'] = chunk['NORADID'].astype(str).str.strip()
        filtered = chunk[chunk['NORADID'].isin(sample_set)]
        if len(filtered) > 0:
            traj_chunks.append(filtered)
            total_rows += len(filtered)
        if total_rows > 0 and total_rows % 1000000 == 0:
            print(f"  extracted {total_rows} trajectory rows ...")
    
    traj_df = pd.concat(traj_chunks, ignore_index=True)
    traj_df.to_csv(OUTPUT_TRAJECTORY, index=False)
    print(f"saved {OUTPUT_TRAJECTORY}")
    print(f"rows: {len(traj_df)}, satellites: {traj_df['NORADID'].nunique()}")

    # 3. extract the sampled satellites from the TLE history
    print(f"\n{'='*60}")
    print("step 3: extract TLEs")
    print("=" * 60)
    tle_df = pd.read_csv(TLE_FILE)
    tle_df['norad_id'] = tle_df['norad_id'].astype(str).str.strip()
    # filter
    tle_df = tle_df[tle_df['norad_id'].isin(sample_set)]
    tle_df.to_csv(OUTPUT_TLE, index=False)
    print(f"saved {OUTPUT_TLE}")
    print(f"records: {len(tle_df)}, satellites: {tle_df['norad_id'].nunique()}")

    # 4. filter the window pairs to the same 500 satellites
    print(f"\n{'='*60}")
    print("step 4: filter window pairs")
    print("=" * 60)
    pairs_df = pd.read_csv(PAIRS_FILE)
    pairs_df['norad_id'] = pairs_df['norad_id'].astype(str).str.strip()
    pairs_df = pairs_df[pairs_df['norad_id'].isin(sample_set)]
    pairs_df.to_csv(OUTPUT_PAIRS, index=False)
    print(f"saved {OUTPUT_PAIRS}")
    print(f"satellites: {pairs_df['norad_id'].nunique()}, pairs: {len(pairs_df)}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    print("three files written:")
    print(f"  1. {OUTPUT_TRAJECTORY}")
    print(f"  2. {OUTPUT_TLE}")
    print(f"  3. {OUTPUT_PAIRS}")
    print("\npoint the training script CONFIG at:")
    print(f"  'trajectory_csv': '{OUTPUT_TRAJECTORY}',")
    print(f"  'tle_csv': '{OUTPUT_TLE}',")
    print(f"  'pairs_csv': '{OUTPUT_PAIRS}',")
    print("training then reads only these small files, which starts much faster.")

if __name__ == '__main__':
    main()