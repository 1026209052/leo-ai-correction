import pandas as pd
import numpy as np

# ==================== config ====================
INPUT_FILE = 'training_satellites_trajectory.csv'   # from the previous step
OUTPUT_FILE = 'maneuver_timepoints.csv'             # detected manoeuvre time points

R_EARTH = 6371.0
GM = 3.986e5
MILE_TO_KM = 1.60934
THRESHOLD_FACTOR = 5

# ==================== load ====================
print("Reading training-satellite trajectory ...")
df = pd.read_csv(INPUT_FILE)
df['pubtime'] = pd.to_datetime(df['pubtime'], format='%Y/%m/%d %H:%M')
df = df.sort_values(['NORADID', 'pubtime'])

# SPEEDMIS is in miles/s
df['velocity_kms'] = df['SPEEDMIS'] * MILE_TO_KM

# ==================== per-satellite manoeuvre detection ====================
all_maneuvers = []  # every detected manoeuvre time point

for norad_id, group in df.groupby('NORADID'):
    if len(group) < 3:
        continue
    
    group = group.copy()
    group['r'] = R_EARTH + group['ALTITUDEKM']
    group['specific_energy'] = 0.5 * group['velocity_kms']**2 - GM / group['r']
    group['energy_diff'] = group['specific_energy'].diff()
    
    std = group['energy_diff'].std()
    if pd.isna(std) or std == 0:
        continue
    
    threshold = THRESHOLD_FACTOR * std
    
    # the first difference is NaN -> treat as False
    maneuver_mask = abs(group['energy_diff']) > threshold
    maneuver_mask = maneuver_mask.fillna(False)
    
    # collect the time points
    maneuver_times = group.loc[maneuver_mask, 'pubtime']
    
    for t in maneuver_times:
        all_maneuvers.append({
            'NORADID': norad_id,
            'maneuver_time': t
        })
    
        print(f"satellite {norad_id}: {len(maneuver_times)} manoeuvre time points")

# ==================== save ====================
if len(all_maneuvers) == 0:
    print("No manoeuvre time points detected; check the data or the threshold.")
else:
    maneuver_df = pd.DataFrame(all_maneuvers)
    maneuver_df = maneuver_df.sort_values(['NORADID', 'maneuver_time'])
    maneuver_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nsaved to: {OUTPUT_FILE}")
    print(f"{len(maneuver_df)} manoeuvre time points in total")
    
    # show a few time points per satellite
    for sat in maneuver_df['NORADID'].unique():
        sat_data = maneuver_df[maneuver_df['NORADID'] == sat]
        print(f"\nsatellite {sat}, first 5 manoeuvre time points:")
        print(sat_data['maneuver_time'].head(5).to_string(index=False))