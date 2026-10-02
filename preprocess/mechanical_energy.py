import pandas as pd
import numpy as np

# ==================== config ====================
DATA_FILE = 'starlink_trajectory_sgp4.csv'      # input trajectory
R_EARTH = 6371.0                     # mean Earth radius (km)
GM = 3.986e5                         # Earth gravitational parameter (km^3/s^2)
MILE_TO_KM = 1.60934                # miles -> km
THRESHOLD_FACTOR = 5                 # detection threshold, in standard deviations

# ==================== 1. load ====================
print("Reading data ...")
df = pd.read_csv(DATA_FILE)
print(f"loaded {len(df)} rows")

# ==================== 2. cleaning ====================
print("Cleaning ...")

# required columns must exist
required_cols = ['NORADID', 'pubtime', 'ALTITUDEKM', 'SPEEDMIS']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"missing column: {col}. Columns found: {df.columns.tolist()}")

# coerce numeric types
df['SPEEDMIS'] = pd.to_numeric(df['SPEEDMIS'], errors='coerce')
df['ALTITUDEKM'] = pd.to_numeric(df['ALTITUDEKM'], errors='coerce')

# rows before cleaning
before_clean = len(df)

# drop missing values
df = df.dropna(subset=['SPEEDMIS', 'ALTITUDEKM'])

# drop implausible values
df = df[df['SPEEDMIS'] > 0]
df = df[df['ALTITUDEKM'] > 50]
df = df[df['ALTITUDEKM'] < 50000]

after_clean = len(df)
print(f"cleaning done: dropped {before_clean - after_clean} rows, {after_clean} left")

# ==================== 3. time parsing and sorting ====================
print("Parsing timestamps ...")
try:
    df['pubtime'] = pd.to_datetime(df['pubtime'], format='%Y/%m/%d %H:%M')
except Exception:
    try:
        df['pubtime'] = pd.to_datetime(df['pubtime'])
    except Exception as e:
        raise ValueError(f"cannot parse the pubtime column: {e}")

df = df.sort_values(['NORADID', 'pubtime'])

# ==================== 4. speed unit conversion ====================
df['velocity_kms'] = df['SPEEDMIS'] * MILE_TO_KM

# ==================== 5. per-satellite manoeuvre detection ====================
print("Detecting manoeuvres ...")
print("-" * 50)

summary_list = []
total_satellites = df['NORADID'].nunique()
processed_count = 0

for norad_id, group in df.groupby('NORADID'):
    processed_count += 1
    if processed_count % 20 == 0:
        print(f"progress: {processed_count}/{total_satellites}")
    
    # skip satellites with too few data points
    if len(group) < 3:
        continue
    
    group = group.copy()
    
    # geocentric radius
    group['r'] = R_EARTH + group['ALTITUDEKM']
    
    # specific mechanical energy
    group['specific_energy'] = 0.5 * group['velocity_kms']**2 - GM / group['r']
    
    # first difference
    group['energy_diff'] = group['specific_energy'].diff()
    
    # threshold
    std = group['energy_diff'].std()
    if pd.isna(std) or std == 0:
        continue
    
    threshold = THRESHOLD_FACTOR * std
    
    # count manoeuvres
    maneuver_mask = abs(group['energy_diff']) > threshold
    # the first difference is NaN
    maneuver_mask = maneuver_mask.fillna(False)
    maneuver_count = maneuver_mask.sum()
    
    summary_list.append({
        'NORADID': norad_id,
        '数据点数': len(group),
        '机动次数': maneuver_count,
        '平均高度_km': round(group['ALTITUDEKM'].mean(), 2),
        '高度最小值_km': round(group['ALTITUDEKM'].min(), 2),
        '高度最大值_km': round(group['ALTITUDEKM'].max(), 2)
    })

# ==================== 6. output ====================
print("-" * 50)

if len(summary_list) == 0:
    print("ERROR: no satellite met the criteria.")
    print("Check (1) enough data points, (2) plausible speed values.")
else:
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('机动次数', ascending=False)
    
    # summary
    total_sats = len(summary_df)
    sats_with_maneuver = sum(summary_df['机动次数'] > 0)
    total_maneuvers = summary_df['机动次数'].sum()
    
    print("\n" + "=" * 60)
    print("                MANOEUVRE DETECTION SUMMARY")
    print("=" * 60)
    print(f"  satellites:            {total_sats}")
    print(f"  with manoeuvres:       {sats_with_maneuver}")
    print(f"  without manoeuvres:    {total_sats - sats_with_maneuver}")
    print(f"  manoeuvre time points: {total_maneuvers}")
    print(f"  mean per satellite:    {total_maneuvers/total_sats:.2f}")
    print("=" * 60)
    
    print(f"\n10 most manoeuvred satellites:")
    print("-" * 60)
    top10 = summary_df.head(10)
    for i, row in top10.iterrows():
        print(f"  {row['NORADID']}:  {row['机动次数']} manoeuvres, "
              f"mean altitude {row['平均高度_km']} km, "
              f"{row['数据点数']} data points")
    
    # altitude distribution
    print(f"\nmanoeuvres by altitude band:")
    print("-" * 60)
    height_bins = [(50, 500), (500, 900), (900, 2000), (2000, 50000)]
    for low, high in height_bins:
        subset = summary_df[(summary_df['平均高度_km'] >= low) & (summary_df['平均高度_km'] < high)]
        if len(subset) > 0:
            print(f"  {low}-{high} km: {len(subset)} satellites, "
                  f"{sum(subset['机动次数'] > 0)} with manoeuvres, "
                  f"{subset['机动次数'].sum()} time points")
    
    # save the summary
    summary_df.to_csv('maneuver_summary_starlink.csv', index=False, encoding='utf-8-sig')
    print(f"\nsummary written to maneuver_summary_starlink.csv")