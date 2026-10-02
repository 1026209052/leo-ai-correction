"""tle_to_trajectory.py -- turn a TLE history into an altitude/speed time series.

Pipeline step 1 of the data-preparation chain (see ../README.md, Part B).

For every TLE the orbit is propagated with SGP4 over [epoch - 1 h, epoch + 12 h]
at a 5-minute step. The per-TLE predictions are then stitched into one continuous
series using the rule "at any instant the newest TLE wins", i.e. at every instant the
series shows what the then-current TLE predicts. Because SGP4 cannot
follow a manoeuvre, the stitched series shows a step in the specific mechanical
energy exactly at the manoeuvre; that step is what mechanical_energy.py and
tranning_data_extract_500_pair.py detect.

Input  : starlink_tle.csv   (columns: norad_id, tle)
Output : starlink_trajectory_sgp4.csv
             (columns: NORADID, pubtime, ALTITUDEKM, LATITUDE, LONGITUDE,
                       SPEEDMIS, velocity_kms)
         tle_errors.csv     (records that failed to propagate)
"""
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

# ==================== config ====================
TLE_FILE = 'starlink_tle.csv'                     # input: TLE history (norad_id, tle)
OUTPUT_FILE = 'starlink_trajectory_sgp4.csv'      # output: stitched 5-min series
ERROR_LOG_FILE = 'tle_errors.csv'   # log of failed records

STEP_MINUTES = 5                    # sampling interval (minutes)
HOURS_BEFORE = 1                    # hours propagated backwards from each TLE epoch
HOURS_AFTER = 12                    # hours propagated forward from each TLE epoch
R_EARTH = 6371.0                    # mean Earth radius (km)

# ==================== 1. read the TLE history ====================
print("Reading TLE history ...")
df = pd.read_csv(TLE_FILE)
print(f"records: {len(df)}")
print(f"columns: {df.columns.tolist()}")

# required columns
if 'norad_id' not in df.columns:
    raise ValueError("missing column 'norad_id' in the CSV")
if 'tle' not in df.columns:
    raise ValueError("missing column 'tle' in the CSV")

# ==================== 2. TLE cleaning / splitting ====================
def clean_tle_text(tle_text):
    """Strip quotes and normalise newlines in the TLE text."""
    text = str(tle_text).strip()
    text = text.replace('"', '').replace("'", '')
    text = text.replace('\\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
    return text

def split_tle(tle_text):
    """
    Split a TLE blob into its two lines.
    Returns (line1, line2); (None, None) on failure.
    """
    text = clean_tle_text(tle_text)
    lines = text.split('\n')
    
    line1, line2 = None, None
    for line in lines:
        line = line.strip()
        if line.startswith('1 '):
            line1 = line
        elif line.startswith('2 '):
            line2 = line
    
    # no standard prefix found: fall back to the first two lines
    if line1 is None and len(lines) >= 2:
        line1 = lines[0].strip()
        if not line1.startswith('1'):
            line1 = '1 ' + line1
    if line2 is None and len(lines) >= 2:
        line2 = lines[1].strip()
        if not line2.startswith('2'):
            line2 = '2 ' + line2
    
    return line1, line2

# ==================== 3. epoch parsed from line 1 ====================
def extract_epoch_from_tle(line1):
    """Parse the epoch from TLE line 1 (does not rely on satellite.epoch)."""
    try:
        epoch_str = line1[18:32].strip()
        year = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])
        
        # year: 50-99 -> 1950-1999, 00-49 -> 2000-2049
        if year >= 50:
            full_year = 1900 + year
        else:
            full_year = 2000 + year
        
        day_int = int(day_of_year)
        day_frac = day_of_year - day_int
        
        jan1 = datetime(full_year, 1, 1)
        epoch_dt = jan1 + timedelta(days=day_int - 1, hours=day_frac * 24)
        return epoch_dt
    except:
        return None

# ==================== 4. SGP4 propagation ====================
def tle_to_trajectory(norad_id, line1, line2, step_minutes=5,
                      hours_before=1, hours_after=12):
    """Propagate a single TLE into discrete trajectory points."""
    try:
        satellite = Satrec.twoline2rv(line1, line2)
        epoch = extract_epoch_from_tle(line1)
        
        if epoch is None:
            return pd.DataFrame()
        
        start_time = epoch - timedelta(hours=hours_before)
        end_time = epoch + timedelta(hours=hours_after)
        
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        num_steps = total_minutes // step_minutes + 1
        
        results = []
        for i in range(num_steps):
            t = start_time + timedelta(minutes=i * step_minutes)
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
            e, r, v = satellite.sgp4(jd, fr)
            
            if e != 0:
                continue
            
            x, y, z = r
            vx, vy, vz = v
            
            velocity = np.sqrt(vx**2 + vy**2 + vz**2)
            altitude = np.sqrt(x**2 + y**2 + z**2) - R_EARTH
            
            lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
            lon = np.degrees(np.arctan2(y, x))
            
            results.append({
                'NORADID': norad_id,
                'pubtime': t.strftime('%Y/%m/%d %H:%M'),
                'ALTITUDEKM': round(altitude, 6),
                'LATITUDE': round(lat, 6),
                'LONGITUDE': round(lon, 6),
                'SPEEDMIS': round(velocity / 1.60934, 6),
                'velocity_kms': round(velocity, 6)
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        raise e

# ==================== 4.5 explicit TLE ordering (do not rely on file order) ====================
# The stitching below uses drop_duplicates(keep='first'): for a given instant it keeps the row
# that was encountered first. To reproduce 'the newest TLE at that instant wins', the newest
# TLE must be encountered first. Sorting explicitly here makes the result independent of the
# row order of the input file (it used to rely on starlink_tle.csv being sorted newest-first).
_ep = df['tle'].map(lambda s: extract_epoch_from_tle(split_tle(s)[0] or ''))
_na = int(_ep.isna().sum())
if _na:
    print('  WARNING: %d rows have an unparsable TLE epoch; sorted last' % _na)
df = (df.assign(_epoch=_ep)
        .sort_values(['norad_id', '_epoch'], ascending=[True, False], na_position='last')
        .drop(columns='_epoch')
        .reset_index(drop=True))
print('sorted by (norad_id, TLE epoch) descending: %d rows' % len(df))

# ==================== 5. batch processing ====================
print("\nPropagating TLEs ...")
all_trajectories = []
error_records = []

success = 0
fail = 0

for idx, row in df.iterrows():
    norad_id = row['norad_id']
    tle_text = row['tle']
    
    # split the TLE
    line1, line2 = split_tle(tle_text)
    
    # check the split
    if not line1 or not line2:
        fail += 1
        error_records.append({
            'index': idx,
            'norad_id': norad_id,
            'error_type': 'TLE split failed (empty line)',
            'tle_preview': str(tle_text)[:100]
        })
        continue
    
    # check the line prefixes
    if not line1.strip().startswith('1'):
        fail += 1
        error_records.append({
            'index': idx,
            'norad_id': norad_id,
            'error_type': f'line1 has an unexpected prefix: {line1[:15]}',
            'tle_preview': str(tle_text)[:100]
        })
        continue
    
    if not line2.strip().startswith('2'):
        fail += 1
        error_records.append({
            'index': idx,
            'norad_id': norad_id,
            'error_type': f'line2 has an unexpected prefix: {line2[:15]}',
            'tle_preview': str(tle_text)[:100]
        })
        continue
    
    # propagate
    try:
        traj = tle_to_trajectory(norad_id, line1, line2,
                                 STEP_MINUTES, HOURS_BEFORE, HOURS_AFTER)
        if len(traj) > 0:
            all_trajectories.append(traj)
            success += 1
        else:
            fail += 1
            error_records.append({
                'index': idx,
                'norad_id': norad_id,
                'error_type': 'SGP4 returned an empty trajectory',
                'tle_preview': str(tle_text)[:100]
            })
    except Exception as e:
        fail += 1
        error_records.append({
            'index': idx,
            'norad_id': norad_id,
            'error_type': f'SGP4 error: {str(e)[:100]}',
            'tle_preview': str(tle_text)[:100]
        })
    
    # progress
    if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
        print(f"  {idx+1}/{len(df)} | ok: {success} | failed: {fail}")

# ==================== 6. error log ====================
if error_records:
    error_df = pd.DataFrame(error_records)
    error_df.to_csv(ERROR_LOG_FILE, index=False)
    print(f"\nfailed records written to: {ERROR_LOG_FILE}")
    print("failure reasons:")
    print(error_df['error_type'].value_counts().to_string())

# ==================== 7. merge and save ====================
if len(all_trajectories) == 0:
    print("\nERROR: no trajectory points produced. Check the TLE data.")
else:
    print("\nMerging ...")
    result_df = pd.concat(all_trajectories, ignore_index=True)
    result_df = result_df.sort_values(['NORADID', 'pubtime'])
    result_df = result_df.drop_duplicates(subset=['NORADID', 'pubtime'])
    
    print(f"trajectory points: {len(result_df)}")
    print(f"satellites: {result_df['NORADID'].nunique()}")
    print(f"time range: {result_df['pubtime'].min()} .. {result_df['pubtime'].max()}")
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"saved to: {OUTPUT_FILE}")
    print('Stitching complete; ready for manoeuvre detection.')