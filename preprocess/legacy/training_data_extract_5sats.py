import pandas as pd

# Input trajectory (stitched SGP4 series generated from the TLE history)
trajectory_file = 'starlink_trajectory_sgp4.csv'

# NORAD IDs of the 5 training satellites
training_sats = [64689, 64728, 64667, 64735, 64614]

# load the full trajectory file
print("Reading trajectory data ...")
df = pd.read_csv(trajectory_file)
print(f"raw rows: {len(df)}")

# keep only the training satellites
df_train = df[df['NORADID'].isin(training_sats)]
print(f"rows kept: {len(df_train)}")
print(f"satellites: {df_train['NORADID'].unique().tolist()}")

# save
output_file = 'training_satellites_trajectory.csv'
df_train.to_csv(output_file, index=False)
print(f"saved to: {output_file}")