# `preprocess/` — from a TLE history to the training samples

This folder turns the raw TLE history into the `preprocessed_samples/` set that the models are trained
on. It is the first part of the pipeline; the training and evaluation scripts live in `../analysis/`.

## The chain

```
                       starlink_tle.csv            <- the only external input (Space-Track)
                        (columns: norad_id, tle)
                                |
   (1) tle_to_trajectory.py     |  propagate every TLE with SGP4 over [epoch-1h, epoch+12h]
                                |  at 5-min steps, then stitch: the newest TLE wins at every instant
                                v
                     starlink_trajectory_sgp4.csv  (NORADID, pubtime, ALTITUDEKM, LATITUDE,
                                |                   LONGITUDE, SPEEDMIS, velocity_kms)
   (2) mechanical_energy.py     |  per-satellite detection of steps in the specific mechanical
                                |  energy (threshold = 5 sigma of the energy difference)
                                v
                  maneuver_summary_starlink.csv    (also deposited in ../figures/data/)
                                |
   (3) training_data_extract_500_pair.py
                                |  stratified sample of 500 manoeuvre-active satellites
                                |  (high/mid/low = 50/200/250), per-satellite detection,
                                |  and conditioning/target window pairs (288 + 864 points)
                                v
                     training_pairs_500.csv
                                |
   (4) preprocess_500_satellites.py
                                |  re-derives the same 500 satellites with the same seed and
                                |  materialises the subsets used from here on
                                v
   starlink_trajectory_500.csv + starlink_tle_500.csv + training_pairs_500_final.csv
                                |
   (5) preprocess_sgp4_residuals.py
                                |  SGP4 baseline = latest TLE BEFORE the manoeuvre;
                                |  reference trajectory = the stitched series AFTER it;
                                |  residual = reference - SGP4
                                v
                     preprocessed_samples/
                     ├── sample_{idx}_cond.npy    (288 x 2)  conditioning window
                     ├── sample_{idx}_res.npy     (864 x 2)  training target residual
                     ├── sample_{idx}_true.npy    (864 x 2)  reference trajectory
                     ├── sample_{idx}_sgp4.npy    (864 x 2)  SGP4 forecast
                     ├── scaler_cond_mean/std.npy, scaler_res_mean/std.npy
                     └── sample_metadata.csv      (sample_idx, norad_id)
                                |
   (6) mechanical_energy_no.py  |  appends 500 zero-manoeuvre satellites x 12 windows, with the
                                |  reference set equal to the SGP4 forecast and a zero target
                                v
                     preprocessed_samples/sample_metadata.csv  (manoeuvre + non-manoeuvre)
```

## Required inputs

| Input | Where it comes from | Deposited here? |
|---|---|---|
| `starlink_tle.csv` | Space-Track (<https://www.space-track.org>). Columns `norad_id`, `tle` (an
extra `change_time` column is ignored). One row per TLE; any row order is fine — step 1 sorts it
explicitly. | **Yes, via Zenodo** — the complete 4,572,892-row / 8,430-satellite history
(753 MB) ships with the Zenodo record, because it is too large for Git. See
[`../data/README.md`](../data/README.md). |

Every other file in the chain is produced by these scripts, and `maneuver_summary_starlink.csv` is also
deposited in `../figures/data/` so that the Part-A analyses can run without re-running the chain.

## Scripts

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `tle_to_trajectory.py` | `starlink_tle.csv` | `starlink_trajectory_sgp4.csv`, `tle_errors.csv` |
| 2 | `mechanical_energy.py` | `starlink_trajectory_sgp4.csv` | `maneuver_summary_starlink.csv` |
| 3 | `training_data_extract_500_pair.py` | `maneuver_summary_starlink.csv`, `starlink_trajectory_sgp4.csv`, `starlink_tle.csv` | `training_pairs_500.csv` |
| 4 | `preprocess_500_satellites.py` | the three above + `training_pairs_500.csv` | `starlink_trajectory_500.csv`, `starlink_tle_500.csv`, `training_pairs_500_final.csv` |
| 5 | `preprocess_sgp4_residuals.py` | the three `*_500` files | `preprocessed_samples/` |
| 6 | `mechanical_energy_no.py` | `starlink_trajectory_sgp4.csv`, `maneuver_summary_starlink.csv`, `preprocessed_samples/sample_metadata.csv` | appends to `preprocessed_samples/` |
| — | `analyze_tle_age.py` | the three `*_500` files | `tle_age_analysis.png` (diagnostic only) |

## Design decisions worth knowing

* **The SGP4 baseline and the reference trajectory are time-ordered but independent.** The baseline is
  propagated from the latest TLE *before* the manoeuvre, while the reference trajectory comes from the
  TLEs published *after* it (see the manuscript, §4.1.1). This is what makes the residual a
  manoeuvre-induced quantity rather than a data-source artefact.
* **Non-manoeuvre samples are negative by construction.** For the 500 zero-manoeuvre satellites the
  reference is set equal to the SGP4 forecast and the training target is zero, i.e. "output no correction
  where SGP4 is trustworthy" — not an observed zero error (manuscript §4.1).
* **The 3-day TLE-freshness filter is justified, not assumed.** `preprocess_sgp4_residuals.py` discards
  samples whose TLE is older than `MAX_TLE_AGE_DAYS = 3`. `analyze_tle_age.py` is the diagnostic behind
  that number: it bins the samples by TLE age and reports the first bin whose median SGP4 start error
  exceeds three times the baseline (TLE age < 2 days). Residual guard rails are |d altitude| <= 500 km
  and |d velocity| <= 1 km/s.
* **Steps 3 and 4 sample the same 500 satellites.** Both call the same stratified sampler with the same
  seed; step 4 exists to materialise the subsets and to consistency-check the window-pair file.
* **Satellites, not samples, are counted downstream.** A satellite all of whose samples are filtered out
  in steps 5–6 disappears from the dataset: 500 sampled manoeuvre-active satellites yield the 382 that
  survive the filters, and 500 zero-manoeuvre satellites yield 497 (12 windows each = 5,964 samples).

## A note on the data files

`maneuver_summary_starlink.csv` uses **Chinese column names** (`数据点数`, `机动次数`, `平均高度_km`,
`高度最小值_km`, `高度最大值_km`). They are part of the deposited data and are therefore kept verbatim in
the code that reads the file; everything else in these scripts is in English.

## `legacy/`

`legacy/training_data_extract_5sats.py`, `legacy/maneuver_timepoints.py` and `legacy/training_pairs.py`
are an earlier **5-satellite debug branch**. They are superseded by steps 3–5 and are kept only for
traceability. They are not part of the chain above.

## Running it

```bash
cd preprocess
cp ../data/starlink_tle.csv .           # after downloading it from the Zenodo record
python tle_to_trajectory.py             # needs starlink_tle.csv in this directory
python mechanical_energy.py
python training_data_extract_500_pair.py
python preprocess_500_satellites.py
python preprocess_sgp4_residuals.py     # writes preprocessed_samples/
python mechanical_energy_no.py          # must run AFTER step 5
```

Requirements: `numpy`, `pandas`, `sgp4`; `matplotlib` for `analyze_tle_age.py`.

`starlink_trajectory_sgp4.csv` and `maneuver_summary_starlink.csv` are the large intermediates: the
processes are chunked where possible, but step 3 reads the full trajectory file.

> Note: `analyze_tle_age.py` ends with `plt.show()`, which blocks until the window is closed. Comment it
> out for headless runs.
