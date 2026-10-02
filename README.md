# Code and derived data — *Risk-constrained deployment of AI correction for LEO orbit prediction*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23098904.svg)](https://doi.org/10.5281/zenodo.23098904)

This repository accompanies the manuscript

> Yang K. **Risk-constrained deployment of AI correction for LEO orbit prediction: Physics-inspired soft
> masking and fail-safe gating.** *Reliability Engineering & System Safety* (submitted).

It contains the analysis code and the derived data needed to reproduce **every number and figure**
reported in the paper.

---

## What is *not* included, and why

| Not included | Reason |
|---|---|
| The raw TLE archive | [`data/starlink_tle.csv`](data/README.md) is the complete 4.57 M-row / 8,430-satellite history. It is 753 MB, which exceeds GitHub's 100 MB per-file limit, **and it is not redistributed anywhere**: Space-Track's User Agreement forbids transferring U.S. Government data to a third party. It is excluded via `.gitignore`; download your own copy from Space-Track — see [`data/README.md`](data/README.md). |
| Model checkpoints (`.pth`) | Large; can be regenerated with the Part-B scripts. |

---

## Repository layout

```
.
├── README.md
├── LICENSE                      # MIT (code); derived data are CC BY 4.0 — see Licence
├── requirements.txt
├── run_all_analyses.py          # runs every Part-A script and prints PASS/FAIL
├── .gitignore                     # excludes the 753 MB TLE archive (not redistributed; see data/README.md)
├── data/                        # raw TLE input (753 MB, not redistributed)
├── preprocess/                  # data-preparation chain (TLE -> training samples)
│   └── legacy/                  # superseded 5-satellite debug branch
├── analysis/                    # all analysis code
│   ├── <20 Part-A scripts>      # runnable here: reproduce every reported number
│   ├── <19 Part-B scripts>      # training / evaluation: need the preprocessed dataset + a GPU
│   └── out/                     # intermediate results consumed by Part A
└── figures/
    ├── data/                    # derived per-sample and summary CSV files (canonical)
    ├── make_fig2_data.py
    ├── build_figures.ps1
    └── fig{1..5}_*.{tex,pdf,png}
```

---

## Quick start

```bash
pip install -r requirements.txt     # numpy is sufficient for Part A
python run_all_analyses.py          # prints one table per script, then PASS 20 / 20
```

On Windows, if your console is not UTF-8, set the encoding first — several scripts print
Chinese-language console tables and would otherwise raise `UnicodeEncodeError`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_all_analyses.py
```

Verified on Python 3.12 with numpy 2.3 (no GPU, no raw data required).

---

## Part A — Reproduce every reported number

All 20 scripts read **only** from `figures/data/*.csv` and `analysis/out/*.csv`. Each prints its result
table to stdout; none of them needs a GPU or the raw TLE archive.

| Script | Reproduces | Primary input |
|---|---|---|
| `layer2_fmea.py` | §3.2.4 — probabilistic FMEA basic events: `p_H(theta)`, its CI, and the T2 decomposition (Tables 1–2) | `data/fig2_cdf.csv`, `data/sample_metadata.csv` |
| `c5_event_rate.py` | §3.2.4 / §5.5.3 — uncertainty of the manoeuvre event rate (events per satellite-year) | `data/maneuver_summary_starlink.csv`, `data/satellite_observation_spans.csv` |
| `layer2_table7_mc.py` | §5.5.1 — threshold scan: accuracy, safety and operating risk (Table 8) + Monte-Carlo T2 remapping | `data/fig2_cdf_all.csv`, `data/pareto_samples.csv` |
| `layer2_table8_cost.py` | §5.5.2 — one-sided Wilson risk calibration (Table 9) and the expected-cost rows of Table 2(b) | `data/fig2_cdf_all.csv`, `data/monte_carlo_results_v3.csv` |
| `check_table9_def.py` | §5.8 — convention behind the SGP4 column of the collision-warning failure indicator (Table 12) | `data/pareto_samples.csv` |
| `why_denominator.py` | §5.5.2 — why the degradation-median denominator differs between methods (Table 9 note) | `data/fig2_cdf.csv` |
| `c2_cluster_bootstrap.py` | Appendix B — satellite-level (cluster) paired bootstrap for the tail metrics, incl. the LSTM | `data/fig2_cdf.csv`, `data/per_sample_eta_lstm.csv`, `data/sample_metadata.csv` |
| `c2c_a02_extra.py` | Appendix B + §5.1 — LSTM vs LSTM+soft-mask cluster-paired bootstrap; degradation median on the intersection of degraded samples | `data/fig2_cdf_all.csv`, `data/per_sample_eta_lstm.csv` |
| `c2d_a02_wilson.py` | Appendix B — Wilson intervals and tail-ratio quantiles at the deployed setting | `data/fig2_cdf.csv` |
| `c3_block_crc.py` | §5.5.2 — satellite-level (block) vs i.i.d. Wilson risk calibration | `data/pareto_samples.csv`, `data/sample_metadata.csv` |
| `c4_tau_max_bootstrap.py` | §5.9 — cluster bootstrap interval and effective sample size for the empirical zero-danger-miss boundary | `figures/data/*.csv` |
| `design_rule_alpha_tau.py` | §5.9 — the joint `(alpha, tau)` risk-constrained design rule | `out/alpha_sweep_metrics_inner_val_ckpt.csv`, `data/fig2_cdf_all.csv`, `data/pareto_samples.csv` |
| `fig5_design_rule_data.py` | Figure 5 input data (tau side model-free; alpha side from the alpha sweep) | same as above |
| `tail_analysis.py` | Achieved damage floor, bootstrap CIs, and the FMEA cross-threshold table | `data/fig2_cdf.csv` |
| `crosscheck_eta.py` | Independent recomputation of the six eta-derived metrics, used to cross-check the master verifier | `data/fig2_cdf.csv` |
| `table5_strata.py` | §5.2 — performance stratified by SGP4 baseline RMSE (Table 6) | `data/fig2_cdf.csv`, `data/pareto_samples.csv` |
| `a1_table6_definition.py` | §5.1 — how the two additional columns of the main comparison table (Table 5) are defined | `data/alpha_ablation_ensemble_results.csv`, `data/fig2_cdf.csv` |
| `c1_v5_analysis.py` | §5.7 — tail metrics on the corrected independent-source run (SpaceX operational ephemeris) | `data/spacex_v5_verification.csv` |
| `c1_independent_tail.py` | §5.7 — tail metrics on the earlier independent-source run | `data/spacex_68p4_verification.csv` |
| `s57_from_csv.py` | §5.7 — every quantity of Table 11 recomputed from the independent-source CSV | `data/spacex_v5_verification.csv` |

### Sanity check: `fig2_cdf.csv`

`figures/data/fig2_cdf.csv` holds the per-sample `eta` of the final model on the 817-sample
large-manoeuvre group, and reproduces the paper's headline tail statistics: superior-to-SGP4
65.5 % → 82.7 %, severe degradation 9.3 % → 3.3 %, worst single sample −199.8 % → −77.6 %. It is the
input to every Part-A script, including `tail_analysis.py` and `make_fig2_data.py`.

---

## Part B — Retraining the models

The training and evaluation scripts are included for completeness, but **the full chain cannot be run
from this repository alone.** It requires:

1. the **preprocessed dataset** (`preprocessed_samples/`, `preprocessed_samples_masked/`) — 879 satellites,
   14,799 samples. The data-preparation scripts are in [`preprocess/`](preprocess/), but that folder is
   **not self-contained** (see below);
2. a **CUDA GPU**: the full sweep is 11 alpha values x 6 random seeds, plus the standard-MSE, Huber,
   quantile and LSTM baselines.

### Data preparation (`preprocess/`)

See [`preprocess/README.md`](preprocess/README.md) for the full chain diagram, the required inputs and the
design decisions behind each step. In short, the chain is **complete** and needs only one external input:
the TLE history `starlink_tle.csv` (columns `norad_id`, `tle`), which you export yourself from Space-Track —
see [`data/README.md`](data/README.md) and [Data provenance](#data-provenance).

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `tle_to_trajectory.py` | `starlink_tle.csv` (753 MB; obtain from Space-Track — see [`data/README.md`](data/README.md)) | `starlink_trajectory_sgp4.csv` — every TLE propagated with SGP4 over [epoch − 1 h, epoch + 12 h] at 5-min steps, then stitched so that *the newest TLE wins at every instant* |
| 2 | `mechanical_energy.py` | `starlink_trajectory_sgp4.csv` | `maneuver_summary_starlink.csv` — per-satellite manoeuvre detection from jumps in specific mechanical energy (threshold = 5σ of the energy difference). *Also deposited in `figures/data/`* |
| 3 | `training_data_extract_500_pair.py` | `maneuver_summary_starlink.csv`, `starlink_trajectory_sgp4.csv`, `starlink_tle.csv` | `training_pairs_500.csv` — stratified sample of 500 manoeuvre-active satellites (high/mid/low frequency = 50/200/250), manoeuvre detection per satellite, and conditioning/target window pairs (288 + 864 points) |
| 4 | `preprocess_500_satellites.py` | the same three files + `training_pairs_500.csv` | `starlink_trajectory_500.csv`, `starlink_tle_500.csv`, `training_pairs_500_final.csv` — re-derives the identical 500-satellite sample (same seed) and materialises the subsets |
| 5 | `preprocess_sgp4_residuals.py` | the three `*_500` files | `preprocessed_samples/sample_{idx}_{cond,res,true,sgp4}.npy`, `scaler_*.npy`, `preprocessed_samples/sample_metadata.csv` — SGP4 baseline from the latest TLE *before* the manoeuvre, reference trajectory from the stitched series *after* it, residual = reference − SGP4 |
| 6 | `mechanical_energy_no.py` | `starlink_trajectory_sgp4.csv`, `maneuver_summary_starlink.csv`, `preprocessed_samples/sample_metadata.csv` | appends 500 zero-manoeuvre satellites × 12 windows to `preprocessed_samples/`, with the reference set equal to the SGP4 forecast and a zero training target by construction |

Step 5 applies the TLE-freshness filter (**≤ 3 days**, justified by `preprocess/analyze_tle_age.py`) and
residual guard rails on altitude (≤ 500 km) and velocity (≤ 1 km/s).

`analyze_tle_age.py` is a diagnostic script used to justify the 3-day TLE-freshness threshold: it bins
samples by TLE age and reports the first bin whose median SGP4 start error exceeds three times the
baseline (TLE age < 2 days).

`legacy/training_data_extract_5sats.py`, `legacy/maneuver_timepoints.py` and `legacy/training_pairs.py`
belong to an earlier **5-satellite debug branch**; they are superseded by steps 3–5 and are kept only
for traceability.

Sampling protocol, in one paragraph: manoeuvre events are detected per satellite from jumps in the
specific mechanical energy of the stitched trajectory series, using a threshold of five standard
deviations of the energy difference; the SGP4 baseline is propagated from the **latest TLE before** the
manoeuvre and the reference trajectory from the **TLE updated after** it, so the two are time-ordered and
independent (§4.1.1). Non-manoeuvre samples take the SGP4 forecast as the reference and a zero training
target by construction (§4.1).

| Script | Role |
|---|---|
| `train_alpha_inner_val.py` | soft-mask alpha ablation; early stopping on an inner validation set, with the evaluation satellites hard-isolated |
| `train_loss_baselines.py` | standard-MSE / Huber / quantile-regression baselines |
| `train_lstm_inner_val.py` | Bi-LSTM backbone |
| `select_alpha_inner_val.py` | alpha selection on the inner validation set |
| `eval_alpha_sweep.py`, `eval_loss_baselines.py` | evaluation, using exactly the conventions of `verify_all_tables.py` |
| `verify_all_tables.py` | master verifier: recomputes all main tables from the checkpoints |
| `export_eta.py`, `export_eta_lstm.py`, `export_eta_perseed.py`, `export_eta_qr.py` | export per-sample eta for each family |
| `gate_oracle_vs_proxy.py`, `fault_injection.py` | proxy-vs-Oracle gating; FMEA fault injection |
| `run_switch_a02.py`, `collect_switch_a02.py` | the deployed-configuration refit / collection driver |
| `validate_with_space_v5.py` | independent verification against the SpaceX operational ephemeris |
| `boot_alpha_pairs.py`, `debug_ensemble.py`, `ding_check.py` | auxiliary checks |

Training protocol, in one paragraph: the evaluation satellites are selected first (879 satellite ids,
`np.unique`, `RandomState(42)`, first 10 % = 87 satellites) and are then **excluded from every stage of
training** — fitting, early stopping and monitoring. Early stopping uses an inner validation split drawn
from the remaining training satellites only. Checkpoints are named
`error_predictor_alpha{alpha:.1f}_seed{seed}.pth` for the alpha family, `error_predictor_seed{seed}.pth`
for the standard-MSE baseline and `lstm_seed{seed}.pth` for the LSTM.

---

## Part C — Regenerating the figures

Figures 2–5 are drawn with TikZ:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File figures\build_figures.ps1
```

`build_figures.ps1` compiles each `figures/*.tex` with `pdflatex` in a temporary directory and copies the
resulting `fig{2..5}_*.pdf` and `.png` back. Figure 1 is a schematic supplied as vector PDF; its TikZ
source is not part of this release.

The Figure 2 ECDF grid is regenerated with:

```bash
python figures/make_fig2_data.py
```

This reads `figures/data/fig2_cdf.csv`, the canonical per-sample file (see the caveat above).

---

## Data provenance

* **Raw TLE data** — Space-Track (<https://www.space-track.org>), 18th Space Defense Squadron. Free, but
  requires an account. The complete history used here (4,572,892 TLEs, 8,430 satellites, epochs
  2024-03-10 … 2025-08-09) is **not redistributed**, here or on Zenodo: Space-Track's
  User Agreement forbids transferring U.S. Government data to any other entity without prior express
  approval (10 U.S.C. 2274(c)(2)). Export it yourself — see [`data/README.md`](data/README.md).
  The same agreement states that a public TLE "should not be used for conjunction assessment
  prediction"; the manuscript uses the collision-risk quantity only as an illustrative analogy (section 5.8).
* **Derived data** in `figures/data/` — produced by the authors from those TLEs (879 satellites,
  14,799 samples; 8,835 manoeuvre samples). Released here under CC BY 4.0.
* **Independent validation ephemeris** — public SpaceX operational ephemeris files; see §5.7 of the
  manuscript.
* **`satellite_observation_spans.csv`** — per-satellite observation span (`first_time`, `last_time`,
  `duration_years`) for all 8,430 screened satellites; used as the exposure denominator of the
  manoeuvre event-rate estimate (§3.2.4, §5.5.3). Together with `maneuver_summary_starlink.csv`
  it feeds `analysis/c5_event_rate.py`.

---

## Licence

* **Code** (`analysis/`, `figures/*.py`, `figures/*.ps1`, `run_all_analyses.py`) — MIT, see [LICENSE](LICENSE).
* **Derived data** (`figures/data/`) — CC BY 4.0.
* **Raw TLE data** (`data/starlink_tle.csv`) is third-party U.S. Government data (18 SDS, distributed
  via Space-Track). The authors assert no rights over it and claim no licence over it. It is excluded
  from Git and **not redistributed**, here or on Zenodo - see [`data/README.md`](data/README.md).

---

## How to cite

If you use this code or data, please cite the paper and the archived release:

```
Yang K. Code for "Risk-constrained deployment of AI correction for LEO orbit prediction:
Physics-inspired soft masking and fail-safe gating" [software]. Zenodo; 2026.
https://doi.org/10.5281/zenodo.23098904

The raw TLE input has **no DOI** and is not distributed. Cite its source directly:

Space-Track. Two-Line Element Sets. https://www.space-track.org (accessed 2026).
```

---

## Verification note

`run_all_analyses.py` exits with a non-zero status if any script fails. All 20 Part-A scripts pass
(**PASS 20 / 20**) — verified both in the working repository and from a copy of this repository placed
in an unrelated directory, so the suite depends only on files that ship with it.
