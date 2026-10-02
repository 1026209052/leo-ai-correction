# `data/` — raw input

## `starlink_tle.csv` — the full TLE history behind the manuscript

| | |
|---|---|
| Source | Space-Track, 18th Space Defense Squadron — <https://www.space-track.org> (free account required) |
| Columns | `norad_id`, `tle`, `change_time`. The two TLE lines live inside a single CSV field, separated by a newline. |
| Rows | 4,572,892 |
| Satellites | **8,430** — the complete screening set of the manuscript |
| TLE epoch range | 2024-03-10 … 2025-08-09 |
| Size | **753 MB** (789,979,062 bytes) |
| sha256 | `0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608` |

Coverage verified on this exact file:

| Set | Satellites | Present in the file |
|---|---|---|
| Screening set (manoeuvre detection) | 8,430 | **100 %** |
| Dataset reported in the manuscript | 879 | **100 %** |

The epoch range starts on 2024-03-10, consistent with the start of the observation spans recorded in
`../figures/data/satellite_observation_spans.csv` (2024-03-11).

---

## ⚠️ This file is **not** in the Git repository

At 753 MB it exceeds GitHub's 100 MB per-file limit (and Git LFS's free tier). It is therefore
**excluded via `../.gitignore`** and distributed with the rest of the release through **Zenodo**:

> **Download:** `starlink_tle.csv` from the Zenodo **dataset** record — `10.5281/zenodo.XXXXXXX`
> <!-- TODO: fill in after uploading; ready-to-paste metadata is in ZENODO_DATASET_RECORD.md.
>      The companion *software* record is 10.5281/zenodo.23098904. -->

Verify the download before using it:

```bash
sha256sum starlink_tle.csv        # expect 0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608
certutil -hashfile starlink_tle.csv SHA256      # Windows
```

Then place it here, or next to the `preprocess/` scripts, and run the chain
(see [`../preprocess/README.md`](../preprocess/README.md)).

---

## Notes

* `preprocess/tle_to_trajectory.py` sorts the rows by TLE epoch itself, so the export order does not
  matter.
* **4,512 of the 4,572,892 rows (0.1 %)** carry an epoch that the parser cannot read from line 1; they are
  skipped during propagation and recorded in `tle_errors.csv`.
* `change_time` is not used by the code; it is kept for traceability.

## Terms of use

TLE data are produced by the 18th Space Defense Squadron and distributed via Space-Track, whose User
Agreement governs their use. Check that agreement before redistributing this file.
