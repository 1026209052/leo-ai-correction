
# `data/` - the raw TLE input

## `starlink_tle.csv` is **not** redistributed

The complete 4,572,892-row / 8,430-satellite Starlink TLE history behind the manuscript is *not* in this
repository and is *not* deposited anywhere, including Zenodo.

| | |
|---|---|
| Source | Space-Track, 18th Space Defense Squadron - <https://www.space-track.org> (free account required) |
| Columns | `norad_id`, `tle`, `change_time` (both TLE lines inside one CSV field, newline-separated) |
| Rows | 4,572,892 |
| Satellites | **8,430** - the complete screening set of the manuscript |
| TLE epoch range | 2024-03-10 to 2025-08-09 |
| Size | 753.4 MB (789,979,062 bytes) |
| sha256 | `0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608` |

**Why.** The Space-Track User Agreement that every registered user accepts states:

> "The User agrees not to transfer any data or technical information received from this website, or
> other U.S. Government source, including the analysis of data, to any other entity without prior
> express approval." (10 U.S.C. 2274(c)(2))

The archive is therefore excluded from Git *and* from any dataset deposit.

**What to do instead.** Export it yourself under your own Space-Track account. The exact object set,
epoch range, retrieval steps and the coverage check you can run afterwards are in
[`DATA_ACCESS.md`](DATA_ACCESS.md).

The sha256 above describes the file the authors used; it is a provenance record, not a download check
(a fresh export will not be byte-identical).

## Coverage verified on the authors' file

| Set | Satellites | Present |
|---|---|---|
| Screening set (manoeuvre detection) | 8,430 | 100 % |
| Dataset reported in the manuscript | 879 | 100 % |

## Notes

* `preprocess/tle_to_trajectory.py` sorts the rows by TLE epoch itself, so the export order does not
  matter.
* **4,512 of the 4,572,892 rows (0.1 %)** carry an epoch the parser cannot read from line 1; they are
  skipped during propagation and recorded in `tle_errors.csv`.
* `change_time` is not used by the code; it is kept for traceability.
