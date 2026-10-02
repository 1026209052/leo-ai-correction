# `data/` — the raw TLE input

## `starlink_tle.csv` is not included

The two-line element history behind the manuscript comes from **Space-Track** (18th Space Defense
Squadron): <https://www.space-track.org> — free, but a registered account is required. Download your
own copy from there and place it in this folder (or next to the `preprocess/` scripts) as
`starlink_tle.csv`. `preprocess/tle_to_trajectory.py` sorts the rows by TLE epoch itself, so the export
order does not matter.

It is **not** redistributed here or anywhere else. The Space-Track User Agreement requires a user

> "not to transfer any data or technical information received from this website, or other U.S. Government
> source, including the analysis of data, to any other entity without prior express approval."
> (10 U.S.C. 2274(c)(2))

**What the paper used** — columns `norad_id, tle, change_time`; 4,572,892 rows over 8,430 satellites;
TLE epochs 2024-03-10 to 2025-08-09; 753.4 MB with sha256
`0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608`. The hash identifies the authors'
file for provenance; a fresh download will not be byte-identical, because Space-Track serves live records.

[`../.gitignore`](../.gitignore) excludes `starlink_tle.csv`, so the archive cannot be committed by
accident.
