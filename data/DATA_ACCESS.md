
# Obtaining the raw TLE input

**The raw TLE archive is intentionally *not* redistributed.** The element sets are U.S. Government
space situational awareness data, and the Space-Track User Agreement that every registered user accepts
forbids passing them on:

> "The User agrees **not to transfer any data or technical information received from this website, or
> other U.S. Government source, including the analysis of data, to any other entity without prior
> express approval.**"
> - Space-Track User Agreement (the agreement itself cites 10 U.S.C. 2274(c)(2))

The same agreement also constrains *use*:

> "A TLE AVAILABLE TO THE PUBLIC SHOULD NOT BE USED FOR CONJUNCTION ASSESSMENT PREDICTION."

Accordingly **no Zenodo dataset DOI is issued for `starlink_tle.csv`**, and the upload helper that was
previously kept here (`zenodo_upload.py`) has been removed. Everything downstream - the preprocessing
chain, the derived per-sample dataset and all analysis code - is public and can be rebuilt from an
export you obtain yourself under your own Space-Track account.

---

## The export behind the manuscript

| | |
|---|---|
| Source | Space-Track, 18th Space Defense Squadron - <https://www.space-track.org> (free account required) |
| Columns | `norad_id`, `tle`, `change_time`. The two TLE lines live inside a single CSV field, separated by a newline. |
| Rows | 4,572,892 |
| Satellites | **8,430** - the complete screening set of the manuscript |
| TLE epoch range | 2024-03-10 to 2025-08-09 |
| Size | 753.4 MB (789,979,062 bytes) |
| sha256 | `0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608` |

The sha256 identifies *the file the authors used*. It is recorded for provenance, not as a download
check: a fresh export will not reproduce the same bytes, because Space-Track serves live records. What
must match for reproduction is the **object set** and the **epoch range**.

## Retrieval recipe

1. Register at <https://www.space-track.org> (free) and accept the User Agreement.
2. Log in and export the historical element sets for the Starlink object set over 2024-03-10 to
   2025-08-09, with columns `norad_id, tle, change_time`. Use the API
   (`/basicspacedata/query/class/gp_history/...`) or the web **Download -> CSV** button; the web export
   caps a single request at **10,000 records**, so batch over the object set.
3. Save as `starlink_tle.csv` next to the `preprocess/` scripts (or here in `data/`).
4. Run the chain in [`../preprocess/README.md`](../preprocess/README.md). Step 1 sorts the rows by TLE
   epoch itself, so the export order does not matter.

## Coverage check after exporting

| Set | Satellites | Expected |
|---|---|---|
| Screening set (manoeuvre detection) | 8,430 | all present |
| Dataset reported in the manuscript | 879 | all present |

The 879-satellite list is itself public, in the derived data
([`../figures/data/sample_metadata.csv`](../figures/data/sample_metadata.csv)), so coverage can be
verified without asking the authors for anything.

## Notes

* 4,512 of the 4,572,892 rows (0.1 %) carry an epoch the parser cannot read from line 1; they are
  skipped during propagation and recorded in `tle_errors.csv`.
* `change_time` is not used by the code; it is kept for traceability.
* [`../.gitignore`](../.gitignore) excludes `starlink_tle.csv`, so the archive cannot be committed by
  accident.

## What *is* released

`figures/data/` holds the **authors' own analysis products** - per-sample prediction errors, gate
statistics, coverage tables. They are released under the repository licence (MIT for code, CC BY 4.0 for
derived data). They contain **no TLE strings**, and none of them can be used to recover orbital element
sets or to support conjunction assessment.
