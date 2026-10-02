# Creating the Zenodo **dataset** record (for `starlink_tle.csv`)

The code already has its own DOI, created automatically by the GitHub integration:

> **software** — `10.5281/zenodo.23098904` (record `1026209052/leo-ai-correction`, release `v1.0.0`)

The 753 MB TLE archive is too large for GitHub, so it needs a **separate Zenodo record**. Two records is
also the right structure: the RESS Guide for Authors shows *Reference to a dataset* and *Reference to
software* as two distinct reference types, and both are then citable independently.

---

## Option A — web upload (simplest)

1. Go to <https://zenodo.org/uploads/new>
2. **Files** → *Upload files* → drag `starlink_tle.csv` in. Expect several minutes for 753 MB; keep the
   tab open. If it stalls, use Option B.
3. Fill in the metadata from the table below.
4. **Publish** → note the DOI you get, then tell the maintainer so the placeholders
   (`10.5281/zenodo.XXXXXXX` in `README.md` and `data/README.md`) can be filled in.

## Option B — API upload (recommended for a 753 MB file)

```bash
# 1. create a token with the `deposit:write` scope:
#    https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 2. dry run against the sandbox first (safe, separate site)
python zenodo_upload.py --sandbox --no-publish

# 3. real upload + publish
python zenodo_upload.py
```

`zenodo_upload.py` sits next to this file. It resumes nothing, so run it on a stable connection; rerun
with a fresh token if it fails.

---

## Metadata

| Field | Value |
|---|---|
| **Resource type** | Dataset |
| **Title** | Starlink two-line element history (2024-03 to 2025-08) for AI orbit-correction experiments |
| **Creators** | Yang, Kai — Tianjin University (ORCID if you have one) |
| **Version** | 1.0.0 |
| **Language** | eng |
| **Keywords** | Starlink; two-line elements; TLE; SGP4; space traffic management; orbit prediction; reliability |
| **License** | see the note below |
| **Related identifiers** | `10.5281/zenodo.23098904` — *is supplemented by* (the code release) |

### Description (copy-paste)

```html
<p>Space-Track two-line element (TLE) history used to build the dataset of the manuscript
<i>Risk-constrained deployment of AI correction for LEO orbit prediction: Physics-inspired soft masking
and fail-safe gating</i> (Reliability Engineering &amp; System Safety).</p>

<p><b>Contents</b></p>
<ul>
  <li>4,572,892 TLE records covering 8,430 Starlink satellites</li>
  <li>Columns: <code>norad_id</code>, <code>tle</code>, <code>change_time</code>. The two TLE lines are
      stored inside a single CSV field, separated by a newline.</li>
  <li>TLE epochs: 2024-03-10 to 2025-08-09</li>
  <li>Format: UTF-8 CSV, 753.4 MB (789,979,062 bytes)</li>
  <li>sha256: <code>0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608</code></li>
</ul>

<p><b>Coverage</b></p>
<ul>
  <li>8,430 satellites &mdash; the complete screening set of the manuscript (100 %)</li>
  <li>879 satellites &mdash; the dataset reported in the manuscript (100 %)</li>
</ul>

<p><b>Reuse</b></p>
<p>Feed this file to <code>preprocess/tle_to_trajectory.py</code> from the companion code release
(DOI <a href="https://doi.org/10.5281/zenodo.23098904">10.5281/zenodo.23098904</a>); see
<code>preprocess/README.md</code> for the six-step preparation chain. The epoch range starts on
2024-03-10, matching the start of the observation spans recorded in the code release.</p>

<p><b>Notes</b></p>
<ul>
  <li>4,512 records (0.1 %) carry an epoch field that cannot be parsed from TLE line 1; they are skipped
      during propagation and recorded in <code>tle_errors.csv</code>.</li>
  <li>The <code>change_time</code> column is not used by the analysis code and is kept for traceability.</li>
</ul>

<p><b>Source and terms</b></p>
<p>The TLE data are produced by the 18th Space Defense Squadron and distributed through Space-Track
(https://www.space-track.org). They are redistributed here solely to make the study reproducible;
users should also observe Space-Track's terms of use.</p>
```

### Which licence?

The data are **third-party** (Space-Track / 18 SDS), so you cannot unilaterally relicense them. Two
workable choices:

* **CC BY 4.0** — simplest, and consistent with how the derived data in the code release are licensed;
  attribution to Space-Track is given explicitly in the description.
* **Other (Open)** — with the same attribution note, if you would rather not assert a CC licence over
  third-party records.

**Before publishing, re-read Space-Track's User Agreement.** If it does not permit redistribution, do not
post the file: instead publish a record containing only the *fetch recipe* (the NORAD ID list and the TLE
epochs) and mark the data as available on request — the derived dataset and code are still fully public,
which is what RESS Option C actually requires.
