#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Upload a large file (e.g. starlink_tle.csv, 753 MB) to a new Zenodo dataset record.

Usage
-----
    # 1. create a personal access token with the `deposit:write` scope:
    #    https://zenodo.org/account/settings/applications/tokens/new/
    # 2. make it visible to this script:
    set   ZENODO_TOKEN=xxxxxxxx                 (Windows cmd)
    $env:ZENODO_TOKEN="xxxxxxxx"                (PowerShell)
    export ZENODO_TOKEN=xxxxxxxx                (bash)
    # 3. run
    python zenodo_upload.py --sandbox --no-publish
    python zenodo_upload.py

Options
-------
    --sandbox      use sandbox.zenodo.org (separate site, safe to experiment on)
    --no-publish   create the record and upload the file, but do not publish it
                   (you can then review it in the Zenodo web interface and publish there)

Why not the web uploader? A 753 MB single-file upload often stalls in a browser.
This script streams the file through Zenodo's bucket API in chunks.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "starlink_tle.csv")
STREAM_CHUNK = 8 << 20                     # 8 MiB per read

SANDBOX = "--sandbox" in sys.argv
NO_PUBLISH = "--no-publish" in sys.argv
API = "https://sandbox.zenodo.org/api" if SANDBOX else "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN", "").strip()

METADATA = {
    "title": ("Starlink two-line element history (2024-03 to 2025-08) "
              "for AI orbit-correction experiments"),
    "upload_type": "dataset",
    "description": (
        "<p>Space-Track two-line element (TLE) history used to build the dataset of the manuscript "
        "<i>Risk-constrained deployment of AI correction for LEO orbit prediction: Physics-inspired "
        "soft masking and fail-safe gating</i> (Reliability Engineering &amp; System Safety).</p>"
        "<p><b>Contents</b></p><ul>"
        "<li>4,572,892 TLE records covering 8,430 Starlink satellites</li>"
        "<li>Columns: <code>norad_id</code>, <code>tle</code>, <code>change_time</code>. The two TLE "
        "lines are stored inside a single CSV field, separated by a newline.</li>"
        "<li>TLE epochs: 2024-03-10 to 2025-08-09</li>"
        "<li>UTF-8 CSV, 753.4 MB (789,979,062 bytes). "
        "sha256 <code>0bb7b6c9fd2324b6c0639ad1414ec0ac126b76aaeda9e9d0676b1798f8c18608</code></li>"
        "</ul>"
        "<p><b>Coverage</b></p><ul>"
        "<li>8,430 satellites &mdash; the complete screening set of the manuscript (100 %)</li>"
        "<li>879 satellites &mdash; the dataset reported in the manuscript (100 %)</li></ul>"
        "<p><b>Reuse</b></p><p>Feed this file to <code>preprocess/tle_to_trajectory.py</code> from the "
        "companion code release (DOI "
        "<a href=\"https://doi.org/10.5281/zenodo.23098904\">10.5281/zenodo.23098904</a>); see "
        "<code>preprocess/README.md</code> for the six-step preparation chain.</p>"
        "<p><b>Notes</b></p><ul>"
        "<li>4,512 records (0.1 %) carry an epoch that cannot be parsed from TLE line 1; they are "
        "skipped during propagation and recorded in <code>tle_errors.csv</code>.</li>"
        "<li><code>change_time</code> is not used by the analysis code.</li></ul>"
        "<p><b>Source and terms</b></p><p>TLE data are produced by the 18th Space Defense Squadron and "
        "distributed through Space-Track (https://www.space-track.org). They are redistributed here "
        "solely to make the study reproducible; users should also observe Space-Track's terms of use.</p>"
    ),
    "creators": [{"name": "Yang, Kai", "affiliation": "Tianjin University"}],
    "keywords": ["Starlink", "two-line elements", "TLE", "SGP4", "space traffic management",
                 "orbit prediction", "reliability"],
    "version": "1.0.0",
    "language": "eng",
    "license": "cc-by-4.0",
    "related_identifiers": [
        {"identifier": "10.5281/zenodo.23098904",
         "relation": "isSupplementedBy", "scheme": "doi",
         "resource_type": "software"},
    ],
}


def call(url, method="GET", payload=None, headers=None, raw=None, timeout=600):
    hdr = {"Authorization": "Bearer " + TOKEN}
    if headers:
        hdr.update(headers)
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise SystemExit("HTTP %s on %s %s\n%s" % (e.code, method, url, detail))


class Progress:
    """File wrapper that reports upload progress."""
    def __init__(self, path):
        self.fh = open(path, "rb")
        self.total = os.path.getsize(path)
        self.sent = 0
        self.t0 = time.time()

    def read(self, n=-1):
        b = self.fh.read(n)
        self.sent += len(b)
        pct = 100.0 * self.sent / self.total
        mb = self.sent / 1048576
        print("\r  uploaded %7.1f / %.1f MB  (%5.1f%%)  %5.1f MB/s"
              % (mb, self.total / 1048576, pct,
                 mb / max(1e-6, time.time() - self.t0)), end="", flush=True)
        return b

    def close(self):
        self.fh.close()
        print()


def main():
    if not TOKEN or TOKEN.startswith("PASTE"):
        raise SystemExit("Set ZENODO_TOKEN first (scope: deposit:write). See the docstring.")
    if not os.path.exists(FILE):
        raise SystemExit("not found: " + FILE)
    size = os.path.getsize(FILE)
    print("target   : %s" % API)
    print("file     : %s (%.1f MB)" % (FILE, size / 1048576))
    print()

    print("[1/4] creating the deposition ...")
    dep = call(API + "/deposit/depositions", method="POST", payload={})
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print("      deposition id = %s" % dep_id)

    print("[2/4] uploading ...")
    call("%s/%s" % (bucket, os.path.basename(FILE)), method="PUT", raw=Progress(FILE),
         headers={"Content-Type": "application/octet-stream",
                  "Content-Length": str(size)})

    print("[3/4] setting metadata ...")
    call("%s/deposit/depositions/%s" % (API, dep_id), method="PUT",
         payload={"metadata": METADATA})

    if NO_PUBLISH:
        print("[4/4] skipped publish (--no-publish)")
        print("      review and publish at: %s/deposit/%s" % (API.replace("/api", ""), dep_id))
        return
    print("[4/4] publishing ...")
    pub = call("%s/deposit/depositions/%s/actions/publish" % (API, dep_id), method="POST")
    doi = pub.get("doi") or pub.get("metadata", {}).get("doi")
    print()
    print("DONE. DOI:", doi)
    print("Record:", pub.get("links", {}).get("record_html", ""))


if __name__ == "__main__":
    main()
