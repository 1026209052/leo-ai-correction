#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run every Part-A analysis script and report pass/fail.

Usage
-----
    PYTHONIOENCODING=utf-8 python run_all_analyses.py

All scripts read only from `figures/data/*.csv` and `analysis/out/*.csv`,
write their console tables, and need neither a GPU nor the raw TLE archive.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "analysis")

SCRIPTS = [
    "a1_table6_definition.py",
    "c1_independent_tail.py",
    "c1_v5_analysis.py",
    "c2_cluster_bootstrap.py",
    "c2c_a02_extra.py",
    "c2d_a02_wilson.py",
    "c3_block_crc.py",
    "c4_tau_max_bootstrap.py",
    "c5_event_rate.py",
    "check_table9_def.py",
    "crosscheck_eta.py",
    "design_rule_alpha_tau.py",
    "fig5_design_rule_data.py",
    "layer2_fmea.py",
    "layer2_table7_mc.py",
    "layer2_table8_cost.py",
    "s57_from_csv.py",
    "table5_strata.py",
    "why_denominator.py",
    "tail_analysis.py",
]

def main():
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    failed = []
    for name in SCRIPTS:
        print("=" * 78)
        print(">>", name)
        print("=" * 78)
        r = subprocess.run([sys.executable, name], cwd=ANALYSIS, env=env)
        if r.returncode != 0:
            failed.append((name, r.returncode))
    print()
    print("=" * 78)
    print("PASS %d / %d" % (len(SCRIPTS) - len(failed), len(SCRIPTS)))
    for name, rc in failed:
        print("  FAILED:", name, "exit", rc)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
