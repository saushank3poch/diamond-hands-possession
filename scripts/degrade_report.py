#!/usr/bin/env python3
"""FCB-8: degrade each dev match with the fixed Config and report achieved input statistics.

Dev matches only. Reads tracking (inputs) only; no reference labels are loaded.
Usage: PYTHONPATH=src python scripts/degrade_report.py [--data ../external/idsse] [--out findings/degradation-report.json]
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession import split  # noqa: E402
from possession.adapters.dfl import load_dfl  # noqa: E402
from possession.degrade import Config, degrade  # noqa: E402


def match_seed(base, match):
    return (base * 1_000_003 + sum(ord(c) * 31 ** i for i, c in enumerate(match))) % 2**32


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "external" / "idsse"))
    ap.add_argument("--out", default=str(ROOT / "findings" / "degradation-report.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    body = split.load()
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
    report = {"split_sha256": body["split_sha256"], "git_commit": git("rev-parse", "HEAD"),
              "worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "matches": []}
    for match in body["dev"]:
        info = next(Path(args.data).glob(f"DFL_02_01_matchinformation_*_{match}.xml"))
        raw = next(Path(args.data).glob(f"DFL_04_03_positions_raw_observed_*_{match}.xml"))
        t0 = time.time()
        clean = load_dfl(info, raw)
        t1 = time.time()
        _, stats = degrade(clean, Config(seed=match_seed(args.seed, match)))
        stats.update(match=match, clean_ball_visible=round(float((~clean.ball_missing)[clean.period > 0].mean()), 4),
                     load_s=round(t1 - t0, 1), degrade_s=round(time.time() - t1, 1))
        stats.pop("config")
        report["matches"].append(stats)
        print(json.dumps(stats), flush=True)
    report["config"] = Config().__dict__
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
