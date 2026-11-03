#!/usr/bin/env python3
"""Build clean/degraded dev caches or a contained target cache; never load references."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from possession.adapters.atd import load_target
from possession.adapters.dfl import load_dfl
from possession.degrade import Config as DegradeConfig, degrade
from possession.feature_cache import cache_features
from possession.features import Config
from possession.split import load as load_split


def match_seed(base, match):
    # Keep identical to the reviewed degradation-report convention.
    return (base * 1_000_003 + sum(ord(c) * 31 ** i for i, c in enumerate(match))) % 2**32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("dev", "target"), required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/features.yaml"))
    parser.add_argument("--data", default=str(ROOT.parent / "external/idsse"))
    args = parser.parse_args()
    if not os.environ.get("POSSESSION_RUN_DIR"):
        raise RuntimeError("run via the research harness")
    run_dir = Path(os.environ["POSSESSION_RUN_DIR"]).resolve()
    if args.source == "target":
        if not os.environ.get("POSSESSION_TARGET_LOG_DIR") or Path(os.environ["POSSESSION_TARGET_LOG_DIR"]).resolve() != run_dir:
            raise RuntimeError("target features require the containment wrapper")
        def deadline(signum, frame):
            raise TimeoutError("target feature pipeline exceeded 600 seconds")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(600)
    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    cfg = Config(**config["features"])
    stats = {"config": asdict(cfg), "source": args.source, "caches": []}

    def write(tracking, name):
        path = run_dir / (name + ".parquet")
        table, hit = cache_features(tracking, path, cfg)
        cached, hit = cache_features(tracking, path, cfg)
        if not hit or not table.equals(cached, check_metadata=True):
            raise ValueError("feature cache round-trip mismatch")
        stats["caches"].append({"name": name, "rows": table.num_rows, "columns": table.num_columns})

    if args.source == "target":
        write(load_target(), "target-features")
    else:
        split = load_split()
        stats["split_sha256"] = split["split_sha256"]
        for match in split["dev"]:
            if match in split["holdout"]:
                raise ValueError("holdout processing is forbidden in dev feature runs")
            def one(prefix):
                paths = list(Path(args.data).glob(f"{prefix}_*_{match}.xml"))
                if len(paths) != 1:
                    raise ValueError("expected exactly one file for the requested dev match")
                return paths[0]
            tracking = load_dfl(one("DFL_02_01_matchinformation"), one("DFL_04_03_positions_raw_observed"))
            write(tracking, match + "-clean")
            degraded, _ = degrade(tracking, DegradeConfig(seed=match_seed(config["seed"], match)))
            write(degraded, match + "-degraded")
    stats["elapsed_s"] = time.monotonic() - started
    (run_dir / "features-run.json").write_text(json.dumps(stats, indent=2) + "\n")
    if args.source == "target":
        if stats["elapsed_s"] >= 600:
            raise TimeoutError("target feature pipeline exceeded 600 seconds")
        signal.alarm(0)
    print("FEATURE_CACHE PASS")


if __name__ == "__main__":
    main()
