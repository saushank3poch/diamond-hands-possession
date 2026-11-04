#!/usr/bin/env python3
"""Pipeline entry points, run through scripts/run_experiment.py (dev) or the target wrapper.

dev-eval: score each configured predictor on dev matches (FCB-9) and write
          $POSSESSION_RUN_DIR/dev_metrics.json. Optional "compare_to" adds per-match
          paired spellF1 differences against an earlier dev_metrics.json.
holdout-eval: FCB-18 one-time holdout evaluation of the frozen candidate. Refuses unless HEAD is
          exactly the freeze tag named in the config's freeze manifest, the worktree is clean,
          the predictor equals the manifest's, and no holdout result exists yet.
submission: not implemented yet (FCB-19).
"""

import argparse
import json
import os
from pathlib import Path
import sys

from possession.runs import seed_generators

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("dev-eval", "holdout-eval", "submission"))
args = parser.parse_args()
seed = int(os.environ["POSSESSION_SEED"])
seed_generators(seed)

if args.stage == "submission":
    print("submission is not implemented yet; no data was read.", file=sys.stderr)
    sys.exit(2)

from possession.evaluation import harness  # noqa: E402

run_dir = Path(os.environ["POSSESSION_RUN_DIR"])
config = json.loads((run_dir / "config.yaml").read_text())

if args.stage == "holdout-eval":
    from possession.holdout_gate import check_gate, record_opened  # noqa: E402
    spec, freeze = check_gate(config)
    report = harness.evaluate_holdout(spec, seed=seed, bootstrap_n=config.get("bootstrap_n", 1000))
    report["freeze"] = freeze
    harness.write({"seed": seed, "reports": {"frozen": report}}, run_dir / "holdout_metrics.json")
    record_opened(run_dir)
    print("HOLDOUT_EVAL wrote", run_dir / "holdout_metrics.json")
    sys.exit(0)
out = {"seed": seed, "reports": {}}
for spec in config["predictors"]:
    name = spec.get("label") or json.dumps(spec, sort_keys=True)
    spec = {k: v for k, v in spec.items() if k != "label"}
    report = harness.evaluate(spec, seed=seed, bootstrap_n=config.get("bootstrap_n", 1000),
                              variants=tuple(config.get("variants", harness.VARIANTS)))
    out["reports"][name] = report
    print("scored", name, flush=True)
if config.get("compare_to"):
    reference = json.loads(Path(config["compare_to"]["path"]).read_text())["reports"][config["compare_to"]["report"]]
    out["paired"] = {name: harness.paired(r, reference) for name, r in out["reports"].items()}
harness.write(out, run_dir / "dev_metrics.json")
print("DEV_EVAL wrote", run_dir / "dev_metrics.json")
