#!/usr/bin/env python3
"""Pipeline entry points, run through scripts/run_experiment.py (dev) or the target wrapper.

dev-eval: score each configured predictor on dev matches (FCB-9) and write
          $POSSESSION_RUN_DIR/dev_metrics.json. Optional "compare_to" adds per-match
          paired spellF1 differences against an earlier dev_metrics.json.
holdout-eval: FCB-18 one-time holdout evaluation of the frozen candidate; see
          possession/holdout_gate.py for the preflight checks and the one-shot reservation.
submission: FCB-19 target CSV with the frozen recipe; target wrapper only (see possession/submission.py).
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
    # Runs only inside scripts/run_target_local.sh (stdout/stderr captured to local files; pass/fail only).
    import signal  # noqa: E402
    import subprocess  # noqa: E402
    signal.alarm(600)  # whole pipeline must finish in < 10 min
    from possession.adapters.atd import load_target  # noqa: E402
    from possession.evaluation import harness  # noqa: E402
    from possession import submission  # noqa: E402

    def validate(path):
        r = subprocess.run([sys.executable, "evaluation/validate_submission.py", str(path), "--rows",
                            str(submission.N_FRAMES)], capture_output=True)
        return r.returncode, r.stdout + r.stderr

    code = submission.build(Path(os.environ["POSSESSION_RUN_DIR"]), seed, harness.load_match,
                            harness.variant_input, harness.gt_labels, load_target, validate)
    sys.exit(0 if code == 0 else 3)

from possession.evaluation import harness  # noqa: E402

run_dir = Path(os.environ["POSSESSION_RUN_DIR"])
config = json.loads((run_dir / "config.yaml").read_text())

if args.stage == "holdout-eval":
    from possession import holdout_gate, split as split_mod  # noqa: E402
    holdout_gate.run_once(
        (run_dir / "config.yaml").read_bytes(), seed, split_mod.load()["split_sha256"], run_dir,
        evaluate=lambda spec, n: harness.evaluate_holdout(spec, seed=seed, bootstrap_n=n),
        write=lambda rep: harness.write({"seed": seed, "reports": {"frozen": rep}},
                                        run_dir / "holdout_metrics.json"))
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
