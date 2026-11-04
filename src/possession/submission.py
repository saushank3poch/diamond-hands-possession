"""FCB-19: build the Barça–Athletic submission with the frozen recipe (configs/freeze.json).

Training: exactly `target_training` in the freeze manifest (5 dev matches from configs/split.yaml,
primary labels, default degraded variant with dev-CV per-match seeds, seed 0). The holdout and
the target never enter fitting. Prediction: the fitted model reads target tracking only, inside
the containment wrapper. Output: `frame,label` for every video frame; X outside the published
halves (the adapter sets period 0 there, and predictors return X for period 0).

Runs only under scripts/run_target_local.sh, which discards stdout/stderr and reports pass/fail.
No target-derived number is printed or returned to the caller.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from possession import split as split_mod
from possession.predictors import make

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "freeze.json"
N_FRAMES = 199319


def recipe(manifest=None):
    m = manifest or json.loads(MANIFEST.read_text())
    t = m["target_training"]
    if not t.get("decided_before_holdout") or t["labels"] != "primary" or t["seed"] != 0:
        raise ValueError("unexpected target training recipe")
    if not t["input_variant"].startswith("degraded"):
        raise ValueError("unexpected input variant")
    return m["predictor"], "degraded", t["seed"]


def train(spec, variant, seed, load_match, variant_input, gt_labels):
    split = split_mod.load()
    dev = list(split["dev"])
    if set(dev) & set(split["holdout"]):
        raise ValueError("dev and holdout overlap")
    examples = []
    for m in dev:
        tracking, refs = load_match(m, split)
        examples.append((variant_input(tracking, m, variant, seed), gt_labels(refs, tracking, "primary")))
    return make(spec).fit(examples), dev


def predict_target(model, tracking):
    labels = np.asarray(model.predict(tracking))
    if labels.shape != (N_FRAMES,) or not np.isin(labels, list("HADX")).all():
        raise ValueError("prediction does not match the target frame axis")
    if (labels[tracking.period <= 0] != "X").any():
        raise ValueError("labels outside the published halves must be X")
    return labels


def write_csv(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return hashlib.sha256(path.read_bytes()).hexdigest()
