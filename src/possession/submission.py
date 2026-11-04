"""FCB-19: build the Barça–Athletic submission with the frozen recipe (configs/freeze.json).

Training: exactly `target_training` in the freeze manifest (5 dev matches from configs/split.yaml,
primary labels, default degraded variant with dev-CV per-match seeds, seed 0). The holdout and
the target never enter fitting. Prediction: the fitted model reads target tracking only, inside
the containment wrapper. Output: `frame,label` for every video frame; X outside the published
halves (the adapter sets period 0 there, and predictors return X for period 0).

Runs only under scripts/run_target_local.sh, which captures stdout/stderr in local run files and
prints only pass/fail.
No target-derived number is printed or returned to the caller.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from possession import split as split_mod
from possession.predictors import make

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "freeze.json"
FREEZE_TAG = "freeze-e005"
N_FRAMES = 199319
# Everything that determines the fitted model or its inputs must be byte-identical to the tag.
# New runner files (this module, pipeline.py, Makefile, configs/submission.yaml) may differ.
FROZEN_FILES = ("configs/freeze.json", "configs/split.yaml", "requirements.lock", "games.py")
FROZEN_DIRS = ("src/possession/predictors", "src/possession/adapters")
FROZEN_MODULES = ("src/possession/tracking.py", "src/possession/degrade.py", "src/possession/split.py",
                  "src/possession/periods.py", "src/possession/evaluation/harness.py",
                  "src/possession/evaluation/labels.py", "src/possession/evaluation/references.py")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def verify_freeze(git=_git, root=ROOT):
    """Refuse model drift from the freeze tag; run before any data is loaded.

    Returns the freeze identity (tag, tag commit, HEAD) to record with the output.
    """
    tag_commit = git("rev-parse", f"{FREEZE_TAG}^{{commit}}").decode().strip()
    head = git("rev-parse", "HEAD").decode().strip()
    paths = list(FROZEN_FILES) + list(FROZEN_MODULES)
    for d in FROZEN_DIRS:
        tagged = sorted(p for p in git("ls-tree", "-r", "--name-only", FREEZE_TAG, d).decode().split() if p.endswith(".py"))
        present = sorted(str(p.relative_to(root)) for p in (root / d).rglob("*.py"))
        if tagged != present:
            raise ValueError(f"model drift: file set differs from {FREEZE_TAG} in {d}")
        paths += tagged
    for rel in paths:
        if (root / rel).read_bytes() != git("show", f"{FREEZE_TAG}:{rel}"):
            raise ValueError(f"model drift: {rel} differs from {FREEZE_TAG}")
    m = json.loads((root / MANIFEST.relative_to(ROOT)).read_bytes())
    if m["tag"] != FREEZE_TAG:
        raise ValueError("manifest names another tag")
    if hashlib.sha256((root / "requirements.lock").read_bytes()).hexdigest() != m["requirements_lock_sha256"]:
        raise ValueError("requirements.lock differs from the manifest")
    return {"tag": FREEZE_TAG, "tag_commit": tag_commit, "head": head}


VARIANTS = {"degraded": "degraded (FCB-8 default DegradeConfig)"}
ELIGIBILITY = ROOT / "configs" / "eligibility.json"
FALLBACK_KEY = "fallback_if_degradation_ruled_ineligible"


def eligibility(path=None):
    """Organiser ruling switch. True (default) = frozen degraded recipe; False = prespecified clean
    fallback from the freeze manifest. Only a literal boolean is accepted."""
    e = json.loads((path or ELIGIBILITY).read_text())
    allowed = e.get("degradation_statistics_allowed")
    if type(allowed) is not bool:
        raise ValueError("eligibility switch must be a boolean")
    return allowed


def recipe(manifest=None, split=None, degradation_allowed=None):
    """Resolve the exact frozen recipe; refuse anything that is not literally it.

    With the organiser switch off, resolve the prespecified clean fallback instead: same
    predictor, matches, labels and seed, clean inputs. The fallback text must be present in the
    (tag-bound) manifest; it is not chosen from any score.
    """
    m = manifest or json.loads(MANIFEST.read_text())
    split = split or split_mod.load()
    allowed = eligibility() if degradation_allowed is None else degradation_allowed
    t = m["target_training"]
    if not t.get("decided_before_holdout") or t["labels"] != "primary" or t["seed"] != 0:
        raise ValueError("unexpected target training recipe")
    variant = t["input_variant"].split(",")[0].strip()
    if variant != VARIANTS["degraded"]:
        raise ValueError("input variant is not the frozen default degraded variant")
    if m.get("split_sha256") != split["split_sha256"]:
        raise ValueError("split differs from the manifest")
    if m["predictor"] != {"name": "short_spell_x", "mode": "carry"}:
        raise ValueError("predictor is not the frozen E005 candidate")
    if not split["dev"] or set(split["dev"]) & set(split["holdout"]):
        raise ValueError("dev must be nonempty and disjoint from holdout")
    if allowed:
        return m["predictor"], "degraded", t["seed"], list(split["dev"])
    fallback = t.get(FALLBACK_KEY, "")
    if "clean variant" not in fallback or "same matches, labels, seed" not in fallback:
        raise ValueError("manifest has no prespecified clean fallback")
    return m["predictor"], "clean", t["seed"], list(split["dev"])


def train(spec, variant, seed, dev, load_match, variant_input, gt_labels):
    split = split_mod.load()
    if list(split["dev"]) != list(dev) or set(dev) & set(split["holdout"]):
        raise ValueError("training matches differ from the frozen dev list")
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


def build(run_dir, seed, load_match, variant_input, gt_labels, load_target, validate):
    """verify freeze -> resolve recipe -> train on dev -> load/predict target -> write -> validate.

    The freeze and recipe checks run before any dev, holdout or target file is opened.
    `validate(path)` returns (exit_code, log_bytes). Returns the validator exit code.
    """
    run_dir = Path(run_dir)
    freeze = verify_freeze()
    spec, variant, train_seed, dev = recipe()
    if train_seed != seed:
        raise ValueError("run seed differs from the frozen recipe")
    model, dev = train(spec, variant, train_seed, dev, load_match, variant_input, gt_labels)
    labels = predict_target(model, load_target())
    out = run_dir / "submission.csv"
    digest = write_csv(out, labels)
    code, log = validate(out)
    (run_dir / "validate.log").write_bytes(log)
    (run_dir / "submission.json").write_text(json.dumps(
        {"sha256": digest, "spec": spec, "variant": variant, "train": dev, "freeze": freeze,
         "degradation_statistics_allowed": variant == "degraded", "validator_exit": code}) + "\n")
    return code
