"""FCB-9: score a predictor on dev matches with the official scorer (evaluator side).

- Dev matches only (configs/split.yaml); holdout matches are refused.
- Leave-one-match-out: the predictor is fitted on the other dev matches, then predicts the
  held-out one from tracking alone. Reference labels never reach `predict`.
- Inputs: clean tracking and degraded variants (FCB-8). Labels: primary (raw DFL flags) and
  the experimental 2 s proxy, reported side by side (FCB-5).
- Scoring: official `evaluation/possession_metrics.evaluate` per match, plus a pooled score over
  all dev matches (matches concatenated with an X separator so nothing crosses them) with the
  official 1000-draw bootstrap (5th-95th percentile, i.e. a 90% interval).
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from possession import split as split_mod
from possession.adapters.dfl import load_dfl
from possession.degrade import Config as DegradeConfig, degrade
from possession.evaluation.labels import proxy_labels, reference_labels
from possession.evaluation.references import References, load_dfl_references
from possession.predictors import make
from possession.tracking import Tracking

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402

DATA = ROOT.parent / "external" / "idsse"
CACHE = ROOT / "runs" / "cache" / "dfl"

# Degradation variants: the default plus two alternative gap-length assumptions (robustness band).
VARIANTS = {
    "clean": None,
    "degraded": {},
    "degraded_short_gaps": {"ball_gap_mean_s": 0.6, "player_gap_mean_s": 1.0},
    "degraded_long_gaps": {"ball_gap_mean_s": 3.0, "player_gap_mean_s": 4.0},
}
LABEL_SETS = ("primary", "proxy_2s")


def match_seed(base, key):
    # Same rule as scripts/degrade_report.py, so the default variant reproduces FCB-8's report.
    return (base * 1_000_003 + sum(ord(c) * 31 ** i for i, c in enumerate(key))) % 2**32


def _files(match):
    def one(prefix):
        hits = list(DATA.glob(f"{prefix}_*_{match}.xml"))
        if len(hits) != 1:
            raise FileNotFoundError(prefix)
        return hits[0]
    return one("DFL_02_01_matchinformation"), one("DFL_04_03_positions_raw_observed")


def load_match(match, split):
    """(Tracking, References) for a dev match, cached under ignored runs/cache."""
    if match not in split["dev"] or match in split["holdout"]:
        raise ValueError("only dev matches may be evaluated here")
    key = hashlib.sha256((split["split_sha256"] + match).encode()).hexdigest()[:16]
    t_path, r_path = CACHE / f"{match}-{key}.tracking.npz", CACHE / f"{match}-{key}.refs.npz"
    if t_path.exists() and r_path.exists():
        tracking = Tracking.load(t_path)
        with np.load(r_path, allow_pickle=False) as z:
            refs = References(z["frame"], z["alive"], z["owner"], z["unknown"], (), ())
        return tracking, refs
    info, raw = _files(match)
    tracking = load_dfl(info, raw)
    refs = load_dfl_references(raw, tracking)
    CACHE.mkdir(parents=True, exist_ok=True)
    tracking.save(t_path)
    with open(r_path, "wb") as h:
        np.savez_compressed(h, frame=refs.frame, alive=refs.alive, owner=refs.owner, unknown=refs.unknown)
    return tracking, refs


def gt_labels(refs, tracking, label_set):
    labels = reference_labels(refs) if label_set == "primary" else proxy_labels(refs, tracking.period)
    return np.where(tracking.period > 0, labels, "X")


def variant_input(tracking, match, variant, seed):
    if VARIANTS[variant] is None:
        return tracking
    key = match if variant == "degraded" else f"{match}:{variant}"
    cfg = DegradeConfig(seed=match_seed(seed, key), **VARIANTS[variant])
    return degrade(tracking, cfg)[0]


def _summary(result):
    t25 = result["transitions"][25]
    return {
        "spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
        "spell_ci90": [result["spell_bootstrap_low"], result["spell_bootstrap_high"]],
        "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"],
        "spell_count_error": result["spell_count_error"],
        "pred_d_share": None, "alive_dead_f1": result["alive_dead_f1"],
        "fragmentation": result["fragmentation"],
        "transition_f1_25": t25.get("f1"), "mean_abs_latency_25": t25.get("mean_abs_latency"),
        "per_class_f1": {k: v["f1"] for k, v in result["per_class"].items()},
    }


def evaluate(spec, seed=0, bootstrap_n=1000, variants=tuple(VARIANTS), label_sets=LABEL_SETS):
    split = split_mod.load()
    dev = list(split["dev"])
    data = {m: load_match(m, split) for m in dev}
    report = {"split_sha256": split["split_sha256"], "predictor": spec, "seed": seed,
              "bootstrap_n": bootstrap_n, "results": {}}
    for variant in variants:
        inputs = {m: variant_input(data[m][0], m, variant, seed) for m in dev}
        preds = {}
        for fold in split["dev_folds"]:
            held = fold["validate"]
            predictor = make(spec)
            # Training examples: other dev matches only, primary labels, same input variant.
            predictor.fit([(inputs[m], gt_labels(data[m][1], data[m][0], "primary")) for m in fold["train"]])
            preds[held] = np.asarray(predictor.predict(inputs[held]))
        for label_set in label_sets:
            per_match, gts, prs = {}, [], []
            for m in dev:
                gt = gt_labels(data[m][1], data[m][0], label_set)
                pr = preds[m]
                if len(pr) != len(gt) or not np.isin(pr, list("HADX")).all():
                    raise ValueError("predictor output does not match the frame axis")
                g, p = pm.label_codes(list(gt)), pm.label_codes(list(pr))
                s = _summary(pm.evaluate(g, p, bootstrap_n=0))
                active = gt != "X"
                s["pred_d_share"] = float((pr[active] == "D").mean())
                s["gt_d_share"] = float((gt[active] == "D").mean())
                per_match[m] = s
                gts += [gt, np.array(["X"])]
                prs += [pr, np.array(["X"])]
            pooled = pm.evaluate(pm.label_codes(list(np.concatenate(gts))),
                                 pm.label_codes(list(np.concatenate(prs))), bootstrap_n=bootstrap_n)
            ps = _summary(pooled)
            gt_all, pr_all = np.concatenate(gts), np.concatenate(prs)
            act = gt_all != "X"
            ps["pred_d_share"] = float((pr_all[act] == "D").mean())
            ps["gt_d_share"] = float((gt_all[act] == "D").mean())
            report["results"][f"{variant}/{label_set}"] = {"pooled": ps, "per_match": per_match}
    return report


def paired(report, reference, key="degraded/primary"):
    """Per-match spellF1 differences (candidate - reference) on the same matches and variant."""
    a, b = report["results"][key]["per_match"], reference["results"][key]["per_match"]
    diffs = {m: a[m]["spell_f1"] - b[m]["spell_f1"] for m in a if m in b}
    return {"key": key, "diffs": diffs, "mean": float(np.mean(list(diffs.values()))),
            "n_positive": sum(d > 0 for d in diffs.values()), "n": len(diffs),
            "pooled_ci90_candidate": report["results"][key]["pooled"]["spell_ci90"],
            "pooled_ci90_reference": reference["results"][key]["pooled"]["spell_ci90"]}


def write(report, path):
    Path(path).write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
