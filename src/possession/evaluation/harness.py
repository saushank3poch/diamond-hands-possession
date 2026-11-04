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
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from possession import split as split_mod
from possession.adapters.dfl import load_dfl
from possession.degrade import Config as DegradeConfig, degrade
from possession.evaluation.restarts import (edge_timing, restart_counts, summarize as summarize_restarts,
                                            summarize_timing)
from possession.evaluation.labels import proxy_labels, reference_labels
from possession.evaluation.references import References, load_dfl_references
from possession.predictors import make
from possession.tracking import Tracking

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402

DATA = ROOT.parent / "external" / "idsse"
CACHE = ROOT / "runs" / "cache" / "dfl"
MANIFEST = ROOT / "configs" / "external" / "idsse_manifest.json"
CACHE_SCHEMA = 2
SRC = ROOT / "src" / "possession"
# Code that determines cached inputs/references; any change invalidates the cache.
CACHE_CODE = (SRC / "adapters" / "dfl.py", SRC / "evaluation" / "references.py", SRC / "tracking.py",
              ROOT / "requirements.lock")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cache_identity(match, files):
    """Everything that determines the cached (tracking, references) pair for a match."""
    manifest = {f["name"]: f["sha256"] for f in json.loads(MANIFEST.read_text())["files"]}
    sources = []
    for path in files:
        st = path.stat()
        sources.append({"name": path.name, "manifest_sha256": manifest.get(path.name),
                        "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CACHE_CODE}
    return {"schema": CACHE_SCHEMA, "match": match, "sources": sources, "code": code}


def scorer_sha256():
    return hashlib.sha256((ROOT / "evaluation" / "possession_metrics.py").read_bytes()).hexdigest()

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


def load_match(match, split, holdout=False):
    """(Tracking, References) for a dev match, cached under ignored runs/cache.

    The cache key binds the source files (manifest SHA-256, size, mtime), the parsing code,
    the dependency lock and a schema version. On a miss the source bytes are verified against
    the checksum manifest before parsing. Tracking/reference alignment is checked on every load.
    """
    if holdout:
        if match not in split["holdout"] or match in split["dev"]:
            raise ValueError("holdout=True accepts holdout matches only")
    elif match not in split["dev"] or match in split["holdout"]:
        raise ValueError("only dev matches may be evaluated here")
    files = _files(match)
    identity = cache_identity(match, files)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    t_path, r_path = CACHE / f"{match}-{key}.tracking.npz", CACHE / f"{match}-{key}.refs.npz"
    if t_path.exists() and r_path.exists():
        tracking = Tracking.load(t_path)
        with np.load(r_path, allow_pickle=False) as z:
            refs = References(z["frame"], z["alive"], z["owner"], z["unknown"], (), ())
    else:
        for path, src in zip(files, identity["sources"]):
            if src["manifest_sha256"] is None or _sha256(path) != src["manifest_sha256"]:
                raise ValueError("source file does not match the checksum manifest")
        tracking = load_dfl(*files)
        refs = load_dfl_references(files[1], tracking)
        CACHE.mkdir(parents=True, exist_ok=True)
        tracking.save(t_path)
        with open(r_path, "wb") as h:
            np.savez_compressed(h, frame=refs.frame, alive=refs.alive, owner=refs.owner, unknown=refs.unknown)
    n = len(tracking.frame)
    if not (np.array_equal(refs.frame, tracking.frame) and len(refs.alive) == n
            and len(refs.owner) == n and len(refs.unknown) == n):
        raise ValueError("references are not aligned with tracking")
    return tracking, refs


def scored_sequence(gt, pred, period):
    """Evaluator-only: lay out active periods with one X separator after each, so no spell,
    transition or abstention fill crosses a period (or match) boundary. Real frames are kept;
    frames outside active periods are dropped (they are X in the truth and never graded)."""
    gts, prs = [], []
    cuts = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if period[a] > 0:
            gts += [gt[a:b], np.array(["X"])]
            prs += [pred[a:b], np.array(["X"])]
    return np.concatenate(gts), np.concatenate(prs)


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


def _score_variant(report, variant, matches, data, preds, label_sets, bootstrap_n):
    """Score one input variant's predictions per match and pooled (shared by dev and holdout)."""
    for label_set in label_sets:
        per_match, gts, prs, timing = {}, [], [], {}
        for m in matches:
            gt = gt_labels(data[m][1], data[m][0], label_set)
            pr = preds[m]
            if len(pr) != len(gt) or not np.isin(pr, list("HADX")).all():
                raise ValueError("predictor output does not match the frame axis")
            g_seq, p_seq = scored_sequence(gt, pr, data[m][0].period)
            s = _summary(pm.evaluate(pm.label_codes(list(g_seq)), pm.label_codes(list(p_seq)), bootstrap_n=0))
            active = g_seq != "X"
            s["pred_d_share"] = float((p_seq[active] == "D").mean())
            s["gt_d_share"] = float((g_seq[active] == "D").mean())
            s["restart_owner"] = summarize_restarts(restart_counts(gt, pr, data[m][0].period))
            timing[m] = edge_timing(gt, pr, data[m][0].period)
            s["edge_timing"] = summarize_timing([timing[m]])
            per_match[m] = s
            gts.append(g_seq)
            prs.append(p_seq)
        gt_all, pr_all = np.concatenate(gts), np.concatenate(prs)
        pooled = pm.evaluate(pm.label_codes(list(gt_all)), pm.label_codes(list(pr_all)), bootstrap_n=bootstrap_n)
        ps = _summary(pooled)
        act = gt_all != "X"
        ps["pred_d_share"] = float((pr_all[act] == "D").mean())
        ps["gt_d_share"] = float((gt_all[act] == "D").mean())
        counts = {field: sum(v["restart_owner"][field] for v in per_match.values())
                  for field in ("n", "correct", "team_predictions", "pred_d", "pred_x")}
        ps["restart_owner"] = summarize_restarts(counts)
        ps["edge_timing"] = summarize_timing([timing[m] for m in matches])
        report["results"][f"{variant}/{label_set}"] = {"pooled": ps, "per_match": per_match}


def evaluate(spec, seed=0, bootstrap_n=1000, variants=tuple(VARIANTS), label_sets=LABEL_SETS):
    split = split_mod.load()
    dev = list(split["dev"])
    data = {m: load_match(m, split) for m in dev}
    report = {"split_sha256": split["split_sha256"], "predictor": spec, "seed": seed,
              "bootstrap_n": bootstrap_n, "results": {},
              "provenance": {"split_sha256": split["split_sha256"], "seed": seed, "dev": dev,
                             "variants": {v: VARIANTS[v] for v in variants},
                             "degrade_defaults": asdict(DegradeConfig()),
                             "label_sets": list(label_sets), "bootstrap_n": bootstrap_n,
                             "scorer_sha256": scorer_sha256(),
                             "labels_sha256": hashlib.sha256((SRC / "evaluation" / "labels.py").read_bytes()).hexdigest(),
                             "degrade_sha256": hashlib.sha256((SRC / "degrade.py").read_bytes()).hexdigest(),
                             "inputs": {m: cache_identity(m, _files(m)) for m in dev}}}
    for variant in variants:
        inputs = {m: variant_input(data[m][0], m, variant, seed) for m in dev}
        preds = {}
        for fold in split["dev_folds"]:
            held = fold["validate"]
            predictor = make(spec)
            # Training examples: other dev matches only, primary labels, same input variant.
            predictor.fit([(inputs[m], gt_labels(data[m][1], data[m][0], "primary")) for m in fold["train"]])
            preds[held] = np.asarray(predictor.predict(inputs[held]))
        _score_variant(report, variant, dev, data, preds, label_sets, bootstrap_n)
    return report


def evaluate_holdout(spec, seed=0, bootstrap_n=1000, variants=tuple(VARIANTS), label_sets=LABEL_SETS):
    """FCB-18 one-time holdout evaluation of a frozen predictor spec (evaluator side).

    For each input variant the predictor is fitted once on all dev matches (primary labels, same
    variant, exactly as in dev folds) and predicts each holdout match from tracking alone. Holdout
    references are used only for scoring. The run-once guard and freeze checks live in the
    pipeline stage that calls this.
    """
    split = split_mod.load()
    dev, holdout = list(split["dev"]), list(split["holdout"])
    if not holdout or set(dev) & set(holdout):
        raise ValueError("holdout must be nonempty and disjoint from dev")
    data = {m: load_match(m, split) for m in dev}
    held = {m: load_match(m, split, holdout=True) for m in holdout}
    report = {"split_sha256": split["split_sha256"], "predictor": spec, "seed": seed,
              "bootstrap_n": bootstrap_n, "results": {}, "train": dev, "holdout": holdout,
              "provenance": {"split_sha256": split["split_sha256"], "seed": seed,
                             "variants": {v: VARIANTS[v] for v in variants},
                             "degrade_defaults": asdict(DegradeConfig()),
                             "label_sets": list(label_sets), "bootstrap_n": bootstrap_n,
                             "scorer_sha256": scorer_sha256(),
                             "labels_sha256": hashlib.sha256((SRC / "evaluation" / "labels.py").read_bytes()).hexdigest(),
                             "degrade_sha256": hashlib.sha256((SRC / "degrade.py").read_bytes()).hexdigest(),
                             "inputs": {m: cache_identity(m, _files(m)) for m in dev + holdout}}}
    for variant in variants:
        train = [(variant_input(data[m][0], m, variant, seed), gt_labels(data[m][1], data[m][0], "primary"))
                 for m in dev]
        predictor = make(spec)
        predictor.fit(train)
        preds = {m: np.asarray(predictor.predict(variant_input(held[m][0], m, variant, seed))) for m in holdout}
        _score_variant(report, variant, holdout, held, preds, label_sets, bootstrap_n)
    return report


PAIRED_MUST_MATCH = ("split_sha256", "seed", "dev", "variants", "degrade_defaults", "label_sets",
                     "bootstrap_n", "scorer_sha256", "labels_sha256", "degrade_sha256", "inputs")


def paired(report, reference, key="degraded/primary"):
    """Per-match spellF1 differences (candidate - reference) on identical matches and settings.

    Refuses incompatible runs: the dev match sets must be identical and nonempty, and split,
    seed, degradation, labels, bootstrap and scorer provenance must match exactly.
    """
    pa, pb = report.get("provenance"), reference.get("provenance")
    if not pa or not pb:
        raise ValueError("both runs need provenance")
    for field in PAIRED_MUST_MATCH:
        if pa.get(field) != pb.get(field):
            raise ValueError(f"incompatible runs: {field} differs")
    if key not in report["results"] or key not in reference["results"]:
        raise ValueError("comparison key missing")
    a, b = report["results"][key]["per_match"], reference["results"][key]["per_match"]
    if not a or set(a) != set(b):
        raise ValueError("match sets differ or are empty")
    diffs = {m: a[m]["spell_f1"] - b[m]["spell_f1"] for m in sorted(a)}
    return {"key": key, "diffs": diffs, "mean": float(np.mean(list(diffs.values()))),
            "n_positive": sum(d > 0 for d in diffs.values()), "n": len(diffs),
            "pooled_ci90_candidate": report["results"][key]["pooled"]["spell_ci90"],
            "pooled_ci90_reference": reference["results"][key]["pooled"]["spell_ci90"],
            "note": "pooled CIs resample possessions within the dev set, not matches"}


def write(report, path):
    Path(path).write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
