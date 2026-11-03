"""Match-level dev/holdout split, drawn from file names only (no match records read).

Procedure (fixed in FCB-7 before any label inspection):
- one untouched holdout match per competition stratum, drawn with a seeded RNG;
- all tuning uses leave-one-match-out folds over the remaining dev matches;
- the holdout is opened once, in FCB-18, on a frozen candidate.
"""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "external" / "idsse_manifest.json"
SPLIT = ROOT / "configs" / "split.yaml"
SEED = 20261103  # prespecified before the draw: the date FCB-7 ran
POSITIONS = re.compile(r"^DFL_04_03_positions_raw_observed_(DFL-COM-\d+)_(DFL-MAT-[A-Z0-9]+)\.xml$")


def matches_from_manifest(manifest_path=MANIFEST):
    """(competition, match) pairs taken from position file names in the checksum manifest."""
    files = json.loads(Path(manifest_path).read_text())["files"]
    pairs = sorted({m.groups() for f in files if (m := POSITIONS.match(f["name"]))})
    if len({match for _, match in pairs}) != len(pairs):
        raise ValueError("a match appears in more than one competition")
    return pairs


def draw(pairs, seed=SEED, per_stratum=1):
    """Deterministic stratified draw; returns (holdout, dev), each a sorted match-ID list."""
    rng = random.Random(seed)
    holdout = []
    for competition in sorted({c for c, _ in pairs}):
        stratum = sorted(m for c, m in pairs if c == competition)
        if len(stratum) <= per_stratum:
            raise ValueError("a stratum would leave no dev match")
        holdout += rng.sample(stratum, per_stratum)
    dev = sorted(m for _, m in pairs if m not in holdout)
    return sorted(holdout), dev


def build(manifest_path=MANIFEST, seed=SEED):
    pairs = matches_from_manifest(manifest_path)
    holdout, dev = draw(pairs, seed)
    body = {
        "source": "DFL/IDSSE figshare 28196177 v1",
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "method": "stratified by competition, 1 holdout match per stratum, random.Random(seed).sample",
        "seed": seed,
        "competition_of": {m: c for c, m in pairs},
        "holdout": holdout,
        "dev": dev,
        "dev_folds": [{"validate": m, "train": [d for d in dev if d != m]} for m in dev],
        "rules": [
            "Holdout matches: mechanical integrity checks only until FCB-18 (no label, outcome or error inspection).",
            "All tuning and selection: leave-one-match-out over dev; matches, not halves, are the unit.",
            "Changing this file requires owner approval.",
        ],
    }
    body["split_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def load(path=SPLIT):
    body = json.loads(Path(path).read_text())
    check = dict(body)
    digest = check.pop("split_sha256")
    if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != digest:
        raise ValueError("split file was edited without regenerating its hash")
    return body
