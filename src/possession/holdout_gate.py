"""FCB-18 holdout gate: the holdout is opened exactly once, on the frozen candidate.

Preflight (`check_gate`), all before any holdout file is read:
- fixed, tracked paths only: the manifest is always configs/freeze.json and the run config must be
  byte-identical to configs/holdout.yaml; both must be byte-identical to their blobs at the
  freeze tag (no external, ignored or path-supplied manifests);
- HEAD is exactly the freeze tag's commit and the worktree is clean;
- requirements.lock and the split hash match the manifest;
- effective run settings (seed in config and environment, bootstrap count) match the manifest;
- no reservation exists yet.

Reservation (`reserve`): an atomic O_CREAT|O_EXCL marker under runs/holdout/ is written with
status "started" before any holdout access. Any existing marker (started, failed or completed)
refuses every later attempt; there is no automatic retry. Rework after a reservation needs owner
approval (D5) and is reported as holdout-contaminated. `finish` records failed/completed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "configs/freeze.json"
CONFIG = "configs/holdout.yaml"
MARKER = ROOT / "runs" / "holdout" / "RESERVATION.json"
RESULT = ROOT / "findings" / "results" / "HOLDOUT_metrics.json"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def _fail(msg):
    raise SystemExit("holdout gate: " + msg)


def check_gate(run_config_bytes, env_seed, split_sha256, git=_git):
    if MARKER.exists() or RESULT.exists():
        _fail("holdout already reserved or opened; re-running needs owner approval (D5)")
    manifest_bytes = (ROOT / MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    tag = manifest["tag"]
    head = git("rev-parse", "HEAD").decode().strip()
    tagged = git("rev-parse", f"{tag}^{{commit}}").decode().strip()
    if head != tagged:
        _fail("HEAD is not the freeze tag")
    if git("status", "--porcelain").strip():
        _fail("worktree is not clean")
    if git("show", f"{tag}:{MANIFEST}") != manifest_bytes:
        _fail("freeze manifest differs from its blob at the tag")
    tagged_config = git("show", f"{tag}:{CONFIG}")
    if run_config_bytes != tagged_config or (ROOT / CONFIG).read_bytes() != tagged_config:
        _fail("run config differs from the tracked holdout config at the tag")
    config = json.loads(run_config_bytes)
    if set(config) != {"seed", "bootstrap_n"}:
        _fail("holdout config may set only seed and bootstrap_n")
    run = manifest["holdout_run"]
    if config["seed"] != run["seed"] or env_seed != run["seed"] or config["bootstrap_n"] != run["bootstrap_n"]:
        _fail("effective run settings differ from the manifest")
    if hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest() != manifest["requirements_lock_sha256"]:
        _fail("requirements.lock differs from the manifest")
    if split_sha256 != manifest["split_sha256"]:
        _fail("split differs from the manifest")
    return manifest["predictor"], {"tag": tag, "commit": head, "manifest": MANIFEST, "run": run}


def reserve(run_dir):
    """Atomically claim the single holdout attempt; raises if anyone has claimed it already."""
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        _fail("holdout already reserved; re-running needs owner approval (D5)")
    with os.fdopen(fd, "w") as f:
        json.dump({"status": "started", "run_dir": str(run_dir)}, f)
        f.write("\n")


def finish(status, run_dir, error=None):
    if status not in ("completed", "failed"):
        raise ValueError("status must be completed or failed")
    rec = {"status": status, "run_dir": str(run_dir)}
    if error:
        rec["error_type"] = type(error).__name__  # type only: messages could carry data content
    tmp = MARKER.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    os.replace(tmp, MARKER)


def run_once(run_config_bytes, env_seed, split_sha256, run_dir, evaluate, write, git=_git):
    """Preflight, reserve, evaluate, record. `evaluate(spec, bootstrap_n)` reads the holdout."""
    spec, freeze = check_gate(run_config_bytes, env_seed, split_sha256, git=git)
    reserve(run_dir)  # one-shot claim before any holdout access
    try:
        report = evaluate(spec, freeze["run"]["bootstrap_n"])
        report["freeze"] = freeze
        write(report)
    except BaseException as exc:
        finish("failed", run_dir, exc)
        raise
    finish("completed", run_dir)
    return report
