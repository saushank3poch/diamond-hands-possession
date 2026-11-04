"""FCB-18 holdout gate: the holdout is opened exactly once, on the frozen candidate.

Checks (all must pass before any holdout file is read):
- the config names a freeze manifest; its tag exists and HEAD is exactly that tag's commit;
- the worktree is clean;
- the predictor in the config equals the manifest's predictor;
- no holdout result has been recorded (marker under findings/ or runs/holdout/).
`record_opened` writes the ignored runs/holdout/OPENED marker; the committed result file under
findings/results/ is the durable record. Rework after opening needs owner approval (D5).
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / "runs" / "holdout" / "OPENED"
RESULT = ROOT / "findings" / "results" / "HOLDOUT_metrics.json"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def already_opened():
    return MARKER.exists() or RESULT.exists()


def check_gate(config, git=_git):
    if already_opened():
        raise SystemExit("holdout already opened; re-running needs owner approval (D5)")
    if "freeze" not in config or "predictors" in config:
        raise SystemExit("holdout config must name a freeze manifest and no other predictors")
    manifest = json.loads((ROOT / config["freeze"]).read_text())
    tag, spec = manifest["tag"], manifest["predictor"]
    if config.get("predictor", spec) != spec:
        raise SystemExit("config predictor differs from the freeze manifest")
    head, tagged = git("rev-parse", "HEAD"), git("rev-parse", f"{tag}^{{commit}}")
    if head != tagged:
        raise SystemExit("HEAD is not the freeze tag")
    if git("status", "--porcelain"):
        raise SystemExit("worktree is not clean")
    return spec, {"tag": tag, "commit": head, "manifest": config["freeze"]}


def record_opened(run_dir):
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({"run_dir": str(run_dir)}) + "\n")
