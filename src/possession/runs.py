"""Local run records and deterministic process configuration (no asset loading)."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def seed_generators(seed):
    """Called by Python pipeline entry points; child processes receive the same seed."""
    random.seed(seed)
    if importlib.util.find_spec("numpy") is None:
        return  # The scaffold and its tests use only the standard library.
    import numpy as np
    np.random.seed(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def run(config_path, command, mode="dev", run_dir=None):
    """Execute one trusted local command, recording success or failure in runs/."""
    if mode == "target":
        # Only the shell wrapper sets this after redirecting both output streams.
        if not run_dir or os.environ.get("POSSESSION_TARGET_LOG_DIR") != str(run_dir):
            raise ValueError("target runs require scripts/run_target_local.sh")
        run_dir = Path(run_dir)
    else:
        parent = ROOT / "runs" / "dev"
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))

    start = time.monotonic()
    code = 70
    try:
        raw_config = Path(config_path).read_bytes()
        (run_dir / "config.yaml").write_bytes(raw_config)
        # JSON is a strict subset of YAML 1.2; keep scaffold configs dependency-free.
        config = json.loads(raw_config)
        seed = config["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not command:
            raise ValueError("a command is required")
        lock = (ROOT / "requirements.lock").read_bytes()
        write_json(run_dir / "run.json", {
            "mode": mode,
            "git_commit": git("rev-parse", "HEAD"),
            "worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "config_sha256": hashlib.sha256(raw_config).hexdigest(),
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
            "python": sys.version,
            "seed": seed,
            "command": command,
        })
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": str(seed), "POSSESSION_SEED": str(seed),
                    "POSSESSION_RUN_DIR": str(run_dir),
                    "PYTHONPATH": str(ROOT / "src")})
        # Pipeline children must call seed_generators for random/numpy. Hash seeding
        # takes effect at interpreter startup; setting it in a running process cannot.
        with (run_dir / "process.log").open("ab") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        code = result.returncode if result.returncode >= 0 else 128 - result.returncode
    except Exception:
        with (run_dir / "process.log").open("a") as log:
            traceback.print_exc(file=log)
    finally:
        write_json(run_dir / "metrics.json", {
            "status": "passed" if code == 0 else "failed",
            "exit_code": code,
            "elapsed_seconds": time.monotonic() - start,
            "model_metrics": None,
        })
    return code
