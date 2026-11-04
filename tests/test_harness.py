"""Exercise real processes in disposable checkouts without copying any match assets."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = "SYNTHETIC_PRIVATE_PAYLOAD"


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fcb synthetic ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "checkout"
        # No checkout: no data/ or ground-truth files materialized or opened.
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout",
                        str(ROOT), str(self.repo)], check=True, capture_output=True)
        for name in ("src", "scripts", "configs"):
            shutil.copytree(ROOT / name, self.repo / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("Makefile", "requirements.lock", ".gitignore"):
            shutil.copy2(ROOT / name, self.repo / name)
        self.env = os.environ.copy()
        self.env["PYTHON"] = sys.executable
        self.env.pop("POSSESSION_TARGET_LOG_DIR", None)

    def invoke(self, command, target=True, config="configs/scaffold.yaml", env=None):
        if target:
            prefix = ["sh", "scripts/run_target_local.sh", config, "--"]
        else:
            prefix = [sys.executable, "scripts/run_experiment.py", "--config", config, "--"]
        return subprocess.run(prefix + command, cwd=self.repo, env=env or self.env,
                              capture_output=True, text=True, timeout=20)

    def records(self, mode="target"):
        return sorted((self.repo / "runs" / mode).glob("run-*"))

    def assert_contained(self, result, code):
        self.assertEqual(result.returncode, code)
        status = "PASS" if code == 0 else "FAIL"
        self.assertEqual(result.stdout, f"TARGET_RUN {status} exit_code={code}\n")
        self.assertEqual(result.stderr, "")
        self.assertNotIn(PAYLOAD, result.stdout + result.stderr)
        self.assertFalse((self.repo / "data").exists())

    def test_stdout_stderr_exception_and_exit_codes(self):
        for body, code in [
            (f"import sys; print('{PAYLOAD}'); print('{PAYLOAD}', file=sys.stderr)", 0),
            (f"import sys; print('{PAYLOAD}'); sys.exit(7)", 7),
            (f"raise RuntimeError('{PAYLOAD}')", 1),
        ]:
            with self.subTest(code=code):
                result = self.invoke([sys.executable, "-c", body])
                self.assert_contained(result, code)
        records = self.records()
        self.assertEqual(len(records), 3)
        for folder in records:
            self.assertIn(PAYLOAD, (folder / "process.log").read_text())
            record = json.loads((folder / "run.json").read_text())
            self.assertEqual(len(record["git_commit"]), 40)
            self.assertEqual(len(record["lock_sha256"]), 64)
            self.assertEqual(record["seed"], 0)
            self.assertEqual((folder / "config.yaml").read_bytes(),
                             (self.repo / "configs/scaffold.yaml").read_bytes())
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertIsNone(metrics["model_metrics"])
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((folder / "process.log").stat().st_mode), 0o600)
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                         cwd=self.repo, text=True)
        self.assertNotIn("runs/", status)

    def test_timing_output_is_local(self):
        self.assert_contained(self.invoke(["/usr/bin/time", sys.executable, "-c",
                                           f"print('{PAYLOAD}')"]), 0)
        log = (self.records()[0] / "process.log").read_text()
        self.assertIn(PAYLOAD, log)
        self.assertIn("real", log)

    def test_python_bootstrap_failure_is_local(self):
        env = dict(self.env, PYTHON=f"/missing/{PAYLOAD}")
        self.assert_contained(self.invoke(["unused"], env=env), 127)
        self.assertIn(PAYLOAD, (self.records()[0] / "bootstrap.log").read_text())

    def test_missing_command_and_invalid_config_fail_closed(self):
        self.assert_contained(self.invoke([f"/missing/{PAYLOAD}"]), 70)
        (self.repo / "bad.yaml").write_text(PAYLOAD)
        self.assert_contained(self.invoke([sys.executable, "-c", "pass"], config="bad.yaml"), 70)
        for folder in self.records():
            self.assertEqual(json.loads((folder / "metrics.json").read_text())["status"], "failed")

    def test_invalid_seed_does_not_launch_command(self):
        for seed in (True, -1, 2**32, "0"):
            (self.repo / "bad.yaml").write_text(json.dumps({"seed": seed}))
            self.assert_contained(self.invoke([sys.executable, "-c", "raise SystemExit(99)"],
                                             config="bad.yaml"), 70)

    def test_child_has_no_saved_caller_output_descriptor_or_stdin(self):
        body = """import os, sys
assert sys.stdin.read() == ''
try:
    os.write(3, b'SYNTHETIC_PRIVATE_PAYLOAD')
except OSError:
    pass
else:
    raise AssertionError('caller descriptor inherited')
"""
        self.assert_contained(self.invoke([sys.executable, "-c", body]), 0)

    def test_seeded_dev_runs_repeat_and_record_failures(self):
        body = """import os, random
from possession.runs import seed_generators
seed_generators(int(os.environ['POSSESSION_SEED']))
print(random.random(), hash('same-input'))
"""
        for _ in range(2):
            result = self.invoke([sys.executable, "-c", body], target=False)
            self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records("dev")
        self.assertEqual(len(records), 2)
        self.assertEqual((records[0] / "process.log").read_bytes(),
                         (records[1] / "process.log").read_bytes())
        failed = self.invoke([sys.executable, "-c", "raise SystemExit(9)"], target=False)
        self.assertEqual(failed.returncode, 9)

    def test_make_entrypoints_fail_without_touching_assets(self):
        for stage in ("dev-eval", "submission"):
            result = subprocess.run(["make", stage, f"PYTHON={sys.executable}"],
                                    cwd=self.repo, env=self.env, capture_output=True,
                                    text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            if stage == "submission":
                # Implemented since FCB-19: without data it fails generically, printing only pass/fail.
                self.assertRegex(result.stdout, r"^TARGET_RUN FAIL exit_code=[1-9][0-9]*\n$")
                self.assertNotIn("not implemented", result.stdout + result.stderr)
        self.assertFalse((self.repo / "data").exists())

    def test_target_cli_cannot_accidentally_bypass_wrapper(self):
        result = subprocess.run([sys.executable, "scripts/run_experiment.py", "--mode", "target",
                                 "--", sys.executable, "-c", f"print('{PAYLOAD}')"],
                                cwd=self.repo, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(PAYLOAD, result.stdout + result.stderr)

    def test_log_directory_symlink_is_rejected(self):
        destination = Path(self.temp.name) / "elsewhere"
        destination.mkdir()
        (self.repo / "runs").symlink_to(destination, target_is_directory=True)
        self.assert_contained(self.invoke([sys.executable, "-c", "raise SystemExit(99)"]), 64)
        self.assertEqual(list(destination.iterdir()), [])

    def test_missing_wrapper_arguments_are_generic_failure(self):
        self.assert_contained(self.invoke([]), 64)

    def test_child_signal_is_failure_not_success(self):
        self.assert_contained(self.invoke([sys.executable, "-c",
                              "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"]), 143)
        metrics = json.loads((self.records()[0] / "metrics.json").read_text())
        self.assertEqual(metrics["exit_code"], 143)
        self.assertEqual(metrics["status"], "failed")


if __name__ == "__main__":
    unittest.main()
