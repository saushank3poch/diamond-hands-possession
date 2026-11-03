# Local research harness

FCB-1 adds run bookkeeping and output containment only. It does not implement a
model, read match assets, score public labels, or generate a submission.

The FCB-2 contract tests exercise the official scorer and therefore require
NumPy and pandas. Set up the pinned environment below, then run:

```sh
make test
```

The run harness itself has no third-party dependencies or network calls. `requirements.lock`
pins the existing scorer environment (Python 3.14). Install it when using the
official scorer and contract tests, outside the offline run itself:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
make test PYTHON=.venv/bin/python
```

Make prefers `.venv/bin/python` when it exists; otherwise it uses `python3`.
Missing scorer dependencies fail the suite, rather than silently skipping it.
Contract tests generate all observations, reference sequences, XML and CSV files
in temporary directories. They never load target assets or the public GT.

`possession.periods.predict_periods` creates fresh predictor state for each
half-open block, validates returned labels, and writes X outside those blocks.
Only published periods or observable discontinuities may define its blocks.
Reference labels and masks belong to the evaluator and are not accepted by this
helper. FCB-2 checks that boundary using a stateful test double; real model and
adapter leakage checks must also be exercised when FCB-4/FCB-12 supply them.

Configurations in `configs/*.yaml` use the JSON subset of YAML 1.2, parsed with
the standard library. `seed` is required, an integer from 0 to 2**32 - 1.
This avoids adding a YAML library just for one scaffold setting.

## Development commands

```sh
python3 scripts/run_experiment.py --config configs/scaffold.yaml -- python3 -c 'print("synthetic")'
make dev-eval
```

The second command deliberately fails with exit code 2 until FCB-9 implements
evaluation. It records that failure; it does not substitute fake model scores.
Every launched command gets a unique `runs/dev/run-*/` directory containing:

- `config.yaml`: exact input configuration bytes.
- `run.json`: commit, worktree-dirty flag (including untracked files), config/lock hashes, Python version,
  seed and command. A dirty flag is not a snapshot; freeze runs from clean commits.
- `process.log`: stdout/stderr and any runner exception.
- `metrics.json`: status, exit code, elapsed time and `model_metrics: null`.

Children receive `PYTHONHASHSEED`, `POSSESSION_SEED`, `POSSESSION_RUN_DIR`, and
the source package on `PYTHONPATH`. Python pipeline entry points must call
`possession.runs.seed_generators(int(os.environ["POSSESSION_SEED"]))` to seed
`random` and NumPy when installed. Future libraries need their own explicit
determinism settings. Run IDs and elapsed time vary; model output reproducibility
will be checked when a model exists. A missing/invalid config is a failed run.

## Target commands stay local

```sh
sh scripts/run_target_local.sh configs/scaffold.yaml -- python3 -c 'print("synthetic only")'
make submission
```

`make submission` deliberately fails until FCB-19 provides the actual pipeline.
Never bypass the wrapper for a command that touches target assets, including
adapters, feature generation, timing, validation and failure diagnosis.

The wrapper redirects stdout/stderr before launching Python, closes its saved
output descriptor in the child, and disconnects child stdin. The caller receives
only `TARGET_RUN PASS exit_code=0` or `TARGET_RUN FAIL exit_code=N`. Nonzero
status propagates to the shell (Make additionally prints its generic error).
Private logs go to `runs/target/run-*/` with restrictive permissions. Python
startup errors go to `bootstrap.log`; pipeline output goes to `process.log`.
Bootstrap/config failures may have incomplete provenance; they are never successes.
Do not read target logs back into agent tools or publish them.

This is output containment for trusted local programs, not an OS sandbox.
Commands must not upload data, open `/dev/tty`, start detached processes, or write
artefacts outside ignored local output directories. Run the wrapper normally,
without shell tracing. Do not pass restricted contents as shell command text;
pass local file paths. No target command is exercised by the synthetic tests.

Keep match-data links intact. Store generated outputs in `runs/`, `artifacts/`
or `submissions/`; these and `external/` are ignored. The upstream data exclusions
remain in force. Claude reviews the FCB-1 branch before merge; later tickets
supply adapters, contract tests, evaluation and model implementations.
