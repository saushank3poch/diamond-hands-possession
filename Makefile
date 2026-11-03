PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CONFIG ?= configs/scaffold.yaml

.PHONY: test dev-eval submission
test:
	@PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m unittest discover -s tests -v

dev-eval:
	@"$(PYTHON)" scripts/run_experiment.py --config "$(CONFIG)" -- "$(PYTHON)" scripts/pipeline.py dev-eval

# All target execution must enter through this wrapper, including future adapters.
submission:
	@PYTHON="$(PYTHON)" sh scripts/run_target_local.sh "$(CONFIG)" -- "$(PYTHON)" scripts/pipeline.py submission

# Bootstrap does not depend on an existing virtual environment.
BOOTSTRAP_PYTHON ?= python3.14
.PHONY: setup
setup:
	"$(BOOTSTRAP_PYTHON)" -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock
	.venv/bin/python -m pip check
