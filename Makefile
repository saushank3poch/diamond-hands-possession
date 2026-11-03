PYTHON ?= python3
CONFIG ?= configs/scaffold.yaml

.PHONY: test dev-eval submission
test:
	@PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m unittest discover -s tests -v

dev-eval:
	@"$(PYTHON)" scripts/run_experiment.py --config "$(CONFIG)" -- "$(PYTHON)" scripts/pipeline.py dev-eval

# All target execution must enter through this wrapper, including future adapters.
submission:
	@PYTHON="$(PYTHON)" sh scripts/run_target_local.sh "$(CONFIG)" -- "$(PYTHON)" scripts/pipeline.py submission
