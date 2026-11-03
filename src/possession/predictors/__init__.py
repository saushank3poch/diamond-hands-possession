"""Label-free predictors. Each maps a Tracking to one H/A/D/X label per frame.

Interface: `make(spec)` returns an object with `fit(examples)` (examples are
(Tracking, codes) pairs from training matches only; may be ignored) and
`predict(tracking) -> np.ndarray[str]`. Frames outside active periods must be X.
Predictors never import the evaluation or adapter packages.
"""
from possession.predictors.baseline import NearestPlayer
from possession.predictors.deadball import DeadBall

REGISTRY = {"nearest_player": NearestPlayer, "deadball": DeadBall}


def make(spec):
    spec = dict(spec)
    name = spec.pop("name")
    if name not in REGISTRY:
        raise ValueError("unknown predictor")
    return REGISTRY[name](**spec)
