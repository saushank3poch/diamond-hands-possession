"""Label-free predictors. Each maps a Tracking to one H/A/D/X label per frame.

Interface: `make(spec)` returns an object with `fit(examples)` (examples are
(Tracking, codes) pairs from training matches only; may be ignored) and
`predict(tracking) -> np.ndarray[str]`. Frames outside active periods must be X.
Predictors never import the evaluation or adapter packages.
"""
from possession.predictors.baseline import NearestPlayer
from possession.predictors.deadball import DeadBall

from possession.predictors.turnover import Turnover

from possession.predictors.restart import RestartOwner
from possession.predictors.xpolicy import ShortSpellX
from possession.predictors.edgeoffset import EdgeOffset

REGISTRY = {"nearest_player": NearestPlayer, "deadball": DeadBall, "turnover": Turnover, "restart_owner": RestartOwner,
            "short_spell_x": ShortSpellX, "edge_offset": EdgeOffset}


def make(spec):
    spec = dict(spec)
    name = spec.pop("name")
    if name not in REGISTRY:
        raise ValueError("unknown predictor")
    return REGISTRY[name](**spec)
