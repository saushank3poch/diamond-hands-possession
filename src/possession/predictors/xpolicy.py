"""FCB-14 short-spell abstention on top of E002 DeadBall (prespecified, E005).

Inside each decoded live run, any owner spell shorter than `min_spell_s` that is not the run's
first spell is treated as unreliable. With `mode="x"` it is written as X (abstain); with
`mode="carry"` it is overwritten with the owner carried into it. The official scorer fills X
forward, so both modes give the same spellF1; they differ only on frame metrics (macroF1, the
first tie-breaker). D still comes only from the dead mask. The 1 s default equals E002's
minimum live-run length and is fixed a priori, not tuned.
"""
import numpy as np
from possession.predictors.deadball import DeadBall, FPS


def suppress_short_spells(owner, min_len, mode):
    """Return a copy of one live run's owner labels with short non-first spells replaced."""
    if mode not in ("x", "carry"):
        raise ValueError("mode must be 'x' or 'carry'")
    out = np.array(owner, dtype="<U1").copy()
    n = len(out)
    if n == 0:
        return out
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    stops = np.r_[starts[1:], n]
    carried = out[0]
    for i, (s, e) in enumerate(zip(starts, stops)):
        label = owner[s]
        if (i > 0 and label in ("H", "A") and label != carried and e - s < min_len
                and carried in ("H", "A")):
            out[s:e] = "X" if mode == "x" else carried
        elif label in ("H", "A"):
            carried = label
    return out


class ShortSpellX(DeadBall):
    def __init__(self, min_spell_s=1.0, mode="x", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("x", "carry"):
            raise ValueError("mode must be 'x' or 'carry'")
        self.min_spell, self.mode = int(round(float(min_spell_s) * FPS)), mode

    def owner_run(self, tracking, codes, start, stop):
        owner = super().owner_run(tracking, codes, start, stop)
        return suppress_short_spells(owner, self.min_spell, self.mode)
