"""Turn DFL reference flags into challenge-style H/A/D labels (evaluator only).

DFL `BallPossession` follows the last touch (FCB-5 audit): the new team is within
1.5 m at 93-99% of live switches, and 16-26% of switches flip back within 2 s.
The challenge counts only *controlled* possession: touches, deflections and won
aerials without control do not change possession (CHALLENGE.md). The transform
below confirms a live change of team only when the new team holds the flag for at
least `confirm_s` seconds, taken from CHALLENGE.md's "keeps the ball for a few
seconds" proxy. It is fixed before evaluation and never tuned on public GT.
"""
import numpy as np

FPS = 25
CONFIRM_S = 2.0  # prespecified in FCB-5; sensitivity at 1 s and 3 s is reported, not selected


def raw_hd(alive, owner):
    """D where the ball is dead, the flagged owner where alive, X where unknown."""
    return np.where(alive == 0, "D", np.where(alive == 1, owner, "X"))


def confirm_control(labels, period, confirm_s=CONFIRM_S, fps=FPS):
    """Absorb unconfirmed live switches into the previous owner, per period.

    A team spell that directly follows the other team (no D between) and lasts
    fewer than confirm_s seconds is relabelled as the previous owner. The first
    spell after a D or a period start is a restart and always kept. X frames
    (unknown reference) pass through unchanged and break the chain.
    """
    labels = np.asarray(labels)
    out = labels.copy()
    min_len = int(round(confirm_s * fps))
    for p in np.unique(period):
        if not p:
            continue
        rows = np.flatnonzero(period == p)
        seq = labels[rows]
        edges = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [len(seq)]))
        owner = None
        for a, b in zip(starts, stops):
            v = seq[a]
            if v in ("D", "X"):
                owner = None
                continue
            if owner is None or v == owner or (b - a) >= min_len:
                owner = v
            else:
                out[rows[a:b]] = owner
    return out


def challenge_labels(references, period, confirm_s=CONFIRM_S):
    return confirm_control(raw_hd(references.alive, references.owner), period, confirm_s)
