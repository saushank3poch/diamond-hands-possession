"""DFL reference flags to H/A/D labels (evaluator only).

Primary reference: `raw_hd`, i.e. DFL flags as published (the source defines
possession by the player in control of the ball). `confirm_control` is an
EXPERIMENTAL proxy only (a minimum-duration filter on live team switches); it is
not accepted ground truth and must never be the sole basis for a model decision.
See findings/dev-labels-audit.md.
"""
import numpy as np

FPS = 25
CONFIRM_S = 2.0  # experimental proxy parameter only


def raw_hd(alive, owner, unknown=None):
    """D where the ball is dead, the flagged owner where alive, X where unknown or masked."""
    labels = np.where(alive == 0, "D", np.where(alive == 1, owner, "X"))
    if unknown is not None:
        labels = np.where(np.asarray(unknown, dtype=bool), "X", labels)
    return labels


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


def reference_labels(references):
    """Primary reference: raw DFL flags with the unknown mask applied."""
    return raw_hd(references.alive, references.owner, references.unknown)


def proxy_labels(references, period, confirm_s=CONFIRM_S):
    """Experimental proxy; the unknown mask is applied first, so masked frames stay X and break state."""
    return confirm_control(reference_labels(references), period, confirm_s)
