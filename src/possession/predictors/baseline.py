"""The organisers' nearest-player baseline, re-implemented on the common Tracking schema.

Same rule as starter/baseline.py: per frame with a visible ball, the team of the nearest
visible player; a centred +/- window majority vote between H and A only (ties to H);
D where neither team has a vote. Differences from the starter, both by our contract:
the vote never crosses a period boundary, and frames outside active periods are X.
"""
import numpy as np

from possession.periods import predict_periods


def nearest_team(tracking):
    """Per frame: 'H', 'A' or '' (no visible ball or no visible player)."""
    d = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.argmin(axis=1)
    seen = np.isfinite(d.min(axis=1))
    return np.where(seen, tracking.teams[best], "")


def majority(codes, window):
    if window <= 0:
        return np.where(codes == "", "D", codes)
    home = np.r_[0, np.cumsum(codes == "H")]
    away = np.r_[0, np.cumsum(codes == "A")]
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    return np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))


class NearestPlayer:
    def __init__(self, window=50):
        if type(window) is not int or window < 0:
            raise ValueError("window must be a nonnegative integer")
        self.window = window

    def fit(self, examples):
        return self  # untuned baseline

    def predict(self, tracking):
        codes = nearest_team(tracking)
        period = tracking.period
        cuts = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
        blocks = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if period[a] > 0]
        return np.array(predict_periods(codes, blocks, lambda: lambda rows: majority(np.asarray(rows), self.window)))
