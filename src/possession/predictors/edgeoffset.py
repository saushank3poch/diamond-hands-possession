"""E006 dead-edge offset on top of E005 (prespecified, 2026-11-04).

`fit()` fits E005 on the training matches, predicts each training match in-sample, pairs predicted
and reference dead-run edges one-to-one (same period, |offset| <= 3 s, greedy by smallest
offset), and stores the median signed latency (pred - ref, whole frames) for stoppages
(live->dead) and restarts (dead->live). `predict()` moves each dead run's start earlier by the
stoppage median and its end later by minus the restart median, clipped to its period, then runs
owner assignment on the adjusted live runs. Reference labels are used only inside `fit()`, on
training matches. Nothing is hand-tuned; the 3 s pairing window is fixed a priori.

Prespecified edge cases: no paired training edges -> that offset is 0. A dead run touching its
period's start or end keeps that boundary (there is no edge there), and nothing moves across a
period. Shifted runs that overlap merge into one dead run; a run whose shifted interval is empty
is dropped (its frames become live). In-sample offsets may differ from held-out ones; adoption is
decided by the outer leave-one-match-out comparison only.
"""
import numpy as np
from possession.predictors.deadball import _min_runs, _period_blocks, _runs
from possession.predictors.baseline import nearest_team
from possession.predictors.xpolicy import ShortSpellX

WINDOW = 75


def _median_offset(ref, pred, window=WINDOW):
    pred = np.asarray(pred, dtype=int)
    pairs = []
    for i, r in enumerate(ref):
        lo, hi = np.searchsorted(pred, [r - window, r + window + 1])
        pairs += [(abs(int(pred[j]) - r), i, j, int(pred[j]) - r) for j in range(lo, hi)]
    pairs.sort()
    ur, up, lat = set(), set(), []
    for _, i, j, d in pairs:
        if i not in ur and j not in up:
            ur.add(i); up.add(j); lat.append(d)
    return int(round(float(np.median(lat)))) if lat else 0


def shift_dead(dead, stop_shift, release_shift):
    """Move each dead run [s, e) to [s - stop_shift, e - release_shift), within the block.

    Runs touching the block start/end keep that boundary (no edge there). Overlaps merge;
    a run that would vanish is dropped.
    """
    n = len(dead)
    out = np.zeros(n, dtype=bool)
    for s, e in _runs(dead):
        if not dead[s]:
            continue
        ns = s if s == 0 else s - stop_shift
        ne = e if e == n else e - release_shift
        ns, ne = max(0, ns), min(n, ne)
        if ne > ns:
            out[ns:ne] = True
    return out


class EdgeOffset(ShortSpellX):
    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "carry")
        super().__init__(**kwargs)
        self.stop_shift, self.release_shift = 0, 0

    def _dead_blocks(self, tracking):
        p = self.dead_probability(tracking)
        for a, b in _period_blocks(tracking.period):
            dead = _min_runs(np.nan_to_num(p[a:b], nan=0.0) >= self.threshold, self.min_dead, self.min_live)
            yield a, b, dead

    def fit(self, examples):
        examples = list(examples)
        super().fit(examples)
        return self._fit_offsets(examples)

    def _fit_offsets(self, examples):
        """Median signed edge latency on (in-sample) training matches; 0 if nothing pairs."""
        stops_r, stops_p, rel_r, rel_p = [], [], [], []
        offset = 0
        for tracking, codes in examples:
            codes = np.asarray(codes)
            for a, b, dead in self._dead_blocks(tracking):
                g = codes[a:b]
                ok = np.isin(g, ["H", "A", "D"])
                # Pair only inside contiguous reference-known sub-blocks of this period
                # (training references only; never used by predict()).
                cut = np.r_[True, ok[1:] != ok[:-1]]
                starts = np.flatnonzero(cut)
                for s0, s1 in zip(starts, np.r_[starts[1:], len(g)]):
                    if not ok[s0]:
                        continue
                    gd, pd = g[s0:s1] == "D", dead[s0:s1]
                    ge = np.flatnonzero(np.diff(gd.astype(np.int8)) != 0) + 1
                    pe = np.flatnonzero(np.diff(pd.astype(np.int8)) != 0) + 1
                    stops_r += (offset + ge[gd[ge]]).tolist()
                    rel_r += (offset + ge[~gd[ge]]).tolist()
                    stops_p += (offset + pe[pd[pe]]).tolist()
                    rel_p += (offset + pe[~pd[pe]]).tolist()
                    offset += (s1 - s0) + 10 * WINDOW  # keep sub-blocks and matches apart
        self.stop_shift = _median_offset(stops_r, sorted(stops_p))
        self.release_shift = _median_offset(rel_r, sorted(rel_p))
        return self

    def predict(self, tracking):
        n = len(tracking.frame)
        out = np.full(n, "X", dtype="<U1")
        codes = nearest_team(tracking)
        for a, b, dead in self._dead_blocks(tracking):
            dead = shift_dead(dead, self.stop_shift, self.release_shift)
            seg = np.full(b - a, "D", dtype="<U1")
            for s, e in _runs(dead):
                if not dead[s]:
                    seg[s:e] = self.owner_run(tracking, codes, a + s, a + e)
            out[a:b] = seg
        return out
