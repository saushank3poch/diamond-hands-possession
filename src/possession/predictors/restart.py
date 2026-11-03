"""FCB-13 local restart-taker candidate; tracking only, with conservative fallback.

All constants are prespecified engineering assumptions. The ball nearest in time
within +/-0.5s of a predicted live edge anchors the restart point. Require repeated
near-stationary ball observations and an unambiguous nearby team. Near a touchline
(excluding corners), penalise candidates standing inside the pitch: a thrower is
normally on or outside that line. This is a soft geometric cue, not a restart tag.
"""
import numpy as np
from possession.predictors.deadball import DeadBall, FPS


def restart_team(tracking, start, stop, geometry=True):
    """Return H/A or X for ambiguous/insufficient evidence, never reference labels.

    Clip context to the same active period and the live-run end. Pre-edge context
    can include the preceding dead interval, but never a preceding period.
    """
    period = tracking.period[start]
    if period <= 0:
        return "X"
    left = start
    while left > max(0, start - FPS // 2) and tracking.period[left - 1] == period:
        left -= 1
    right = min(stop, start + FPS // 2 + 1)
    while right > start + 1 and tracking.period[right - 1] != period:
        right -= 1
    rows = np.arange(left, right)
    ball = tracking.ball[rows]
    seen = np.isfinite(ball).all(axis=1)
    if seen.sum() < 3:
        return "X"
    observed = np.flatnonzero(seen)
    anchor = ball[observed[np.argmin(np.abs(rows[observed] - start))]]
    stable = seen & (np.linalg.norm(ball - anchor, axis=1) <= 1.0)
    if stable.sum() < 3:
        return "X"
    players = tracking.players[rows[stable]]
    dist = np.linalg.norm(players - anchor, axis=-1)
    cost = np.where(np.isfinite(dist), dist, np.inf)
    half_l, half_w = tracking.pitch / 2
    touchline = abs(anchor[1]) >= half_w - 1.5 and abs(anchor[0]) < half_l - 5
    if geometry and touchline:
        # Interior receivers should not outrank the taker solely through proximity.
        inside = np.maximum(0.0, half_w - np.abs(players[..., 1]))
        cost += np.where(np.isfinite(inside), 0.75 * inside, 0.0)
    scores, raw = [], []
    for team in ("H", "A"):
        mask = tracking.teams == team
        if not mask.any():
            return "X"
        chosen = cost[:, mask].argmin(axis=1)
        scores.append(cost[:, mask][np.arange(len(cost)), chosen])
        raw.append(dist[:, mask][np.arange(len(cost)), chosen])
    scores, raw = np.column_stack(scores), np.column_stack(raw)
    supported = np.isfinite(scores).all(axis=1)
    if supported.sum() < 3:
        return "X"
    scores, raw = scores[supported], raw[supported]
    med = np.median(scores, axis=0)
    best = int(med.argmin())
    if np.median(raw[:, best]) > 3.0 or med[1-best] - med[best] < 0.75:
        return "X"
    if np.mean(scores[:, best] < scores[:, 1-best]) < 0.7:
        return "X"
    return "HA"[best]


class RestartOwner(DeadBall):
    def __init__(self, geometry=True, **kwargs):
        super().__init__(**kwargs)
        if type(geometry) is not bool:
            raise ValueError("geometry must be boolean")
        self.geometry = geometry

    def owner_run(self, tracking, codes, start, stop):
        owner = super().owner_run(tracking, codes, start, stop)
        team = restart_team(tracking, start, stop, self.geometry)
        if team != "X":
            # Correct only the initial baseline-owner spell, capped at one second.
            # Never replace a later baseline switch or change the dead mask.
            changes = np.flatnonzero(owner[1:] != owner[:-1]) + 1
            end = min(len(owner), FPS, int(changes[0]) if len(changes) else len(owner))
            owner[:end] = team
        return owner
