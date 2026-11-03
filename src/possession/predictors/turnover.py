"""FCB-12: conservative control evidence and offline confirmation within live runs.

No model-selection labels enter this module. DeadBall.fit trains only its existing
stoppage detector; the ownership thresholds below are prespecified engineering
assumptions. D is emitted exclusively by the parent detector.
"""
import numpy as np

from possession.predictors.deadball import DeadBall, FPS


def displacement_velocity(xy, half=12, cap=12.0):
    """Centred displacement over 0.96 s, requiring observations throughout the window.

    The caller supplies ONE live run. Missing samples break support, preventing a
    slot reappearing after a gap from supplying motion evidence. Small uninterrupted
    identity swaps remain indistinguishable from motion; the speed cap is only a guard.
    """
    out = np.full(xy.shape, np.nan)
    if len(xy) <= 2 * half:
        return out
    good = np.isfinite(xy).all(axis=-1)
    counts = np.concatenate([np.zeros_like(good[:1], dtype=int), np.cumsum(good, axis=0)])
    supported = counts[2 * half + 1:] - counts[:-2 * half - 1] == 2 * half + 1
    v = (xy[2 * half:] - xy[:-2 * half]) / (2 * half / FPS)
    valid = supported & (np.linalg.norm(v, axis=-1) <= cap)
    out[half:-half] = np.where(valid[..., None], v, np.nan)
    return out


def control_evidence(tracking, start, stop, radius=2.5, margin=0.5, relative_speed=4.0):
    """H/A only for close, uncontested, motion-supported ball control; else X.

    Motion: relative velocity <=4 m/s OR ball speed <=3 m/s (slow reception).
    Neither a lone proximity observation nor missing observations establish control.
    Distances/motion use the same slot; no repaired coordinates or reference flags.
    """
    players, ball = tracking.players[start:stop], tracking.ball[start:stop]
    distance = np.linalg.norm(players - ball[:, None, :], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    out = np.full(len(ball), "X", dtype="<U1")
    if players.shape[1] == 0:
        return out
    best = distance.argmin(axis=1)
    rows = np.arange(len(ball))
    nearest = distance[rows, best]
    team = tracking.teams[best]
    opponent = np.min(np.where(tracking.teams[None, :] != team[:, None], distance, np.inf), axis=1)
    pv = displacement_velocity(players)
    bv = displacement_velocity(ball, cap=60.0)
    relative = np.linalg.norm(pv[rows, best] - bv, axis=-1)
    slow = np.linalg.norm(bv, axis=-1) <= 3.0
    # Missing opponent positions do not prove uncontested control.
    good = (nearest <= radius) & np.isfinite(opponent) & (opponent >= nearest + margin)
    good &= np.isfinite(relative) & ((relative <= relative_speed) | slow)
    out[good] = team[good]
    return out


def confirmed_owner(evidence, frames=10, backdate=True):
    """Keep the owner across loose/contested observations; confirm sustained opposition.

    X interrupts the pending candidate, never the established owner. A single touch
    cannot switch (frames>=2). Confirmations are back-dated to evidence onset offline.
    The first confirmed owner back-fills the run head, a provisional restart rule.
    Return labels and (onset, confirmation, team) events for synthetic lag checks.
    """
    if type(frames) is not int or frames < 2:
        raise ValueError("confirmation requires at least two frames")
    evidence = np.asarray(evidence)
    if evidence.ndim != 1 or not np.isin(evidence, list("HAX")).all():
        raise ValueError("evidence must be a vector of H/A/X")
    out = np.full(len(evidence), "X", dtype="<U1")
    owner, pending, onset, count = "X", "X", 0, 0
    events = []
    for i, team in enumerate(evidence):
        if team == "X" or team == owner:
            pending, count = "X", 0
        else:
            if team != pending:
                pending, onset, count = team, i, 0
            count += 1
            if count >= frames:
                initial = owner == "X"
                owner = team
                events.append((onset, i, str(team)))
                boundary = 0 if initial else (onset if backdate else i)
                out[boundary:i + 1] = owner
                pending, count = "X", 0
        out[i] = owner
    return out, events


class Turnover(DeadBall):
    def __init__(self, confirmation_s=0.4, backdate=True, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(confirmation_s) or confirmation_s * FPS < 2:
            raise ValueError("confirmation_s must be finite and at least 2 frames")
        if type(backdate) is not bool:
            raise ValueError("backdate must be boolean")
        self.confirmation_frames = int(np.ceil(confirmation_s * FPS))
        self.backdate = backdate

    def owner_run(self, tracking, codes, start, stop):
        evidence = control_evidence(tracking, start, stop)
        return confirmed_owner(evidence, self.confirmation_frames, self.backdate)[0]
