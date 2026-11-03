"""FCB-11: dead-ball detector fused from tracking-only evidence, plus the baseline owner vote.

Per frame, a logistic model (numpy, L2) estimates P(dead) from windowed tracking signals in
four groups, each removable for ablation:
  ball      - ball missing, time since last seen, last-seen distance to the nearest line or off
              the pitch, windowed ball speed (missing ball alone is one input, never the rule)
  motion    - windowed (1 s displacement) team speeds; per-frame derivatives are avoided because
              position jitter makes them meaningless (FCB-11 finding)
  formation - team compactness, centroid separation, own-half shares (kick-off shape), players
              crowding the last-seen ball (corner/free-kick/wall shapes)
  restart   - change in team and ball speed between the past and the future few seconds
              (re-acceleration at a restart, deceleration at a stoppage)
Each signal is also averaged over centred 1 s / 4 s / 12 s windows within a period (offline).

Decoding: P(dead) smoothed over 1 s, thresholded at the value that maximises dead-frame F1 on
the training matches, then dead runs < min_dead_s and live runs < min_live_s are removed
(fixed a priori). Owner: the nearest-player majority vote, run separately inside each live run
so it never crosses a stoppage. No Smart Tagging. No reference data reaches `predict`.
"""
import numpy as np

from possession.predictors.baseline import majority, nearest_team

FPS = 25
GROUPS = ("ball", "motion", "formation", "restart")
CONTEXT_S = (1, 4, 12)


def _period_blocks(period):
    cuts = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if period[a] > 0]


def _rolling_mean(x, half, blocks):
    """Centred NaN-aware mean over +/- half frames, never crossing a block."""
    out = np.full(len(x), np.nan)
    for a, b in blocks:
        v = x[a:b]
        ok = np.isfinite(v)
        cs = np.r_[0, np.cumsum(np.where(ok, v, 0.0))]
        cn = np.r_[0, np.cumsum(ok)]
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - half), np.minimum(b - a, i + half + 1)
        n = cn[hi] - cn[lo]
        out[a:b] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out


def _window_speed(xy, blocks, k=12, cap=12.0):
    """Speed from displacement over +/- k frames (same slot visible at both ends), per slot."""
    out = np.full(xy.shape[:-1], np.nan)
    for a, b in blocks:
        if b - a <= 2 * k:
            continue
        seg = xy[a:b]
        d = np.linalg.norm(seg[2 * k:] - seg[:-2 * k], axis=-1) / (2 * k / FPS)
        out[a + k:b - k] = np.where(d <= cap, d, np.nan)
    return out


def _nanmedian_rows(v):
    with np.errstate(all="ignore"):
        ok = np.isfinite(v).any(axis=1)
        res = np.full(v.shape[0], np.nan)
        if ok.any():
            res[ok] = np.nanmedian(v[ok], axis=1)
    return res


def _carry_last(xy, blocks):
    """Last observed position, carried forward within a block (NaN before the first sighting)."""
    out = np.full(xy.shape, np.nan)
    for a, b in blocks:
        seg = xy[a:b]
        seen = np.isfinite(seg).all(axis=1)
        idx = np.where(seen, np.arange(b - a), -1)
        idx = np.maximum.accumulate(idx)
        ok = idx >= 0
        out[a:b][ok] = seg[idx[ok]]
    return out


def signals(tracking):
    """Dict of group -> list of (name, per-frame array). Tracking inputs only."""
    period = tracking.period
    blocks = _period_blocks(period)
    n = len(period)
    half_l, half_w = tracking.pitch / 2
    ball = tracking.ball
    missing = tracking.ball_missing.astype(float)
    last = _carry_last(ball, blocks)
    # Seconds since the ball was last seen (capped), within a block.
    age = np.full(n, np.nan)
    for a, b in blocks:
        seen = ~tracking.ball_missing[a:b]
        idx = np.maximum.accumulate(np.where(seen, np.arange(b - a), -1))
        age[a:b] = np.where(idx >= 0, (np.arange(b - a) - idx) / FPS, np.nan)
    age = np.minimum(age, 30.0)
    line_dist = np.minimum(half_l - np.abs(last[:, 0]), half_w - np.abs(last[:, 1]))  # < 0 = off pitch
    ball_speed = _window_speed(ball[:, None, :], blocks, k=6, cap=45.0)[:, 0]

    teams = tracking.teams
    sp = _window_speed(tracking.players, blocks)
    h_speed, a_speed = _nanmedian_rows(sp[:, teams == "H"]), _nanmedian_rows(sp[:, teams == "A"])
    all_speed = _nanmedian_rows(sp)

    def team_shape(t):
        p = tracking.players[:, teams == t]
        with np.errstate(all="ignore"):
            c = np.nanmean(p, axis=1)
            spread = np.sqrt(np.nanmean(np.sum((p - c[:, None]) ** 2, axis=-1), axis=1))
        return c, spread, p
    hc, hspread, hp = team_shape("H")
    ac, aspread, ap = team_shape("A")
    sep = np.linalg.norm(hc - ac, axis=1)
    side = np.sign(hc[:, 0] - ac[:, 0])  # which half each team occupies, from geometry only
    with np.errstate(all="ignore"):
        h_own = np.nanmean(np.where(np.isfinite(hp[..., 0]), (hp[..., 0] * side[:, None]) > 0, np.nan), axis=1)
        a_own = np.nanmean(np.where(np.isfinite(ap[..., 0]), (ap[..., 0] * side[:, None]) < 0, np.nan), axis=1)
        d_last = np.linalg.norm(tracking.players - last[:, None, :], axis=-1)
    crowd = np.sum(np.nan_to_num(d_last, nan=np.inf) < 10.0, axis=1).astype(float)
    crowd[~np.isfinite(last).all(axis=1)] = np.nan
    centre_dist = np.linalg.norm(last, axis=1)

    def delta(x, s=3):
        past = _rolling_mean(np.r_[x[s * FPS:], np.full(s * FPS, np.nan)], 0, blocks)  # future value
        fut = np.r_[np.full(s * FPS, np.nan), x[:-s * FPS]]                          # past value
        return past - fut

    return {
        "ball": [("ball_missing", missing), ("ball_age_s", age), ("ball_line_dist", line_dist),
                 ("ball_off_pitch", (line_dist < 0).astype(float)), ("ball_speed", ball_speed)],
        "motion": [("speed_all", all_speed), ("speed_h", h_speed), ("speed_a", a_speed)],
        "formation": [("spread_h", hspread), ("spread_a", aspread), ("centroid_sep", sep),
                      ("own_half_h", h_own), ("own_half_a", a_own), ("crowd_last_ball", crowd),
                      ("ball_centre_dist", centre_dist)],
        "restart": [("speed_change_3s", delta(all_speed)), ("ball_speed_change_3s", delta(ball_speed))],
    }


def design(tracking, groups):
    blocks = _period_blocks(tracking.period)
    sig = signals(tracking)
    cols, names = [], []
    for g in groups:
        for name, x in sig[g]:
            x = np.asarray(x, float)
            cols.append(x)
            names.append(name)
            for s in CONTEXT_S:
                cols.append(_rolling_mean(x, s * FPS // 2, blocks))
                names.append(f"{name}_m{s}s")
    return np.column_stack(cols), names


def _fit_logistic(X, y, l2=1.0, iters=25):
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + reg * w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg + 1e-6)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def _runs(mask):
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(mask)]
    return list(zip(starts, stops))


def _min_runs(dead, min_dead, min_live):
    """Remove short dead runs, then short live runs (both by flipping them)."""
    out = dead.copy()
    for value, min_len in ((True, min_dead), (False, min_live)):
        for a, b in _runs(out):
            if out[a] == value and b - a < min_len and a > 0 and b < len(out):
                out[a:b] = not value
    return out


class DeadBall:
    def __init__(self, groups=GROUPS, window=50, min_dead_s=2.0, min_live_s=1.0, l2=1.0, stride=5):
        groups = tuple(groups)
        if not groups or any(g not in GROUPS for g in groups):
            raise ValueError("unknown feature group")
        self.groups, self.window = groups, int(window)
        self.min_dead, self.min_live = int(min_dead_s * FPS), int(min_live_s * FPS)
        self.l2, self.stride = float(l2), int(stride)

    def _standard(self, X):
        Z = (X - self.mu) / self.sd
        miss = ~np.isfinite(Z)
        return np.c_[np.where(miss, 0.0, Z), miss[:, self.miss_cols].astype(float)]

    def fit(self, examples):
        Xs, ys = [], []
        for tracking, codes in examples:
            X, _ = design(tracking, self.groups)
            keep = (tracking.period > 0) & np.isin(codes, ["H", "A", "D"])
            idx = np.flatnonzero(keep)[:: self.stride]
            Xs.append(X[idx])
            ys.append((np.asarray(codes)[idx] == "D").astype(float))
        X, y = np.vstack(Xs), np.concatenate(ys)
        with np.errstate(all="ignore"):
            self.mu = np.nanmean(X, axis=0)
            self.sd = np.nanstd(X, axis=0)
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 0), self.sd, 1.0)
        self.miss_cols = np.flatnonzero(np.isnan(X).mean(axis=0) > 0.01)
        Z = self._standard(X)
        self.w = _fit_logistic(Z, y, self.l2)
        # Threshold: maximise dead-frame F1 on the training matches (never the held-out match).
        p = self._prob_from_Z(Z)
        best, self.threshold = -1, 0.5
        for t in np.linspace(0.2, 0.8, 25):
            pred = p >= t
            tp = np.sum(pred & (y == 1))
            f1 = 2 * tp / max(pred.sum() + y.sum(), 1)
            if f1 > best:
                best, self.threshold = f1, float(t)
        return self

    def _prob_from_Z(self, Z):
        return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(Z)), Z] @ self.w, -30, 30)))

    def dead_probability(self, tracking):
        X, _ = design(tracking, self.groups)
        p = self._prob_from_Z(self._standard(X))
        blocks = _period_blocks(tracking.period)
        return _rolling_mean(p, FPS // 2, blocks)

    def predict(self, tracking):
        n = len(tracking.frame)
        out = np.full(n, "X", dtype="<U1")
        p = self.dead_probability(tracking)
        codes = nearest_team(tracking)
        for a, b in _period_blocks(tracking.period):
            dead = _min_runs(np.nan_to_num(p[a:b], nan=0.0) >= self.threshold, self.min_dead, self.min_live)
            seg = np.full(b - a, "D", dtype="<U1")
            for s, e in _runs(dead):
                if not dead[s]:
                    seg[s:e] = majority(codes[a + s:a + e], self.window)
            out[a:b] = seg
        return out
