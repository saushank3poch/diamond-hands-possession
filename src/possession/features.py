"""Tracking-only geometry and motion features. No possession inference or reference inputs."""
from dataclasses import dataclass

import numpy as np

from possession.tracking import Tracking

FPS = 25
BASE_COLUMNS = (
    "frame", "period", "ball_x", "ball_y", "ball_observed", "ball_missing",
    "ball_interpolated", "ball_position_valid", "ball_quality", "ball_vx", "ball_vy",
    "ball_speed", "ball_velocity_valid", "ball_ax", "ball_ay", "ball_observation_age_s",
    "ball_out_of_bounds", "ball_exiting",
)
TEAM_COLUMNS = (
    "observed_count", "observation_quality", "centroid_x", "centroid_y", "depth", "width",
    "line_x_min", "line_x_max", "compactness", "mean_speed", "mean_acceleration", "motion_count",
    "attacking_line", "defending_line", "nearest_distance", "comotion_speed_difference",
)
FEATURE_COLUMNS = BASE_COLUMNS + tuple(team + "_" + name for team in ("H", "A") for name in TEAM_COLUMNS)


@dataclass(frozen=True)
class Config:
    # Prespecified engineering assumptions, not tuned on match scores.
    max_ball_gap_frames: int = 5
    max_player_speed_mps: float = 12.0
    max_ball_speed_mps: float = 60.0
    exit_horizon_s: float = 0.2

    def __post_init__(self):
        if type(self.max_ball_gap_frames) is not int or self.max_ball_gap_frames < 0:
            raise ValueError("max_ball_gap_frames must be a nonnegative integer")
        for value in (self.max_player_speed_mps, self.max_ball_speed_mps, self.exit_horizon_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("motion limits and horizon must be positive and finite")


def blocks(period):
    """Contiguous active observation blocks; period 0 is never interpolated."""
    cuts = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if period[a] > 0]


def interpolate_ball(tracking, max_gap):
    ball = tracking.ball.copy()
    ball[tracking.period == 0] = np.nan
    interpolated = np.zeros(len(ball), dtype=bool)
    for start, stop in blocks(tracking.period):
        known = np.flatnonzero(np.isfinite(ball[start:stop]).all(axis=1)) + start
        for left, right in zip(known[:-1], known[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                fraction = np.arange(1, gap + 1)[:, None] / (gap + 1)
                ball[left + 1:right] = ball[left] + fraction * (ball[right] - ball[left])
                interpolated[left + 1:right] = True
    return ball, interpolated


def derivative(values, period, limit=None):
    """Backward difference, only across adjacent finite samples in the same active period."""
    out = np.full(values.shape, np.nan, dtype=float)
    delta = np.diff(values, axis=0) * FPS
    same = (period[1:] == period[:-1]) & (period[1:] > 0)
    valid = np.isfinite(delta).all(axis=-1)
    valid &= same.reshape((-1,) + (1,) * (valid.ndim - 1))
    if limit is not None:
        valid &= np.linalg.norm(delta, axis=-1) <= limit
    out[1:] = np.where(valid[..., None], delta, np.nan)
    return out


def mean_finite(values, axis=1):
    valid = np.isfinite(values)
    total = np.where(valid, values, 0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def extreme(values, maximum=False):
    valid = np.isfinite(values)
    filled = np.where(valid, values, -np.inf if maximum else np.inf)
    result = filled.max(axis=1) if maximum else filled.min(axis=1)
    return np.where(valid.any(axis=1), result, np.nan)


def compute(tracking: Tracking, cfg: Config = Config()):
    """Fixed per-frame column mapping; NaN means unavailable, never evidence of a stoppage.

    Interpolation is offline (uses both endpoints). Quality values are provenance
    indicators, not calibrated probabilities. Slot motion can still be wrong when
    an identity swap is small enough to pass the displacement guard.
    """
    n = len(tracking.frame)
    active = tracking.period > 0
    ball, interpolated = interpolate_ball(tracking, cfg.max_ball_gap_frames)
    observed = ~tracking.ball_missing & active
    ball_valid = np.isfinite(ball).all(axis=1)
    velocity = derivative(ball, tracking.period, cfg.max_ball_speed_mps)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = derivative(velocity, tracking.period)
    players = tracking.players.copy()
    players[~active] = np.nan
    player_velocity = derivative(players, tracking.period, cfg.max_player_speed_mps)
    player_acceleration = derivative(player_velocity, tracking.period)
    result = {
        "frame": tracking.frame.copy(), "period": tracking.period.copy(),
        "ball_x": ball[:, 0], "ball_y": ball[:, 1],
        "ball_observed": observed, "ball_missing": ~observed,
        "ball_interpolated": interpolated, "ball_position_valid": ball_valid,
        "ball_quality": observed.astype(float) + 0.5 * interpolated,
        "ball_vx": velocity[:, 0], "ball_vy": velocity[:, 1],
        "ball_speed": speed, "ball_velocity_valid": np.isfinite(speed),
        "ball_ax": acceleration[:, 0], "ball_ay": acceleration[:, 1],
    }
    # Since last observed ball, reset at each period. Never carry through halftime.
    age = np.full(n, np.nan)
    for start, stop in blocks(tracking.period):
        last = None
        for i in range(start, stop):
            if observed[i]:
                last = i
            if last is not None:
                age[i] = (i - last) / FPS
    result["ball_observation_age_s"] = age
    half = tracking.pitch / 2
    outside = (np.abs(ball) > half).any(axis=1)
    projected = ball + cfg.exit_horizon_s * velocity
    result["ball_out_of_bounds"] = np.where(ball_valid, outside.astype(float), np.nan)
    # Exiting: currently in bounds, projected beyond a boundary at the fixed horizon.
    exit_valid = ball_valid & np.isfinite(velocity).all(axis=1)
    result["ball_exiting"] = np.where(exit_valid, ((~outside) &
                                       (np.abs(projected) > half).any(axis=1)).astype(float), np.nan)

    for team in ("H", "A"):
        select = tracking.teams == team
        p = players[:, select]
        pv = player_velocity[:, select]
        pa = player_acceleration[:, select]
        count = np.isfinite(p).all(axis=2).sum(axis=1)
        prefix = team + "_"
        result[prefix + "observed_count"] = count
        result[prefix + "observation_quality"] = np.minimum(count / 11.0, 1.0)
        if p.shape[1] == 0:
            # Same schema for a source with no slots for one team.
            p = np.full((n, 1, 2), np.nan)
            pv = pa = p.copy()
        centre = mean_finite(p)
        low_x, high_x = extreme(p[:, :, 0]), extreme(p[:, :, 0], True)
        low_y, high_y = extreme(p[:, :, 1]), extreme(p[:, :, 1], True)
        result.update({prefix + "centroid_x": centre[:, 0], prefix + "centroid_y": centre[:, 1],
                       prefix + "depth": high_x - low_x, prefix + "width": high_y - low_y,
                       prefix + "line_x_min": low_x, prefix + "line_x_max": high_x,
                       prefix + "compactness": np.sqrt(mean_finite(np.sum((p - centre[:, None]) ** 2, axis=2))),
                       prefix + "mean_speed": mean_finite(np.linalg.norm(pv, axis=2)),
                       prefix + "mean_acceleration": mean_finite(np.linalg.norm(pa, axis=2)),
                       prefix + "motion_count": np.isfinite(pv).all(axis=2).sum(axis=1)})
        direction = tracking.home_direction * (1 if team == "H" else -1)
        result[prefix + "attacking_line"] = np.where(direction == 1, high_x,
                                                       np.where(direction == -1, -low_x, np.nan))
        result[prefix + "defending_line"] = np.where(direction == 1, low_x,
                                                       np.where(direction == -1, -high_x, np.nan))
        dist = np.linalg.norm(p - ball[:, None], axis=2)
        nearest = np.where(np.isfinite(dist), dist, np.inf).argmin(axis=1)
        rows = np.arange(n)
        nearest_dist = dist[rows, nearest]
        result[prefix + "nearest_distance"] = nearest_dist
        relative_velocity = pv[rows, nearest] - velocity
        result[prefix + "comotion_speed_difference"] = np.linalg.norm(relative_velocity, axis=1)
    return result
