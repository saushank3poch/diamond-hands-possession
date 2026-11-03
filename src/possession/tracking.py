"""Label-free 25 Hz tracking in centred metres, +y towards the top touchline."""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Tracking:
    frame: np.ndarray
    source_frame: np.ndarray
    period: np.ndarray
    slot_ids: np.ndarray
    teams: np.ndarray
    players: np.ndarray
    ball: np.ndarray
    home_direction: np.ndarray
    pitch: np.ndarray

    def __post_init__(self):
        # Own the buffers: callers cannot mutate predictors via a shared reference table.
        for field in fields(self):
            value = np.array(getattr(self, field.name), copy=True)
            if value.dtype.hasobject:
                raise ValueError("object arrays are not tracking fields")
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)
        n, slots = len(self.frame), len(self.slot_ids)
        for name in ("frame", "source_frame", "period", "home_direction"):
            a = getattr(self, name)
            if a.shape != (n,) or a.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer frame vector")
        if not np.array_equal(self.frame, np.arange(n)) or n == 0:
            raise ValueError("frames must be nonempty and contiguous from zero")
        if np.any(self.period < 0) or not np.isin(self.home_direction, [-1, 0, 1]).all():
            raise ValueError("invalid period or home direction")
        if self.slot_ids.shape != (slots,) or len(set(self.slot_ids)) != slots:
            raise ValueError("slot IDs must be unique")
        if self.teams.shape != (slots,) or not np.isin(self.teams, ["H", "A"]).all():
            raise ValueError("each slot must belong to H or A")
        if self.pitch.shape != (2,) or not np.isfinite(self.pitch).all() or np.any(self.pitch <= 0):
            raise ValueError("pitch must contain positive length and width in metres")
        for name, shape in (("players", (n, slots, 2)), ("ball", (n, 2))):
            a = getattr(self, name)
            if a.shape != shape or a.dtype.kind != "f" or np.isinf(a).any():
                raise ValueError(f"invalid {name} coordinates")
            if np.any(np.isnan(a).any(axis=-1) != np.isnan(a).all(axis=-1)):
                raise ValueError("a missing observation needs both coordinates NaN")
        for p in np.unique(self.period):
            rows = np.flatnonzero(self.period == p)
            if p and (np.any(np.diff(rows) != 1)
                      or np.any(np.diff(self.source_frame[rows]) != 1)
                      or len(np.unique(self.home_direction[rows])) != 1):
                raise ValueError("each period must be a contiguous source block with one direction")

    @property
    def player_missing(self):
        return np.isnan(self.players).any(axis=-1)

    @property
    def ball_missing(self):
        return np.isnan(self.ball).any(axis=-1)

    def home_attacking_coordinates(self):
        """Rotate each known period so home attacks +x; unknown direction fails closed.

        Non-period rows have no attacking orientation and return NaN.
        Directions must come from observation metadata or an automatic geometry
        method, never reference labels or hand annotations of the target match.
        """
        active = self.period > 0
        if np.any(self.home_direction[active] == 0):
            raise ValueError("home attacking direction is unknown")
        signs = np.where(active, self.home_direction, np.nan)
        return self.players * signs[:, None, None], self.ball * signs[:, None]

    def save(self, path: Path):
        """Persist only the fixed tracking allowlist, never labels or arbitrary metadata."""
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            expected = {f.name for f in fields(cls)}
            if set(archive.files) != expected:
                raise ValueError("unexpected tracking archive fields")
            return cls(**{key: archive[key] for key in expected})


def coordinates(x, y):
    """Partial/nonfinite observations are missing, never zero or interpolated."""
    xy = np.array([float(x) if x not in (None, "") else np.nan,
                   float(y) if y not in (None, "") else np.nan])
    return xy if np.isfinite(xy).all() else np.full(2, np.nan)


def period_vectors(n, blocks, directions=None):
    """Blocks are (period, start, stop), half-open, from observation metadata only."""
    period = np.zeros(n, dtype=np.int64)
    direction = np.zeros(n, dtype=np.int64)
    previous, seen = 0, set()
    directions = directions or {}
    for p, start, stop in blocks:
        if (any(type(v) is not int for v in (p, start, stop)) or p <= 0
                or p in seen or not previous <= start < stop <= n):
            raise ValueError("invalid period blocks")
        d = directions.get(p, 0)
        if type(d) is not int or d not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 (unknown), or +1")
        period[start:stop], direction[start:stop] = p, d
        previous = stop
        seen.add(p)
    if set(directions) - seen:
        raise ValueError("direction supplied for absent period")
    return period, direction
