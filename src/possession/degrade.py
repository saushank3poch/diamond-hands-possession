"""FCB-8: make clean dev tracking look like the target's automatic tracking.

Operates on a `Tracking` copy only. Reference labels are never touched: this module
does not import the evaluation package and takes no label argument.

Calibration targets are the published DATA.md input statistics, and nothing else:
  - ball visible on 59.3% of graded frames (BALL_VISIBLE)
  - mean 20.5 tracked objects over graded frames (MEAN_OBJECTS), out of ~22 on the pitch
  - 46 track slots, i.e. 23 per team (SLOTS_PER_TEAM); slots are tracklets, not people
  - occasional duplicates (more than 22 objects)
Parameters DATA.md does not publish (gap lengths, how strongly crowding and ball
speed drive ball loss, jitter, duplicate rate) are fixed ASSUMPTIONS in `Config`. They are
never fitted to target-match data or scores.
"""
from dataclasses import dataclass, asdict

import numpy as np

from possession.tracking import Tracking

FPS = 25
BALL_VISIBLE = 0.593   # DATA.md: 94,214 / 158,857 graded frames
MEAN_OBJECTS = 20.5    # DATA.md: mean tracked objects over graded frames
SLOTS_PER_TEAM = 23    # DATA.md: 46 track slots in the tactical ATD


@dataclass(frozen=True)
class Config:
    seed: int = 0
    ball_visible: float = BALL_VISIBLE
    mean_objects: float = MEAN_OBJECTS
    slots_per_team: int = SLOTS_PER_TEAM
    # --- assumptions (unpublished) ---
    ball_gap_mean_s: float = 1.5      # mean length of a ball-missing burst
    crowd_radius_m: float = 3.0       # players within this radius count as crowding the ball
    crowd_weight: float = 0.6         # logit increase per crowding player
    speed_weight: float = 0.08        # logit increase per m/s of ball speed (aerial/fast-ball proxy)
    out_of_bounds_missing: float = 0.9  # chance a frame with the ball off the pitch is missing
    player_gap_mean_s: float = 2.0    # mean length of a player-missing burst
    tracklet_break_per_min: float = 0.5  # extra fragmentation of continuous player tracks
    duplicate_rate: float = 0.0001    # per present player-frame chance of starting a ghost burst ("occasionally")
    duplicate_mean_s: float = 1.0
    duplicate_offset_m: float = 1.0
    jitter_player_m: float = 0.15
    jitter_ball_m: float = 0.30


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _markov_mask(rng, p_on, q_off):
    """Two-state chain: from visible, go missing with prob p_on[t]; from missing, recover with q_off."""
    n = len(p_on)
    u = rng.random(n)
    out = np.zeros(n, dtype=bool)
    missing = False
    for t in range(n):
        missing = (u[t] >= q_off) if missing else (u[t] < p_on[t])
        out[t] = missing
    return out


def _ball_entry_logits(tracking, cfg):
    ball = tracking.ball
    d = np.linalg.norm(tracking.players - ball[:, None, :], axis=-1)
    crowd = np.sum(np.nan_to_num(d, nan=np.inf) < cfg.crowd_radius_m, axis=1)
    v = np.zeros(len(ball))
    v[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    v = np.nan_to_num(v, nan=0.0)
    return cfg.crowd_weight * crowd + cfg.speed_weight * np.minimum(v, 40.0)


def _calibrate_offset(target_missing, logits, q_off):
    """Pick the base logit so the chain's expected stationary missing share hits the target."""
    lo, hi = -20.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = _sigmoid(mid + logits)
        share = np.mean(p / (p + q_off))
        lo, hi = (mid, hi) if share < target_missing else (lo, mid)
    return (lo + hi) / 2


def degrade(tracking: Tracking, cfg: Config = Config()):
    """Return (degraded Tracking, stats). Input is not modified; labels are out of scope."""
    rng = np.random.default_rng(cfg.seed)
    n = len(tracking.frame)
    in_period = tracking.period > 0
    half_l, half_w = tracking.pitch / 2

    # 1. Ball dropout: bursty, more likely when crowded or fast, near-certain off the pitch.
    q_ball = 1.0 / (cfg.ball_gap_mean_s * FPS)
    logits = _ball_entry_logits(tracking, cfg)
    d_ball = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=-1)
    crowd = np.sum(np.nan_to_num(d_ball, nan=np.inf) < cfg.crowd_radius_m, axis=1)
    already = tracking.ball_missing
    off_pitch = (np.abs(tracking.ball[:, 0]) > half_l) | (np.abs(tracking.ball[:, 1]) > half_w)
    off_pitch &= ~already
    forced = off_pitch & (rng.random(n) < cfg.out_of_bounds_missing)
    target = 1.0 - cfg.ball_visible
    # Account for frames already missing or forced missing, then calibrate the chain on the rest.
    free = in_period & ~already & ~forced
    need = (target * in_period.sum() - (already | forced)[in_period].sum()) / max(free.sum(), 1)
    need = float(np.clip(need, 0.0, 0.999))
    base = _calibrate_offset(need, logits[free], q_ball) if need > 0 else -50.0
    # One deterministic correction pass: the stationary formula ignores burst/edge effects.
    state = rng.bit_generator.state
    trial = _markov_mask(rng, _sigmoid(base + logits), q_ball)
    got = trial[free].mean() if free.any() else 0.0
    if need > 0 and got > 0:
        base = _calibrate_offset(float(np.clip(need * need / got, 0, 0.999)), logits[free], q_ball)
    rng.bit_generator.state = state
    chain = _markov_mask(rng, _sigmoid(base + logits), q_ball)
    ball_missing = already | forced | (chain & in_period)
    ball = tracking.ball + rng.normal(0, cfg.jitter_ball_m, tracking.ball.shape)
    ball[ball_missing] = np.nan

    # 2. Player dropout so the mean visible count hits MEAN_OBJECTS (before duplicates).
    present = ~tracking.player_missing
    on_pitch = present[in_period].sum(axis=1).mean()
    expected_ghosts = cfg.duplicate_rate * cfg.duplicate_mean_s * FPS * on_pitch
    keep = min((cfg.mean_objects - expected_ghosts) / max(on_pitch, 1e-9), 1.0)
    q_pl = 1.0 / (cfg.player_gap_mean_s * FPS)
    p_on = (1 - keep) * q_pl / max(keep, 1e-9)  # stationary missing share = 1 - keep
    visible = present.copy()
    for j in range(present.shape[1]):
        if present[:, j].any():
            visible[:, j] &= ~_markov_mask(rng, np.full(n, p_on), q_pl)
    players = tracking.players + rng.normal(0, cfg.jitter_player_m, tracking.players.shape)

    # 3. Fragment into tracklets and re-slot into a fixed per-team pool (identity-free slots).
    breaks = rng.random(visible.shape) < cfg.tracklet_break_per_min / (60 * FPS)
    teams_out, slots_out = [], []
    pools = {}
    for team in ("H", "A"):
        pools[team] = list(range(len(teams_out), len(teams_out) + cfg.slots_per_team))
        teams_out += [team] * cfg.slots_per_team
    n_slots = len(teams_out)
    out_xy = np.full((n, n_slots, 2), np.nan)

    # Collect tracklets (and ghost bursts), then assign slots in start-time order so a slot
    # is reused only after its previous occupant has ended.
    segments = []  # (start, stop, player, is_ghost)
    for j in range(visible.shape[1]):
        v = visible[:, j]
        cut = v & (~np.r_[False, v[:-1]] | breaks[:, j])
        starts = np.flatnonzero(cut)
        stops = np.r_[starts[1:], n]
        for a, b in zip(starts, stops):
            end_vis = np.flatnonzero(~v[a:b])
            segments.append((a, a + (end_vis[0] if len(end_vis) else b - a), j, False))
    dup_len = max(int(cfg.duplicate_mean_s * FPS), 1)
    for a, j in zip(*np.nonzero(visible & (rng.random(visible.shape) < cfg.duplicate_rate))):
        segments.append((a, min(a + dup_len, n), j, True))
    segments.sort(key=lambda seg: (seg[0], seg[3], seg[2]))

    occupied_until = np.full(n_slots, -1)
    tracklets = ghosts = lost = 0
    for a, b, j, ghost in segments:
        team = tracking.teams[j]
        free_slots = [s_ for s_ in pools[team] if occupied_until[s_] < a]
        if not free_slots:
            lost += 1  # pool exhausted: the tracker drops this tracklet entirely
            continue
        slot = free_slots[rng.integers(len(free_slots))]
        seg = visible[a:b, j]
        xy = players[a:b, j][seg]
        if ghost:
            xy = xy + rng.normal(0, cfg.duplicate_offset_m, 2)
            ghosts += 1
        else:
            tracklets += 1
        out_xy[a:b][seg, slot] = xy
        occupied_until[slot] = b - 1

    degraded = Tracking(tracking.frame, tracking.source_frame, tracking.period,
                        np.array([f"slot{i + 1}" for i in range(n_slots)]), np.array(teams_out),
                        out_xy, ball, tracking.home_direction, tracking.pitch)
    vis_count = (~np.isnan(out_xy).any(axis=-1))[in_period].sum(axis=1)
    stats = {
        "config": asdict(cfg),
        "ball_visible_share": round(float((~ball_missing)[in_period].mean()), 4),
        "ball_visible_target": cfg.ball_visible,
        "mean_objects": round(float(vis_count.mean()), 2),
        "mean_objects_target": cfg.mean_objects,
        "share_frames_over_22": round(float((vis_count > 22).mean()), 4),
        "tracklets": tracklets,
        "ghost_bursts": ghosts,
        "tracklets_lost_pool_full": lost,
        "clean_mean_on_pitch": round(float(on_pitch), 2),
        # Bias check: ball loss should be higher when the ball is crowded (>= 3 players within radius).
        "ball_missing_share_crowded": round(float(ball_missing[in_period & (crowd >= 3)].mean()), 4)
        if (in_period & (crowd >= 3)).any() else None,
        "ball_missing_share_uncrowded": round(float(ball_missing[in_period & (crowd == 0)].mean()), 4)
        if (in_period & (crowd == 0)).any() else None,
    }
    return degraded, stats
