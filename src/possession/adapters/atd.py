"""Strict streaming tactical EPTS adapter, following starter/baseline.py's format."""
import os
import xml.etree.ElementTree as ET

import numpy as np

from possession.tracking import Tracking, coordinates, period_vectors


def load_atd(meta_path, raw_path, *, n_frames, pitch, blocks, directions=None):
    """Read only under the trusted local containment runner, including synthetic tests.

    Coordinates remain in a fixed pitch orientation; home_direction records the
    direction in that system (+1 right, -1 left, 0 unknown), never inferred from GT.
    """
    log_dir = os.environ.get("POSSESSION_TARGET_LOG_DIR")
    if not log_dir or os.environ.get("POSSESSION_RUN_DIR") != log_dir:
        raise RuntimeError("ATD requires the target containment wrapper")
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    slots = list(root.iter("Player"))
    if len(teams) != 2 or len(set(teams)) != 2 or not slots:
        raise ValueError("expected two teams and at least one slot")
    if any(slot.get("teamId") not in teams for slot in slots):
        raise ValueError("unknown slot team")
    slot_ids = [slot.get("id") for slot in slots]
    if any(slot is None for slot in slot_ids):
        raise ValueError("missing slot ID")
    period, direction = period_vectors(n_frames, blocks, directions)
    players = np.full((n_frames, len(slots), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with open(raw_path) as handle:
        for row, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or row >= n_frames or int(parts[0]) != row + 1:
                raise ValueError("ATD must contain ordered frame IDs 1..n")
            groups = parts[1].rstrip(";").split(";")
            if len(groups) != len(slots):
                raise ValueError("ATD slot count does not match metadata")
            for i, group in enumerate(groups + [parts[2]]):
                values = group.rstrip(";").split(",")
                if len(values) != 3:
                    raise ValueError("expected x,y,speed")
                xy = coordinates(values[0], values[1])
                xy = (xy - .5) * np.array([pitch[0], -pitch[1]])
                if i == len(slots):
                    ball[row] = xy
                else:
                    players[row, i] = xy
            count += 1
    if count != n_frames:
        raise ValueError("ATD row count mismatch")
    return Tracking(np.arange(n_frames), np.arange(1, n_frames + 1), period,
                    np.array(slot_ids), np.array(["H" if s.get("teamId") == teams[0]
                                                 else "A" for s in slots]),
                    players, ball, direction, np.array(pitch, dtype=float))


def load_target():
    # Import only published metadata, never GT offsets for ATD frame mapping.
    from games import GAME, GAMES, asset_paths, half_video_blocks
    paths = asset_paths()
    return load_atd(paths["atd_meta"], paths["atd_raw"],
                    n_frames=GAMES[GAME]["n_video_frames"], pitch=GAMES[GAME]["pitch"],
                    blocks=[(p, start, last + 1) for p, start, last in half_video_blocks(GAME)])
