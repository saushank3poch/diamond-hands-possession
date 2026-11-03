"""Raw DFL reference flags, kept outside every predictor input.

The numeric convention is provisional until FCB-5 qualifies label semantics.
Unknown raw values survive for that audit; no H/A/D scoring labels are emitted.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from possession.adapters.dfl import SECTIONS


@dataclass(frozen=True, slots=True)
class References:
    frame: np.ndarray
    alive: np.ndarray  # -1 unknown, 0 dead, 1 alive
    owner: np.ndarray  # H/A/X; raw flags retained below
    unknown: np.ndarray
    raw_status: tuple[str | None, ...]
    raw_owner: tuple[str | None, ...]


def load_dfl_references(raw_path, tracking):
    keys = {(int(p), int(f)): i for i, (p, f) in
            enumerate(zip(tracking.period, tracking.source_frame, strict=True))}
    n = len(tracking.frame)
    status, owner = [None] * n, [None] * n
    seen = set()
    period, is_ball = None, False
    root = None
    for event, elem in ET.iterparse(raw_path, events=("start", "end")):
        if root is None:
            root = elem
        if event == "start" and elem.tag == "FrameSet":
            period = SECTIONS.get(elem.get("GameSection"))
            is_ball = elem.get("TeamId") == "BALL"
        elif event == "end" and elem.tag == "Frame":
            if is_ball:
                key = period, int(elem.attrib["N"])
                if key not in keys or key in seen:
                    raise ValueError("reference frame missing from tracking or duplicated")
                seen.add(key)
                i = keys[key]
                status[i], owner[i] = elem.get("BallStatus"), elem.get("BallPossession")
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            root.clear()
            is_ball = False
    alive = np.array([{"0": 0, "1": 1}.get(s, -1) for s in status], dtype=np.int8)
    side = np.array([{"1": "H", "2": "A"}.get(o, "X") for o in owner])
    unknown = (alive == -1) | ((alive == 1) & (side == "X"))
    return References(tracking.frame.copy(), alive, side, unknown, tuple(status), tuple(owner))
