"""DFL observations: kloppy 3.19.1 metadata plus lossless coordinate streaming.

The full kloppy tracking loader drops ball-less frames and defaults missing X/Y
and unknown reference fields. We deliberately use only its pinned metadata API.
"""
import xml.etree.ElementTree as ET

import numpy as np
from lxml import objectify
from kloppy.infra.serializers.event.sportec.deserializer import sportec_metadata_from_xml_elm

from possession.tracking import Tracking, coordinates, period_vectors

SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def observations(path):
    """Allowlist only identity/frame/coordinates; reference attributes never leave this reader."""
    section = obj = None
    root = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if root is None:
            root = elem
        if event == "start" and elem.tag == "FrameSet":
            section = SECTIONS.get(elem.get("GameSection"))
            if section is None:
                raise ValueError("unknown DFL period")
            obj = "ball" if elem.get("TeamId") == "BALL" else elem.get("PersonId")
        elif event == "end" and elem.tag == "Frame":
            if section is None or obj is None:
                raise ValueError("DFL frame outside identified frameset")
            yield section, obj, int(elem.attrib["N"]), coordinates(elem.get("X"), elem.get("Y"))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            section = obj = None
            root.clear()


def load_dfl(meta_path, raw_path, *, directions=None):
    with open(meta_path, "rb") as handle:
        meta = sportec_metadata_from_xml_elm(objectify.fromstring(handle.read()))
    if meta.fps != 25:
        raise ValueError("DFL adapter requires 25 Hz")
    ids, teams = [], []
    for label, team in zip(("H", "A"), meta.teams, strict=True):
        for player in team.players:
            ids.append(player.player_id)
            teams.append(label)
    slot_of = {slot: i for i, slot in enumerate(ids)}
    official_ids = {o.official_id for o in (meta.officials or [])}
    spans = {}
    # Two streaming passes avoid millions of dicts and preserve gaps without interpolation.
    for p, obj, frame, _ in observations(raw_path):
        if obj in official_ids:
            continue
        if obj != "ball" and obj not in slot_of:
            raise ValueError("DFL object missing from team metadata")
        low, high = spans.get(p, (frame, frame))
        spans[p] = min(low, frame), max(high, frame)
    if not spans:
        raise ValueError("empty DFL observations")
    blocks, sources, offset = [], [], {}
    n = 0
    for p, (low, high) in sorted(spans.items()):
        size = high - low + 1
        blocks.append((p, n, n + size))
        sources.extend(range(low, high + 1))
        offset[p] = n - low
        n += size
    period, direction = period_vectors(n, blocks, directions)
    players, ball = np.full((n, len(ids), 2), np.nan), np.full((n, 2), np.nan)
    seen = np.zeros((n, len(ids) + 1), dtype=bool)
    for p, obj, frame, xy in observations(raw_path):
        if obj in official_ids:
            continue
        row, slot = frame + offset[p], len(ids) if obj == "ball" else slot_of[obj]
        if seen[row, slot]:
            raise ValueError("duplicate DFL object/frame")
        seen[row, slot] = True
        if obj == "ball":
            ball[row] = xy
        else:
            players[row, slot] = xy
    return Tracking(np.arange(n), np.array(sources), period, np.array(ids), np.array(teams),
                    players, ball, direction, np.array([meta.x_max, meta.y_max]))
