#!/usr/bin/env python3
"""FCB-5: audit DFL reference flags against the challenge's H/A/D semantics.

Dev matches only (configs/split.yaml). Holdout matches are never opened here.
Writes aggregate statistics only (no frame-level rows) to a JSON report.

Usage: PYTHONPATH=src python scripts/audit_dfl_labels.py [--data ../external/idsse] [--out findings/dev-labels-audit.json]
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession import split  # noqa: E402
from possession.adapters.dfl import SECTIONS, load_dfl  # noqa: E402
from possession.evaluation.references import load_dfl_references  # noqa: E402

FPS = 25
RESTART_EVENTS = ("KickOff", "ThrowIn", "FreeKick", "GoalKick", "CornerKick", "Penalty", "RefereeBall")


def files(data, match):
    def one(prefix):
        hits = list(Path(data).glob(f"{prefix}_*_{match}.xml"))
        if len(hits) != 1:
            raise FileNotFoundError(f"{prefix} for {match}")
        return hits[0]
    return one("DFL_02_01_matchinformation"), one("DFL_04_03_positions_raw_observed"), one("DFL_03_02_events_raw")


def parse_time(text):
    return datetime.fromisoformat(text).timestamp()


def ball_times(raw_path, tracking):
    """Wall-clock time per tracking row, from the ball FrameSet (evaluator-side helper)."""
    keys = {(int(p), int(f)): i for i, (p, f) in enumerate(zip(tracking.period, tracking.source_frame))}
    t = np.full(len(tracking.frame), np.nan)
    period, is_ball, root = None, False, None
    for event, elem in ET.iterparse(raw_path, events=("start", "end")):
        if root is None:
            root = elem
        if event == "start" and elem.tag == "FrameSet":
            period, is_ball = SECTIONS.get(elem.get("GameSection")), elem.get("TeamId") == "BALL"
        elif event == "end" and elem.tag == "Frame":
            if is_ball:
                i = keys.get((period, int(elem.attrib["N"])))
                if i is not None:
                    t[i] = parse_time(elem.attrib["T"])
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            root.clear()
    return t


def runs(values):
    """(start, stop, value) maximal runs of a 1-D array."""
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(starts, stops)]


def summarise(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "median": round(float(np.median(x)), 2),
            "p10": round(float(np.percentile(x, 10)), 2), "p90": round(float(np.percentile(x, 90)), 2)}


def audit_match(data, match):
    info, raw, events_path = files(data, match)
    tracking = load_dfl(info, raw)
    refs = load_dfl_references(raw, tracking)
    t = ball_times(raw, tracking)
    out = {"match": match}

    # 1. Raw value sets and unknowns.
    out["raw_BallStatus"] = dict(Counter(refs.raw_status))
    out["raw_BallPossession"] = dict(Counter(refs.raw_owner))
    out["frames"] = int(len(tracking.frame))
    out["periods"] = {int(p): int((tracking.period == p).sum()) for p in np.unique(tracking.period) if p}
    out["unknown_share"] = round(float(refs.unknown.mean()), 4)
    dead = refs.alive == 0
    out["owner_during_dead"] = dict(Counter(refs.owner[dead].tolist()))

    # 2. Home/away mapping: match-info home team vs adapter order, and empirical nearest-player agreement.
    root = ET.parse(info).getroot()
    home_id = next(root.iter("General")).get("HomeTeamId")
    home_players = {p.get("PersonId") for t_ in root.iter("Team") if t_.get("TeamId") == home_id
                    for p in t_.iter("Player")}
    adapter_home = {s for s, team in zip(tracking.slot_ids, tracking.teams) if team == "H"}
    out["adapter_home_equals_matchinfo_home"] = adapter_home == home_players
    alive_known = (refs.alive == 1) & ~refs.unknown & ~tracking.ball_missing
    d = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    close = d.min(axis=1) < 1.5
    nearest_team = tracking.teams[nearest]
    sel = alive_known & close
    out["nearest_within_1p5m_agrees_with_flag"] = round(float((nearest_team[sel] == refs.owner[sel]).mean()), 4)
    out["frames_used_for_agreement"] = int(sel.sum())

    # 3. H/A/D sequence per period, possessions and dead intervals.
    label = np.where(refs.alive == 0, "D", np.where(refs.alive == 1, refs.owner, "X"))
    poss_dur, dead_dur, same_team_restart, restarts_total = [], [], 0, 0
    switch_new_team_dist, flipback_2s, switches = [], 0, 0
    total_time = 0
    for p in np.unique(tracking.period):
        if not p:
            continue
        rows = np.flatnonzero(tracking.period == p)
        seq = label[rows]
        total_time += len(rows) / FPS
        rr = runs(seq)
        for k, (a, b, v) in enumerate(rr):
            if v in ("H", "A"):
                poss_dur.append((b - a) / FPS)
            elif v == "D":
                dead_dur.append((b - a) / FPS)
                if 0 < k < len(rr) - 1 and rr[k - 1][2] in ("H", "A") and rr[k + 1][2] in ("H", "A"):
                    restarts_total += 1
                    same_team_restart += rr[k - 1][2] == rr[k + 1][2]
            # Direct live switches H<->A (no D between).
            if k and v in ("H", "A") and rr[k - 1][2] in ("H", "A") and rr[k - 1][2] != v:
                switches += 1
                i = rows[a]
                team_mask = tracking.teams == v
                dd = d[i][team_mask]
                switch_new_team_dist.append(float(dd.min()) if np.isfinite(dd).any() else np.nan)
                if b - a < 2 * FPS and k + 1 < len(rr) and rr[k + 1][2] == rr[k - 1][2]:
                    flipback_2s += 1
    poss = np.array(poss_dur)
    out["possessions"] = summarise(poss)
    out["possessions_per_5min"] = round(len(poss) / (total_time / 300), 2)
    out["possession_share_under_1s"] = round(float((poss < 1).mean()), 4)
    out["possession_share_under_2s"] = round(float((poss < 2).mean()), 4)
    out["dead_intervals"] = summarise(dead_dur)
    out["dead_frame_share"] = round(float((label == "D").sum() / (tracking.period > 0).sum()), 4)
    out["dead_between_possessions"] = restarts_total
    out["dead_same_team_both_sides_share"] = round(same_team_restart / max(restarts_total, 1), 4)
    out["live_switches"] = switches
    dist = np.array(switch_new_team_dist)
    dist = dist[np.isfinite(dist)]
    out["new_team_nearest_dist_at_switch_m"] = summarise(dist)
    out["switch_new_team_within_1p5m_share"] = round(float((dist < 1.5).mean()), 4) if len(dist) else None
    out["switch_flips_back_within_2s_share"] = round(flipback_2s / max(switches, 1), 4)

    # 4. Restart events (human, other games) vs dead->alive transitions in the flags.
    ev_root = ET.parse(events_path).getroot()
    ev_times = sorted(parse_time(ev.get("EventTime")) for ev in ev_root.iter("Event")
                      if any(ch.tag in RESTART_EVENTS for ch in ev))
    starts = []
    for p in np.unique(tracking.period):
        if not p:
            continue
        rows = np.flatnonzero(tracking.period == p)
        a = refs.alive[rows]
        edge = np.flatnonzero((a[1:] == 1) & (a[:-1] == 0)) + 1
        starts += [t[rows[e]] for e in edge if np.isfinite(t[rows[e]])]
    starts = np.array(sorted(starts))
    matched = [bool(len(starts)) and float(np.min(np.abs(starts - e))) <= 3.0 for e in ev_times]
    out["restart_events"] = len(ev_times)
    out["dead_to_alive_transitions"] = int(len(starts))
    out["restart_events_with_transition_within_3s_share"] = round(float(np.mean(matched)), 4) if matched else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "external" / "idsse"))
    ap.add_argument("--out", default=str(ROOT / "findings" / "dev-labels-audit.json"))
    args = ap.parse_args()
    body = split.load()
    report = {"split_sha256": body["split_sha256"], "dev_only": True, "matches": []}
    for match in body["dev"]:
        if match in body["holdout"]:
            raise SystemExit("refusing to audit a holdout match")
        report["matches"].append(audit_match(args.data, match))
        print("audited", match, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
