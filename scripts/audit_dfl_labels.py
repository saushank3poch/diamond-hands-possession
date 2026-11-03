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
from possession.evaluation.labels import confirm_control, reference_labels  # noqa: E402
import subprocess  # noqa: E402

FPS = 25
RESTART_EVENTS = ("KickOff", "ThrowIn", "FreeKick", "GoalKick", "CornerKick", "Penalty", "RefereeBall")
CONTROL_EVENTS = ("Play", "ShotAtGoal", "BallClaiming")
MATCH_WINDOW_S = 5.0
TAUS = (1.0, 2.0, 3.0)


def read_events(events_path, home_id):
    """(time, type, side, attrs) for every DFL event child; side is H/A/None."""
    out = []
    for ev in ET.parse(events_path).getroot().iter("Event"):
        t = parse_time(ev.get("EventTime"))
        for ch in ev:
            team = ch.get("Team") or ch.get("WinnerTeam")
            side = None if team is None else ("H" if team == home_id else "A")
            out.append((t, ch.tag, side, dict(ch.attrib)))
    return sorted(out, key=lambda e: e[0])


def one_to_one(event_times, transition_times, window=MATCH_WINDOW_S):
    """Greedy one-to-one pairing by smallest |dt| within window; signed latency = transition - event."""
    pairs = sorted((abs(tt - et), i, j) for i, et in enumerate(event_times)
                   for j, tt in enumerate(transition_times) if abs(tt - et) <= window)
    used_e, used_t, lat = set(), set(), []
    for _, i, j in pairs:
        if i in used_e or j in used_t:
            continue
        used_e.add(i)
        used_t.add(j)
        lat.append(transition_times[j] - event_times[i])
    return lat, len(event_times) - len(used_e), len(transition_times) - len(used_t)


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
    label = reference_labels(refs)
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

    # 4. Restart events (human DFL events) vs dead->alive flag transitions: DFL internal consistency only.
    events = read_events(events_path, home_id)
    ev_times = [e[0] for e in events if e[1] in RESTART_EVENTS]
    starts = []
    for p in np.unique(tracking.period):
        if not p:
            continue
        rows = np.flatnonzero(tracking.period == p)
        a = refs.alive[rows]
        edge = np.flatnonzero((a[1:] == 1) & (a[:-1] == 0)) + 1
        starts += [t[rows[e]] for e in edge if np.isfinite(t[rows[e]])]
    lat, unmatched_events, unmatched_transitions = one_to_one(ev_times, sorted(starts))
    out["restart_events"] = len(ev_times)
    out["dead_to_alive_transitions"] = len(starts)
    out["restart_one_to_one"] = {"window_s": MATCH_WINDOW_S, "matched": len(lat),
                                 "unmatched_events": unmatched_events,
                                 "unmatched_transitions": unmatched_transitions,
                                 "signed_latency_s": summarise(lat)}

    # 5. Control-specific evidence at live switches: independent human event semantics.
    ev_t = np.array([e[0] for e in events])
    groups = {"short_flipback": Counter(), "held_2s": Counter()}
    for p in np.unique(tracking.period):
        if not p:
            continue
        rows = np.flatnonzero(tracking.period == p)
        rr = runs(label[rows])
        for k, (a, b, v) in enumerate(rr):
            if not (k and v in ("H", "A") and rr[k - 1][2] in ("H", "A") and rr[k - 1][2] != v):
                continue
            t0, t1 = t[rows[a]], t[rows[b - 1]]
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            short = (b - a) < 2 * FPS and k + 1 < len(rr) and rr[k + 1][2] == rr[k - 1][2]
            held = (b - a) >= 2 * FPS
            if not (short or held):
                continue
            # Fixed window around the switch start, identical for short and held spells (no length bias).
            lo, hi = np.searchsorted(ev_t, t0 - 2.0), np.searchsorted(ev_t, t0 + 1.0)
            window = events[lo:hi]
            control = any(e[2] == v and (e[1] in CONTROL_EVENTS or (e[1] == "TacklingGame"
                          and e[3].get("PossessionChange") == "true")) for e in window)
            nochange = any(e[1] == "TacklingGame" and e[3].get("PossessionChange") == "false" for e in window)
            cls = "control_event_by_new_team" if control else ("duel_no_possession_change" if nochange
                                                               else "no_relevant_event")
            groups["short_flipback" if short else "held_2s"][cls] += 1
    out["switch_event_evidence"] = {g: dict(c) for g, c in groups.items()}
    phases = {e[3].get("BallPossessionPhase") for e in events if e[3].get("BallPossessionPhase")}
    out["event_possession_phases"] = len(phases)
    out["flag_possessions"] = int(len(poss))

    # 6. Experimental proxy sensitivity (not a reference; reported for transparency only).
    sens = {}
    primary = reference_labels(refs)
    for tau in TAUS:
        lab = confirm_control(primary, tracking.period, tau)
        n, T = 0, 0
        for p in np.unique(tracking.period):
            if not p:
                continue
            seq = lab[tracking.period == p]
            T += len(seq) / FPS
            n += sum(1 for a, b, v in runs(seq) if v in ("H", "A"))
        sens[f"tau_{tau:g}s"] = {"possessions_per_5min": round(n / (T / 300), 2),
                                 "frames_changed": int((lab != primary).sum())}
    out["proxy_sensitivity"] = sens
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "external" / "idsse"))
    ap.add_argument("--out", default=str(ROOT / "findings" / "dev-labels-audit.json"))
    args = ap.parse_args()
    body = split.load()
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
    report = {"split_sha256": body["split_sha256"], "dev_only": True,
              "git_commit": git("rev-parse", "HEAD"),
              "worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
              "matches": []}
    for match in body["dev"]:
        if match in body["holdout"]:
            raise SystemExit("refusing to audit a holdout match")
        report["matches"].append(audit_match(args.data, match))
        print("audited", match, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
