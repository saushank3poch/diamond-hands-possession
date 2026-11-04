"""Oracle-time restart diagnostics: evaluation only; never used by predictors."""
import numpy as np


def restart_counts(gt, pred, period):
    """Count predictions exactly at reference D->H/A edges within active periods.

    Excludes period heads, GT-X gaps and transitions across period boundaries.
    D/X predictions count as wrong in accuracy; team_accuracy conditions on H/A
    predictions and must be accompanied by team_coverage. No timing alignment or
    oracle boundary is supplied to the predictor. This is end-to-end accuracy at
    reference restarts, not accuracy given a correctly detected restart.
    """
    gt, pred, period = np.asarray(gt), np.asarray(pred), np.asarray(period)
    if gt.ndim != 1 or pred.shape != gt.shape or period.shape != gt.shape:
        raise ValueError("restart metrics require aligned vectors")
    edge = (gt[:-1] == "D") & np.isin(gt[1:], ["H", "A"])
    edge &= (period[1:] > 0) & (period[1:] == period[:-1])
    idx = np.flatnonzero(edge) + 1
    return {"n": len(idx), "correct": int(np.sum(pred[idx] == gt[idx])),
            "team_predictions": int(np.sum(np.isin(pred[idx], ["H", "A"]))),
            "pred_d": int(np.sum(pred[idx] == "D")), "pred_x": int(np.sum(pred[idx] == "X"))}


def summarize(counts):
    return {**counts, "accuracy": counts["correct"] / counts["n"] if counts["n"] else None,
            "team_coverage": counts["team_predictions"] / counts["n"] if counts["n"] else None,
            "team_accuracy": counts["correct"] / counts["team_predictions"] if counts["team_predictions"] else None}


# -- edge timing audit (evaluation only) -------------------------------------------------------

EDGE_WINDOW = 75  # +/-3 s at 25 fps, fixed a priori; pairs farther apart count as unmatched


def _alive(codes, period):
    """Alive mask with X carried from the previous label inside each period (scorer-like)."""
    codes = np.asarray(codes).copy()
    for p in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == p)
        seg = codes[idx]
        known = seg != "X"
        if known.any():
            carry = np.maximum.accumulate(np.where(known, np.arange(len(seg)), -1))
            carry[carry < 0] = np.flatnonzero(known)[0]
            codes[idx] = seg[carry]
    return np.isin(codes, ["H", "A"])


def _edges(alive, period, kind):
    same = (period[1:] > 0) & (period[1:] == period[:-1])
    if kind == "release":   # dead -> live
        e = ~alive[:-1] & alive[1:]
    else:                   # live -> dead
        e = alive[:-1] & ~alive[1:]
    return np.flatnonzero(e & same) + 1


def match_edges(ref, pred, period, window=EDGE_WINDOW):
    """One-to-one greedy matching by smallest |offset|, same period, |offset| <= window.

    Returns signed latencies (pred - ref, frames) of matched pairs and unmatched counts.
    """
    pred_arr = np.asarray(pred, dtype=int)
    pairs = []
    for i, r in enumerate(ref):
        lo, hi = np.searchsorted(pred_arr, [r - window, r + window + 1])
        for j in range(lo, hi):
            q = int(pred_arr[j])
            if period[q] == period[r]:
                pairs.append((abs(q - r), i, j, q - r))
    pairs.sort()
    used_r, used_p, lat = set(), set(), []
    for _, i, j, d in pairs:
        if i not in used_r and j not in used_p:
            used_r.add(i); used_p.add(j); lat.append(d)
    return {"n_ref": len(ref), "n_pred": len(pred), "latencies": lat,
            "unmatched_ref": len(ref) - len(lat), "unmatched_pred": len(pred) - len(lat)}


def edge_timing(gt, pred, period, window=EDGE_WINDOW):
    """Release (D->live) and stop (live->D) edge audit against the reference.

    Reference edges exclude GT-X gaps and period boundaries; reference X frames are treated as
    not alive only for edge detection, and edges touching a GT-X frame are dropped.
    """
    gt, pred, period = np.asarray(gt), np.asarray(pred), np.asarray(period)
    out = {}
    ok = (gt[:-1] != "X") & (gt[1:] != "X")
    g_alive, p_alive = np.isin(gt, ["H", "A"]), _alive(pred, period)
    for kind in ("release", "stop"):
        ref = _edges(g_alive, period, kind)
        ref = ref[ok[ref - 1]]
        prd = _edges(p_alive, period, kind)
        out[kind] = match_edges(ref.tolist(), prd.tolist(), period, window)
    return out


def summarize_timing(parts):
    """Pool per-match edge_timing results into summary statistics (no raw frames)."""
    out = {}
    for kind in ("release", "stop"):
        lat = np.concatenate([np.asarray(p[kind]["latencies"], float) for p in parts]) if parts else np.array([])
        n_ref = sum(p[kind]["n_ref"] for p in parts)
        n_pred = sum(p[kind]["n_pred"] for p in parts)
        um_r = sum(p[kind]["unmatched_ref"] for p in parts)
        um_p = sum(p[kind]["unmatched_pred"] for p in parts)
        q = (lambda x: float(np.percentile(lat, x))) if len(lat) else (lambda x: None)
        out[kind] = {"n_ref": n_ref, "n_pred": n_pred, "matched": int(len(lat)),
                     "unmatched_ref": um_r, "unmatched_pred": um_p,
                     "median": q(50), "p10": q(10), "p25": q(25), "p75": q(75), "p90": q(90),
                     "mean": float(lat.mean()) if len(lat) else None,
                     "share_late": float((lat > 0).mean()) if len(lat) else None,
                     "share_within_8": float((np.abs(lat) <= 8).mean()) if len(lat) else None}
    return out
