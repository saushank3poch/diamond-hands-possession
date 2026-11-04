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


def valid_blocks(gt, period):
    """Contiguous [start, stop) spans within one active period where the reference is known."""
    ok = (np.asarray(period) > 0) & (np.asarray(gt) != "X")
    cut = np.r_[True, (period[1:] != period[:-1]) | (ok[1:] != ok[:-1])]
    starts = np.flatnonzero(cut)
    stops = np.r_[starts[1:], len(gt)]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if ok[a]]


def _block_edges(alive, a, kind):
    if kind == "release":   # dead -> live
        e = ~alive[:-1] & alive[1:]
    else:                   # live -> dead
        e = alive[:-1] & ~alive[1:]
    return (np.flatnonzero(e) + 1 + a).tolist()


def _carry_x(seg):
    """Forward-fill X inside one block, back-filling its head (scorer-like)."""
    known = seg != "X"
    if not known.any():
        return seg
    carry = np.maximum.accumulate(np.where(known, np.arange(len(seg)), -1))
    carry[carry < 0] = np.flatnonzero(known)[0]
    return seg[carry]


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

    Everything happens inside contiguous reference-valid blocks (one active period, no GT-X):
    prediction X is carried within the block, edges are extracted within the block (so no edge
    touches an unknown reference frame), and pairs never cross a block. Predicted edges in
    unknown regions are ignored, not counted as spurious.
    """
    gt, pred, period = np.asarray(gt), np.asarray(pred), np.asarray(period)
    parts = {k: {"n_ref": 0, "n_pred": 0, "latencies": [], "unmatched_ref": 0, "unmatched_pred": 0}
             for k in ("release", "stop")}
    for a, b in valid_blocks(gt, period):
        g_alive = np.isin(gt[a:b], ["H", "A"])
        p_alive = np.isin(_carry_x(pred[a:b].copy()), ["H", "A"])
        blk = np.ones(b - a + 1, int)  # single pseudo-period: pairs stay inside the block
        for kind in parts:
            r = _block_edges(g_alive, 0, kind)
            q = _block_edges(p_alive, 0, kind)
            m = match_edges(r, q, blk, window)
            for f in ("n_ref", "n_pred", "unmatched_ref", "unmatched_pred"):
                parts[kind][f] += m[f]
            parts[kind]["latencies"] += m["latencies"]
    return parts


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
