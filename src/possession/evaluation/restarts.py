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
