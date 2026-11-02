# Evaluation and submission contract review

Reviewed: 2026-11-02. Scope: documentation and synthetic correctness checks only.

## Evidence and revision

Reviewed against local commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
`git ls-remote upstream HEAD refs/heads/main` returned that same commit on
2026-11-02; no newer upstream main revision was found. Official source:
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027).
The existing [baseline analysis](baseline-analysis.md) accurately describes the
nearest-player classifier and centred H/A voting in [baseline.py](../starter/baseline.py).
This note adds the scoring details needed before evaluating improvements.

Sources: [challenge](../CHALLENGE.md), [rules](../RULES.md),
[data](../DATA.md), [evaluation](../EVALUATION.md),
[submission](../SUBMISSION.md), [constants](../games.py),
[metric implementation](../evaluation/possession_metrics.py),
[ranking runner](../evaluation/run_evaluation.py),
[format validator](../evaluation/validate_submission.py), and
[ATD loader](../starter/load_atd.py). Function names below identify the precise
implementation authority. No public-label candidate comparison was run.

## Timeline and output

- Emit exactly 199,319 data rows, with `frame,label` header and ordered integer
  frames 0 through 199318. Use H (Barcelona alive), A (Athletic alive), D (dead),
  or X (abstain); never leave blank labels.
- Participant ATD uses `video_frame = frame_id - 1`. Do not apply the per-half
  `IdFrame` offsets from `games.py` to the supplied ATD: the loader explains that
  those offsets describe the separate source counter used to derive ground truth.
- Homography frames already use video indexing. Smart Tagging uses tactical-video
  seconds, converted with `round(seconds * 25)`. Broadcast has a separate timeline.
- Published graded ranges are inclusive 11185–90651 and 110808–190197. Internally,
  `valid_runs()` derives half-open intervals from GT non-X frames. On the public
  slice it therefore uses only the published five minutes, not the full half.
- Loose or contested live balls retain the previous team until opponent control;
  a deflection alone is not a turnover. D separates possessions even when the
  same team restarts. These are semantic rules, not tuned duration thresholds.

## Exact ranking semantics

`run_evaluation.rank_key()` orders by descending spellF1, descending macroF1,
descending transition F1 at the loosest requested tolerance, ascending mean
absolute transition latency, then filename. The default tolerances are 12 and
25 frames. The signed mean latency printed in the board is not that tie-break.

`possession_spells()` extracts maximal H/A intervals, excluding D, separately
inside each GT-valid run. Each predicted/true pair must have the same team and
overlap in time. For half-open intervals, temporal IoU is intersection length
divided by union length. `spell_overlaps()` sorts candidates by descending IoU,
then GT index and prediction index. `matched_gt_flags()` greedily consumes each
side at most once, accepting IoU equal to the threshold. This is the implemented
matching procedure; do not substitute a global assignment optimizer.

At each threshold 0.3, 0.4, 0.5, 0.6, 0.7, unmatched predictions are false
positives and unmatched true spells are false negatives. The ranking score is
the unweighted mean of those five F1 values, not only F1 at 0.5 and not mAP.
Short and long spells each count once. False D can split a possession; missed
D can merge two possessions. Dead-ball detection therefore affects spellF1 even
though D intervals themselves are not matched possessions.

## Abstention edge cases

`evaluate()` keeps raw predictions for frame metrics, and calls
`fill_abstentions()` before temporal metrics:

1. Forward-fill X using the latest non-X label within the same GT-valid run.
2. Back-fill leading X using that run's first non-X prediction.
3. Leave an entirely X run unchanged. It produces no predicted H/A spells.
4. Never carry labels across GT-X gaps.

The carried label can be D as well as H or A. Thus X is not a neutral gap at a
restart or turnover. A leading X can also import a future prediction backward.
Frame macroF1 still charges X as a false negative of the true class. GT-X frames
are excluded from scored frame metrics regardless of the prediction.

The standalone validator has no GT, so its descriptive possession count treats
predicted X as a gap. That count can differ from the scorer's filled count;
it is not evidence of a broken scorer or a failed format check.

## Secondary metrics and caveats

- `macro_f1()` averages H/A/D F1 equally; an absent class scores zero. Its
  docstring still calls it the ranking metric, but the runner actually ranks
  on spellF1. Treat that sentence as stale commentary.
- `alive_dead_f1()` uses alive as the positive class. `team_accuracy()` reports
  both a strict GT-alive denominator and a conditional both-alive denominator.
- `match_transitions()` defaults to destination-label matching, not matching
  both origin and destination. It pairs earliest feasible events one-to-one
  within the frame tolerance; the latency is prediction minus GT, positive for
  delay. Maximum-cardinality matching does not guarantee minimum latency.
- Fragmentation counts all constant-label segments, including D, after filling;
  it is not just a ratio of H/A possession counts.
- Possession-share error uses raw predictions and each side's own alive total
  on graded frames. With no predicted alive frames, `share_defined` is false;
  the numeric sentinel must not be presented as a measured share error.
- Bootstrap macroF1 resamples GT segments. Bootstrap spellF1 resamples matched
  GT-possession flags with prediction count fixed; it does not rematch each
  sample. Neither is a frame-independent interval or permission to tune on
  the public slice. Published noise bands are full-match reference estimates,
  not universal significance cutoffs.

## Synthetic checks executed

Executed against the revision above with the existing repository `.venv/bin/python`
(numpy/pandas available), without opening the public GT CSV or match assets.
Letters below are invented sequences, not annotations of this match.

| Synthetic GT | Prediction | Filled prediction | spellF1 | macroF1 |
|---|---|---|---:|---:|
| HHH | XHX | HHH | 1.000000 | 0.166667 |
| HHH | XXX | XXX | 0.000000 | 0.000000 |
| HHDAA | HXXXA | HHHHA | 0.600000 | 0.444444 |
| HHXAA | HXXXA | HHXAA | 1.000000 | 0.444444 |
| HHAADD | HHHADD | HHHADD | 0.700000 | 0.822222 |
| HHHH | HHHH | HHHH | 1.000000 | 0.333333 |

These cases also bound two broad statements in EVALUATION.md: a constant guess
is not mathematically guaranteed to score zero on every possible sequence, and
a one-frame timing error is not universally free. Those published examples
refer to their reference evaluation; small synthetic spells show otherwise.
An all-H perfect toy sequence gets macroF1 1/3 because A and D are absent.

To reproduce from the repository root:

```python
from evaluation import possession_metrics as pm

cases = [("HHH", "XHX"), ("HHH", "XXX"), ("HHDAA", "HXXXA"),
         ("HHXAA", "HXXXA"), ("HHAADD", "HHHADD"), ("HHHH", "HHHH")]
for truth, prediction in cases:
    gt = pm.label_codes(list(truth))
    pred = pm.label_codes(list(prediction))
    filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
    result = pm.evaluate(gt, pred)
    print(truth, prediction, "".join(pm.LABELS[i] for i in filled),
          round(result["spell_f1_mean"], 6), round(result["macro_f1"], 6))
```

## Boundaries for subsequent experiments

Keep model/threshold selection on accessible other-match data with documented
provenance, licensing and label semantics, or use permitted automatic learning.
Use synthetic cases for indexing, missingness, state continuity and metric
correctness. Use the public window only for the correctness checks allowed by
RULES.md; do not compare candidate scores there to choose a method. Published
baseline full-match scores remain organiser-reported reference values, not
locally reproduced performance or evidence for a new model.

A submission must be regenerated as the same file from code, pinned dependencies
and documented commands. Preserve the existing data links; do not publish assets
or reconstructive derived data. No model, prediction, tracking correction,
submission, or upstream source change was made by this review.
