# FCB-5: dev-label eligibility audit (DFL / IDSSE)

Run: 2026-11-03. Split: `configs/split.yaml`. **Dev matches only**: J03WN1, J03WOH, J03WPY, J03WQQ, J03WR9. The holdout matches (J03WMX, J03WOY) were not opened.
Reproduce with `PYTHONPATH=src .venv/bin/python scripts/audit_dfl_labels.py`.
Raw aggregates: `dev-labels-audit.json` and `dev-labels-transform-sensitivity.json`.

## Verdict

| Source | Verdict |
|---|---|
| DFL `BallStatus` → D | **Accept as-is** |
| DFL `BallPossession` → H/A | **Accept with transform**: confirm-control, τ = 2 s, prespecified (below) |
| Metrica, SkillCorner | Not audited here (FCB-6). Metrica Game 1 events were already inspected in the semantics survey, so that game is not untouched |

## Evidence (ranges over the 5 dev matches)

### 1. Integrity and value sets
- `BallStatus` ∈ {0, 1} and `BallPossession` ∈ {1, 2} on every ball frame.
- Unknown share is 0.0 in all 5 matches.
- Frames per match: 137k–147k, in 2 periods.

### 2. Home/away mapping
- Adapter home slots equal match-info `HomeTeamId` players in 5/5 matches.
- Where a player is within 1.5 m of the ball in live play, the flagged team matches that player's team on 87–92% of frames.
- So `BallPossession` 1 = home holds. The remainder is consistent with flags lagging or leading a touch.

### 3. Dead ball
- **Restart timing:** 88–95% of restart events (human DFL events on other games: kick-off, throw-in, free kick, goal kick, corner, penalty, referee ball) have a dead→alive flag transition within 3 s.
- **Counts:** restart events and dead→alive transitions agree within ±2 per match.
- **Same-team restarts:** 24–43% of dead intervals have the same team on both sides. D therefore splits same-team possessions, as the challenge requires.
- **D share:** 41–52% of period frames; median dead interval is 19–24 s.
- **Owner during dead frames:** DFL keeps an owner value here. We map every dead frame to D regardless.

### 4. Possession is last-touch, not controlled
- At live H↔A switches, the new team has a player within 1.5 m on 93–99% of switches (median 0.6 m).
- 16–26% of switches flip back within 2 s.
- 25–34% of possessions last under 2 s.
- Raw flags give 19–26 possessions per 5 min, with a median of 3–5 s.

CHALLENGE.md excludes touches without control, deflections and won aerials, so raw `BallPossession` over-segments relative to the target convention.

## Transform (evaluator-only; `src/possession/evaluation/labels.py`)

`confirm_control`, applied per period:

- A spell for the other team that directly follows a team spell (no D between) is kept only if it lasts at least **τ = 2 s**. Otherwise it is relabelled as the previous owner.
- The first spell after a D or a period start is a restart and is always kept.
- A confirmed switch keeps its original start frame, so control is back-dated to the first touch.
- Unknown (X) frames break the chain. Synthetic tests are in `tests/test_reference_labels.py`.

**τ was fixed from CHALLENGE.md's "keeps the ball for a few seconds" proxy** and written into the code before the sensitivity run. It is not selected by comparing against the public 5-minute GT or any target statistic. Disclosure: the published DATA.md summary of the public window (17 possessions, mean 14.2 s) was known to the authors. It was not used as a target, and no τ was chosen to match it.

Sensitivity is for robustness reporting only, never for selection:

| τ | possessions / 5 min | median spell (s) | share < 2 s |
|---|---|---|---|
| raw | 18.9–26.5 | 3.1–4.7 | 25–34% |
| 1 s | 16.3–21.0 | 4.4–5.6 | 15–19% |
| **2 s** | **13.6–16.7** | **5.8–7.8** | **2–6%** |
| 3 s | 11.3–14.1 | 6.9–10.6 | 0.5–6% |

FCB-9 reports dev results under τ = 2 s as primary, and τ = 1 s and 3 s as a robustness band. A component that wins only under one τ is not a win.

## Not material / deferred

- **DFL y-axis sign vs ATD's flipped y:** possession features use distances, the pitch bounds and the per-period attack direction, all of which are invariant to a y mirror. Not material.
- **Pass-based control proxy** ("more than two consecutive passes"): not used, because the transform is duration-only. Pass events could refine it later, but only with owner approval of a protocol change.
- **Remaining risk:** DFL dead-ball timing is DFL's operator/automatic convention, not Metrica's corrected-tracking convention. Boundary latencies measured on DFL carry that offset uncertainty, of the order of the 3 s restart window above.
