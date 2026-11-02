# Baseline possession estimation: behaviour and limitations

Reviewed: 2026-11-02. Re-verified line by line against `starter/baseline.py` at upstream commit `f8f2629` (latest on `origin/main` as of 2026-11-02).

Source: [`starter/baseline.py`](../starter/baseline.py), with pitch dimensions from [`games.py`](../games.py).

## Overview

The baseline assigns each frame to the team of the nearest tracked player to the ball, then optionally smooths those assignments. It has no trained model and no explicit representation of control, passes, tackles, or stoppages.

## Processing flow

### 1. Map tracking slots to teams

`team_of_each_track()` reads the tracking XML and maps each slot to home or away. For this match, home is Barcelona and away is Athletic Club.

Slots are tracklets, not permanent player identities. The baseline does not follow individuals across frames, so an identity switch within the same team does not inherently break its logic. Duplicate or misplaced tracks can still distort the nearest-player decision.

### 2. Assign a team independently in each frame

`nearest_player_labels()`:

1. Starts every frame as internally unanswered (`NOBODY = -1`).
2. Skips frames where the ball is missing. Per [`DATA.md`](../DATA.md), the ball is missing on 43% of all frames and 41% of graded frames, in bursts.
3. Converts normalised player and ball coordinates into metres using a 105.3 × 68.0 metre pitch.
4. Calculates squared Euclidean distance between the ball and each observed player.
5. Assigns the nearest player's team. An exact distance tie keeps the lower slot index, because the comparison is strict (`<`).
6. Maps the tracking frame number to the submission frame by subtracting one.

There is no maximum control-distance threshold: even a player 15 metres away can win. Player and ball speed fields are ignored. If no player is usable, the frame remains unanswered. Lines that do not split into exactly three `:`-separated fields are skipped silently, so those frames also stay unanswered.

### 3. Smooth the assignments

Without smoothing, unanswered frames become `D`. These are missing observations, not verified dead-ball states. The baseline never outputs `X`.

With smoothing, `smooth()` examines a centred window around each frame:

- Count home (`H`) and away (`A`) assignments separately.
- Select the team with the larger count.
- Resolve equal positive counts in favour of home.
- Output `D` only when neither team has any assignments in the window.

**Missing observations do not vote for `D`. The vote is between H and A only.** One team observation can therefore fill an otherwise unanswered window.

| Variant | Context on either side | Full window away from recording edges |
|---|---:|---:|
| `baseline_raw` | None | 1 frame |
| `baseline_smooth25` | ±1 second | 51 frames |
| `baseline_smooth50` | ±2 seconds | 101 frames |
| `baseline_smooth125` | ±5 seconds | 251 frames |

The suffix is the window's half-width in frames. The video runs at 25 FPS. Smoothing uses future as well as past observations and cumulative sums for efficient window counts.

The window runs over the whole recording. It knows nothing about the graded ranges, half-time, or the kick-offs. Votes from ungraded frames, before, between and after the graded halves, can therefore leak into the first and last 2 seconds of each graded half for `smooth50`.

Because only H and A vote, smoothed `D` survives only where a ball gap is longer than the full window (more than 101 frames, about 4 s, for `smooth50`). Short stoppages that the tracking does see the ball through are never `D` at all. In practice, the published floor has almost no dead-ball detection.

### 4. Write the submission

`write_submission()` writes a `frame,label` CSV with one row per tactical-video frame (199,319 rows), labelled H/A/D everywhere, including ungraded frames. It never writes `X`. How the scorer treats non-`X` labels on ungraded frames is covered in [`evaluation-contract-review.md`](evaluation-contract-review.md): non-`X` labels on GT-`X` frames are excluded from scoring. `build()` computes the nearest-player assignments once, and the command-line entry point applies each smoothing variant to those same assignments.

## Verified synthetic examples

These examples were run through the existing `smooth()` function with a half-width of two frames. They are synthetic illustrations, not manual labels for the provided match. Raw `D` below means the internal unanswered value rendered without smoothing.

| Raw output | Smoothed output | Effect |
|---|---|---|
| `DDHDD` | `HHHHH` | One home observation fills surrounding gaps |
| `HHAHH` | `HHHHH` | A brief away assignment disappears |
| `HDDDA` | `HHHAA` | An uncertain gap becomes an inferred transition |

Removing a brief away assignment helps if it was an error, but hurts if it represented a genuine short possession. The smoother cannot distinguish those cases.

## Limitations

| Baseline assumption | Potential failure |
|---|---|
| The nearest player controls the ball | Passes, deflections, and aerial balls produce false turnovers |
| Any nearest-player distance is acceptable | A team is assigned when nobody is plausibly controlling the ball |
| Missing ball means raw `D` | Tracking failures become false stoppages |
| Nearby team observations override missing evidence | Possession can extend through actual stoppages |
| One fixed smoothing window works everywhere | Short possessions disappear and boundaries shift |
| Every assignment has equal weight | Dubious distant observations count as much as clear control |

The baseline does not use Smart Tagging, homography, either video, movement features, or observation-confidence estimates. Its only temporal reasoning is the centred vote; it does not explicitly detect control changes or possession boundaries.

## Published full-match results

These values are reported by the organisers in the baseline source. They are **not full-match results computed locally**; the private full-match ground truth is unavailable.

| Variant | macroF1 | spellF1 |
|---|---:|---:|
| Raw | 0.6532 | 0.0486 |
| Smooth25 | 0.6752 | 0.1996 |
| Smooth50 | 0.6724 | **0.2433** |
| Smooth125 | 0.6356 | 0.2242 |

Across the sweep macroF1 moves by about 0.04, while spellF1 moves by a factor of five. That is why the board is ranked on spellF1, and why frame accuracy is a poor proxy for our objective. Smoothing substantially improves possession recovery, but more smoothing eventually hurts. Per the `baseline.py` docstring, `smooth125` is the best of the four on the public five minutes "by a wide margin", yet it loses to `smooth50` on the full match. The public window holds only 17 possessions ([`EVALUATION.md`](../EVALUATION.md)). It ranks these variants differently, so its scores must not be used to choose a smoothing setting. Since commit `f8f2629` (2026-10-23), this is a rule, not just advice: [`RULES.md`](../RULES.md) §2 says the public window is "not training data and not a tuning set".

## Implications for our solution

The clearest hypotheses to investigate are:

1. Distinguish missing observations from evidence that play has stopped.
2. Require evidence of controlled possession before changing teams.
3. Weight evidence by its quality and physical plausibility.
4. Preserve possession through passes and uncertain intervals without erasing genuine short possessions.
5. Use additional automatic signals where they help resolve stoppages and control changes.

How these map to our planned components, in priority order:

| Hypothesis | Planned component | Baseline failure it targets |
|---|---|---|
| 1 | Dead-ball detectors (out of bounds, synchronised deceleration, restart formations) | Missing ball becomes `D`; real stoppages are smoothed over |
| 2 | Turnover rule (sustained opposing control or a completed pass) | Deflections and aerials flip the team |
| 3 | Distance threshold, co-motion, and gap-aware evidence weights | Far-away nearest players vote with full weight |
| 4 | Ball-free owner inference plus an HSMM decoder with duration priors | One fixed window erases short possessions and shifts boundaries |
| 5 | Smart Tagging priors; own ball detector as a stretch goal | Unused assets |

These are design hypotheses, not demonstrated improvements. Each becomes an entry in [`experiments.md`](experiments.md), with a falsifiable dev-set threshold, before it is claimed. No possession solution was implemented during this review.
