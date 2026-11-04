# Ball possession from automatic tracking: technical document

*More than a Hack 2027, Metrica Sports × FC Barcelona. Team: **diamond hands**. Repository: https://github.com/saushank3poch/diamond-hands-possession (private until submission).*

## 1. Summary

We infer `H`/`A`/`D` for every frame of Barça–Athletic from Metrica's automatic tactical tracking alone. The pipeline has three stages:

1. **Dead-ball detector.** A logistic model estimates, per frame, the probability that the ball is out of play from windowed tracking signals. That probability is decoded into dead and live runs with minimum durations.
2. **Owner inside each live run.** A nearest-player majority vote, confined to the live run so it never crosses a stoppage.
3. **Short-spell suppression.** Inside each live run, an owner spell shorter than 1 s, other than the run's first spell, that disagrees with the owner carried into it is overwritten with that owner.

Everything is fitted and selected on other matches (DFL / IDSSE open data), with the inputs degraded to resemble Metrica's automatic tracking. On the Barça match the frozen model only predicts.

| model (degraded dev input, primary labels) | spellF1, dev CV (5 matches) | spellF1, one-time holdout (2 matches) |
|---|---|---|
| nearest-player baseline, ±2 s vote (organisers' rule) | 0.227 | not evaluated |
| + dead-ball detector (E002) | 0.332 | not evaluated |
| **+ short-spell suppression (E005, selected and frozen)** | **0.389** [0.373, 0.406] | **0.359** [0.334, 0.385] |

Intervals are the official scorer's 90% bootstrap over possessions. **None of these numbers is a Barça–Athletic score.** We never scored the Barça match, and we did not use its public five-minute ground truth to choose anything.

## 2. Approach, and why this rather than something else

We treated the task as **boundary detection**, not frame classification. spellF1 matches whole possessions by temporal IoU, so a possession split by a spurious flip and a missed stoppage both hurt it. Two observations motivated our hypotheses; neither established the largest error source by itself. First, the baseline predicted D on 1.6% of dev frames, against 45% in truth, which led to a dead-ball detector (E002). Second, E002 predicted 2,578 possessions where the truth had 2,094, which led to suppressing short owner flips (E005).

**Dead-ball detector.** This is an L2-regularised logistic regression in numpy, on four groups of signals. Each signal is averaged over centred 1 s, 4 s and 12 s windows within a period.
- **Ball:** missing, time since last seen, last-seen distance to the nearest line or off the pitch, windowed speed and its 3 s change, crowding around the last-seen ball, distance from the centre spot. A missing ball is one input, never the rule.
- **Motion:** 1 s *displacement* team speeds. Per-frame derivatives are swamped by position jitter. On one dev match under our assumed jitter, per-frame mean team speed was 6.1 m/s dead vs 6.2 m/s alive; 1 s displacement gave 1.1 vs 2.3.
- **Formation:** compactness, centroid separation and own-half shares.
- **Restart:** the change in team speed between the past and the future 3 s.

P(dead) is smoothed over 1 s, then thresholded at the value that maximises dead-frame F1 on the *training* matches. Dead runs shorter than 2 s and live runs shorter than 1 s are then removed, except runs touching a period edge, which are kept; both limits were fixed in advance.

**Why not other choices.** Everything below was prespecified and tested on dev, and every result is in `findings/experiments.md`:
- **A turnover confirmation state machine (E003).** The default 0.4 s window failed the criterion and predicted 803 possessions against 2,094 true. The 0.2 s window improved the means, but its short-gap interval overlapped the reference, so it was not adopted.
- **A restart-taker rule (E004)** gave +0.001 to +0.002, with every interval overlapping.
- **A learned constant offset on dead-ball edges (E006)** re-centred matched-edge timing. The spellF1 changes were small and mixed (−0.0015 to +0.0009), and degraded was negative.
- **Writing X instead of the carried owner (E005).** On these dev runs it gave the same spellF1 and a slightly lower macroF1 (the first tie-breaker), so the frozen model writes the carried owner.

## 3. Assets: used and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical automatic tracking (ATD), Barça–Athletic | **yes** | The only model input. Frame mapping `video_frame = frame_id − 1`; published half boundaries are used as metadata. |
| Smart Tagging | no | Permitted as an input. Dropped because our development matches have no comparable automatic tags, so we could not validate its contribution. |
| Homography, tactical and broadcast video | no | Not needed by a tracking-only model. A video ball detector was scoped but not built. We never compared our output with the video. |
| Broadcast tracking | no | Out of scope. |
| Public 5-minute ground truth | no | Not read by any code or test of ours. Its conventions were not used to tune anything. |
| DFL / IDSSE open data (7 matches) | **yes** | Development and holdout; see §4. |
| Metrica open sample games ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)) | dropped | Cloned and partly audited. Their events do not convert cleanly to H/A/D: touch-start possession, RECOVERY includes blocks, BALL OUT timing. They were therefore not used for training or selection. |

## 4. Training data

| dataset | source | who labelled it | how much | role |
|---|---|---|---|---|
| DFL / IDSSE open data | Bassek et al. (2025), *Sci Data* 12:195; [doi:10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) v1; CC BY 4.0 | Positions are TRACAB automatic tracking with DFL operator QC. The per-frame `BallStatus` and `BallPossession` flags are produced by DFL. | 7 matches (2022/23 Bundesliga and 2. Bundesliga), about 140k frames each. 5 dev matches: 2,094 possessions and 487 restarts. 2 holdout matches: 967 possessions. | Training and model selection on the 5 dev matches; one-time holdout on the other 2 |

**None of it comes from the provided game.**

**Label definition (primary reference):** D where `BallStatus` = 0, otherwise the team named by `BallPossession`, with DFL's unknown mask applied as X. The flag value 1 is the match-info home team in 5/5 dev matches. Where a player is within 1.5 m of the ball in live play, the flag names that player's team on 87–92% of frames. A 2 s confirm-control variant was reported for transparency only and never drove a decision. The DFL and Metrica conventions could not be compared, because we have no Metrica-side truth. See `findings/dev-labels-audit.md`.

**Degrading inputs to resemble the target.** Each dev match is degraded in a seeded, deterministic way (`src/possession/degrade.py`):
- ball dropout in bursts, biased to crowded and fast moments, to 59.3% ball visibility;
- player dropout to a mean of 20.5 objects;
- tracklet fragmentation into a 46-slot pool with identity loss;
- occasional ghost duplicates;
- 0.15 m player and 0.30 m ball jitter.

The calibration targets are only the input statistics published in `DATA.md`. Gap lengths and the other shape parameters are fixed assumptions, and three gap-length scenarios are reported as a robustness band. Whether using these published statistics counts as calibration is an open question for the organisers; see §6.

**Validation protocol.**
- The split was fixed by a seeded draw before any label inspection (`configs/split.yaml`).
- Selection used leave-one-match-out over the 5 dev matches, with paired per-match differences.
- A candidate was adopted only if, in all four input scenarios, its mean paired spellF1 difference was positive with a majority of matches positive, **and** its pooled 90% interval did not overlap the reference's. These intervals resample possessions, not matches; with 5 matches the criterion is a conservative screen, not a significance test.
- The holdout was opened once, on the frozen candidate (tag `freeze-e005`), through a gate that refuses reruns (`src/possession/holdout_gate.py`).

## 5. Possession rules implemented

- **Dead ball:** D wherever the decoded dead mask is on. D comes **only** from the dead-ball detector, never from a missing ball or a missing owner vote. Dead spells under 2 s and live spells under 1 s are removed, except those touching a period edge.
- **Owner:** in each live run, frames with a visible ball vote for the team of the nearest visible player. A centred ±2 s majority (ties to H) decides each frame. The vote never crosses a stoppage or a period boundary.
- **Loose balls and challenges:** frames with no vote carry the previous voted owner forward within the live run, and the run's head is back-filled from its first vote. After the first spell, an opposing spell shorter than 1 s is overwritten by the owner carried into it. The heuristic does not know whether such a spell was a touch, a deflection, a lost challenge or real brief control, so it also erases some genuine short possessions (see §6).
- **Restarts:** the owner at a restart is the first owner voted in the new live run. We tested a dedicated taker rule, including throw-in geometry (E004); it was not adopted.
- **Outside the published halves:** X. A live run with no owner evidence at all is also X. For spellF1 the scorer fills X forward within each graded run and back-fills a run's head from its first label; a run that is all X stays unfilled.

## 6. Limitations

- **Possessions are undercounted on unseen matches:** 719 predicted vs 967 true on the holdout (74%), against 88% on dev. Short-spell suppression probably contributes by merging real short possessions; that cause is a hypothesis, not established. It is the clearest thing to investigate next.
- **Dead-ball timing is noisy.** On degraded dev input, matched restarts have a median latency of −5 frames and matched stoppages +3. The spread is roughly ±20–40 frames (10th to 90th percentile). About 11–15% of real restarts and stoppages remain unmatched under one-to-one matching within ±3 s. Re-centring the median gave only small mixed changes (E006). Reducing the spread probably needs better evidence, such as a video ball detector; that was not tested.
- **Transfer is unmeasured.** The degraded DFL inputs are a model of Metrica's automatic tracking, not the real thing. DFL and Metrica may also define possession differently.
- **Open eligibility question.** Whether DATA.md input statistics may set the degradation rates is unresolved; sending the question and any response are unconfirmed at the time of writing. If they may not, `configs/freeze.json` prespecifies a fallback that retrains the same model on clean inputs. The fallback is **not implemented** in the runner, which currently refuses the clean recipe. Activating it would need a controlled runner update tied to the organisers' answer, plus contained validation.
- **Small evaluation base:** 5 dev matches and 2 holdout matches. The bootstrap intervals resample possessions, not matches.

## 7. Reproduction

Requirements: Python 3.14, the competition data bundle, and the IDSSE files.

```bash
git clone https://github.com/saushank3poch/diamond-hands-possession.git && cd diamond-hands-possession
make setup                                    # venv from requirements.lock, then pip check
make test                                     # synthetic tests only
# Data: put the competition bundle in ./data (or export MTAH_DATA_DIR=/path/to/bundle),
# and the IDSSE files (figshare 28196177 v1) in ../external/idsse.
# configs/external/idsse_manifest.json checks their SHA-256.
make submission                               # prints TARGET_RUN PASS or FAIL
```

`make submission` does the following:
1. It refuses to run unless the model code, split, lock and freeze manifest are byte-identical to the tag `freeze-e005`.
2. It trains the frozen recipe: 5 dev matches, primary labels, default degraded inputs, seed 0.
3. It predicts the Barça match and writes `runs/target/<run>/submission.csv`.
4. It runs the official `evaluation/validate_submission.py --rows 199319` on that exact file.

Two runs in our checkout and one fresh clone with `make setup` produced byte-identical files. The whole pipeline runs in under 10 minutes on a laptop.

Development results are reproduced with `make dev-eval CONFIG=configs/dev_xpolicy.yaml` (E005). Every experiment's config, commit and metrics are in `findings/results/`.

**Repository requirements for reproduction.** The preflight checks against git, so a reproducing repository must keep the whole executable dependency set (`src/`, `scripts/`, `configs/`, `evaluation/`, `games.py`, `Makefile`, `requirements.lock`) and the `freeze-e005` tag with its blobs. A script-only export would not reproduce this runner.

## References

- Bassek, Rein, Weber, Memmert (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. Data: [doi:10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) (CC BY 4.0).
- Challenge rules and submission requirements: [github.com/metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) (`RULES.md`, `SUBMISSION.md`, `EVALUATION.md`).
- Metrica sample data (examined, dropped): [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
