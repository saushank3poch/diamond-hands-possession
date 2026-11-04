# Experiment log

One hypothesis per entry, written as falsifiable **before** the run. Failures are logged too.

Every claim reports, on the degraded dev set with official `--bootstrap 1000` intervals (5th–95th percentile, i.e. 90%):
- spellF1
- macroF1
- possession count vs truth
- D share
- boundary latency

Overlapping intervals mean no win (saushank's standing rule). Non-overlap is necessary but not sufficient, because individual intervals are not a paired significance test. Claim improvements only with a suitable paired comparison on held-out other matches. The ~0.04 noise band is the organisers' full-match reference, not a dev-set cutoff. The public 5-minute GT is never a metric here.

Template:

```
## E### — <short name> (YYYY-MM-DD, commit <hash>, runs/<id>/)
Hypothesis: <change> raises dev spellF1 by ≥ <x>.
Change:
Dev metrics (90% interval): spellF1 | macroF1 | poss count (truth) | D share | latency
Match diagnostics (label-free): poss count | mean duration | D share | flip rate
Verdict: confirmed / rejected / inconclusive — <one line>
```

---

## E000: Reference baseline (published numbers only)

No local run. These are the organisers' full-match figures from `starter/baseline.py`, recorded as the floor:

| Variant | macroF1 | spellF1 |
|---|---:|---:|
| raw | 0.6532 | 0.0486 |
| smooth25 | 0.6752 | 0.1996 |
| smooth50 | 0.6724 | **0.2433** |
| smooth125 | 0.6356 | 0.2242 |

Next step: once the dev set exists, run the same four variants on it and log the result as a dev-set baseline. Do **not** adjust the degradation until the dev ordering matches these figures: they are scores against the private target-match GT, and fitting to them would be selection against test-derived results. Calibrate degradation only to the published input statistics in `DATA.md`.

---

## E001: Nearest-player baseline on dev (2026-11-03, commit 27d8b82, run `run-anyiij4h`)

**Hypothesis:** none. This is the reference point later experiments are compared against, not a claim.

**Setup:**
- 5 DFL dev matches, leave-one-match-out (the baseline is untuned, so fitting is a no-op).
- Official scorer with evaluator-only X separators after every period, so nothing crosses a half or match boundary.
- Pooled over matches with the official 1000-draw bootstrap (5th–95th percentile, 90%).
- Primary labels are the raw DFL flags.
- Command: `make dev-eval CONFIG=configs/dev_baseline.yaml` on a clean tree.
- Full output: `results/E001_dev_metrics.json`; provenance: `results/E001_run.json`.

**Limitation of the interval:** the official pooled bootstrap resamples possessions within these 5 matches, not matches. It understates between-match variation, so it is not a generalisation interval. Per-match paired differences are reported alongside it.

Pooled spellF1 [90% CI], primary labels:

| scenario | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|
| clean | 0.097 [0.093, 0.101] | 0.301 [0.289, 0.312] | 0.315 [0.300, 0.329] | 0.220 [0.206, 0.235] |
| degraded (default) | 0.018 [0.017, 0.020] | 0.177 [0.169, 0.184] | 0.227 [0.217, 0.238] | 0.197 [0.184, 0.210] |
| degraded, short gaps | 0.012 [0.011, 0.013] | 0.200 [0.191, 0.208] | 0.252 [0.240, 0.263] | 0.206 [0.192, 0.218] |
| degraded, long gaps | 0.022 [0.021, 0.023] | 0.180 [0.172, 0.188] | 0.214 [0.204, 0.224] | 0.198 [0.186, 0.210] |

Paired per-match spellF1 difference vs smooth50, within each scenario (mean, and number of matches better):

| variant | clean | degraded | short gaps | long gaps |
|---|---|---|---|---|
| raw | −0.219 (0/5) | −0.209 (0/5) | −0.240 (0/5) | −0.192 (0/5) |
| smooth25 | −0.014 (0/5) | −0.051 (0/5) | −0.052 (0/5) | −0.034 (0/5) |
| smooth125 | −0.095 (0/5) | −0.029 (1/5) | −0.044 (0/5) | −0.014 (3/5) |

**Other metrics, degraded default, smooth50:**
- macroF1 0.381.
- Possessions 3,166 predicted vs 2,094 true.
- D share 1.6% predicted vs 45.2% true.
- Mean |latency| at τ = 25: 12.5 frames.

**Reading:**
- **How to compare methods:** a later method is compared by its candidate-minus-baseline difference *within each scenario*, then by whether that difference holds across all four scenarios. The spread of the baseline's own score across scenarios (0.214–0.252 for smooth50) describes how sensitive the score level is to the degradation assumption. It is not a threshold for method gains.
- **Seeds:** variant seeds differ by scenario, so part of that spread is realisation noise, not the assumption alone.
- **macroF1 is comparable** across variants because the truth is the same. Raw-degraded's higher macroF1 (0.557) reflects a real trade-off: it outputs D whenever the ball is missing (40.8% D), which catches many true stoppages but fragments possessions (spellF1 0.018).
- **Dead balls:** the baseline almost never outputs D (1.6% vs 45%). That motivates FCB-11 (dead-ball detection), but it does not establish how much spellF1 dead-ball detection can actually gain.
- **Observation only:** the dev ordering (smooth50 best on degraded) happens to match the organisers' published full-match ordering. Nothing was calibrated toward it.

**Proxy labels (2 s)** are in the JSON for transparency and are not used for selection.

---

## E002: Dead-ball detector, FCB-11 (2026-11-03, commit 99f05f0, run `run-fjwo1893`; revision 2)

**Hypothesis (written before the first run):** fusing tracking-only dead-ball evidence into the smooth50 baseline raises spellF1 over smooth50 in all 4 dev scenarios. Concretely: D where the fused score says dead, and the nearest-player vote confined to live runs elsewhere. In each scenario the mean paired per-match difference should be positive, with a majority of matches positive.

**Revision 2** fixes three defects found in review, then re-runs:
1. Restart deltas crossed period boundaries.
2. Missing-ball frames inside decoded live runs could still become D through the owner vote.
3. The "no ball" ablation still contained ball-derived signals.

Now:
- live-run ownership never emits D: it carries within the run, and X if there is no vote at all;
- every ball-derived signal is in the ball group;
- the ball-free detector is tested to be invariant to ball coordinates.

The new tests fail on the revision 1 code (5 of 11) and pass now.

**Change:** `predictors/deadball.py`. A per-frame logistic model is trained in leave-one-match-out folds. It uses windowed signals in 4 groups:
- **ball:** missing, time since seen, last-seen distance to a line or off the pitch, windowed ball speed and its 3 s change, crowding around the last-seen ball, distance from the centre spot;
- **motion:** 1 s displacement team speeds;
- **formation:** team spread, centroid separation, own-half shares;
- **restart:** team speed change over ±3 s, within the period.

Each signal is also averaged over centred 1, 4 and 12 s windows within the period. Decoding:
- P(dead) is smoothed over 1 s;
- the threshold maximises dead-frame F1 on the *training* matches only;
- the minimum run lengths are fixed a priori: dead 2 s, live 1 s.

No Smart Tagging.

**Command:** `make dev-eval CONFIG=configs/dev_deadball.yaml`, clean tree. It is paired against E001 smooth50; the provenance check passed. Output: `results/E002_dev_metrics.json`.

**Pooled spellF1 [90% CI]** (primary labels). Δ = mean paired per-match difference vs smooth50, with the number of matches better:

| | clean | degraded | short gaps | long gaps |
|---|---|---|---|---|
| smooth50 (E001) | 0.315 | 0.227 | 0.252 | 0.214 |
| **deadball_all** | **0.431** [0.416, 0.446], +0.117 (5/5) | **0.332** [0.319, 0.346], +0.105 (5/5) | **0.358** [0.344, 0.371], +0.106 (5/5) | **0.324** [0.311, 0.338], +0.111 (5/5) |
| only_ball | 0.428, +0.114 (5/5) | 0.323, +0.096 (5/5) | 0.352, +0.100 (5/5) | 0.319, +0.105 (5/5) |
| detector_without_ball | 0.355, +0.040 (5/5) | 0.275, +0.048 (5/5) | 0.295, +0.043 (5/5) | 0.268, +0.054 (5/5) |
| only_motion | 0.334, +0.019 (4/5) | 0.266, +0.038 (5/5) | 0.287, +0.034 (4/5) | 0.261, +0.047 (5/5) |
| no_motion | 0.429 | 0.328 | 0.357 | 0.319 |
| no_formation | 0.430 | 0.330 | 0.359 | 0.322 |
| no_restart | 0.432 | 0.330 | 0.358 | 0.323 |

**deadball_all, degraded, other metrics:**

| metric | deadball_all | smooth50 |
|---|---|---|
| alive/dead F1 | 0.960 | — |
| D share (true 45.2%) | 44.2% | 1.6% |
| macroF1 | 0.826 | 0.381 |
| possessions (true 2,094) | 2,578 | 3,166 |
| mean \|latency\| at τ = 25 | 10.3 frames | 12.5 frames |

**Verdict: confirmed on dev.** The improvement holds in every scenario, on every match, and the pooled intervals do not overlap. This is a dev result only; it does not establish target performance.

**Ablation reading:**
- **The ball group carries most of the gain:** only_ball scores 0.323 vs 0.332 on degraded.
- **Motion, formation and restart are not individually established.** Dropping any one changes spellF1 by ≤ 0.004. Their combined contribution (all minus only_ball) is +0.003 to +0.009, inside the intervals. No gain is claimed for them; they stay in as cheap redundancy.
- **A ball-free detector still beats smooth50** on dev in every scenario: +0.040 to +0.054, 5/5 matches each. The owner vote still uses the ball. This is a dev ablation, **not a lower bound on target performance**.

**Main risk: transfer.**
- The ball group's strength partly reflects two things: DFL keeps tracking the ball when it is off the pitch, and FCB-8 assumes an off-pitch ball goes missing 90% of the time.
- The target ATD's dead-ball behaviour is unpublished, and dev data cannot resolve it.
- Contained, label-free target diagnostics (FCB-19) can flag an implausible output, such as predicted D share or stoppages per half. They can neither validate transfer nor select a model.

**Other notes:**
- **Windowed motion:** per-frame speeds are swamped by FCB-8's per-frame position jitter. The motion group uses 1 s displacement speeds.
- **Possession over-count remains:** 2,578 predicted vs 2,094 true on degraded. That is the next lever, the turnover rules (FCB-12).


## E003: Control-confirmed ownership (prespecified, 2026-11-03)

Hypothesis: replacing E002 live ownership with sustained control evidence raises primary dev spellF1 in all four scenarios, with positive mean paired per-match differences, a majority positive, and non-overlapping pooled 90% intervals versus E002. Default confirmation is 0.4 s; 0.2 and 0.8 s are prespecified sensitivity ablations, not proxy-label-derived minima. A 0.4 s delayed-output variant isolates back-dating. No target or holdout run.

Evidence: observed ball within 2.5 m of the nearest player, at least 0.5 m advantage over an observed opponent, and supported 0.96 s displacement motion (relative speed <=4 m/s or ball speed <=3 m/s). Missing/contested evidence retains an established owner and resets a pending switch. Confirmations are back-dated; the first confirmed owner fills the live-run head (provisional restart assignment, to be assessed in FCB-13). Unknown runs stay X; only the unchanged dead detector emits D. All limits are engineering assumptions fixed before the run.

Confirmation lag for continuous evidence is (N-1)/25: 0.16/0.36/0.76 s for 5/10/20 frames. Back-dating removes this output delay relative to evidence onset, not relative to unknown true control. Motion uses future samples, so all variants remain offline. Official transition latency and synthetic onset/confirmation tests measure complementary effects.

Command: `make dev-eval CONFIG=configs/dev_turnover.yaml`. Producer `deb0c1627667ae97770a12ee5092a903cc8e99cd`, clean tree, `run-4la912mx`. Full metrics: `results/E003_dev_metrics.json`; provenance: `results/E003_run.json`.

### E003 results and adoption decision

Primary pooled spellF1 [90% possession-bootstrap interval]; mean paired per-match delta vs E002 and number positive. These intervals are conditional on the five dev matches, not cross-match generalisation intervals.

| candidate | clean | degraded | short gaps | long gaps |
|---|---|---|---|---|
| turnover_04 | 0.481 [0.460, 0.501], +0.051 (5/5) | 0.326 [0.307, 0.346], -0.005 (2/5) | 0.301 [0.282, 0.321], -0.055 (1/5) | 0.338 [0.318, 0.358], +0.016 (4/5) |
| turnover_02 | 0.540 [0.520, 0.559], +0.110 (5/5) | 0.385 [0.366, 0.405], +0.052 (4/5) | 0.374 [0.352, 0.394], +0.017 (3/5) | 0.391 [0.370, 0.411], +0.067 (5/5) |
| turnover_08 | 0.398 [0.377, 0.419], -0.032 (1/5) | 0.255 [0.237, 0.274], -0.076 (0/5) | 0.223 [0.206, 0.242], -0.134 (0/5) | 0.272 [0.253, 0.291], -0.051 (0/5) |
| turnover_04_delayed | 0.470 [0.449, 0.490], +0.040 (5/5) | 0.318 [0.299, 0.338], -0.014 (2/5) | 0.294 [0.275, 0.314], -0.062 (0/5) | 0.332 [0.312, 0.351], +0.010 (4/5) |

**Verdict: the prespecified default hypothesis is rejected. Do not adopt this ownership replacement; E002 deadball_all remains selected.** The default 0.4 s candidate regresses on default and short-gap degradation. The 0.2 s sensitivity candidate improves the paired mean in every scenario, but its short-gap interval overlaps E002 (3/5 matches positive), so it does not meet the all-scenario criterion either. Retain it as an experimental candidate, not a validated replacement. No further thresholds were searched after this run. The 2 s proxy was not used to choose a window.

**Why this is not simply a fragmentation win:** default degradation yields only 803 predicted spells with 0.4 s (994 with 0.2 s), versus 2,094 references and E002's 2,578. Reducing excessive flips also suppresses real changes or fails to establish ownership. Counts alone do not establish better possession accuracy. Requiring continuous motion support in noisy, fragmented tracks is restrictive; this is a plausible mechanism, not a measured causal attribution. Initial-owner inference is also changed and remains provisional until FCB-13.

**Lag ablation:** the 10-frame window confirms 9 frames (0.36 s) after sustained evidence begins; back-dating restores the recorded boundary to evidence onset. Synthetic tests establish the exact 4/9/19-frame delays for 5/10/20-frame windows. At default degradation, the delayed 0.4 s variant scores 0.318 versus 0.326 back-dated, with matched transition mean absolute latency 9.41 versus 9.15 frames (E002: 10.3). That observed latency covers matched H/A/D transitions and excludes unmatched ones; it is not an unbiased control-boundary error or proof of a gain. The two intervals overlap. All variants, including delayed output, remain offline because displacement evidence uses future samples.

**Other default-degraded metrics:** the 0.4 s candidate has macroF1 0.828 and D share 44.25%; 0.2 s has macroF1 0.839 and the same D share. The dead detector and its training are unchanged. No target/holdout execution, no public-GT access. This completes the state-machine implementation and window-size experiment; it does not satisfy the empirical improvement hypothesis.


## E004: Local restart-owner inference (prespecified, 2026-11-03)

Hypothesis: local taker evidence with touchline geometry improves reference-restart frame accuracy and primary spellF1 versus selected E002 in all four dev scenarios, with positive paired spellF1 means, majority of matches positive and non-overlapping pooled 90% intervals. Compare full geometry and distance-only ablation; rerun unchanged E002 to add restart diagnostics. No target/holdout run; no FCB12 candidate adopted.

Prespecified rule: anchor the ball nearest in time within +/-0.5s of each predicted live edge, clipped to the current period and live-run end. Require >=3 observed ball frames within1m of anchor and both teams visible. Choose the team with lower median nearest-player cost, within3m of anchor, >=0.75m cost advantage and >=70% frame votes. Near a touchline (within1.5m, excluding5m at each goal line), add0.75 times the player's distance inside that line to discourage selecting a receiver over an on-line thrower. This is an assumed geometry cue, not a known restart type. Weak/ambiguous evidence falls back to E002. Override only the initial baseline-owner spell, capped at1s. D and later owner votes remain unchanged.

Separate diagnostic: exact reference D->H/A edges inside a period, excluding GT-X gaps and period heads. Report correct/total, H/A prediction coverage, conditional H/A accuracy and D/X misses. These oracle-time labels are evaluator-only and never reset or seed a predictor. The diagnostic conflates restart detection timing and ownership; it is not conditional accuracy given correct boundaries.

Command: `make dev-eval CONFIG=configs/dev_restart.yaml`. Producer `041d6355e9190a871fbbc7735ab5464942c509fb`, clean tree, `run-ms9utn7a`. Full metrics/provenance: `results/E004_dev_metrics.json` and `results/E004_run.json`.

### E004 results and adoption decision

E002 reproduction matches every previously reported pooled and per-match metric exactly, on primary and proxy labels in all four scenarios. New restart diagnostics do not change the existing scoring. Primary pooled spellF1 [90% possession-bootstrap interval], mean paired delta vs E002 (matches positive):

| candidate | clean | degraded | short gaps | long gaps |
|---|---|---|---|---|
| restart_geometry | 0.4330 [0.4173, 0.4487], +0.0020 (5/5) | 0.3336 [0.3203, 0.3475], +0.0012 (4/5) | 0.3599 [0.3459, 0.3733], +0.0019 (4/5) | 0.3257 [0.3125, 0.3391], +0.0014 (4/5) |
| restart_distance | 0.4327 [0.4172, 0.4483], +0.0017 (5/5) | 0.3336 [0.3203, 0.3475], +0.0012 (4/5) | 0.3599 [0.3459, 0.3733], +0.0019 (4/5) | 0.3256 [0.3124, 0.3390], +0.0013 (4/5) |

Exact-reference-time restart ownership (487 within-period D->H/A edges; D/X count as wrong):

| candidate | clean correct | degraded correct | short gaps correct | long gaps correct |
|---|---|---|---|---|
| e002_reference | 341/487 (70.02%) | 253/487 (51.95%) | 213/487 (43.74%) | 263/487 (54.00%) |
| restart_geometry | 352/487 (72.28%) | 255/487 (52.36%) | 216/487 (44.35%) | 270/487 (55.44%) |
| restart_distance | 350/487 (71.87%) | 255/487 (52.36%) | 216/487 (44.35%) | 269/487 (55.24%) |

**Verdict: inconclusive; do not adopt. E002 remains selected.** All mean paired spellF1 changes are positive, but every interval overlaps E002, failing the prespecified adoption criterion. No additional thresholds were searched. These possession-bootstrap intervals are conditional on these five dev matches, not generalisation intervals. Neither the slight accuracy increase nor positive point estimates establish a reliable improvement.

On default degradation the full rule gives 255/487 exact-edge correct (52.36%) versus E002's253/487 (51.95%). Both predict a team at333/487 edges (68.38% coverage); conditional H/A accuracy is76.58% vs75.98%. Both predict D at154 edges and X at0: these misses include detection timing, so this is not pure taker-identification accuracy. The denominator excludes period heads and unknown gaps. No boundaries were given to the predictor.

**Geometry ablation:** geometry and distance-only outputs have identical aggregate scores in default and short-gap degradation. Geometry adds two correct owners on clean inputs and one on long-gap inputs. This does not establish a geometry contribution or separately validate throw-in accuracy; no event-type labels were used. The implementation supplies the tracking-only throw-in cue and a synthetic taker/receiver test, not a measured throw-in-specific success claim.

**Fragmentation risk:** default-degraded predicted spells rise from2578 to2599, while truth has2094. The one-second correction can revert to the previous baseline owner at its cap, creating a synthetic switch; that is a plausible cost of this design, not a measured causal attribution for every extra spell. Do not change the cap or rerun a search based on this result in E004.

The dead detector, model training and labels are unchanged. All experiments are dev-only; no target/holdout/public-GT processing. FCB13 implementation and separate restart evaluation are complete, but the empirical adoption criterion is not met. Keep the candidate experimental.

## A005: X policy, FCB-14 (scorer analysis; revised 2026-11-04 after review)

**What holds:** the official scorer computes spellF1 and every temporal metric on `fill_abstentions(pred)`. That function forward-fills X inside each graded run and back-fills only the run head. So an X scores **the same as the label it carries**, and an X that opens a live run after D scores as D. `tests/test_x_policy.py` pins this on synthetic sequences.

**Withdrawn (review `RESEARCH/FCB_14_X_POLICY_REVIEW_2026_11_04.md`):** the first version said no X policy could raise spellF1 over E002, that X had "no headroom", and that it affected "only frame metrics". All three are wrong.
- Replacing a wrong label with X *can* raise spellF1 relative to the original prediction. Truth `HHHHHH`, prediction `HHAAHH` scores 0.1; `HHXXHH` scores 1.0.
- macroF1 is the first tie-breaker (`CHALLENGE.md`). X can raise it when it replaces wrong labels, because it removes false positives.
- X can also harm both metrics, by erasing a real short possession.

Regression cases for both directions are in `tests/test_x_policy.py`. The correct statement is narrower: for spellF1, any X rewrite has an equivalent carried-label rewrite. Whether a given rewrite helps E002 is an empirical question, tested in E005.


## E005: Short-spell suppression inside live runs, FCB-14 (prespecified, 2026-11-04; commit 4230757, run `run-vura5btv`)

**Hypothesis (posted before the run):** inside each decoded live run of E002 `DeadBall`, any owner spell shorter than 1 s that is not the run's first spell becomes X. *As implemented,* a short spell that already agrees with the carried owner stays explicit; only disagreeing short spells are replaced. This affects the X variant's frame metrics, not the carry output. This raises primary dev spellF1 over E002 `deadball_all` in all 4 scenarios, with positive mean paired per-match differences and a majority of matches positive. The 1 s equals E002's minimum live length and was fixed a priori. No other threshold was run. A `carry` variant writes the carried owner instead of X. It isolates the macroF1 effect where both variants score the same spellF1. That equality is not universal: the scorer's fill restarts at reference-X gaps while the predictor's carry does not, so the two can differ (regression `test_x_and_carry_differ_across_reference_x_gap`: truth `XXXHHXXX`, X variant scores 0, carry 1).

**Change:** `src/possession/predictors/xpolicy.py` (`ShortSpellX`, via the E002 `owner_run` hook); `configs/dev_xpolicy.yaml`. D, the dead mask and training are unchanged.

**Results** (primary labels; pooled spellF1 [90% CI, resampling possessions]; Δ = mean paired per-match difference vs E002, matches positive):

| scenario | E002 | short-spell (X = carry) | Δ | spells pred / true (E002 → new) | macroF1 E002 / X / carry | latency (frames) |
|---|---|---|---|---|---|---|
| clean | .431 [.416, .446] | **.466** [.448, .483] | +.034 (5/5) | 2010 → 1616 / 2094 | .8680 / .8684 / .8693 | 8.25 → 7.78 |
| degraded | .332 [.319, .346] | **.389** [.373, .406] | +.057 (5/5) | 2578 → 1838 / 2094 | .8257 / .8262 / .8279 | 10.32 → 10.06 |
| short gaps | .358 [.344, .371] | **.405** [.389, .421] | +.048 (5/5) | 2423 → 1746 / 2094 | .8386 / .8388 / .8399 | 10.64 → 10.16 |
| long gaps | .324 [.311, .338] | **.373** [.357, .388] | +.049 (5/5) | 2437 → 1764 / 2094 | .8155 / .8159 / .8170 | 10.50 → 10.11 |

D share is unchanged in every scenario (dead mask untouched). The X and carry variants have equal spellF1 in every reported per-match and pooled result here. This is observed on these dev scenarios, not guaranteed in general (see the hypothesis note).

**Verdict: hypothesis confirmed.** All four scenarios have positive mean paired differences with 5/5 matches positive. The pooled intervals do not overlap E002's in any scenario.

**Variant choice:** carry has higher macroF1 than X in all four scenarios (+0.0009 to +0.0017), with identical spellF1. Proposed selection: **`short_spell_carry`** as the new baseline (E005), on the first tie-breaker. The macroF1 differences are small and have no interval; they decide only the tie-break.

**Caveats:**
- Possessions are now undercounted (1838 vs 2094 on degraded), where E002 overcounted (2578). Real short possessions are being merged; a different threshold could trade this off, but none was searched.
- Dev only. Target transfer is unproven; the threshold was not chosen on any target data.

## A006: Dead-edge timing audit of E005 (evaluator only, 2026-11-04; commit 18d94ba, run `run-bj7a8dky`)

`possession.evaluation.restarts.edge_timing` pairs predicted and reference dead-run edges one-to-one (same period, |offset| ≤ 75 frames = 3 s, fixed a priori, greedy by smallest offset). It reports signed latency (predicted − reference) for matched pairs, plus missed and spurious counts. It is evaluation-only; predictors never see it.

Primary labels, pooled over the 5 dev matches:

| scenario | edge | ref / pred / matched | missed / spurious | median | p10 / p90 | share \|lat\| ≤ 8 |
|---|---|---|---|---|---|---|
| clean | restart (D→live) | 487 / 500 / 460 | 27 / 40 | −3 | −21 / 0 | 0.70 |
| clean | stoppage (live→D) | 495 / 509 / 460 | 35 / 49 | 0 | −7 / +31 | 0.70 |
| degraded | restart | 487 / 507 / 432 | 55 / 75 | −5 | −27 / +17 | 0.46 |
| degraded | stoppage | 495 / 517 / 420 | 75 / 97 | +3 | −16 / +38 | 0.43 |
| short gaps | restart | 487 / 510 / 442 | 45 / 68 | −3 | −24 / +21 | 0.42 |
| short gaps | stoppage | 495 / 519 / 442 | 53 / 77 | +2 | −17 / +33 | 0.48 |
| long gaps | restart | 487 / 521 / 424 | 63 / 97 | −5 | −30 / +10 | 0.53 |
| long gaps | stoppage | 495 / 530 / 411 | 84 / 119 | +3 | −16 / +39 | 0.41 |

**Reading (scoped per review):**
- *Matched* restarts have an early median and matched stoppages a slightly late one. The ±3 s pairing excludes missed and spurious edges, and the spread includes many late releases.
- Similar total edge counts do not show that the right events were found. The missed and spurious counts above are the evidence.
- The ±8-frame share is descriptive only. spellF1 has no frame cutoff: it matches whole possessions by temporal IoU at thresholds 0.3–0.7.
- This supersedes the earlier "releases late" reading of the E004 restart counts, which was withdrawn.

## E006: Training-only median dead-edge offset (prespecified, 2026-11-04; commit 97451eb, run `run-a32u9rtq`)

**Hypothesis (posted before the run):** shifting E005's dead-run edges by the median signed latency estimated in-sample on the training matches raises primary dev spellF1 over E005 `short_spell_carry` in all 4 scenarios, with a positive mean paired difference and a majority of matches positive. Single variant.

**Change:** `src/possession/predictors/edgeoffset.py` (`EdgeOffset`); `configs/dev_edge_offset.yaml`. The edge cases were prespecified and tested synthetically in `tests/test_edge_offset.py`:
- no paired edges → offset 0;
- period boundaries are kept, and nothing moves across a period;
- overlapping runs merge;
- runs whose shifted interval is empty are dropped.

**Results** (pooled spellF1 [90% CI]; Δ = mean paired per-match difference vs E005, matches positive):

| scenario | E005 | E006 | Δ | restart median / ≤8 share: E005 → E006 |
|---|---|---|---|---|
| clean | .4657 [.448, .483] | .4664 [.449, .484] | +.0009 (3/5) | −3 / .70 → 0 / .77 |
| degraded | .3893 [.373, .406] | .3878 [.372, .404] | −.0015 (1/5) | −5 / .46 → 0 / .53 |
| short gaps | .4053 [.389, .421] | .4061 [.389, .422] | +.0009 (2/5) | −3 / .42 → 0 / .46 |
| long gaps | .3725 [.357, .388] | .3726 [.357, .388] | +.0001 (1/5) | −5 / .53 → 0 / .60 |

The E005 reference reproduced its earlier metrics exactly. The offset re-centres matched medians to 0 and lifts the restart ±8-frame share, but stoppage shares barely move.

**Verdict: rejected; E005 remains selected.** The all-scenario criterion fails: degraded is negative (1/5 positive), and only clean has a majority of matches positive (3/5). All intervals overlap.

**Interpretation (not tested further):** re-centring the median leaves the wide spread untouched (p10/p90 roughly ±20–40 frames), and that spread, together with missed and spurious edges, dominates. A constant shift changes IoU very little on multi-second spells. No further offsets or windows were searched.

**Correction (review `RESEARCH/FCB_E006_REVIEW_2026_11_04.md`, commit ed52396, run `run-111gha3c`):** the audit and `EdgeOffset._fit_offsets` now extract edges, fill prediction X and pair edges only inside contiguous reference-valid blocks: one active period, no GT-X. Predicted edges in unknown regions are ignored, and no pair crosses a GT-X gap. Regressions are in `tests/test_edge_timing.py` (both reviewer probes and the per-block X fill) and `tests/test_edge_offset.py` (unknown regions in training references). `predict()` never receives these boundaries.

**Affected reports:** I reran `configs/dev_edge_offset.yaml` at ed52396. Every report and paired result is identical to the committed `E006_dev_metrics.json`, including the proxy label sets. The rerun's `e005_reference` results are identical to the committed A006 audit results. The A006 and E006 numbers above therefore stand unchanged; the rerun record is `findings/results/E006_rerun_run.json`.

## FCB-18 pre-freeze (2026-11-04): holdout runner, not yet opened

- **Freeze manifest:** `configs/freeze.json`, tracked. It holds:
  - the predictor (`short_spell_x`, `mode: carry`, defaults otherwise);
  - the `requirements.lock` hash and the split hash;
  - the holdout run settings (seed 0, bootstrap 1000);
  - every component, marked as either *included and evaluated in dev CV* or *explicitly excluded*;
  - **the target training recipe, fixed before opening:** the 5 dev matches only (the holdout is never used for fitting), primary labels, the default degraded variant with the dev-CV per-match seeds, and seed 0. Its headline holdout number is `degraded/primary`. If the organisers rule the published-statistics degradation ineligible, the prespecified fallback is the clean variant, reported as `clean/primary`.

  The tag `freeze-e005` goes on the reviewed commit.
- **Runner:** `make holdout-eval` → `scripts/pipeline.py holdout-eval` → `holdout_gate.run_once` → `harness.evaluate_holdout`. For each of the four input variants, it fits once on all 5 dev matches (primary labels, same variant) and predicts each holdout match from tracking alone. Holdout references are used only for scoring, through the same `_score_variant` code as dev. The four reports are prespecified robustness checks; the submission recipe above is not chosen from them.
- **Preflight** (`src/possession/holdout_gate.py`), all checked before any holdout read:
  - only the fixed tracked paths are accepted. The run config must be byte-identical to `configs/holdout.yaml` and may set only `seed` and `bootstrap_n`. Both files must equal their blobs at the tag;
  - HEAD must equal the tag's commit, and the worktree must be clean;
  - the lock and split hashes, and the effective seed (config and environment) and bootstrap count, must match the manifest.
- **One shot:** an atomic `O_CREAT|O_EXCL` reservation (`runs/holdout/RESERVATION.json`, status `started`) is written after preflight and before any holdout access. It is then updated to `failed` (exception type only) or `completed`. Any existing reservation, or a committed `findings/results/HOLDOUT_metrics.json`, refuses every later attempt; there is no automatic retry. `load_match` refuses holdout matches unless called with `holdout=True`, and refuses dev matches on that path.
- **Refactor check:** `configs/dev_xpolicy.yaml` was rerun at be5297f. Every E005 report and paired result is identical to the committed `E005_dev_metrics.json`. The comparison excludes `edge_timing`, a field added after E005.
- **Tests:** `tests/test_holdout.py` uses a spy predictor, a fake git and temporary markers, with no holdout data. It covers:
  - reviewer probes: a manifest selecting another predictor, a wrong lock hash, an altered seed or extra config keys;
  - a gate failure, which never reserves or evaluates;
  - a failure after a mocked read, which records `failed` and blocks a retry;
  - competing reservations, where one wins.

  172 tests OK.

## H001: One-time holdout evaluation of `freeze-e005` (2026-11-04; tag `freeze-e005` = 66da3d6, run `run-0exz6gbj`)

Opened exactly once through `make holdout-eval`; the reservation ended `completed`, and the run was clean (`worktree_dirty: false`). Training used the 5 dev matches; the holdout was DFL-MAT-J03WMX and DFL-MAT-J03WOY. These results are reported as they came out, with no re-selection (D5).

Primary labels; pooled spellF1 [90% CI, resampling possessions; only 2 matches, so not a cross-match interval]:

| scenario | holdout spellF1 | per match (WMX / WOY) | dev CV (E005) | macroF1 | spells pred / true | D share pred / true |
|---|---|---|---|---|---|---|
| clean | .449 [.421, .478] | .452 / .447 | .466 [.448, .483] | .868 | 664 / 967 | .433 / .440 |
| **degraded (headline)** | **.359 [.334, .385]** | .376 / .340 | .389 [.373, .406] | .823 | 719 / 967 | .426 / .440 |
| short gaps | .395 [.370, .420] | .400 / .389 | .405 [.389, .421] | .836 | 743 / 967 | .431 / .440 |
| long gaps | .349 [.325, .374] | .347 / .352 | .373 [.357, .388] | .814 | 712 / 967 | .423 / .440 |

**Reading:**
- The headline degraded holdout score is 0.030 below dev CV, and the intervals overlap. That is consistent with modest optimism from dev selection, but two matches cannot separate it from match-to-match variation.
- The ordering across scenarios matches dev.
- Possessions are undercounted more on holdout (74% of true) than on dev (88%). This is the clearest remaining weakness: real short possessions are being merged.
- The D share is close to truth.
- Target (Barça–Athletic) performance remains unmeasured. The organiser eligibility question on the degradation statistics is still open.

## FCB-19: target submission runner (2026-11-04; commit fade003)

- **Command:** `make submission`, which runs `scripts/run_target_local.sh configs/submission.yaml -- pipeline.py submission`. It trains exactly the frozen `target_training` recipe from `configs/freeze.json`: the 5 dev matches, primary labels, the default degraded variant, seed 0. It then predicts the Barça–Athletic match from tracking alone, writes `runs/target/<run>/submission.csv` with X outside the published halves, and runs `evaluation/validate_submission.py --rows 199319` on that exact file. Validator output stays in the contained run directory, and the wrapper prints only pass/fail. There is a 600 s alarm.
- **Data placement for a clean clone:** the match files are not distributed with the repo. Place them under `data/` or set `MTAH_DATA_DIR` (see `games.py`). The DFL dev data is expected at `../external/idsse`.
- **Results (pass/fail only; no target output or diagnostic was read):**
  - two runs in the checkout: `TARGET_RUN PASS`, validator exit 0 in both, byte-identical CSVs (sha256 equal);
  - a fresh clone at fade003 with an empty dev cache, so dev sources were re-parsed: the first attempt failed with `FileNotFoundError` because the target data was not placed. With `MTAH_DATA_DIR` set: `TARGET_RUN PASS`, validator exit 0, and a CSV byte-identical to the checkout's.
  - The clone reused the checkout's virtualenv rather than running `make setup`, so it does not test dependency installation.
- **Process note:** these target runs happened before Codex's review of the runner; the request for a review first arrived while they were running. No target content was inspected, and they were reproducibility checks, not selection. Any change from review will be followed by a rerun.

**Revision after review (`RESEARCH/FCB_19_SUBMISSION_REVIEW_2026_11_04.md`, commit cb2054e):**
- `submission.build` runs `verify_freeze` and `recipe` before any dev, holdout or target file is opened.
  - `verify_freeze` requires the manifest, split, lock, `games.py`, every predictor and adapter module (same file set) and the model-input modules to be byte-identical to `freeze-e005`. New runner files may differ.
  - `recipe` resolves the exact frozen variant string and refuses anything else, including `degraded_long_gaps` and `clean`. It also checks the split hash and the predictor, and returns the frozen dev IDs that `train` must match.
  - The freeze identity (tag, tag commit, HEAD) is recorded in the local `submission.json`.
- Refusal tests (manifest selecting `nearest_player`; drift in code, lock or split; an extra predictor file; seed mismatch) show no loader runs.
- Wording: the wrapper *captures* stdout and stderr in local run files and prints only pass/fail; it does not discard them.
- **Fresh-environment acceptance (pass/fail and equality only):** a fresh clone at cb2054e with `make setup` from `requirements.lock` (`pip check` clean), the full suite at 183 OK, then a contained `make submission` with `MTAH_DATA_DIR` set gave `TARGET_RUN PASS` with validator exit 0. Its CSV is byte-identical to the checkout's run at cb2054e, which is byte-identical to the earlier fade003 runs.

## FCB-20 clean fallback switch (2026-11-04; not run on the target)

`configs/eligibility.json` records the organiser ruling on using published DATA.md statistics for the degradation rates (question sent 2026-11-04) as `pending`, `allowed` or `disallowed`, plus `response_source`. A settled ruling must cite its source; a pending ruling has none.
- **`pending` is not approval.** It keeps the frozen degraded recipe and is recorded as pending. No answer is not organiser clearance; the submission decision stays with the owner.
- `allowed` keeps the degraded recipe.
- `disallowed` selects the clean fallback prespecified in the tag-bound `configs/freeze.json`: the same predictor, 5 dev matches, primary labels and seed 0, with clean inputs. It refuses if the fallback text is missing.

The ruling and its source are recorded in the local `submission.json`, and the eventual answer is cited here. The fallback's only holdout evidence is `clean/primary` in H001, which is on clean inputs and not representative of noisy target tracking. Tests are in `tests/test_submission.py` (`EligibilitySwitchTests`: pending and both rulings, plus invalid values and sources). No target run was done with the fallback.
