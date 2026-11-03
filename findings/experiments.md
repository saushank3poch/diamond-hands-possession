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

Command: `make dev-eval CONFIG=configs/dev_turnover.yaml`. Results pending. E002 remains the selected baseline until evidence supports adoption.
