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

## E002: Dead-ball detector, FCB-11 (2026-11-03, commit a6f4c1c, run `run-524wnvyt`)

**Hypothesis (written before the run):** fusing tracking-only dead-ball evidence into the smooth50 baseline raises spellF1 over smooth50 in all 4 dev scenarios. Concretely: D where the fused score says dead, and the nearest-player vote confined to live runs elsewhere. In each scenario the mean paired per-match difference should be positive, with a majority of matches positive.

**Change:** `predictors/deadball.py`. A per-frame logistic model is trained in leave-one-match-out folds. It uses windowed signals in 4 groups:
- **ball:** missing, time since seen, last-seen distance to a line or off the pitch, windowed ball speed;
- **motion:** 1 s displacement team speeds;
- **formation:** team spread, centroid separation, own-half shares, crowding around the last-seen ball;
- **restart:** speed change between the past and future 3 s.

Each signal is also averaged over centred 1, 4 and 12 s windows. Decoding:
- P(dead) is smoothed over 1 s;
- the threshold maximises dead-frame F1 on the *training* matches only;
- the minimum run lengths are fixed a priori: dead 2 s, live 1 s.

No Smart Tagging.

**Command:** `make dev-eval CONFIG=configs/dev_deadball.yaml`, clean tree. It is paired against E001 smooth50; the provenance check passed. Output: `results/E002_dev_metrics.json`.

**Pooled spellF1 [90% CI]** (primary labels). Δ = mean paired per-match difference vs smooth50, with the number of matches better:

| | clean | degraded | short gaps | long gaps |
|---|---|---|---|---|
| smooth50 (E001) | 0.315 | 0.227 | 0.252 | 0.214 |
| **deadball_all** | **0.432** [0.416, 0.448], Δ +0.117 (5/5) | **0.318** [0.306, 0.332], Δ +0.091 (5/5) | **0.358**, Δ +0.105 (5/5) | **0.297**, Δ +0.083 (5/5) |
| only_ball | 0.427, +0.113 (5/5) | 0.307, +0.080 (5/5) | 0.353, +0.101 (5/5) | 0.291, +0.078 (5/5) |
| no_ball | 0.360, +0.045 (5/5) | 0.273, +0.046 (5/5) | 0.311, +0.059 (5/5) | 0.259, +0.045 (5/5) |
| only_motion | 0.334, +0.019 (4/5) | 0.255, +0.027 (5/5) | 0.286, +0.033 (4/5) | 0.241, +0.026 (5/5) |
| no_motion | 0.428 | 0.315 | 0.357 | 0.292 |
| no_formation | 0.429 | 0.315 | 0.356 | 0.297 |
| no_restart | 0.437 | 0.316 | 0.355 | 0.300 |

**deadball_all, degraded, other metrics:**

| metric | deadball_all | smooth50 |
|---|---|---|
| alive/dead F1 | 0.952 | — |
| D share (true 45.2%) | 45.5% | 1.6% |
| macroF1 | 0.820 | 0.381 |
| possessions (true 2,094) | 2,687 | 3,166 |
| mean \|latency\| at τ = 25 | 10.3 frames | 12.5 frames |

**Verdict: confirmed on dev**, with the caveats below. The improvement holds in every scenario, on every match, and the intervals do not overlap.

**Ablation reading (honest):**
- **The ball group carries the gain.** only_ball reaches 0.307 vs 0.318 for all groups on degraded.
- **The other groups are unproven.** Dropping any one of motion, formation or restart changes spellF1 by ≤ 0.006, which is inside the intervals. Their combined contribution (all minus only_ball) is +0.005 to +0.011, also inside the interval. They stay in for now as cheap redundancy, but no gain is claimed for them.
- **Without the ball group the detector still helps.** Motion, formation and restart alone give +0.045 to +0.059 (5/5 matches in every scenario). That is the robustness floor if target ball behaviour differs from the simulation.

**Main risk: dependence on the simulator.**
- The ball group's strength partly reflects two things: DFL tracking keeps the ball's position when it is off the pitch, and FCB-8's assumption that an off-pitch ball goes missing 90% of the time.
- The target ATD's behaviour for dead or off-pitch balls is unpublished. If the target loses the ball far from the lines, or keeps tracking it during stoppages, the ball features transfer less well, and the no_ball floor matters more.
- This cannot be resolved on dev. Label-free target diagnostics (FCB-19), such as predicted D share and stoppage count per half, will show whether the detector behaves plausibly on the target.

**Other notes:**
- **Windowed motion (FCB-11 finding):** per-frame speeds are swamped by FCB-8's per-frame position jitter (6.1 vs 6.2 m/s dead vs alive degraded); 1 s displacement speeds recover the contrast (1.1 vs 2.3). The motion group uses windowed speeds.
- **Possession over-count remains:** 2,687 predicted vs 2,094 true on degraded. That is the next lever, the turnover rules (FCB-12).
