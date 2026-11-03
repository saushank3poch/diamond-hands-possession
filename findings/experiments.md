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

## E001: Nearest-player baseline on dev (2026-11-03, commit e373162, run `run-05u_3h40`)

**Hypothesis:** none. This is the reference point every later experiment is compared against, not a claim.

**Setup:**
- 5 DFL dev matches, leave-one-match-out (the baseline is untuned, so fitting is a no-op).
- Official scorer, pooled over matches, with the 1000-draw bootstrap (90% interval).
- Primary labels are the raw DFL flags. The 2 s proxy is shown for transparency only.
- Command: `make dev-eval CONFIG=configs/dev_baseline.yaml`.
- Full output: `results/E001_dev_metrics.json`; provenance: `results/E001_run.json`.

Primary labels, pooled spellF1 [90% CI]:

| variant | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|
| clean | 0.097 [0.093, 0.101] | 0.301 [0.289, 0.312] | **0.314** [0.299, 0.328] | 0.218 [0.204, 0.234] |
| degraded (default) | 0.018 [0.017, 0.020] | 0.177 [0.169, 0.184] | **0.227** [0.216, 0.238] | 0.196 [0.184, 0.209] |
| degraded, short gaps | 0.012 | 0.200 | **0.251** | 0.205 |
| degraded, long gaps | 0.022 | 0.180 | **0.213** | 0.197 |

**Other metrics, degraded default, smooth50:**
- macroF1 0.381.
- Possessions predicted 3,164 vs 2,094 true.
- Predicted D share 1.6% vs 45.2% true.
- Mean |latency| at τ = 25: 12.5 frames.

**Reading:**
- The robustness band across gap assumptions for smooth50 is 0.213–0.251. A future gain smaller than that spread may reflect the degradation assumption rather than the method.
- The baseline almost never outputs D (1.6% vs 45%). That makes dead-ball detection (FCB-11) the largest open lever, consistent with the plan.
- macroF1 is not comparable across variants here: raw-degraded scores 0.557 only because missing-ball frames become D, and those correlate with real stoppages.
- **Observation only, not used for anything:** the dev ordering (smooth50 > smooth125 > smooth25 > raw on degraded) happens to match the organisers' published full-match ordering. Per FCB-8, nothing was calibrated toward it, and it will not be used to adjust the degradation.

**Proxy labels (2 s), degraded, smooth50:** 0.241. Shown for transparency only; not used for selection.
