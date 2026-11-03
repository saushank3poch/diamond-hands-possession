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
