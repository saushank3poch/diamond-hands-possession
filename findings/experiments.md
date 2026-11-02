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
