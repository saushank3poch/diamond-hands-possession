# FCB-5: dev-label eligibility audit (DFL / IDSSE), revision 2

Run: 2026-11-03. Code commit: see `git_commit` in `dev-labels-audit.json`; the run is from a clean tree.

- **Split:** `configs/split.yaml`. **Dev matches only**: J03WN1, J03WOH, J03WPY, J03WQQ, J03WR9. The holdout matches (J03WMX, J03WOY) were not opened.
- **Reproduce:** `PYTHONPATH=src .venv/bin/python scripts/audit_dfl_labels.py`. This one command writes every number below to `dev-labels-audit.json`, including the proxy sensitivity, with split hash and git provenance.

Revision 1 claimed DFL possession is "last-touch" and accepted a 2 s filter as the reference. Codex's review showed that conclusion was not supported, and it is **withdrawn**.

## Verdict

| Item | Verdict |
|---|---|
| DFL `BallStatus` → D | **Accept.** Internally consistent with DFL restart events (§3) |
| DFL `BallPossession` → H/A | **Accept raw flags as the primary reference**, with a measured caveat (§4) |
| 2 s confirm-control filter | **Experimental proxy only.** Never the sole basis of a model decision; reported for transparency |
| DFL ↔ Metrica convention | **Unmeasured.** Neither dead-ball timing nor the possession definition can be compared without Metrica-side truth |
| Metrica, SkillCorner | Not audited here (FCB-6). Metrica Game 1 events were already inspected, so that game is not untouched |

The primary reference is `possession.evaluation.labels.reference_labels`: the raw flags, with the unknown mask applied as X.

## Evidence (5 dev matches)

### 1. Integrity
- Value sets: `BallStatus` ∈ {0, 1} and `BallPossession` ∈ {1, 2}. Unknown share 0.0.
- 137k–147k frames per match, in 2 periods.

### 2. Home/away
- The flag value 1 is the match-info `HomeTeamId` side in 5/5 matches.
- Where a player is within 1.5 m of the ball in live play, the flag names that player's team on 87–92% of frames.

### 3. Dead ball: DFL internal consistency only
Restart events (kick-off, throw-in, free kick, goal kick, corner, penalty, referee ball) were paired one-to-one with dead→alive transitions, greedy by |Δt| within 5 s.

| | per match |
|---|---|
| matched | 89–100 |
| unmatched events | 2–4 |
| unmatched transitions | 0–5 |
| median signed latency (transition − event) | −1.25 to +1.98 s |

The latency sign varies by match. That is consistent with the paper's note that manual event timing is offset (−0.37 ± 1.82 s), so this is **not a measure of flag timing error**. It is also **not a DFL–Metrica offset**.

Same-team restarts: 24–43% of dead intervals have the same team on both sides. D share is 41–52% of period frames.

### 4. Possession: does the raw flag follow control?

**Source definition:** the IDSSE paper states that "the ball possession is defined by the player in control of the ball" (Bassek et al. 2025, Position data).

**Independent check:** DFL's human event stream, which is produced separately from the tracking flags.
- At each live H↔A switch, I looked for events in a fixed window [−2 s, +1 s] around the switch start, after aligning each match's event clock by its restart median latency.
- **Control** = the new team has a `Play`, `ShotAtGoal` or `BallClaiming` event, or wins a `TacklingGame` with `PossessionChange="true"`.
- **Duel without change** = a `TacklingGame` with `PossessionChange="false"`.

Pooled over the 5 dev matches:

| switch type | n | control by new team | duel without possession change | no relevant event |
|---|---|---|---|---|
| held ≥ 2 s | 1081 | 29.5% | 15.2% | 55.3% |
| short, flips back < 2 s | 329 | 23.7% | 27.7% | 48.6% |

**Interpretation:**
- Short flip-back switches coincide with a no-change duel more often than held switches: an excess of 12.5 points, about 41 switches, or ≈3% of all 1410 live switches.
- Their control-event rate is only modestly lower.
- So most short spells are not distinguishable from brief controlled possession by this evidence, and a blanket duration filter would also erase legitimate short control. That is why the filter is not the reference.

**Caveat:** about 50% of switches have no relevant event within ±2 s. Manual event timing jitter is comparable to short spell lengths, so this check bounds the problem rather than resolving it. **Estimated raw-flag disagreement with the challenge's "no change on a touch or duel" rule: on the order of 3% of live switches. Unverified against Metrica.**

## Experimental proxy (not a reference)

`confirm_control(τ)`: a live switch counts only if the new team holds the flag for at least τ. It is applied after the unknown mask, which stays X and breaks state (regression tests in `tests/test_reference_labels.py`).

Possessions per 5 min (raw: 18.9–26.5):

| τ | possessions / 5 min |
|---|---|
| 1 s | 16.3–21.0 |
| 2 s | 13.6–16.7 |
| 3 s | 11.3–14.1 |

FCB-9 may report proxy scores alongside the primary, but selection uses the primary reference. Matching a confirmation-window predictor to a confirmation-window reference would reward the imposed rule.

## Not material / deferred
- **DFL y-axis sign vs ATD:** not material. Possession features use distances, pitch bounds and the per-period attack direction, all invariant to a y mirror.
- **`BallPossessionPhase`:** event phase counts (896–1120 per match) use a different granularity from flag possessions (370–500). Recorded but not interpreted.
