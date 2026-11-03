# FCB-5: dev-label eligibility audit (DFL / IDSSE), revision 2

Run: 2026-11-03. Code commit: see `git_commit` in `dev-labels-audit.json`; the run is from a clean tree.

- **Split:** `configs/split.yaml`. **Dev matches only**: J03WN1, J03WOH, J03WPY, J03WQQ, J03WR9. The holdout matches (J03WMX, J03WOY) were not opened.
- **Reproduce:** `PYTHONPATH=src .venv/bin/python scripts/audit_dfl_labels.py`. This one command writes every number below to `dev-labels-audit.json`, including the proxy sensitivity, with split hash and git provenance.

Revision 1 claimed DFL possession is "last-touch" and accepted a 2 s filter as the reference. Codex's review showed that conclusion was not supported, and it is **withdrawn**.

## Verdict

| Item | Verdict |
|---|---|
| DFL `BallStatus` → D | **Accept.** Internally consistent with DFL restart events (§3) |
| DFL `BallPossession` → H/A | **Accept raw flags as the primary reference** for other-match development. This is defensible from the source definition, but it is not measured equivalence to Metrica (§4) |
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

Denominators: there are **1,599 live H↔A switches** across the 5 dev matches. 1,410 fall into the two groups below. The other **189 are excluded**: spells under 2 s that do not flip back to the previous team (they end in a D, an X or the period end).

Pooled over the 5 dev matches:

| switch type | n | control by new team | duel without possession change | no relevant event |
|---|---|---|---|---|
| held ≥ 2 s | 1081 | 29.5% | 15.2% | 55.3% |
| short, flips back < 2 s | 329 | 23.7% | 27.7% | 48.6% |

**Interpretation (exploratory DFL-internal associations only):**
- Short flip-back switches co-occur with a no-change duel in the window more often than held switches (27.7% vs 15.2%). Their control-event rate is only modestly lower (23.7% vs 29.5%).
- This is an **association between event-window categories, not a label-error estimate or bound**, for four reasons:
  - a nearby no-change duel need not describe that switch;
  - windows can reuse the same event;
  - "control" takes precedence when both types occur;
  - the held group is not a known-correct baseline.
- About half of all switches have no relevant event, so the error frequency is not identified.
- The association does not support a blanket duration filter as truth, because it would also erase spells with control events.

**Label and convention uncertainty remains unquantified**, both for raw-flag vs challenge rules and for DFL vs Metrica.

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
