# Methodology, data provenance and rule compliance

Reviewed: 2026-11-02, against upstream commit `f8f2629` (verified against official upstream main as of 2026-11-02).

This file is the source for the technical document's "training data", "assets used" and "reproduction" sections. [`SUBMISSION.md`](../SUBMISSION.md) §3 requires every labelled dataset to be listed with its source, who labelled it and how much. None of it may come from the provided game.

## 1. Upstream rule changes we must design around

On 2026-10-22/23, two commits (`c073f29`, `f8f2629`) tightened the rules. Here is what changed and what it means for us:

| Change (source) | Before | Now | Consequence for us |
|---|---|---|---|
| Hand labels ([`RULES.md`](../RULES.md) §1) | "Small amounts of hand-labelled data used to train or validate a model are fine" | **No hand labels on the provided game, in any amount, for any purpose, including validation.** Hand labels on other games are fine in any amount. | No validation set may be built by hand on our match. Our dev set must come from other games. |
| Corrections (§1) | Not stated | Hand-correcting the provided tracking (ball positions, identity switches) is hand annotation | All tracking repair (gap filling, ID stitching) must be code |
| Frame-tied rules (§1) | Not stated | No "rules and constants tied to specific frames of the game" | No manually observed restart, stoppage or possession frames. The organisers' published half boundaries ([`DATA.md`](../DATA.md) "The two halves": "boundaries, not answers") are metadata and may be used as published. Restarts and stoppages within a half must be detected by code |
| Public 5-min GT (§2, [`EVALUATION.md`](../EVALUATION.md), [`FAQ.md`](../FAQ.md)) | Advice not to rank on it | **Rule:** "not training data and not a tuning set" | Correctness checks only: format, alignment, label semantics |
| Human data about the game (§2) | "other source of corrected possession data" | Any human-produced data about the provided game (event data from Opta, StatsBomb, Wyscout, LaLiga or others; corrected tracking; annotations) is banned | Never pull public event feeds for Barça–Athletic |
| External data (CHALLENGE, OVERVIEW, FAQ) | "External data" | "External data **about other games**" | Metrica sample games are allowed and must be documented |
| Pseudo-labels (§1) | Not stated | Allowed if "produced by an algorithm, provided no person decided any of them" | Our pseudo-label rule must be fully specified in code and documented |

Unchanged and still binding: [`RULES.md`](../RULES.md) "The data" says not to redistribute the data or publish derived data that would reconstruct it. The submission repository must be public, so `data/` stays ignored (`.gitignore` covers `data/*` except `data/README.md`).

## 2. Third-party LLM clause: open question for the owner

[`FAQ.md`](../FAQ.md) cites the Terms and Conditions: participants "shall not upload, disclose or otherwise make available to any third-party hosted or cloud-based LLM any videos, datasets, confidential information or other restricted materials … or any extracts thereof."

We are cloud-hosted LLM agents. Our current control is hard rule 1: no row of anything under `data/` ever enters a conversation. Match processing and any resulting diagnostics remain local until the unresolved disclosure question below is settled.

**Decision (saushank, 2026-11-02):** all match analysis stays local. Diagnostics computed on the match are not returned to the agents, and only the final script is published at the end. The original question is kept below for the record.

**Original question:** do aggregate diagnostics (possession count, D share, ball-visible share) computed on the match count as "extracts"? Two facts bear on it:

- They cannot reconstruct the data.
- The organisers publish the same kinds of numbers themselves (for example, ball visible on 56.9% of frames in [`DATA.md`](../DATA.md)).

Per the decision above, do not return match diagnostics to these agents; the documentation review can proceed using public documentation and synthetic examples. Everything we tune is tuned on the dev set (other games). The final terms you accepted at enrolment are the authority; we have not read them.

## 3. Data inventory and provenance

| Asset | Origin | Human-produced? | Allowed use | Status |
|---|---|---|---|---|
| Tactical ATD tracking, our match | Metrica automatic tracking | No | Model input | In `data/` (not opened in chat) |
| Smart Tagging, our match | Metrica automatic tagging ("not ground truth and not derived from it", FAQ) | No | Model input and priors | In `data/` |
| Homography, our match | Provided | No (assumed automatic; to confirm in DATA.md) | Model input | In `data/` |
| Broadcast tracking | Provided | No | Out of scope for now | `../broadcast/` |
| Public 5-min GT (frames 11185–18684) | Organisers, from corrected tracking | **Yes** | Correctness checks only | `evaluation/gt/` |
| Private full-match GT | Organisers | Yes | Never | Not available |
| Metrica open sample games (`github.com/metrica-sports/sample-data`) | Metrica; tracking plus manually produced event data | Events: yes, but **other games** | Secondary dev source, once its conversion is audited (FCB-6) | Cloned 2026-11-02 at `e706dd5` to `../external/metrica-sample-data` (233 MB). 3 anonymised games: G1/G2 CSV tracking plus events, G3 EPTS plus JSON events. No licence file; the README asks users to "be responsible" and to "acknowledge the source". Events are not ready-made H/A/D (touch-start possession, RECOVERY includes blocks, BALL OUT timing): see the Codex audit notes |
| **DFL / IDSSE open data** (Bassek et al. 2025, *Sci Data* 12:195; figshare DOI 10.6084/m9.figshare.28196177, v1) | TRACAB Gen5 automatic tracking, then post-processing and operator QC of identities (DFL); events manually annotated by DFL operators | Positions: automatic with QC; per-frame `BallStatus`/`BallPossession`: DFL-produced; events: human. All about **other games** | Primary dev source, pending the FCB-5 label audit | Retrieved 2026-11-03 to `../external/idsse` (2.6 GB, 21 files: 7 matches × match info, raw events, observed positions; 2 Bundesliga and 5 2. Bundesliga matches, 2022/23). All 21 MD5s match figshare. SHA-256 manifest: `configs/external/idsse_manifest.json`. CC BY 4.0: cite the paper in the technical doc |

Every dev-set label we derive (H/A/D from sample-game events) is human-produced data about other games. That is allowed, but the technical document must state, per §3 of SUBMISSION: source (Metrica sample-data, pinned commit), who labelled it (Metrica's annotators, as published), and how much (games, frames, possessions). We record those figures here when the dev set is built.

## 4. Validation protocol (proposed; nothing below is implemented yet; per saushank's standing instruction, changes need owner approval)

1. **Dev labels (proposed, not implemented).** Derive per-frame H/A/D from sample-game events. Before claiming these are usable possession labels, check their semantics (whether events mark ball-out, restarts and possession changes consistent with [`CHALLENGE.md`](../CHALLENGE.md)'s rules) and their licence. The derivation rule will be documented in code and in this file.
2. **Degradation.** Make the sample-game tracking look like ours:
   - drop the ball on about 41% of frames, in bursts biased toward crowded or contested moments;
   - inject identity switches;
   - fragment tracks;
   - add position jitter.

   Calibrate to published statistics from [`DATA.md`](../DATA.md) only (56.9% visible across the whole recording versus roughly 59% on graded frames; keep these denominators distinct; burst structure as described there). Do not calibrate to anything computed from our match beyond what DATA.md publishes, so that our match data never becomes a tuning target.
3. **Scoring.** Use the official `evaluation/possession_metrics.py` on the dev set, with `--bootstrap 1000`. The official interval is the 5th–95th percentile (a 90% interval; `BOOTSTRAP_PERCENTILES = (5, 95)`). Report spellF1, macroF1, possession count vs truth, D share and boundary latency. Overlapping intervals mean no claim (saushank's standing rule). Non-overlap is necessary, not sufficient: individual intervals are not a formal test of the paired improvement. The ~0.04 spellF1 noise band is the organisers' full-match reference estimate, not a universal cutoff; report dev-set uncertainty; an improvement claim needs a suitable paired comparison and an explicit independent-match split.
4. **Our match.** Track label-free diagnostics only (possession count, mean duration, D share, flip rate, agreement with Smart Tagging phases). They exist to catch broken pipelines and are never objectives.
5. **Public GT.** Format, alignment and label-convention checks only, never model selection.

Known risk: the sample games are a different competition, camera and tracking system, so dev gains may not transfer one-for-one. We mitigate by matching degradation to DATA.md's published statistics and by preferring rules whose logic does not depend on a specific tracking vendor.

## 5. Pseudo-label provenance (to fill in when built)

For the ball-free owner classifier, record:

- the exact rule (for example: ball visible, one player within ~1 m for N frames, no opponent within R m), with its constants;
- the fact that the constants were chosen on the dev set, never on our match or the public GT;
- the counts produced: pseudo-labelled frames per class, as aggregates only.

## 6. Technical document map

| SUBMISSION §3 section | Source in `findings/` |
|---|---|
| Approach | `experiments.md` verdicts, plus the architecture section to be written (detectors, pseudo-label classifier, HSMM decoder: all proposed, none implemented) |
| Assets used and dropped | §3 above, plus experiment ablations |
| Training data | §3 and §5 above |
| Possession rules | To be written as the detectors land (loose balls, challenges, restarts) |
| Limitations | `baseline-analysis.md` failure table, plus residual dev-set errors |
| Reproduction | Planned, not built: `make submission` from a clean clone; pinned `requirements.lock`; fixed seeds |
