# Tracking feature cache (FCB-10)

`possession.features.compute(tracking, Config())` produces 50 fixed columns at
25 Hz. Its only input is the common label-free Tracking schema. It does not
load reference labels, event files, Smart Tagging or public GT. This ticket adds
observations for later detectors, not a possession model or a measured gain.

## Feature definitions

- **Ball:** centred metric position, backward velocity (m/s), acceleration
  (m/s²), raw observed/missing flags, interpolated flag, position/velocity
  validity and time since last observed ball within the period.
- **Quality:** ball 1 for observed, 0.5 for interpolated and 0 for unavailable;
  team quality is min(observed slots / 11, 1). These are availability indicators,
  not calibrated confidence probabilities. Counts can include duplicate tracks.
- **Team geometry:** centroid, width (y span), depth (x span), minimum/maximum x,
  RMS distance from centroid (compactness), observed count. Attacking/defending
  line coordinates use the supplied attack direction, with increasing x toward
  that team's goal of attack. They stay NaN when direction is unknown; no target
  direction is inferred by hand. Min/max lines are extents, not inferred tactical
  defensive lines.
- **Team motion:** mean per-slot speed and acceleration magnitude over valid
  derivatives, with the valid velocity count. Acceleration requires three
  consecutive valid position samples. Positions are never discarded merely
  because the motion guard rejects a derivative.
- **Ball/team relation:** minimum distance to each team and velocity-vector
  difference magnitude between the ball and that frame's nearest team slot
  (0 means matching velocity). All-missing teams yield NaN, not distance zero.
- **Bounds:** out-of-bounds is 1 only beyond the physical pitch rectangle;
  exiting is 1 when currently inside and constant-velocity projection crosses
  a boundary within the configured horizon. Missing evidence yields NaN.
  These are geometry indicators, not dead-ball decisions.

## Missingness and assumptions

Only bounded ball gaps of at most 5 frames (0.2 s) are linearly interpolated.
Both endpoints must be observed within the same active period. There is no
leading/trailing fill or interpolation through period 0/halftime. Interpolation
is **offline**, using a future endpoint, and remains explicitly flagged.

Derivatives use adjacent finite samples in the same active period. A slot gap,
period boundary or excessive displacement invalidates velocity; invalid velocity
also breaks acceleration. Player speed above 12 m/s and ball speed above 60 m/s
are rejected for derivatives, not clipped. These limits, gap length, quality
weights and 0.2 s exit horizon are prespecified engineering assumptions, not
selected by scores. Gap, speed limits and exit horizon are in
`configs/features.yaml`; quality weights are fixed in the documented code. FCB-9/component
experiments must assess any sensitivity. Small identity swaps can still pass the
speed guard. No persistent player identity or reliable acceleration is claimed
for such swaps.

Period-0 geometry/motion is unavailable (NaN), counts/availability are zero.
The raw ball-missing flag is retained even when interpolation supplies a position.
Consumers must use validity/quality fields rather than treating NaN as zero.

## Cache and reproducibility

`possession.feature_cache.cache_features(tracking, path, cfg)` returns an Arrow
table and a cache-hit flag. The Parquet metadata binds the cache to every Tracking
array, feature config, source code and dependency lock. Changed input, config,
code or lock rebuilds it. Expected metadata plus an invalid frame axis or extra
columns raises an error. Writes are atomic; caches contain only the declared
columns. Parquet uses nulls for missing values (returned as NaN in NumPy), so
round-trip equality is well-defined. This is stale-cache detection, not an
adversarial integrity/signature system.

PyArrow 26.0.0 (Apache-2.0, https://arrow.apache.org/docs/python/) is pinned for
Parquet IO. Its installed table/Parquet API was exercised by synthetic round-trip
and invalidation tests. All artifacts stay in gitignored local run directories.

## Commands

Install with `make setup`; test with `make test`. Use the generic harness so config,
Git/dependency hashes and stdout/stderr are recorded:

```sh
.venv/bin/python scripts/run_experiment.py --config configs/features.yaml -- \
  .venv/bin/python scripts/cache_features.py --source dev --config configs/features.yaml
```

This creates separate clean/degraded caches for only the five reserved dev IDs,
using the reviewed deterministic per-match degradation seed rule. No references
or holdout records are read. The local `features-run.json` records cache shape and
elapsed time; the run harness records code/config provenance.

Target execution, including cache reads, must use containment:

```sh
PYTHON=.venv/bin/python sh scripts/run_target_local.sh configs/features.yaml -- \
  .venv/bin/python scripts/cache_features.py --source target --config configs/features.yaml
```

The target command has a 600-second alarm around adapter loading, feature
computation, cache write/read and round-trip verification. PASS means those steps
finished within the bound. The only result returned to agents is wrapper status;
never read target logs, metadata, runtime JSON or Parquet back through agent tools.
This does not measure the future final model pipeline, which is not implemented.
`make submission` remains a deliberate failure until its later ticket.
