# Tracking adapters (FCB-4)

## Contract

`possession.tracking.Tracking` contains only frame IDs, periods, slots, team
membership, coordinates, pitch dimensions and home attack direction. Frame is
contiguous from zero at 25 Hz. `source_frame` retains the vendor counter. Player
slots are observation slots, not a promise of persistent identity. Coordinates
are centred metres, +x right and +y up. Off-pitch coordinates are retained.
Missing or partially missing/nonfinite observations become two NaNs, with explicit
`player_missing` and `ball_missing` properties. There is no interpolation.

ATD uses `frame_id - 1`, retains all video rows, and uses published half blocks
(period 0 outside them). DFL concatenates the observed periods in order, retaining
all frame-counter gaps within each period as missing observations. No time is
invented between periods; `period` is the reset boundary and `source_frame` is the
join key. DFL period spans use the union of player and ball observations, never
ball status/ownership or reference masks. This is a 25 Hz adapter, not a resampler.

Both adapters keep a fixed pitch orientation. `home_direction` is +1/-1 for
home attacking right/left, or 0 when unknown. Optional per-period directions must
come from observation metadata or an automatic geometry method. Neither adapter
guesses from labels or assumes that a missing direction means right.
`home_attacking_coordinates()` rotates both axes for leftward periods and fails
if any active period's direction is unknown. A later geometry feature can supply
an automatic direction rule; no target direction has been manually annotated.

NPZ persistence is versioned by its exact field allowlist, without pickle or
arbitrary metadata. Arrays are copied and read-only on construction. Missing flags
are derived from coordinates so they cannot disagree. Use only ignored local
output paths for real-match archives.

## Separation of references

`possession.evaluation.references.load_dfl_references` is an evaluator-only pass.
It returns alive (-1/0/1), owner (X/H/A), an unknown mask, and the original raw
status/owner strings. Unknown values never default to away or dead. Dead frames
can be known even with unknown owner. These are provisional vendor conventions,
not a conclusion that DFL possession is eligible for challenge scoring. No H/A/D
answer key is generated; FCB-5 owns that decision after FCB-7 reserves holdout IDs.

Feature/model modules belong under `src/possession/` outside `adapters/` and
`evaluation/`. Tests reject imports of either privileged package or kloppy in
those modules, and reject known label-column/attribute accesses. Scripts may
orchestrate input/evaluation separately. This static check is a regression guard,
not a security sandbox against dynamic imports. All real models must extend the
synthetic input-invariance test when introduced. The current test uses a trivial
symbol-output function, not a possession algorithm.

## Loaders and verified upstream behavior

DFL uses the metadata parser from **kloppy 3.19.1**, BSD-3-Clause:
[Sportec metadata implementation](https://github.com/PySport/kloppy/blob/v3.19.1/kloppy/infra/serializers/event/sportec/deserializer.py).
That internal API is pinned and exercised by a synthetic XML fixture. The
[tracking implementation](https://github.com/PySport/kloppy/blob/v3.19.1/kloppy/infra/serializers/tracking/sportec/deserializer.py)
drops frames without ball records, substitutes zero for absent X/Y, and maps
unrecognized possession values to away. Consequently we stream only N/X/Y and
object/period identities ourselves, rather than using its full tracking loader.
References are read separately; the coordinate reader never yields those fields.

Coordinate conventions were checked against kloppy's
[coordinate systems](https://github.com/PySport/kloppy/blob/v3.19.1/kloppy/domain/models/common.py):
Sportec is centred metric +y up; Metrica is normalized top-left +y down. The ATD
reader follows this repository's `starter/baseline.py` fixed x,y,speed slot format
and converts with `(x-.5)*length, (.5-y)*width`. It is deliberately not a generic
EPTS reader: malformed rows, unexpected slot counts and nonsequential IDs fail.

No raw external match records or target assets were parsed during FCB-4. Tests
construct all XML/text in temporary folders. Actual source compatibility remains
to be checked in the subsequent permitted data stages; library support alone is
not evidence of compatibility or label eligibility. No video/model dependencies
or unused Parquet dependency are installed; NPZ is sufficient for this schema.

## Usage

From the repository root, after `make setup`:

```python
# Other matches only; reserve holdouts before exploring reference labels.
from possession.adapters.dfl import load_dfl
tracking = load_dfl(meta_path, positions_path)
tracking.save(local_output_path)
```

Set `PYTHONPATH=src` for direct development commands; the run harness does this.
For target processing, use only the local wrapper (the loader checks its runner
markers before opening files):

```sh
PYTHON=.venv/bin/python sh scripts/run_target_local.sh configs/scaffold.yaml -- \
  .venv/bin/python -c 'import os; from pathlib import Path; from possession.adapters.atd import load_target; load_target().save(Path(os.environ["POSSESSION_RUN_DIR"]) / "tracking.npz")'
```

This is a future execution command, **not run as part of this ticket**. It emits
only wrapper status. Never inspect its local logs or archive through agent tools.
The marker check prevents accidental bypass; it is not an adversarial sandbox.
