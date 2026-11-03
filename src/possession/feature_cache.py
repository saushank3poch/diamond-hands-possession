"""Content-keyed Parquet cache for tracking-only feature tables."""
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from possession.features import Config, FEATURE_COLUMNS, compute
from possession.tracking import Tracking


def fingerprint(tracking, cfg):
    digest = hashlib.sha256()
    for f in fields(Tracking):
        a = getattr(tracking, f.name)
        digest.update(json.dumps([f.name, a.dtype.str, a.shape]).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    root = Path(__file__).resolve().parents[2]
    source = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("features.py"),
                 Path(__file__).with_name("tracking.py"), root / "requirements.lock"):
        source.update(path.read_bytes())
    return {"version": 1, "input_sha256": digest.hexdigest(), "code_sha256": source.hexdigest(),
            "config": asdict(cfg), "fps": 25, "smart_tagging": False}


def cache_features(tracking: Tracking, path, cfg: Config = Config()):
    """Return (table, cache_hit); rebuild on input/config/code/dependency changes.

    Callers must keep match caches in ignored local storage. Target callers must
    execute through the containment wrapper, including reads of existing caches.
    """
    path = Path(path)
    identity = json.dumps(fingerprint(tracking, cfg), sort_keys=True).encode()
    if path.exists():
        table = pq.read_table(path)
        if (table.schema.metadata or {}).get(b"possession_features") == identity:
            if tuple(table.column_names) != FEATURE_COLUMNS:
                raise ValueError("unexpected feature cache columns")
            if table.num_rows != len(tracking.frame) or not np.array_equal(
                    table["frame"].to_numpy(), tracking.frame):
                raise ValueError("invalid feature cache frame axis")
            return table, True
    columns = compute(tracking, cfg)
    if tuple(columns) != FEATURE_COLUMNS:
        raise ValueError("feature schema differs from declared columns")
    # Arrow nulls preserve missingness and compare equal after a Parquet round-trip.
    table = pa.table({key: pa.array(value, from_pandas=True) for key, value in columns.items()})
    table = table.replace_schema_metadata({b"possession_features": identity})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".features-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return table, False
