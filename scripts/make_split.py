#!/usr/bin/env python3
"""Write configs/split.yaml (JSON subset of YAML). Refuses to overwrite a different split."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.split import SPLIT, build

body = build()
text = json.dumps(body, indent=2, sort_keys=True) + "\n"
if SPLIT.exists() and SPLIT.read_text() != text:
    sys.exit("configs/split.yaml exists and differs; changing the split needs owner approval")
SPLIT.write_text(text)
print("holdout:", ", ".join(body["holdout"]))
print("dev:", ", ".join(body["dev"]))
