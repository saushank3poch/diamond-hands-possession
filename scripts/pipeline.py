#!/usr/bin/env python3
"""Reserved entry points. FCB-1 does not evaluate a model or create predictions."""

import argparse
import os
import sys

from possession.runs import seed_generators

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("dev-eval", "submission"))
args = parser.parse_args()
seed_generators(int(os.environ["POSSESSION_SEED"]))
print("{} is not implemented in FCB-1; no data was read.".format(args.stage), file=sys.stderr)
sys.exit(2)
