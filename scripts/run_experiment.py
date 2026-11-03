#!/usr/bin/env python3
"""Run a trusted local command with a configuration and provenance record."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.runs import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaffold.yaml")
    parser.add_argument("--mode", choices=("dev", "target"), default="dev")
    parser.add_argument("--run-dir")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    code = run(args.config, command, args.mode, args.run_dir)
    if args.mode == "dev":
        print("DEV_RUN {} exit_code={}".format("PASS" if code == 0 else "FAIL", code))
    return code


if __name__ == "__main__":
    sys.exit(main())
