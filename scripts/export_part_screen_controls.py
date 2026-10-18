#!/usr/bin/env python3
"""Refresh verified control references for an already prepared grouped campaign."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_group_screen.protocol import load_campaign
from part_screen_common.controls import refresh_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = load_campaign(args.output)
    print(json.dumps(refresh_controls(args.output, spec["donor"]), indent=2))


if __name__ == "__main__":
    main()
