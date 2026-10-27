#!/usr/bin/env python3
"""Read-only: audit selected full-schedule controls and print active dependencies."""
import argparse
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_state_compare import protocol as P
from part_ev_state_capacity import migration as historical

# Isolate the two-control view; do not alter the four-control historical switch.
_spec = importlib.util.spec_from_file_location('part_ev_state_capacity._compare_dependencies', historical.__file__)
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)
verifier.P = P


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = P.load_campaign(args.output)
    root = Path(spec['controls']['path'])
    source, contract = P.control_contract(root)
    print(':'.join(verifier.kept_dependencies(root, source, contract, verifier.Slurm())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
