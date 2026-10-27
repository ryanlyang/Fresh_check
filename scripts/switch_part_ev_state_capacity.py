#!/usr/bin/env python3
"""Read-only switch plan by default; --apply explicitly cancels scoped pending jobs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_state_capacity import protocol as P
from part_ev_state_capacity.migration import Slurm, replacement_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, default=P.DEFAULT_CONTROLS)
    parser.add_argument('--output', type=Path, help='New campaign; required for dependency verification')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true')
    group.add_argument('--dependencies', action='store_true')
    args = parser.parse_args(argv)
    if args.dependencies:
        if args.output is None:
            parser.error('--dependencies requires --output')
        spec = P.load_campaign(args.output)
        args.controls = Path(spec['controls']['path'])
    source, contract = P.control_contract(args.controls)
    result = replacement_status(args.controls, source, contract, Slurm(),
                                apply=args.apply, require_switched=args.dependencies)
    print(':'.join(result['kept_dependencies']) if args.dependencies else json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
