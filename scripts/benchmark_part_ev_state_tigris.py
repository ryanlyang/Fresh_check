#!/usr/bin/env python3
"""Prepare, measure and review one GH200; no train/cancel/install modes."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_state_tigris import protocol as P


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','benchmark','review'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--donor',type=Path)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.task_index = 0
    if args.phase == 'prepare':
        if args.donor is None: parser.error('--donor required')
        with P.file_lock(args.output/'.prepare.lock',blocking=False):
            P.prepare(args.output,args.donor)
        print('Independent Tigris benchmark prepared. SPORC files/jobs unchanged.',flush=True)
    elif args.phase == 'review':
        spec = P.load_campaign(args.output,check_environment=False)
        print(json.dumps(P.review_view(args.output,spec),indent=2,allow_nan=False))
    else:
        from part_ev_state_tigris.benchmark import run
        try:
            run(args)
        except BaseException as error:
            P.write_json(args.output/'benchmark_error.json',dict(
                authorizes_training=False,error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc()))
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
