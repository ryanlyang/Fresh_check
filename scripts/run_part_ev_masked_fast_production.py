#!/usr/bin/env python3
"""Frozen candidate-only preflight, dispatch and production worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from part_ev_masked_fast_production import protocol as P
from part_ev_masked_fast_production.config import POLICY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('inspect','preflight','summary','review','dispatch','resume','train','evaluate','report'))
    parser.add_argument('--output',type=Path);parser.add_argument('--from-speed',type=Path)
    args=parser.parse_args()
    if args.phase=='inspect':
        if args.from_speed is None:parser.error('--from-speed required')
        result=P.inspect_speed(args.from_speed,ROOT)
        P.O.verify_host(result['original_launch'])
        print(json.dumps(result));return 0
    if args.output is None:parser.error('--output required')
    launch=P.load(args.output,worker=True,environment=args.phase not in ('summary','review'))
    if args.phase in ('summary','review'):
        result=P.review(args.output)
        if args.phase=='summary':P.write(args.output/'summary.json',P.hashed(result))
        print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
    if args.phase in ('dispatch','resume'):
        from part_ev_masked_fast_production import submission as S
        print(json.dumps(getattr(S,args.phase)(args.output),indent=2));return 0
    if args.phase=='report':
        from part_ev_masked_fast_production.reporting import report
        print(json.dumps(report(args.output,P.load_production(args.output)),indent=2));return 0
    from part_ev_masked_predictive import runtime as R
    rank,world,device,gpu=R.setup(launch)
    if gpu!=launch['gpu']:raise ValueError('GPU/driver differs from admitted speed study')
    if 'SLURM_JOB_END_TIME' in os.environ:
        R.DEADLINE=time.monotonic()+max(0,float(os.environ['SLURM_JOB_END_TIME'])-time.time()-POLICY['shutdown_margin'])
    P.check_storage(args.output)
    if args.phase=='preflight':
        from part_ev_masked_fast_production.preflight import run
        print(json.dumps(run(args.output,launch,device,gpu),indent=2));return 0
    from part_ev_masked_fast_production.execution import engine
    spec=P.load_production(args.output);E=engine();E.verify_runtime(spec,gpu)
    if args.phase=='train':E.train(args.output,spec,'masked',device,rank,world)
    else:E.evaluate(args.output,spec,'masked',device)
    return 0


if __name__=='__main__':raise SystemExit(main())
