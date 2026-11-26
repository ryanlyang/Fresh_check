#!/usr/bin/env python3
"""Frozen GH200 speed diagnostic; no production entrypoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from part_ev_masked_speed import protocol as D


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('inspect','benchmark','summary','review'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    if args.phase=='inspect':
        if args.reference is None: parser.error('Inspect requires --reference')
        launch,spec,review,files=D.donor(args.reference,ROOT)
        D.P.verify_host(launch)
        print(json.dumps(dict(launch=launch,donor_files=files,review=review)))
        return 0
    if args.output is None: parser.error('Existing --output required')
    launch=D.load(args.output,worker=True)
    for name,expected in launch['donor_files'].items():
        if D.P.sha(Path(launch['reference'])/name)!=expected: raise ValueError('Donor evidence changed')
    if args.phase in ('summary','review'):
        result=D.summarize(args.output)
        if args.phase=='summary':D.P.write(args.output/'summary.json',D.P.hashed(result))
        print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
    original,spec,review,files=D.donor(launch['reference'],ROOT)
    if files!=launch['donor_files'] or original['resources']!=launch['resources']:
        raise ValueError('Diagnostic reference differs')
    D.check_storage(args.output)
    from part_ev_masked_predictive import runtime as R
    from part_ev_masked_speed.benchmark import run
    rank,world,device,gpu=R.setup(launch)
    if world!=1 or gpu!=review['runtime']['gpu']:
        raise ValueError('One GH200 with the donor hardware/driver identity is required')
    run(args.output,launch,spec,review,gpu,device)
    result=D.summarize(args.output)
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
