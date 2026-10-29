#!/usr/bin/env python3
"""One seed-101 layerwise State-EV run on two A100s; original jobs stay untouched."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_layerwise_ddp import protocol as P


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','preflight','review','train','evaluate','report'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--scaling',type=Path)
    parser.add_argument('--reviewed',action='store_true')
    parser.add_argument('--approve',action='store_true')
    args=parser.parse_args(argv)
    args.output=args.output.resolve()
    if args.phase=='prepare':
        if args.scaling is None: parser.error('--scaling is required')
        P.prepare(args.output,args.scaling)
        print('Prepared fresh two-GPU layerwise training. No old campaign changed.')
        return 0
    spec=P.load(args.output,check_environment=args.phase not in ('review','report'))
    if args.phase=='review':
        result=P.approve(args.output,spec,args.reviewed) if args.approve else P.require_preflight(args.output,spec)[1]
        print(json.dumps(result,indent=2))
        return 0
    from part_ev_layerwise_ddp import engine as E
    if args.phase=='report':
        P.require_approval(args.output,spec)
        E.report(args.output,spec)
        return 0
    P.G.C.verify_inventory_metadata(spec)
    world=1 if args.phase=='evaluate' else 2
    rank,world,device=E.B.setup(world)
    try:
        runtime=E.runtime_record(spec,device,world)
        if args.phase=='preflight': E.preflight(args.output,spec,rank,world,device,runtime)
        elif args.phase=='train': E.train(args.output,spec,rank,world,device,runtime)
        else:
            P.require_approval(args.output,spec)
            E.evaluate(args.output,spec,device)
    finally:
        if E.dist.is_initialized(): E.dist.destroy_process_group()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
