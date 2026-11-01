#!/usr/bin/env python3
"""Queue-independent controller for frozen 72-hour DDP continuation chains."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_ddp_segments import protocol as S


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('review','prepare','segment','gate','evaluate','report'))
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--chain',type=Path)
    parser.add_argument('--segments',type=int,default=S.POLICY['default_segments'])
    parser.add_argument('--index',type=int)
    args = parser.parse_args(argv)
    if args.phase=='review':
        if args.campaign is None: parser.error('--campaign required')
        print(json.dumps(S.review(args.campaign)[1],indent=2))
        return 0
    if args.chain is None: parser.error('--chain required')
    args.chain = args.chain.resolve()
    if args.phase=='prepare':
        if args.campaign is None: parser.error('--campaign required')
        plan = S.prepare(args.chain,args.campaign,args.segments)
        print(json.dumps(dict(chain=str(args.chain),segments=plan['segments'],
                              evaluate_hours=plan['evaluate_hours']),indent=2))
        return 0
    plan,spec = S.load(args.chain,check_environment=args.phase not in ('gate','report'))
    from part_ev_layerwise_ddp import engine as E
    from part_ev_ddp_segments.engine import train_segment, allocation_deadline
    campaign = Path(plan['campaign'])
    if args.phase in ('gate','evaluate','report'):
        S.completion_gate(args.chain,plan,spec)
    if args.phase=='gate':
        print('Full training completed and audited; final validation may start.')
        return 0
    if args.phase=='report':
        E.report(campaign,spec)
        return 0
    if args.phase=='segment' and (args.index is None or not 0 <= args.index < plan['segments']):
        parser.error('--index must identify a frozen segment')
    deadline = allocation_deadline() if args.phase=='segment' else None
    S.P.G.C.verify_inventory_metadata(spec)
    rank,world,device = E.B.setup(2 if args.phase=='segment' else 1)
    try:
        runtime = E.runtime_record(spec,device,world)
        if args.phase=='segment':
            train_segment(args.chain,plan,spec,args.index,rank,world,device,runtime,deadline)
        else:
            E.evaluate(campaign,spec,device)
    finally:
        if E.dist.is_initialized(): E.dist.destroy_process_group()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
