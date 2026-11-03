#!/usr/bin/env python3
"""Fresh GH200 campaign admitted by the existing pending benchmark."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_tigris_segments import protocol as P


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','prepare-chain','review','admit','segment','gate','evaluate','report'))
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--benchmark',type=Path)
    parser.add_argument('--benchmark-job')
    parser.add_argument('--chain',type=Path)
    parser.add_argument('--segments',type=int,default=4)
    parser.add_argument('--index',type=int)
    args=parser.parse_args(argv)
    if args.phase in ('prepare','prepare-chain','review'):
        if args.campaign is None: parser.error('--campaign required')
        output=args.campaign.resolve()
        if args.phase=='prepare':
            if args.benchmark is None or args.benchmark_job is None: parser.error('--benchmark and --benchmark-job required')
            P.prepare(output,args.benchmark,args.benchmark_job)
        if args.phase!='review':
            if args.chain is None: parser.error('--chain required')
            print(json.dumps(P.prepare_chain(args.chain,output,args.segments),indent=2))
        else:
            request,_=P.request(output,check_environment=False)
            row=dict(campaign=str(output),benchmark=request['reference'],policy=P.POLICY,admitted=(output/'campaign.json').exists())
            if row['admitted']:
                spec=P.load_campaign(output,check_environment=False)
                row.update(progress=P.snapshot(output,spec),recommended_train_hours=spec['recommended_train_hours'])
            print(json.dumps(row,indent=2))
        return 0
    if args.chain is None: parser.error('--chain required')
    chain=args.chain.resolve()
    # The deadline includes verification, startup, replay and validation.
    from part_ev_tigris_segments import engine as E
    deadline=E.allocation_deadline() if args.phase=='segment' else None
    plan=P.load_chain(chain,check_environment=args.phase not in ('gate','report'))
    if args.phase=='admit':
        print(json.dumps(P.admit(chain,plan),indent=2))
        return 0
    output=Path(plan['campaign'])
    spec=P.load_campaign(output,check_environment=False)  # load_chain already checked environment.
    P.require_admission(chain,plan,spec)
    if args.phase in ('gate','evaluate','report'): P.completion_gate(chain,plan,spec)
    if args.phase=='gate':
        print('All 1M updates and 50 validations completed and audited.')
        return 0
    D=E.create_engine()
    if args.phase=='report': E.report(output,spec,D);return 0
    P.B.verify_inventory_metadata(spec)
    device,runtime=D.gpu_setup(spec)
    if runtime!=P.expected_runtime(spec): raise ValueError('GPU/precision/runtime differs from benchmark')
    if args.phase=='segment':
        E.train_segment(chain,plan,spec,args.index,device,runtime,deadline,engine=D)
    else:
        D.evaluate(SimpleNamespace(output=output,task_index=0))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
