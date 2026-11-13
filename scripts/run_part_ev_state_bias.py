#!/usr/bin/env python3
"""Prepare, benchmark or review only this independent state-bias experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_state_bias import protocol as P


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','benchmark','review','summary'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if args.phase=='prepare':
        P.prepare(output)
        print('Prepared independent benchmark. No historical files or production queues changed.')
        return 0
    if args.phase in ('review','summary'):
        result=P.review(output)
        print(json.dumps(result,indent=2,allow_nan=False))
        if args.phase=='summary': P.write_json(output/'summary.json',result)
        return 0 if result['passed'] else 1
    # Environment validation happens before lazy optimizer imports. Installed
    # metadata is also enumerated only at interpreter roots, not mutated sys.path.
    spec=P.load(output)
    P.verify_inventory_metadata(spec)
    path=output/'benchmark.json'
    if path.exists(): raise FileExistsError('Benchmark evidence exists; use a fresh campaign for a retry')
    from part_ev_state_bias import benchmark as C
    rank=int(os.environ.get('RANK','0'))
    try:
        rank,world,device=C.B.setup(2)
        result=C.run(output,spec,rank,world,device)
        result['content_hash']=P.digest(result)
        C.root_call(lambda:P.write_json(path,result),rank,world)
        if not result['passed']: raise RuntimeError('Memory/numerical benchmark gate failed; inspect evidence')
        if rank==0: print(json.dumps(P.review(output),indent=2))
        return 0
    except BaseException as exc:
        row=dict(complete=False,passed=False,authorizes_training=False,campaign_sha256=spec['content_hash'],
                 rank=rank,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        row['content_hash']=P.digest(row)
        P.write_json(output/f'error_rank_{rank}.json',row)
        if rank==0 and not path.exists(): P.write_json(path,row)
        raise
    finally:
        if C.dist.is_initialized(): C.dist.destroy_process_group()


if __name__=='__main__': raise SystemExit(main())
