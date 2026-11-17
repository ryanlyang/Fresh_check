#!/usr/bin/env python3
"""Prepare, benchmark and review one independent support-aware experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_support import protocol as P


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','benchmark','review','summary'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);output=a.output.resolve()
    if a.phase=='prepare':
        P.prepare(output);print('Prepared benchmark only. No production submitted.');return 0
    if a.phase in ('review','summary'):
        row=P.review(output)
        print(json.dumps(row,indent=2,allow_nan=False))
        if a.phase=='summary':P.write_json(output/'summary.json',row)
        return 0 if row['passed'] else 1
    spec=P.load(output);P.verify_inventory_metadata(spec)
    P.check_storage(spec['launch']['repo'],output)
    path=output/'benchmark.json'
    if path.exists():raise FileExistsError('Evidence exists; use a fresh campaign for a retry')
    from part_ev_support import benchmark as C
    rank=int(os.environ.get('RANK','0'))
    try:
        rank,world,device=C.B.setup(2)
        row=C.run(output,spec,rank,world,device)
        row['content_hash']=P.digest(row)
        C.root_call(lambda:P.write_json(path,row),rank,world)
        review=P.review(output)
        if not review['passed']:raise RuntimeError('Benchmark health gate failed')
        if rank==0:print(json.dumps(review,indent=2))
        return 0
    except BaseException as exc:
        row=dict(complete=False,passed=False,authorizes_training=False,campaign_sha256=spec['content_hash'],
                 rank=rank,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        row['content_hash']=P.digest(row)
        P.write_json(output/f'error_rank_{rank}.json',row)
        if rank==0 and not path.exists():P.write_json(path,row)
        raise
    finally:
        if C.dist.is_initialized():C.dist.destroy_process_group()


if __name__=='__main__':raise SystemExit(main())
