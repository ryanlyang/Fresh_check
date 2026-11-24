#!/usr/bin/env python3
"""Frozen worker entry point; no live source imports in queued phases."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_workspace import protocol as P
from part_ev_workspace.config import PLATFORMS, POLICY


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('inspect','prepare','benchmark','summary','review','dispatch','resume','train','evaluate','report','status'))
    p.add_argument('--output',type=Path);p.add_argument('--platform',choices=tuple(PLATFORMS))
    p.add_argument('--arm',choices=('control','workspace'))
    a=p.parse_args(argv)
    if a.phase!='inspect' and a.output is None:p.error('Frozen --output required')
    if a.phase in ('train','evaluate','resume') and a.arm is None:p.error('Explicit production arm required')
    if a.phase=='inspect' and a.platform is None:p.error('Inspect needs --platform')
    return a


def main(a):
    if a.phase=='inspect':
        if platform.system()!='Linux' or platform.machine()!=PLATFORMS[a.platform]['machine']:
            raise ValueError('Wrong interpreter architecture/platform')
        print(json.dumps(P.runtime_environment()));return 0
    launch=P.load_launch(a.output,worker=True)
    if a.phase=='prepare':P.prepare(a.output);return 0
    if a.phase in ('summary','review'):
        result=P.review(a.output)
        if a.phase=='summary':P.write(a.output/'summary.json',P.hashed(result))
        print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
    if a.phase=='dispatch':
        from part_ev_workspace.submission import dispatch
        print(json.dumps(dispatch(a.output),indent=2));return 0
    if a.phase=='resume':
        from part_ev_workspace.submission import resume
        print(json.dumps(resume(a.output,a.arm),indent=2));return 0
    if a.phase=='status':
        from part_ev_workspace.status import display
        print(json.dumps(display(a.output),indent=2));return 0
    if a.phase=='report':
        from part_ev_workspace.engine import report
        print(json.dumps(report(a.output,P.load_production(a.output,worker=True)),indent=2));return 0
    from part_ev_workspace import runtime as R
    import torch.distributed as dist
    spec=P.load(a.output,worker=True) if a.phase=='benchmark' else P.load_production(a.output,worker=True)
    rank,world,device,gpu=R.setup(launch,training=a.phase!='evaluate')
    try:
        # Enforced even if the Slurm signal arrives late; replay counts as time.
        if 'SLURM_JOB_END_TIME' in os.environ:
            R.DEADLINE=time.monotonic()+max(0,float(os.environ['SLURM_JOB_END_TIME'])-time.time()-POLICY['shutdown_margin'])
        if a.phase=='benchmark':
            from part_ev_workspace.benchmark import run
            run(a.output,spec,launch,device,rank,world,gpu)
        else:
            from part_ev_workspace import engine
            engine.verify_runtime(spec,gpu)
            P.check_storage(launch,production=True)
            if a.phase=='train':engine.train(a.output,spec,a.arm,device,rank,world)
            else:engine.evaluate(a.output,spec,a.arm,device)
    finally:
        if world>1 and dist.is_initialized():dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main(parse()))
