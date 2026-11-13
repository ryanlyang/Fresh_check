#!/usr/bin/env python3
"""Frozen state-extension production worker. Never submits jobs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_state_production import protocol as P


def deadline():
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal(): raise ValueError('GPU phases require a Slurm allocation')
    row=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    match=re.search(r'\bEndTime=(\S+)',row)
    if not match or match[1] in ('Unknown','None'): raise ValueError('Cannot establish allocation deadline')
    left=datetime.fromisoformat(match[1]).timestamp()-time.time()-P.POLICY['shutdown_margin_seconds']
    if left<=0: raise SystemExit(75)
    return time.monotonic()+left


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('inspect','verify','ready','status','train','evaluate','report'),required=True)
    parser.add_argument('--benchmark',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model',choices=tuple(P.PACKAGES))
    a=parser.parse_args(argv)
    if a.phase=='inspect':
        if a.benchmark is None or a.model is None: parser.error('inspect needs benchmark and model')
        print(json.dumps(P.inspect(a.benchmark,a.model),allow_nan=False)); return
    if a.output is None: parser.error('output required')
    plan=P.load(a.output)
    if a.phase=='verify':
        print(json.dumps(dict(verified=True,campaign_sha256=plan['content_hash']))); return

    from part_ev_state_production import engine
    if a.phase=='ready':
        print(json.dumps(engine.ready(a.output,plan))); return
    if a.phase in ('status','report'):
        if a.phase=='report' and hasattr(signal,'SIGALRM'):
            def stop(*_): raise TimeoutError('Report deadline reached; no completion published')
            signal.signal(signal.SIGALRM,stop); signal.alarm(3300)
        with P.file_lock(a.output/'.report.lock',blocking=False) if a.phase=='report' else engine.nullcontext():
            result=getattr(engine,a.phase)(a.output,plan)
        print(json.dumps(result,indent=2,allow_nan=False)); return
    started=time.time()
    rank=int(os.environ.get('RANK','0'))
    try:
        engine.DEADLINE=deadline()
        rank,world,device=engine.B.setup(2 if a.phase=='train' else 1)
        runtime=engine.runtime_record(plan,device,world)
        if a.phase=='train':
            engine.train(a.output,plan,rank,world,device,runtime)
        else:
            engine.evaluate(a.output,plan,device)
    finally:
        if rank==0:
            run=P.run_directory(a.output,plan)
            P.write(run/'sessions'/f"{a.phase}_{os.environ.get('SLURM_JOB_ID','unknown')}_{time.time_ns()}.json",
                dict(phase=a.phase,start=started,elapsed_seconds=time.time()-started,
                    gpu_count=2 if a.phase=='train' else 1,job_id=os.environ.get('SLURM_JOB_ID'),
                    note='Elapsed includes replay; not a success marker. Consult audited completion and Slurm.'))
        if engine.dist.is_initialized(): engine.dist.destroy_process_group()


if __name__=='__main__': main()
