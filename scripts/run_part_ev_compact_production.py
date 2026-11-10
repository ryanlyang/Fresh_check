#!/usr/bin/env python3
"""Frozen compact production worker. Never submits jobs."""
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
from part_ev_compact_production import protocol as P


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
    parser.add_argument('--task-index',type=int,choices=(0,1))
    parser.add_argument('--indices',default='0,1')
    a=parser.parse_args(argv)
    if a.phase=='inspect':
        if a.benchmark is None: parser.error('inspect needs benchmark')
        print(json.dumps(P.inspect(a.benchmark),allow_nan=False)); return
    if a.output is None: parser.error('output required')
    plan=P.load(a.output)
    if a.phase=='verify':
        print(json.dumps(dict(verified=True,campaign_sha256=plan['content_hash']))); return
    from part_ev_compact_production import engine,reporting
    if a.phase=='ready':
        indices=[int(i) for i in a.indices.split(',')]
        if not indices or len(set(indices))!=len(indices) or any(i not in (0,1) for i in indices):
            parser.error('indices must be unique 0,1')
        print(json.dumps(reporting.ready(a.output,plan,indices))); return
    if a.phase in ('status','report'):
        if a.phase=='report' and hasattr(signal,'SIGALRM'):
            def stop(*_): raise TimeoutError('Report deadline reached; no completion published')
            signal.signal(signal.SIGALRM,stop); signal.alarm(3300)
        print(json.dumps(getattr(reporting,a.phase)(a.output,plan),indent=2,allow_nan=False)); return
    if a.task_index is None: parser.error('GPU phase requires task index')
    worker=engine.create_engine(a.output,plan,a.task_index,deadline=deadline())
    started=time.time()
    try:
        if a.phase=='train': engine.train(a.output,plan,a.task_index,worker)
        else: worker.evaluate(a)
    finally:
        run=P.run_directory(a.output,plan['science']['tasks'][a.task_index])
        P.write(run/'sessions'/f"{a.phase}_{os.environ.get('SLURM_JOB_ID','unknown')}_{time.time_ns()}.json",
            dict(phase=a.phase,start=started,elapsed_seconds=time.time()-started,gpu_count=1,
                job_id=os.environ.get('SLURM_JOB_ID'),
                note='Session elapsed includes replay, not a success marker. Consult audited completion and Slurm.'))


if __name__=='__main__': main()
