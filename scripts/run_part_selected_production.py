#!/usr/bin/env python3
"""Frozen production worker; never submits jobs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_production import protocol as P


def deadline():
    """Use allocation end time (not process uptime), allowing 15 minutes to save."""
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal(): raise ValueError('GPU phases must run inside Slurm')
    record = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    match = re.search(r'\bEndTime=(\S+)', record)
    if not match or match[1] in ('Unknown', 'None'):
        raise ValueError('Cannot establish Slurm allocation deadline')
    end = datetime.fromisoformat(match[1]).timestamp()
    remaining = end-time.time()-P.POLICY['shutdown_margin_seconds']
    if remaining <= 0: raise SystemExit(75)
    return time.monotonic()+remaining


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('inspect','train','evaluate','report','status','verify'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--retry', type=Path)
    parser.add_argument('--task-index', type=int, choices=range(7))
    a = parser.parse_args(argv)
    if a.phase == 'inspect':
        if a.benchmark is None or a.retry is None: parser.error('inspect needs benchmark and retry')
        print(json.dumps(P.inspect(a.benchmark, a.retry), allow_nan=False))
        return
    if a.output is None: parser.error('output required')
    plan = P.load(a.output)
    if a.phase == 'verify':
        print(json.dumps(dict(verified=True, campaign_sha256=plan['content_hash'])))
        return
    from part_selected_production import engine, reporting
    if a.phase in ('report', 'status'):
        result = getattr(reporting, a.phase)(a.output, plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    if a.task_index is None: parser.error('GPU phases need task index')
    worker = engine.create_engine(a.output, plan, a.task_index, deadline=deadline())
    started = time.time()
    try:
        if a.phase == 'train': engine.train(a.output, plan, a.task_index, worker)
        else: worker.evaluate(a)
    finally:
        run = P.run_directory(a.output, plan['science']['tasks'][a.task_index])
        P.write(run/'sessions'/f"{a.phase}_{os.environ.get('SLURM_JOB_ID','unknown')}_{time.time_ns()}.json",
                dict(phase=a.phase, start=started, elapsed_seconds=time.time()-started,
                     gpu_count=1, job_id=os.environ.get('SLURM_JOB_ID'),
                     note='Process elapsed includes replay; consult completion markers and Slurm state for success.'))


if __name__ == '__main__': main()
