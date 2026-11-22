#!/usr/bin/env python3
"""Frozen native-GH200 worker and verified benchmark-to-production handoff."""
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
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_rts_tigris import protocol as P


def deadline():
    job = os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal(): raise ValueError('GPU production phases require a Slurm allocation')
    row = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    match = re.search(r'\bEndTime=(\S+)',row)
    if not match or match[1] in ('Unknown','None'): raise ValueError('Cannot determine allocation deadline')
    left = datetime.fromisoformat(match[1]).timestamp()-time.time()-P.PRODUCTION_POLICY['shutdown_margin_seconds']
    if left <= 0: raise SystemExit(75)
    return time.monotonic()+left


def production(output, phase):
    plan = P.load_production(output)
    from part_ev_rts_tigris import production as engine
    root = output/'production'
    if phase == 'report':
        with P.file_lock(root/'.report.lock'):
            return engine.report(root,plan)
    P.check_storage(P.launch_config(output),production=True)
    started = time.time()
    try:
        engine.DEADLINE = deadline()
        rank, world, device = engine.B.setup(1)
        engine.torch.set_num_threads(4)
        runtime = engine.runtime_record(plan,device,world)
        if phase == 'train': engine.train(root,plan,rank,world,device,runtime)
        else: engine.evaluate(root,plan,device)
    finally:
        P.write(P.run_directory(root,plan)/'sessions'/f"{phase}_{os.environ.get('SLURM_JOB_ID','unknown')}_{time.time_ns()}.json",
            dict(phase=phase,started=started,elapsed_seconds=time.time()-started,gpu_count=1,
                 job_id=os.environ.get('SLURM_JOB_ID'),note='Elapsed includes replay; not a success marker.'))
    return dict(phase=phase,complete=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('inspect','inspect_profile_recovery','recover_profile','prepare','benchmark','summary','review','status','dispatch','train','evaluate','report'),required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--from-campaign',type=Path)
    p.add_argument('--model',choices=tuple(P.PACKAGES))
    a = p.parse_args(argv)
    if a.phase == 'inspect_profile_recovery':
        if a.from_campaign is None: p.error('Original Tigris campaign required')
        spec, origin, summary = P.inspect_profile_origin(a.from_campaign)
        print(json.dumps(dict(admission=spec['launch']['admission'], origin=origin, summary=summary),allow_nan=False))
        return 0
    if a.phase == 'inspect':
        if a.benchmark is None or a.model is None: p.error('inspect requires model and benchmark')
        print(json.dumps(P.inspect(a.benchmark,a.model),allow_nan=False)); return 0
    if a.output is None: p.error('output required')
    launch = P.launch_config(a.output)
    if a.phase == 'status':
        from part_ev_rts_tigris.submission import ledger
        result = dict(model=launch['model_kind'],pipeline=ledger(a.output,'pipeline'),production=ledger(a.output,'production'),
            recovery=ledger(a.output,'recovery'),
            summary=P.read(a.output/'summary.json') if (a.output/'summary.json').exists() else None,
            dispatch=P.read(a.output/'dispatch_result.json') if (a.output/'dispatch_result.json').exists() else None,
            note='Read-only metadata, not live Slurm status.')
        print(json.dumps(result,indent=2)); return 0
    if 'profile_recovery' in launch and a.phase in ('prepare','benchmark'):
        raise ValueError('Recovered measurements may not be replaced by a fresh benchmark')
    if a.phase in ('prepare','benchmark','recover_profile'):
        with (a.output/(a.phase+'_started.json')).open('x',encoding='utf-8') as stream:
            json.dump(dict(phase=a.phase,pid=os.getpid()),stream)
            stream.flush(); os.fsync(stream.fileno())
    def timeout(*_): raise TimeoutError('Allocation budget exhausted; no unverified production handoff')
    if hasattr(signal,'SIGALRM') and a.phase not in ('train','evaluate'):
        signal.signal(signal.SIGALRM,timeout)
        seconds = P.POLICY['max_seconds'] if a.phase=='benchmark' else (3300 if a.phase=='report' else P.MINUTES.get(a.phase,30)*60-120)
        signal.alarm(seconds)
    try:
        if a.phase == 'recover_profile':
            result = P.recover_profile(a.output)
        elif a.phase == 'prepare':
            P.prepare(a.output)
            result = dict(prepared=True,authorizes_training=False)
        elif a.phase == 'dispatch':
            from part_ev_rts_tigris.submission import dispatch
            result = dispatch(a.output)
        elif a.phase in ('train','evaluate','report'):
            result = production(a.output,a.phase)
        else:
            if a.phase == 'benchmark':
                spec = P.load(a.output)
                P.check_storage(launch)
                from part_ev_rts_tigris.benchmark import run
                result = run(a.output,spec)
                P.write(a.output/'benchmark.json',P.hashed(result))
            result = P.summary(a.output)
            if a.phase == 'summary': P.write(a.output/'summary.json',result)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 1 if result.get('passed') is False else 0
    except BaseException as exc:
        P.write(a.output/(a.phase+'_error.json'),dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
    finally:
        if hasattr(signal,'SIGALRM'): signal.alarm(0)


if __name__ == '__main__': raise SystemExit(main())
