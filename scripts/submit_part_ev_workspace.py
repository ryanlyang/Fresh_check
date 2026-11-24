#!/usr/bin/env python3
"""Plan/queue independent paired workspace campaigns; never submit on import."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_workspace import protocol as P, submission as S
from part_ev_workspace.config import PLATFORMS, COUNTS, TASKS, POLICY, SCHEMA, science_hash, campaign_budget

ROOT=Path(__file__).resolve().parents[1]


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('plan','benchmark','pipeline','review','dispatch','resume','status'))
    p.add_argument('--platform',choices=tuple(PLATFORMS))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--data',type=Path,default=Path('/home/ryreu/atlas/datasets/JetClass'))
    p.add_argument('--python');p.add_argument('--account',default='reu-aisocial')
    p.add_argument('--max-train-hours',type=int,default=480)
    p.add_argument('--max-evaluate-hours',type=int,default=24)
    p.add_argument('--reviewed',action='store_true')
    p.add_argument('--arm',choices=('control','workspace'))
    a=p.parse_args(argv)
    if a.mode in ('pipeline','dispatch','resume') and not a.reviewed:p.error('Explicit --reviewed authority required')
    if (a.mode=='resume')!=(a.arm is not None):p.error('--arm is required only for explicit resume')
    if a.mode in ('plan','pipeline','benchmark') and a.platform is None:p.error('Fresh launch requires --platform')
    if a.mode in ('review','dispatch','resume','status') and (a.platform is not None or a.python is not None):
        p.error('Use only the frozen --output, not replacement platform/interpreter')
    if not 1<=a.max_train_hours<=480 or not 1<=a.max_evaluate_hours<=24:p.error('Invalid resource ceiling')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.account):p.error('Invalid account')
    base=ROOT/'checkpoints/part_ev_workspace'
    if (not a.output.is_absolute() or a.output.absolute()!=a.output.resolve() or a.output.parent!=base):
        p.error('Output must be a canonical direct child of '+str(base))
    if a.platform and a.python is None:a.python=PLATFORMS[a.platform]['python']
    return a


def launch(a):
    if a.mode in ('review','dispatch','resume','status'):
        saved=P.load_launch(a.output)
        argv=[saved['python'],'-B',a.output/'source/scripts/run_part_ev_workspace.py',
            '--phase',a.mode,'--output',a.output]
        if a.arm:argv+=['--arm',a.arm]
        result=S.command(argv,env=S.environment(saved['python'],saved['platform']))
        print(result);return
    resources=PLATFORMS[a.platform]
    plan=dict(platform=a.platform,tasks=TASKS,parameters=COUNTS,
        schedule='1M attempts, 50 checks, fresh seed 101, original Ranger',resources=resources,
        phases_minutes=S.MINUTES,paired_trainings_per_platform=2,
        auto_production=a.mode=='pipeline',prospective_budget_gib=(campaign_budget()+POLICY['reserve_bytes'])/2**30)
    print(json.dumps(plan,indent=2))
    if a.mode=='plan':print('Read-only. No files or jobs changed.');return
    if platform.system()!='Linux' or platform.machine()!=resources['machine']:
        raise ValueError('Submit on the requested actual Linux cluster')
    if not Path(a.python).is_absolute() or not Path(a.python).is_file():raise ValueError('Missing absolute platform Python')
    if not a.data.is_absolute() or a.data.absolute()!=a.data.resolve():raise ValueError('Canonical absolute dataset root required')
    if a.output.exists():raise FileExistsError('Use a fresh output; inspect old journal before retry')
    env=S.environment(a.python,a.platform)
    # Inspect using the chosen existing interpreter, without installing anything.
    admission=json.loads(S.command([a.python,'-B',ROOT/'scripts/run_part_ev_workspace.py',
        '--phase','inspect','--platform',a.platform],env=env))
    proposal=dict(schema=SCHEMA,science_sha256=science_hash(),platform=a.platform,
        output=str(a.output),repo=str(ROOT),python=str(Path(a.python).resolve()),data=str(a.data),
        account=a.account,resources=resources,tasks=TASKS,policy=POLICY,
        environment=admission,auto_production=a.mode=='pipeline',
        reservation_bytes=campaign_budget() if a.mode=='pipeline' else POLICY['benchmark_bytes'],
        max_train_hours=a.max_train_hours,max_evaluate_hours=a.max_evaluate_hours)
    registry=a.output.parent
    with S.locks(ROOT,registry):
        P.check_storage(proposal)
        phases=['prepare','benchmark','summary']+(['dispatch'] if a.mode=='pipeline' else [])
        previews=[S.request(a.output,proposal,p,script=ROOT/'sbatch/run_part_ev_workspace.sh') for p in phases]
        S.preflight(a.output,proposal,previews)
        frozen=S.freeze(ROOT,a.output,proposal)
        previous=None
        for phase in phases:
            dependency=None if previous is None else ('afterany:' if phase=='summary' else 'afterok:')+previous
            previous=S.submit_one(a.output,frozen,phase,prefix='pipeline',dependency=dependency)
    print('Campaign:',a.output)
    print('Benchmark-gated automatic production authorized.' if a.mode=='pipeline' else 'Benchmarks only; production needs separate reviewed dispatch.')
    print('Existing jobs, campaigns, environments and datasets unchanged.')


if __name__=='__main__':launch(parse())
