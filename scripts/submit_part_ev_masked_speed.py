#!/usr/bin/env python3
"""Queue only a fresh speed diagnostic and CPU summary on Tigris."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from part_ev_masked_predictive import protocol as P, submission as S
from part_ev_masked_speed import protocol as D
from part_ev_masked_speed.config import SCHEMA, VARIANTS, POLICY


def request(output,launch,phase,dependency=None):
    gpu=phase=='benchmark'
    if phase not in ('benchmark','summary'):raise ValueError('Diagnostic phases only')
    row=['sbatch','--parsable','--job-name=evmpspeed_'+phase,'--partition=tigris',
         '--account='+launch['account'],'--nodes=1','--ntasks=1',
         '--cpus-per-task='+('12' if gpu else '2'),'--mem='+('128G' if gpu else '4G'),
         '--time='+('330' if gpu else '15'),'--export=ALL','--kill-on-invalid-dep=yes',
         '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err')]
    if gpu:row+=['--gres=gpu:gh200:1','--signal=B:USR1@300']
    if dependency:row+=['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_masked_speed.sh'),str(output),phase]


def submit(output,launch,phase,dependency=None):
    rows=S.journal(output,'diagnostic')
    if any(r['phase']==phase for r in rows):raise ValueError('Diagnostic already submitted')
    argv=request(output,launch,phase,dependency)
    entry=dict(status='unknown',phase=phase,request=argv,cluster='tigris');rows.append(entry)
    P.write(output/'diagnostic_submissions.json',rows)
    answer=S.command(argv,env=S.environment(launch['python'],'tigris'))
    match=re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?',answer)
    if not match:raise ValueError('Ambiguous submission; reconcile journal, do not retry')
    entry.update(status='recorded',job=match.group(1));P.write(output/'diagnostic_submissions.json',rows)
    print(f"{phase} job: {entry['job']}",flush=True)
    return entry['job']


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('plan','benchmark','review'))
    parser.add_argument('--from-campaign',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if (not args.output.is_absolute() or args.output!=args.output.resolve()
            or args.output.parent!=ROOT/'checkpoints/part_ev_masked_speed'):
        parser.error('Canonical output under checkpoints/part_ev_masked_speed required')
    if args.mode!='review' and args.from_campaign is None:parser.error('--from-campaign required')
    if args.from_campaign is not None and (not args.from_campaign.is_absolute()
            or args.from_campaign!=args.from_campaign.resolve(strict=True)):
        parser.error('Canonical existing donor path required')
    return args


def launch(args):
    output=args.output
    if args.mode=='review':
        saved=D.load(output)
        print(S.command([saved['python'],'-B',output/'source/scripts/run_part_ev_masked_speed.py',
            '--phase','review','--output',output],env=S.environment(saved['python'],'tigris')))
        return
    if output.exists(): raise FileExistsError('Fresh diagnostic output required; old campaigns are immutable')
    original=P.load_launch(args.from_campaign)
    inspection=json.loads(S.command([original['python'],'-B',ROOT/'scripts/run_part_ev_masked_speed.py',
        '--phase','inspect','--reference',args.from_campaign],env=S.environment(original['python'],'tigris')))
    original=inspection['launch']
    proposal=dict(schema=SCHEMA,output=str(output),repo=str(ROOT),platform='tigris',
        python=original['python'],resources=original['resources'],account=original['account'],
        reference=str(args.from_campaign),donor_files=inspection['donor_files'],variants=VARIANTS,policy=POLICY,
        reference_runtime_gpu=inspection['review']['runtime']['gpu'],
        reference_masked=next(m for m in inspection['review']['models'] if m['arm']=='masked'),
        authorizes_training=False)
    needed=D.check_storage(output)
    print(json.dumps(dict(variants=VARIANTS,benchmark_hours=5.5,summary_minutes=15,
        gpus=1,gpu='GH200',host_memory_gib=128,required_free_gib=needed/2**30,
        authorizes_training=False),indent=2))
    if args.mode=='plan':print('Read-only. No files or jobs changed.');return
    with S.locks(ROOT,output.parent):
        D.check_storage(output)
        previews=[request(output,proposal,p) for p in ('benchmark','summary')]
        for row in previews:row[-3]=str(ROOT/'sbatch/run_part_ev_masked_speed.sh')
        S.preflight(output,proposal,previews)
        frozen=S.freeze(ROOT,output,proposal)
        for name in ('part_ev_masked_speed/benchmark.py','part_ev_masked_speed/step.py',
                     'scripts/run_part_ev_masked_speed.py','sbatch/run_part_ev_masked_speed.sh'):
            if not (output/'source'/name).is_file():raise ValueError('Commit new diagnostic files before submission')
        benchmark=submit(output,frozen,'benchmark')
        summary=submit(output,frozen,'summary','afterany:'+benchmark)
        print(json.dumps(dict(campaign=str(output),benchmark=benchmark,summary=summary,
                              authorizes_training=False),indent=2))


if __name__=='__main__':launch(parse())
