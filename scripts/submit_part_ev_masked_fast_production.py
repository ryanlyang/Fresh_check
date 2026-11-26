#!/usr/bin/env python3
"""Fresh fast256 preflight -> admitted long training -> evaluation -> report."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from part_ev_masked_predictive import submission as S
from part_ev_masked_speed import protocol as D
from part_ev_masked_fast_production import protocol as P, submission as F
from part_ev_masked_fast_production.config import SCHEMA, SCIENCE, RESOURCES, TASK, POLICY, execution_hash


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('plan','preflight','pipeline','review','dispatch','resume'))
    parser.add_argument('--from-speed',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-train-hours',type=int,default=480)
    parser.add_argument('--max-evaluate-hours',type=int,default=24)
    parser.add_argument('--reviewed',action='store_true')
    args=parser.parse_args(argv)
    if (not args.output.is_absolute() or args.output!=args.output.resolve()
            or args.output.parent!=ROOT/'checkpoints/part_ev_masked_fast_production'):
        parser.error('Canonical fresh output under checkpoints/part_ev_masked_fast_production required')
    if args.mode in ('plan','preflight','pipeline') and args.from_speed is None:parser.error('--from-speed required')
    if args.from_speed is not None and (not args.from_speed.is_absolute() or args.from_speed!=args.from_speed.resolve(strict=True)):
        parser.error('Canonical existing speed-study path required')
    if args.mode in ('pipeline','dispatch','resume') and not args.reviewed:parser.error('--reviewed required for production authority')
    if not 1<=args.max_train_hours<=480 or not 1<=args.max_evaluate_hours<=24:parser.error('Walltime ceilings outside allowed bounds')
    return args


def launch(args):
    output=args.output
    if args.mode in ('review','dispatch','resume'):
        saved=P.load(output)
        print(S.command([saved['python'],'-B',output/'source/scripts/run_part_ev_masked_fast_production.py',
            '--phase',args.mode,'--output',output],env=S.environment(saved['python'],'tigris')))
        return
    if output.exists():raise FileExistsError('Fresh output required; do not overwrite a prior chain')
    speed=D.load(args.from_speed)
    evidence=json.loads(S.command([speed['python'],'-B',ROOT/'scripts/run_part_ev_masked_fast_production.py',
        '--phase','inspect','--from-speed',args.from_speed],env=S.environment(speed['python'],'tigris')))
    if evidence['choice']['provisional_train_hours']>args.max_train_hours:
        raise ValueError('Provisional speed-study allocation exceeds requested ceiling')
    original=evidence.pop('original_launch')
    proposal=dict(evidence,schema=SCHEMA,execution_sha256=execution_hash(),output=str(output),repo=str(ROOT),
        platform='tigris',python=original['python'],account=original['account'],resources=RESOURCES,
        science=SCIENCE,task=TASK,policy=POLICY,auto_production=args.mode=='pipeline',
        max_train_hours=args.max_train_hours,max_evaluate_hours=args.max_evaluate_hours)
    needed=P.check_storage(output)
    print(json.dumps(dict(model=TASK,global_batch=512,auxiliary_batch=256,production_gpus=1,
        preflight_hours=5.5,provisional_train_hours=evidence['choice']['provisional_train_hours'],
        train_ceiling_hours=args.max_train_hours,required_free_gib=needed/2**30,
        auto_production=proposal['auto_production'],control_submitted=False),indent=2))
    if args.mode=='plan':print('Read-only. No files/jobs changed.');return
    with S.locks(ROOT,output.parent):
        P.check_storage(output)
        following='dispatch' if proposal['auto_production'] else 'summary'
        previews=[F.request(output,proposal,p) for p in ('preflight',following)]
        for row in previews:row[-3]=str(ROOT/'sbatch/run_part_ev_masked_fast_production.sh')
        S.preflight(output,proposal,previews)
        frozen=S.freeze(ROOT,output,proposal)
        for name in ('part_ev_masked_fast_production/protocol.py','part_ev_masked_fast_production/execution.py',
                     'scripts/run_part_ev_masked_fast_production.py','sbatch/run_part_ev_masked_fast_production.sh'):
            if not (output/'source'/name).is_file():raise ValueError('Commit all new files before submission')
        check=F.submit(output,frozen,'preflight',prefix='pipeline')
        dependency=('afterok:' if proposal['auto_production'] else 'afterany:')+check
        follow=F.submit(output,frozen,following,prefix='pipeline',dependency=dependency)
        print(json.dumps(dict(campaign=str(output),preflight=check,**{following:follow},
            note='Existing jobs untouched. Only fast256 may train; dispatch recomputes all admission gates and walltime.'),indent=2))


if __name__=='__main__':launch(parse())
