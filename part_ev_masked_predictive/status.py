"""Read-only progress/accounting; no checkpoint, prediction or ROOT payloads."""
import json
from pathlib import Path
import re
import subprocess
from . import protocol as P
from .config import COUNTS


def submissions(output):
    result=[]
    for path in sorted(Path(output).glob('*_submissions.json')):
        for row in P.read(path):
            result.append(dict(row,journal=path.name))
    return result


def display(output):
    output=Path(output);jobs=submissions(output);arms={}
    for arm in COUNTS:
        run=P.run_directory(output,arm)
        checks=[P.read(p) for p in sorted((run/'validation').glob('check_*.json'))]
        step=max((r['step'] for r in checks),default=0);latest=None
        for path in (output/'logs').glob(f'evmp_*_train_{arm}_*.out'):
            with path.open(encoding='utf-8',errors='replace') as stream:
                for line in stream:
                    if not line.startswith('{'):continue
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if (row.get('stage')=='training' and row.get('task')==P.task(arm)
                            and row['step']>=step):
                        step=row['step'];latest=dict(log=str(path),**row)
        complete=(run/'training_complete.json').exists()
        check=len(checks)+1;target=check*20000
        fraction=max(0.,min(1.,(step-(target-20000))/20000))
        progress=None if complete else dict(next_check=check,target_update=target,
            last_logged_or_validated_update=step,percent_to_boundary=100*fraction,
            validation_pending=step>=target)
        arms[arm]=dict(checks=checks,training_complete_marker=complete,
            evaluation_complete_marker=(run/'full_validation/metrics.json').exists(),
            progress=progress,last_training_log=latest)
    ids={j['job'] for j in jobs if j.get('status')=='recorded' and re.fullmatch(r'\d+',j.get('job',''))}
    scheduler='No recorded jobs'
    if ids:
        try:
            scheduler=subprocess.check_output(['squeue','--jobs='+','.join(sorted(ids)),
                '--noheader','--format=%i|%T|%j|%M|%l|%R'],text=True,stderr=subprocess.STDOUT,timeout=30).strip()
        except (OSError,subprocess.SubprocessError) as exc:scheduler='Unavailable: '+str(exc)
    return dict(arms=arms,submissions=jobs,scheduler=scheduler,scheduler_allocations=allocated_cost(output),
        note='Last observed updates only; validation itself has no live percentage. Marker presence is not a hash audit. Read-only.')


def allocated_cost(output):
    records={j['job']:j for j in submissions(output)
        if j.get('status')=='recorded' and re.fullmatch(r'\d+',j.get('job',''))
        and j['phase'].split('_',1)[0] in ('benchmark','train','evaluate')}
    result=dict(jobs=[],gpu_hours=None,note='Actual scheduler elapsed times GPU count, including retries; excludes pending time.')
    try:
        if not records:raise ValueError('No recorded GPU allocations')
        text=subprocess.check_output(['sacct','-X','--noheader','--parsable2',
            '--jobs='+','.join(sorted(records)),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],
            text=True,stderr=subprocess.STDOUT,timeout=30)
        seen=set()
        for line in text.splitlines():
            fields=line.strip().split('|')
            if len(fields)<4 or fields[0] not in records or fields[0] in seen:
                raise ValueError('Unrecognized accounting row')
            job,state,elapsed,tres=fields[:4];seen.add(job)
            count=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
            seconds=int(elapsed)
            if not count and seconds>0:raise ValueError('Allocated GPU count missing')
            gpus=int(count.group(1)) if count else 0
            result['jobs'].append(dict(job=job,phase=records[job]['phase'],state=state,
                elapsed_seconds=seconds,gpus=gpus,gpu_hours=seconds*gpus/3600))
        if seen!=set(records):raise ValueError('Incomplete scheduler accounting')
        result['gpu_hours']=sum(j['gpu_hours'] for j in result['jobs'])
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        result.update(gpu_hours=None,unavailable=str(exc))
    return result
