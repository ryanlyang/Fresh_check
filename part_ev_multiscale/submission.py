"""Fail-closed independent Slurm chains with durable pre-submit journals."""
from contextlib import ExitStack, contextmanager
import io
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from . import protocol as P
from .config import PLATFORMS, POLICY, TASKS, SCHEMA, science_hash, campaign_budget

MINUTES=dict(prepare=120,benchmark=330,summary=15,dispatch=30,report=60)


def command(argv,**kwargs):
    return subprocess.check_output([str(x) for x in argv],text=True,**kwargs).strip()


def environment(python,platform,*,source=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_','NCCL_','TORCH_NCCL_'))}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES','PYTHONPATH','EVMS_STOP_FILE'):
        env.pop(key,None)
    binary=Path(python).parent;threads=str(PLATFORMS[platform]['threads'])
    env.update(PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS='1',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    if source is not None:env['PYTHONPATH']=str(source)
    return env


def request(output,launch,phase,*,arm=None,dependency=None,plan=None,script=None):
    output=Path(output);r=launch['resources']
    if phase in ('train','evaluate'):
        if arm not in ('control','multiscale') or plan is None:raise ValueError('Unadmitted production task')
        recommendation=next(m for m in plan['models'] if m['arm']==arm)
        minutes=60*recommendation['train_hours' if phase=='train' else 'evaluate_hours']
    else:
        if phase not in MINUTES:raise ValueError('Unknown phase')
        minutes=MINUTES[phase]
    gpu=phase in ('benchmark','train','evaluate');small=phase in ('summary','dispatch','report')
    cpus=2 if small else r['cpus'] if gpu else 8
    mem=8 if phase=='report' else 4 if small else 128 if gpu else 64
    row=['sbatch','--parsable','--job-name=evms_'+launch['platform']+'_'+phase+('' if arm is None else '_'+arm),
        '--partition='+r['partition'],'--account='+launch['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(cpus),'--mem='+str(mem)+'G','--time='+str(minutes),
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes','--signal=B:USR1@900']
    if gpu:row+=['--gres=gpu:'+r['gpu']+':'+str(1 if phase=='evaluate' else r['gpus'])]
    if dependency:row+=['--dependency='+dependency]
    row += [str(script or output/'source/sbatch/run_part_ev_multiscale.sh'),str(output),phase]
    if arm is not None:row.append(arm)
    return row


def check_partition(launch,max_minutes):
    text=command(['scontrol','show','partition',launch['resources']['partition'],'-o'])
    fields=dict(p.split('=',1) for p in text.split() if '=' in p)
    if fields.get('State')!='UP':raise ValueError('Partition is not UP')
    limit=fields.get('MaxTime','')
    if limit=='UNLIMITED':return
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',limit)
    if not match:raise ValueError('Cannot establish live partition walltime')
    days,hours,minutes,seconds=[int(x or 0) for x in match.groups()]
    if max_minutes*60>days*86400+hours*3600+minutes*60+seconds:
        raise ValueError('Requested recommendation exceeds live partition limit')


def preflight(output,launch,requests):
    for tool in ('sbatch','scontrol','squeue'):
        if not __import__('shutil').which(tool):raise ValueError('Missing Slurm command '+tool)
    command(['squeue','--me','--noheader','--format=%i'])
    max_minutes=max(int(next(v.split('=',1)[1] for v in r if v.startswith('--time='))) for r in requests)
    check_partition(launch,max_minutes)
    env=environment(launch['python'],launch['platform'])
    # Slurm's own account/QoS/resource validation, before any real submissions.
    for row in requests:command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)


def freeze(repo,output,proposal):
    repo,output=Path(repo),Path(output)
    if output.exists():raise FileExistsError('Fresh output required; inspect old journals before retry')
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','-z','--untracked-files=all'])
    for record in dirty.decode('utf-8').split('\0'):
        # -z leaves spaces unquoted; rename records additionally contain the
        # original path without a status prefix, which is checked as well.
        name=record[3:] if len(record)>2 and record[2]==' ' else record
        if name.endswith(('.py','.sh','.yaml','.yml')):
            raise ValueError('Commit/push/pull relevant source before freezing: '+name)
    commit=command(['git','-C',repo,'rev-parse','HEAD'])
    blob=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar',commit])
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:
        members=archive.getmembers()
        for m in members:
            p=PurePosixPath(m.name)
            if (p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name
                    or '\n' in m.name or '\r' in m.name or not (m.isdir() or m.isfile())):
                raise ValueError('Unsafe archive member: '+m.name)
        required={'part_ev_multiscale/model.py','scripts/run_part_ev_multiscale.py','sbatch/run_part_ev_multiscale.sh'}
        if not required.issubset({m.name for m in members}):raise ValueError('New files are not committed')
        output.mkdir();(output/'logs').mkdir()
        for m in members:
            path=output/'source'/m.name
            if m.isdir():path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.extractfile(m).read())
    manifest=P.inventory(output/'source');P.write(output/'source_manifest.json',manifest)
    launch=P.hashed(dict(proposal,commit=commit,source_sha256=P.digest(manifest)))
    P.write(output/'launch.json',launch)
    lines=[f'{P.sha(output/n)}  {n}' for n in ('launch.json','source_manifest.json')]
    lines += [f'{v}  source/{k}' for k,v in sorted(manifest.items())]
    (output/'snapshot.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return launch


def journal(output,prefix):
    path=Path(output)/(prefix+'_submissions.json')
    row=P.read(path) if path.exists() else []
    if any(r.get('status')!='recorded' or not re.fullmatch(r'\d+',r.get('job','')) for r in row):
        raise ValueError('Uncertain submission intent; reconcile with Slurm, do not retry')
    if len({r['phase'] for r in row})!=len(row) or len({r['job'] for r in row})!=len(row):
        raise ValueError('Duplicate submission journal')
    return row


def submit_one(output,launch,phase,*,prefix,arm=None,dependency=None,plan=None):
    rows=journal(output,prefix);key=phase+('' if arm is None else '_'+arm)
    if any(r['phase']==key for r in rows):raise ValueError('Phase already submitted')
    argv=request(output,launch,phase,arm=arm,dependency=dependency,plan=plan)
    entry=dict(status='unknown',phase=key,request=argv,cluster=launch['platform'])
    rows.append(entry);P.write(Path(output)/(prefix+'_submissions.json'),rows)
    answer=command(argv,env=environment(launch['python'],launch['platform']))
    match=re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?',answer)
    if not match:raise ValueError('Ambiguous sbatch response; reconcile durable intent')
    entry.update(status='recorded',job=match.group(1));P.write(Path(output)/(prefix+'_submissions.json'),rows)
    with (Path(output)/(prefix+'_job_ids.tsv')).open('a',encoding='utf-8') as stream:
        stream.write(key+'\t'+entry['job']+'\n');stream.flush();os.fsync(stream.fileno())
    print(f"{key} job: {entry['job']}",flush=True)
    return entry['job']


@contextmanager
def locks(repo,registry):
    # Same order as cooperating existing native-GH200/SPORC producers.
    with ExitStack() as guard:
        guard.enter_context(P.file_lock(Path(repo)/'checkpoints/part_ev_state_production/.submission.lock'))
        guard.enter_context(P.file_lock(Path(repo)/'checkpoints/part_ev_rts_tigris/.submission.lock'))
        guard.enter_context(P.file_lock(Path(registry)/'.submission.lock'))
        yield


def dispatch(output):
    output=Path(output).resolve(strict=True);launch=P.load_launch(output,worker=True)
    with locks(launch['repo'],output.parent):
        plan=P.production_spec(output);owner=output.parent/(launch['platform']+'_'+science_hash()+'.production-owner.json')
        if owner.exists() and P.read(owner)['output']!=str(output):
            raise ValueError('Another production already owns this same-platform comparison')
        rows=journal(output,'production')
        keys={'train_control','train_multiscale','evaluate_control','evaluate_multiscale','report'}
        if rows:
            if {r['phase'] for r in rows}!=keys or not owner.exists() or P.checked(output/'production.json')!=plan:
                raise ValueError('Partial production chain; reconcile, never blindly retry')
            return dict(status='already_submitted',jobs={r['phase']:r['job'] for r in rows})
        P.check_storage(launch,production=True)
        requests=[request(output,launch,p,arm=a,plan=plan) for a in ('control','multiscale') for p in ('train','evaluate')]
        requests.append(request(output,launch,'report',plan=plan))
        preflight(output,launch,requests)
        P.write(output/'production.json',plan);P.write(output/'production/campaign.json',plan)
        P.write(owner,dict(output=str(output),campaign_sha256=plan['content_hash']))
        jobs={};evaluations=[]
        for arm in ('control','multiscale'):
            t=submit_one(output,launch,'train',prefix='production',arm=arm,plan=plan)
            e=submit_one(output,launch,'evaluate',prefix='production',arm=arm,plan=plan,dependency='afterok:'+t)
            jobs['train_'+arm]=t;jobs['evaluate_'+arm]=e;evaluations.append(e)
        jobs['report']=submit_one(output,launch,'report',prefix='production',plan=plan,dependency='afterok:'+':'.join(evaluations))
        return dict(status='submitted',jobs=jobs,production=str(output/'production'))


def checkpoint_for_retry(output,plan,arm):
    """Authenticate a stopped run without restoring GPU RNG on the CPU controller."""
    from . import engine, runtime
    import torch
    run=P.run_directory(output,arm)
    if (run/'training_complete.json').exists():
        engine.audit_training(output,plan,arm)
        return True
    if (run/'last.pt').exists():
        base,_,opt,scaler=runtime.fresh(arm,torch.device('cpu'),1,
            training_world=plan['resources']['gpus'])
        # CUDA checkpoints contain an enabled scaler; keep an enabled CPU
        # scaler for state authentication (no CPU training is performed here).
        scaler=torch.amp.GradScaler('cpu',enabled=True)
        engine.restore(run/'last.pt',base,opt,scaler,plan,arm,0,plan['resources']['gpus'])
    elif list((run/'validation').glob('check_*.json')) or list((run/'selected').glob('*.pt')):
        raise ValueError('Training artifacts without a complete restart; reconcile instead of resetting')
    return False


def job_states(jobs):
    """Missing accounting is uncertainty, not permission to launch a duplicate."""
    text=command(['sacct','-X','--noheader','--parsable2','--jobs='+','.join(sorted(jobs)),
                  '--format=JobIDRaw,State'])
    result={}
    for line in text.splitlines():
        fields=line.strip().split('|')
        if len(fields)<2 or fields[0] not in jobs or fields[0] in result:
            raise ValueError('Unrecognized/duplicate Slurm accounting row')
        result[fields[0]]=fields[1].split()[0].rstrip('+')
    if set(result)!=set(jobs):raise ValueError('Accounting incomplete; no retry submitted')
    return result


def resume(output,arm):
    """Explicit same-platform recovery, never an automatic extension/allocation."""
    if arm not in ('control','multiscale'):raise ValueError('Explicit retry arm required')
    output=Path(output).resolve(strict=True);launch=P.load_launch(output,worker=True)
    with locks(launch['repo'],output.parent):
        plan=P.load_production(output,worker=True)
        owner=output.parent/(launch['platform']+'_'+science_hash()+'.production-owner.json')
        if not owner.exists() or P.read(owner)!=dict(output=str(output),campaign_sha256=plan['content_hash']):
            raise ValueError('Production owner differs')
        original=journal(output,'production')
        if {r['phase'] for r in original}!={'train_control','train_multiscale','evaluate_control','evaluate_multiscale','report'}:
            raise ValueError('Original chain incomplete; reconcile first')
        latest={r['phase']:r['job'] for r in original}
        attempts=sorted(output.glob('resume_[0-9][0-9][0-9]_submissions.json'))
        for path in attempts:
            rows=journal(output,path.name.removesuffix('_submissions.json'))
            keys={r['phase'] for r in rows}
            valid=[{'evaluate_'+a,'report'} for a in ('control','multiscale')]
            valid += [{'train_'+a,'evaluate_'+a,'report'} for a in ('control','multiscale')]
            if keys not in valid:
                raise ValueError('Partial retry chain; reconcile, do not retry blindly')
            latest.update({r['phase']:r['job'] for r in rows})
        relevant={latest['train_'+arm],latest['evaluate_'+arm],latest['report']}
        states=job_states(relevant)
        terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
                  'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}
        if any(s not in terminal for s in states.values()):
            raise ValueError('Prior arm/report jobs still active. Review/cancel only obsolete dependencies, then retry')
        if (output/'production/reports/multiscale_report.json').exists():
            raise ValueError('Comparison already reported; no duplicate training')
        complete=checkpoint_for_retry(output,plan,arm)
        phases=['evaluate','report'] if complete else ['train','evaluate','report']
        P.check_storage(launch,production=True)
        preflight(output,launch,[request(output,launch,p,arm=arm if p!='report' else None,plan=plan) for p in phases])
        if len(attempts)>=999:raise ValueError('Retry ledger exhausted')
        prefix=f'resume_{len(attempts)+1:03d}';jobs={};dependency=None
        if not complete:
            jobs['train_'+arm]=submit_one(output,launch,'train',prefix=prefix,arm=arm,plan=plan)
            dependency='afterok:'+jobs['train_'+arm]
        jobs['evaluate_'+arm]=submit_one(output,launch,'evaluate',prefix=prefix,arm=arm,plan=plan,dependency=dependency)
        other='multiscale' if arm=='control' else 'control'
        jobs['report']=submit_one(output,launch,'report',prefix=prefix,plan=plan,
            dependency='afterok:'+jobs['evaluate_'+arm]+':'+latest['evaluate_'+other])
        return dict(status='recovery_submitted',arm=arm,jobs=jobs,restart='latest_complete_state',
                    note='No old jobs cancelled; unchanged model, schedule and platform. Replay consumes walltime.')

