"""Fail-closed scheduling for independent native-GH200 pipelines."""
import os
from pathlib import Path
import re
import subprocess
from . import protocol as P


def command(argv, **kwargs):
    return subprocess.check_output([str(x) for x in argv], text=True, **kwargs).strip()


def environment(python):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_', 'NCCL_'))}
    binary = Path(python).parent
    env.update(PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES','PYTHONPATH','EVRTS_STOP_FILE'):
        env.pop(key, None)
    return env


def check_partition(minutes):
    row = command(['scontrol','show','partition','tigris','-o'])
    fields = dict(item.split('=',1) for item in row.split() if '=' in item)
    if fields.get('State') != 'UP': raise ValueError('Tigris partition is not UP')
    limit = fields.get('MaxTime','')
    if limit == 'UNLIMITED': return
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',limit)
    if not match: raise ValueError('Cannot establish Tigris walltime limit')
    d,h,m,s = (int(v or 0) for v in match.groups())
    if 60*minutes > d*86400+h*3600+m*60+s:
        raise ValueError('Measured recommendation exceeds live Tigris partition limit')


def request(output, launch, phase, dependency=None, *, plan=None):
    output = Path(output)
    if phase == 'recover_profile':
        minutes = 30
    elif phase in P.MINUTES:
        minutes = P.MINUTES[phase]
    elif plan is not None and phase in ('train','evaluate','report'):
        minutes = 60*plan[phase+'_hours']
    else: raise ValueError('Unknown or unadmitted phase')
    gpu = phase in ('benchmark','train','evaluate')
    small = phase in ('summary','dispatch','report','recover_profile')
    row = ['sbatch','--parsable','--job-name=evrts_'+launch['model_kind']+'_'+phase,
        '--partition=tigris','--account='+launch['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task='+('4' if small else '8'), '--mem='+('16G' if small else '128G'),
        '--time='+str(minutes), '--output='+str(output/'logs/%x_%j.out'),
        '--error='+str(output/'logs/%x_%j.err'),'--export=ALL','--kill-on-invalid-dep=yes']
    if gpu: row += ['--gres=gpu:gh200:1','--signal=USR1@900']
    if dependency: row += ['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_rts_tigris.sh'),str(output),phase]


def ledger(output, prefix):
    path = Path(output)/(prefix+'_job_ids.tsv')
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields)!=2 or fields[0] in result or not fields[1].isdecimal():
                raise ValueError('Ambiguous submission ledger; reconcile with Slurm')
            result[fields[0]] = fields[1]
    if len(set(result.values())) != len(result): raise ValueError('Duplicate job identity')
    return result


def submit_one(output, launch, phase, dependency, prefix, *, plan=None):
    row = request(output,launch,phase,dependency,plan=plan)
    # Unknown is intentionally durable before invoking sbatch: a timeout could
    # occur AFTER Slurm accepts the job. Never retry an ambiguous submission.
    intent = Path(output)/(prefix+'_intent.json')
    P.write(intent,dict(status='unknown',phase=phase,request=row))
    job = command(row,env=environment(launch['python'])).split(';')[0]
    if not job.isdecimal(): raise ValueError('Uncertain sbatch response; reconcile journal and Slurm before retry')
    with (Path(output)/(prefix+'_job_ids.tsv')).open('a',encoding='utf-8') as stream:
        stream.write(f'{phase}\t{job}\n'); stream.flush(); os.fsync(stream.fileno())
    P.write(intent,dict(status='recorded',phase=phase,job=job,request=row))
    print(f'{phase} job: {job}',flush=True)
    return job


def dispatch(output):
    output = Path(output).resolve(strict=True)
    launch = P.launch_config(output)
    registry = Path(launch['repo'])/'checkpoints/part_ev_rts_tigris'
    # Shared SPORC lock serializes admission against those producers too.
    with P.file_lock(Path(launch['repo'])/'checkpoints/part_ev_state_production/.submission.lock'), \
            P.file_lock(registry/'.submission.lock'):
        plan = P.production_spec(output)  # Recompute all native-GPU evidence.
        owner = registry/(P.digest(dict(model=launch['model_kind'],
            donor=launch['admission']['donor']['content_hash']))+'.production-owner.json')
        if owner.exists() and P.read(owner)['output'] != str(output):
            raise ValueError('Another production chain already owns this model/reference: '+P.read(owner)['output'])
        intent_path = output/'production_intent.json'
        jobs = ledger(output,'production')
        if intent_path.exists() or jobs:
            intent = P.read(intent_path) if intent_path.exists() else {}
            if (intent.get('status') != 'recorded' or set(jobs) != {'train','evaluate','report'}
                    or intent.get('job') != jobs.get('report') or not owner.exists()
                    or P.checked(output/'production.json') != plan):
                raise ValueError('Partial/uncertain production submission; reconcile recorded jobs, do not resubmit')
            return dict(status='already_submitted',jobs=jobs)
        if (output/'production.json').exists() and P.checked(output/'production.json') != plan:
            raise ValueError('Existing production admission differs')
        P.check_storage(launch,production=True)
        check_partition(60*max(plan[k] for k in ('train_hours','evaluate_hours','report_hours')))
        for phase in ('train','evaluate','report'):
            row = request(output,launch,phase,plan=plan)
            command([row[0],'--test-only',*row[1:]],env=environment(launch['python']),stderr=subprocess.STDOUT)
        P.write(output/'production.json',plan)
        P.write(output/'production/campaign.json',plan)
        P.write(owner,dict(output=str(output),campaign_sha256=plan['content_hash']))
        previous = None
        for phase in ('train','evaluate','report'):
            job = submit_one(output,launch,phase,None if previous is None else 'afterok:'+previous,
                             'production',plan=plan)
            jobs[phase] = job; previous = job
        result = dict(status='submitted',jobs=jobs,production=str(output/'production'),
                      train_hours=plan['train_hours'],evaluate_hours=plan['evaluate_hours'])
        P.write(output/'dispatch_result.json',result)
        return result
