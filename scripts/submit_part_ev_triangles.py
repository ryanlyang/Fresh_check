#!/usr/bin/env python3
"""Independent SPORC triangle-aware benchmark; no production/cancellation modes."""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_triangles import protocol as P

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python'
REFERENCE='/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z'
SOURCES=('part_ev_triangles','part_ev_pair_readout','part_ev_fullscale','part_ev_screen','part_ev_message_study',
    'part_ev_state_speed','part_ev_state_fullscale','part_ev_state_capacity','part_ev_state_compare',
    'part_layerwise_screen','part_ev_gpu_scaling','part_ev_layerwise_ddp','part_ev_state_bias',
    'part_ev_pair_memory','part_ev_extensions_tigris','particle_transformer/data/JetClass/JetClass_full.yaml',
    'scripts/run_part_ev_screen.py','scripts/check_part_ev_screen_amp_recovery.py',
    'scripts/run_part_ev_triangles.py','scripts/submit_part_ev_triangles.py',
    'sbatch/run_part_ev_triangles.sh','sbatch/submit_part_ev_triangles.sh')


def command(args,**kwargs):
    return subprocess.check_output([str(x) for x in args],text=True,**kwargs).strip()


def environment(python,nccl=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_','NCCL_'))}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT',
                'GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME'):
        env.pop(key,None)
    binary=Path(python).parent
    env.update(PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.update(nccl or {})
    return env


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','benchmark','review'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(REFERENCE))
    p.add_argument('--python',default=PYTHON)
    p.add_argument('--account',default='reu-aisocial')
    p.add_argument('--partition',default='tier3')
    a=p.parse_args(argv)
    for name in ('account','partition'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',getattr(a,name)):p.error('Invalid '+name)
    for name in ('output','reference'):
        value=getattr(a,name)
        if not value.is_absolute() or value.absolute()!=value.resolve():p.error('Use absolute paths without symlinks/traversal')
        setattr(a,name,value.resolve())
    if not Path(a.python).is_absolute():p.error('Use an absolute Python path')
    registry=ROOT/'checkpoints/part_ev_triangles'
    if a.output.parent!=registry:p.error('Output must be a direct child of '+str(registry))
    if a.output==a.reference or a.output.is_relative_to(a.reference) or a.reference.is_relative_to(a.output):
        p.error('Output/reference must be disjoint')
    return a


def request(output,launch,phase,dependency=None):
    if phase not in P.MINUTES:raise ValueError('No production mode')
    row=['sbatch','--parsable','--job-name=evtri_'+phase,'--partition='+launch['partition'],
        '--account='+launch['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task='+('2' if phase=='summary' else '8'),
        '--mem='+str(P.HOST_MEMORY_GIB[phase])+'G','--time='+str(P.MINUTES[phase]),
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes']
    if phase=='benchmark':row+=['--gres=gpu:a100:2','--signal=USR1@900']
    if dependency:row+=['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_triangles.sh'),str(output),phase]


def check_partition(partition):
    fields=dict(word.split('=',1) for word in command(['scontrol','show','partition',partition,'-o']).split() if '=' in word)
    if fields.get('PartitionName')!=partition or fields.get('State')!='UP':raise ValueError('Requested partition is not UP')
    limit=fields.get('MaxTime','')
    if limit in ('UNLIMITED','INFINITE'):return
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',limit)
    if not match:raise ValueError('Cannot parse partition limit')
    d,h,m,s=(int(v or 0) for v in match.groups())
    if d*86400+h*3600+m*60+s<19800:raise ValueError('Partition walltime below benchmark request')


def freeze(a):
    if a.output.exists():raise FileExistsError('Use a new output; inspect prior submission ledger before any retry')
    if command(['git','-C',ROOT,'status','--porcelain','--',*SOURCES]):
        raise ValueError('Commit/push/pull relevant source before submission')
    commit=command(['git','-C',ROOT,'rev-parse','HEAD'])
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive',commit])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members=tar.getmembers()
        for item in members:
            target=a.output/'source'/item.name
            if (not target.resolve().is_relative_to(a.output/'source')
                    or not (item.isfile() or item.isdir())):raise ValueError('Unsafe snapshot entry: '+item.name)
        a.output.mkdir();(a.output/'logs').mkdir();(a.output/'source').mkdir()
        for item in members:
            target=a.output/'source'/item.name
            if item.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(tar.extractfile(item).read())
    manifest={p.relative_to(a.output/'source').as_posix():P.file_sha256(p)
              for p in sorted((a.output/'source').rglob('*')) if p.is_file()}
    P.write_json(a.output/'source_manifest.json',manifest)
    row=dict(schema=P.SCHEMA,output=str(a.output),repo=str(ROOT),reference=str(a.reference),
        python=a.python,account=a.account,partition=a.partition,commit=commit,minutes=P.MINUTES,
        host_memory_gib=P.HOST_MEMORY_GIB,
        nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},
        source_sha256=P.digest(manifest),output_budget_bytes=P.BUDGET_BYTES,authorizes_training=False)
    row['content_hash']=P.digest(row)
    P.write_json(a.output/'launch.json',row)
    (a.output/'launch.sha256').write_text(P.file_sha256(a.output/'launch.json')+'  launch.json\n',encoding='utf-8')
    return row


def launch(a):
    if a.mode=='review':
        saved=P.launch_config(a.output)
        print(command([saved['python'],'-B',a.output/'source/scripts/run_part_ev_triangles.py',
            '--phase','review','--output',a.output],env=environment(saved['python'],saved['nccl_environment'])))
        return
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Use SPORC x86_64, not Tigris or Windows')
    if not os.access(a.python,os.X_OK):raise ValueError('Existing SPORC Python is not executable')
    if Path(sys.executable).resolve()!=Path(a.python).resolve():raise ValueError('Run launcher with the chosen SPORC interpreter')
    P.reference(a.reference)
    print(f'Triangle-aware State-EV: {P.PARAMETERS:,} parameters; unchanged full 1M-update/50-check recipe.')
    print('Prepare 2h CPU; benchmark 5h30m on two same-node A100s; summary 15m CPU. No production.')
    print('Host RAM: prepare 64 GiB; benchmark 128 GiB; summary 4 GiB. Profiling only: at most 2 jets / 32 particles per rank.')
    print('SPORC restart v2: exact restoration/RNG/schedules; bounded floating trajectories with repeat control. Fresh output required.')
    if a.mode=='plan':
        print('Read-only. Submission will check aggregate headroom and live Slurm limits.');return
    for tool in ('git','sbatch','scontrol'):
        if shutil.which(tool) is None:raise ValueError('Missing '+tool)
    registry=ROOT/'checkpoints/part_ev_triangles';registry.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.output.exists():raise FileExistsError('Output exists; inspect ledger, do not resubmit blindly')
        needed=P.check_storage(ROOT,a.output);check_partition(a.partition)
        env=environment(a.python,{k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
        proposal=dict(account=a.account,partition=a.partition)
        for phase in P.MINUTES:
            row=request(a.output,proposal,phase);row[-3]=str(ROOT/'sbatch/run_part_ev_triangles.sh')
            command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)
        saved=freeze(a);previous=None
        for phase in P.MINUTES:
            dep=None if previous is None else ('afterany:' if phase=='summary' else 'afterok:')+previous
            row=request(a.output,saved,phase,dep)
            P.write_json(a.output/'submission_intent.json',dict(status='unknown',request=row))
            job=command(row,env=env).split(';')[0]
            if not job.isdecimal():raise ValueError('Uncertain sbatch reply; inspect scheduler and intent before retrying')
            with (a.output/'submitted_job_ids.tsv').open('a',encoding='utf-8') as ledger:
                ledger.write(f'{phase}\t{job}\n');ledger.flush();os.fsync(ledger.fileno())
            P.write_json(a.output/'submission_intent.json',dict(status='recorded',job=job,request=row))
            print(f'{phase} job: {job}',flush=True);previous=job
        print('Benchmark:',a.output)
        print(f'Aggregate headroom: {needed/2**30:.2f} GiB. Existing jobs/data/environments unchanged.')


if __name__=='__main__':launch(parse())
