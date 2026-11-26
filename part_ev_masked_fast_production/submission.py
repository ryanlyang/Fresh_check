"""One long GH200 allocation; fail-closed, durable candidate-only chains."""
from pathlib import Path
import re
from part_ev_masked_predictive import submission as S
from . import protocol as P
from .config import execution_hash

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED',
          'BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}


def request(output,launch,phase,*,plan=None,dependency=None):
    minutes={'preflight':330,'summary':15,'dispatch':30,'report':60}
    if phase in ('train','evaluate'):
        if plan is None:raise ValueError('Admitted production plan required')
        minutes[phase]=60*plan['models'][0]['train_hours' if phase=='train' else 'evaluate_hours']
    if phase not in minutes:raise ValueError('Unknown fast256 phase')
    gpu=phase in ('preflight','train','evaluate')
    row=['sbatch','--parsable','--job-name=evmp256_'+phase,'--partition=tigris',
        '--account='+launch['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task='+('12' if gpu else '2'),'--mem='+('128G' if gpu else '8G' if phase=='report' else '4G'),
        '--time='+str(minutes[phase]),'--export=ALL','--kill-on-invalid-dep=yes',
        '--output='+str(Path(output)/'logs/%x_%j.out'),'--error='+str(Path(output)/'logs/%x_%j.err')]
    if gpu:row+=['--gres=gpu:gh200:1','--signal=B:USR1@900']
    if dependency:row+=['--dependency='+dependency]
    return row+[str(Path(output)/'source/sbatch/run_part_ev_masked_fast_production.sh'),str(output),phase]


def submit(output,launch,phase,*,prefix,plan=None,dependency=None):
    output=Path(output);rows=S.journal(output,prefix)
    if any(r['phase']==phase for r in rows):raise ValueError('Phase already submitted')
    argv=request(output,launch,phase,plan=plan,dependency=dependency)
    entry=dict(status='unknown',phase=phase,request=argv,cluster='tigris');rows.append(entry)
    P.write(output/(prefix+'_submissions.json'),rows)
    answer=S.command(argv,env=S.environment(launch['python'],'tigris'))
    match=re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?',answer)
    if not match:raise ValueError('Ambiguous submission; reconcile journal before retry')
    entry.update(status='recorded',job=match.group(1));P.write(output/(prefix+'_submissions.json'),rows)
    print(f"{phase} job: {entry['job']}",flush=True)
    return entry['job']


def owner_path(output):return Path(output).parent/('fast256_'+execution_hash()+'.production-owner.json')


def verify_existing(output,plan):
    rows=S.journal(output,'production')
    if {r['phase'] for r in rows}!={'train','evaluate','report'}:
        raise ValueError('Partial production chain; reconcile, do not blindly resubmit')
    owner=owner_path(output)
    if not owner.exists() or P.read(owner)!=dict(output=str(output),campaign_sha256=plan['content_hash']):
        raise ValueError('Production owner differs')
    if any(P.checked(Path(output)/n)!=plan for n in ('production.json','production/campaign.json')):
        raise ValueError('Stored production admission differs')
    return rows


def dispatch(output):
    output=Path(output).resolve(strict=True);launch=P.load(output,worker=True,environment=True)
    with S.locks(launch['repo'],output.parent):
        plan=P.production_spec(output);rows=S.journal(output,'production');owner=owner_path(output)
        if rows:
            rows=verify_existing(output,plan)
            return dict(status='already_submitted',jobs={r['phase']:r['job'] for r in rows})
        if owner.exists() or any((output/n).exists() for n in ('production.json','production')):
            raise ValueError('Existing owner/production artifacts; reconcile before dispatch')
        P.check_storage(output)
        S.preflight(output,launch,[request(output,launch,p,plan=plan) for p in ('train','evaluate','report')])
        P.write(output/'production.json',plan);P.write(output/'production/campaign.json',plan)
        P.write(owner,dict(output=str(output),campaign_sha256=plan['content_hash']))
        train=submit(output,launch,'train',prefix='production',plan=plan)
        evaluate=submit(output,launch,'evaluate',prefix='production',plan=plan,dependency='afterok:'+train)
        report=submit(output,launch,'report',prefix='production',plan=plan,dependency='afterok:'+evaluate)
        return dict(status='submitted',jobs=dict(train=train,evaluate=evaluate,report=report),
            production=str(output/'production'),train_hours=plan['models'][0]['train_hours'],
            evaluate_hours=plan['models'][0]['evaluate_hours'],control_submitted=False)


def resume(output):
    """Explicit same-campaign retry from last complete state, never best weights."""
    output=Path(output).resolve(strict=True);launch=P.load(output,worker=True,environment=True)
    with S.locks(launch['repo'],output.parent):
        plan=P.load_production(output);rows=verify_existing(output,plan)
        latest={r['phase']:r['job'] for r in rows}
        previous=sorted(output.glob('resume_[0-9][0-9][0-9]_submissions.json'))
        for path in previous:
            rows=S.journal(output,path.name.removesuffix('_submissions.json'))
            if {r['phase'] for r in rows} not in ({'train','evaluate','report'},{'evaluate','report'}):
                raise ValueError('Partial retry chain; reconcile')
            latest.update({r['phase']:r['job'] for r in rows})
        statuses=S.job_states(set(latest.values()))
        if set(statuses)!=set(latest.values()) or any(s not in TERMINAL for s in statuses.values()):
            raise ValueError('Prior jobs active or accounting incomplete; no retry')
        if (output/'production/reports/fast256_report.json').exists():raise ValueError('Already reported')
        from .execution import engine
        from part_ev_masked_predictive import runtime as R
        import torch
        E=engine();run=P.run_directory(output);complete=(run/'training_complete.json').exists()
        if complete:E.audit_training(output,plan,'masked')
        elif (run/'last.pt').exists():
            s=R.fresh('masked',torch.device('cpu'),1,distributed=False)
            s.scaler=torch.amp.GradScaler('cpu',enabled=True)
            E.restore(run/'last.pt',s,plan,'masked')
        elif list((run/'validation').glob('*.json')) or list((run/'selected').glob('*.pt')):
            raise ValueError('Training files without restart; do not reset')
        phases=['evaluate','report'] if complete else ['train','evaluate','report']
        P.check_storage(output);S.preflight(output,launch,[request(output,launch,p,plan=plan) for p in phases])
        if len(previous)>=999:raise ValueError('Retry ledger exhausted')
        prefix=f'resume_{len(previous)+1:03d}';jobs={};dep=None
        for phase in phases:
            jobs[phase]=submit(output,launch,phase,prefix=prefix,plan=plan,dependency=dep)
            dep='afterok:'+jobs[phase]
        return dict(status='retry_submitted',jobs=jobs,restart='latest_complete_state',control_submitted=False)
