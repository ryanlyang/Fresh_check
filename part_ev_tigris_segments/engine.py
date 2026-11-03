"""Native single-GPU updates, atomic restart state and deadline-safe handoffs."""
import copy
import math
import os
from pathlib import Path
import time

from . import protocol as P
from part_ev_state_tigris.benchmark import create_driver


def create_engine():
    engine,_ = create_driver()  # Private namespace; does not amend the queued benchmark.
    engine.P = P
    engine.lock_finalists = lock_finalists
    return engine


def allocation_deadline(environ=None,*,wall_clock=time.time,monotonic=time.monotonic):
    env = os.environ if environ is None else environ
    try: start,end = float(env['SLURM_JOB_START_TIME']),float(env['SLURM_JOB_END_TIME'])
    except (KeyError,ValueError) as exc: raise ValueError('Need Slurm start/end timestamps') from exc
    if not math.isfinite(start) or not math.isfinite(end) or end-start!=P.POLICY['walltime_seconds']:
        raise ValueError('Training segment must have exactly 72-hour walltime')
    remaining = end-wall_clock()-P.POLICY['shutdown_margin_seconds']
    if not math.isfinite(remaining) or remaining<=0: raise ValueError('No safe runtime remains')
    return monotonic()+remaining


class PlannedStop(Exception):
    pass


class Deadline:
    def __init__(self,engine,deadline,*,clock=time.monotonic):
        self.engine,self.deadline,self.clock = engine,deadline,clock

    def check(self):
        if self.engine.STOP: raise SystemExit(75)  # External cancellation/error is NOT a successful slot.
        if self.clock()>=self.deadline: raise PlannedStop()


def assert_nested_close(D,actual,expected):
    if isinstance(expected,D.torch.Tensor):
        D.torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('Restart state keys differ')
        for key in expected: assert_nested_close(D,actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected): raise ValueError('Restart state lengths differ')
        for a,b in zip(actual,expected): assert_nested_close(D,a,b)
    elif actual!=expected:
        raise ValueError('Restart scalar state differs')


def restart_check(output,spec,runtime,device,D):
    """Small disposable round-trip once, on the actual GH200, before fresh training."""
    directory = Path(output)/'restart_check'
    result_path = directory/'result.json'
    identity = dict(campaign_sha256=spec['content_hash'],runtime=runtime,
                    updates=P.POLICY['startup_restart_updates'],passed=True)
    if result_path.exists():
        if P.read_json(result_path)!=P.hashed(identity): raise ValueError('Restart evidence changed')
        return
    model = D.make_model(P.TASK,device)
    recipe = spec['recipes'][P.TASK['recipe']]
    opt = D.build_optimizer(model,recipe)
    scaler = D.torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    loader,_ = D.build_loader('train',spec['inventory'],512,seed=P.TASK['seed'])
    try:
        batch = next(iter(loader))
        D.seed_everything(P.TASK['seed'])
        warm = P.POLICY['startup_warmup_updates']
        for step in range(warm):
            D.check_stop()
            D.training_step(model,opt,scaler,batch,device,recipe,step)
        D.save_checkpoint(directory/'probe.pt',model,opt,scaler,dict(task=P.TASK,step=warm),spec,runtime)
        def advance():
            rows=[]
            for step in range(warm,warm+P.POLICY['startup_restart_updates']):
                D.check_stop()
                row=D.training_step(model,opt,scaler,batch,device,recipe,step)
                if row['amp_skipped']: raise ValueError('AMP not settled in restart check')
                rows.append(row['loss'])
            return D.torch.tensor(rows)
        expected_loss = advance()
        expected = copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict()))
        _,rng = D.load_checkpoint(directory/'probe.pt',model,opt,scaler,spec,runtime,P.TASK)
        D.restore_rank_state(model,rng)
        assert_nested_close(D,advance(),expected_loss)
        assert_nested_close(D,dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict()),expected)
        P.write_json(result_path,P.hashed(identity))
        D.progress('restart_check_passed',updates=P.POLICY['startup_restart_updates'])
    finally:
        D.shutdown_loader(loader)


def initial_state():
    return dict(task=P.TASK,step=0,validated_checks=0,best_accuracy=-1.,best_check=None,
                best_file=None,best_sha256=None,amp_skipped_steps=0,clipped_steps=0,
                train_seconds=0.,validation_seconds=0.)


def finish_training(output,spec,state,D):
    run=P.task_directory(output)
    selected=D.trusted_load(run/state['best_file'])
    D.atomic_torch_save(selected,run/'best_model_val.pt')
    P.write_json(run/'training_complete.json',dict(state,campaign_sha256=spec['content_hash'],
        runtime=P.expected_runtime(spec),best_checkpoint_sha256=P.file_sha256(run/'best_model_val.pt')))
    P.audit_training(output,spec)
    D.progress('training_complete',**P.TASK)


def train_body(output,spec,runtime,device,D,timings):
    run = P.task_directory(output)
    recipe = spec['recipes'][P.TASK['recipe']]
    model = D.make_model(P.TASK,device)
    optimizer = D.build_optimizer(model,recipe)
    scaler = D.torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    P.write_json(run/'optimizer.json',D.optimizer_metadata(model,optimizer))
    last,state,rng = run/'last.pt',initial_state(),None
    if last.exists(): state,rng = D.load_checkpoint(last,model,optimizer,scaler,spec,runtime,P.TASK)
    if state['step']==recipe['total_steps'] and state['validated_checks']==recipe['epochs']:
        # Recover a crash between the final restart commit and completion marker,
        # without replaying a finished million-step data stream.
        finish_training(output,spec,state,D)
        return
    train_loader,_ = D.build_loader('train',spec['inventory'],512,seed=P.TASK['seed'])
    val_loader = None
    try:
        val_loader,_ = D.build_loader('val',spec['inventory'],512,seed=P.TASK['seed'])
        iterator,val_iterator = iter(train_loader),iter(val_loader)
        if rng is not None:
            started=time.monotonic()
            try:
                D.replay(iterator,state['step'])
                D.replay(val_iterator,state['validated_checks']*(recipe['validation_events_per_epoch']//512))
            finally: timings['replay_seconds']+=time.monotonic()-started
            D.restore_rank_state(model,rng)
        else: D.seed_everything(P.TASK['seed'])
        while True:
            try: D.check_stop()
            except (PlannedStop,SystemExit):
                D.save_checkpoint(last,model,optimizer,scaler,state,spec,runtime)
                raise
            if state['step']//recipe['steps_per_epoch']>state['validated_checks']:
                D.save_checkpoint(last,model,optimizer,scaler,state,spec,runtime)
                # A stop within validation leaves the pre-validation restart intact.
                metric=D.validate(model,val_iterator,device,recipe['validation_events_per_epoch'])
                state['validated_checks']+=1
                state['validation_seconds']+=metric['seconds']
                metric.update(check=state['validated_checks'],step=state['step'])
                if metric['accuracy']>state['best_accuracy']:
                    state.update(best_accuracy=metric['accuracy'],best_check=metric['check'],
                                 best_file=f"selected/check_{metric['check']:02d}.pt")
                    # Keep the previous committed best intact if this transaction is interrupted.
                    candidate=run/state['best_file']
                    D.atomic_torch_save(dict(model=model.state_dict(),task=P.TASK,runtime=runtime,
                        campaign_sha256=spec['content_hash'],validation=metric),candidate)
                    state['best_sha256']=P.file_sha256(candidate)
                P.write_json(run/'validation'/f"check_{metric['check']:02d}.json",metric)
                D.save_checkpoint(last,model,optimizer,scaler,state,spec,runtime)
                D.progress('validation',**P.TASK,**metric)
            if state['step']==recipe['total_steps']:
                finish_training(output,spec,state,D)
                return
            started=time.monotonic()
            row=D.training_step(model,optimizer,scaler,next(iterator),device,recipe,state['step'])
            state['train_seconds']+=time.monotonic()-started
            state['step']+=1
            state['amp_skipped_steps']+=int(row['amp_skipped'])
            state['clipped_steps']+=int(row['clipped'])
            if state['step']%100==0:
                D.progress('training',**P.TASK,step=state['step'],**row,
                           jets_per_second=state['step']*512/state['train_seconds'])
            if state['step']%P.POLICY['checkpoint_interval']==0:
                if state['amp_skipped_steps']>=.05*state['step']:
                    raise RuntimeError('Excessive AMP skips; not a planned handoff')
                D.save_checkpoint(last,model,optimizer,scaler,state,spec,runtime)
    finally:
        D.shutdown_loader(train_loader)
        if val_loader is not None: D.shutdown_loader(val_loader)


def train_segment(chain,plan,spec,index,device,runtime,deadline,*,engine=None,clock=time.monotonic):
    D=create_engine() if engine is None else engine
    output=Path(plan['campaign'])
    P.require_admission(chain,plan,spec)
    if runtime!=P.expected_runtime(spec): raise ValueError('Keep the admitted single-GH200 runtime')
    with P.file_lock(P.task_directory(output)/'.train.lock',blocking=False):
        before=P.snapshot(output,spec)
        if before!=P.previous_snapshot(chain,plan,index): raise ValueError('Checkpoint differs from prior handoff')
        D.check_stop=Deadline(D,deadline,clock=clock).check
        started=time.monotonic()
        timings=dict(replay_seconds=0.)
        planned=False
        if not before['complete']:
            try:
                restart_check(output,spec,runtime,device,D)
                train_body(output,spec,runtime,device,D,timings)
            except PlannedStop: planned=True
        if D.STOP: raise SystemExit(75)
        after=P.snapshot(output,spec)
        if after['complete']: status='already_complete' if before['complete'] else 'complete'
        elif planned and (after['step'],after['validated_checks'])>(before['step'],before['validated_checks']): status='paused'
        else: raise RuntimeError('No committed progress; inspect replay cost/error before continuing')
        row=P.hashed(dict(chain_sha256=plan['content_hash'],index=index,status=status,before=before,after=after,
                          elapsed_seconds=time.monotonic()-started,**timings,job_id=os.environ.get('SLURM_JOB_ID')))
        P.write_json(P.receipt_path(chain,index),row)
        D.progress('segment_handoff',**row)
        return row


def lock_finalists(output,spec):
    output=Path(output)
    with P.file_lock(output/'.selection.lock'):
        if not P.snapshot(output,spec)['complete']: raise ValueError('Full schedule not complete')
        trained=P.audit_training(output,spec)
        row=dict(campaign_sha256=spec['content_hash'],checkpoints=[dict(task=P.TASK,
            checkpoint=str(P.task_directory(output)/'best_model_val.pt'),sha256=trained['checkpoint_sha256'])])
        path=output/'selection/locked_checkpoints.json'
        if path.exists() and P.read_json(path)!=row: raise ValueError('Selected checkpoint changed')
        if not path.exists(): P.write_json(path,row)
        return row


def report(output,spec,D):
    from part_ev_message_study.controls import validate_evaluation
    with P.file_lock(Path(output)/'.report.lock',blocking=False):
        lock_finalists(output,spec)
        trained=P.audit_training(output,spec)
        evaluated,_=validate_evaluation(Path(output),spec,P.TASK,trained,recompute=D.classification_metrics)
        result=P.hashed(dict(schema=P.SCHEMA,campaign_sha256=spec['content_hash'],task=P.TASK,
                             population='official_validation_5M',independent_test=False,model=evaluated))
        directory=Path(output)/'reports'
        P.write_json(directory/'tigris_state_report.json',result)
        m=evaluated['metrics']
        lines=['# Layerwise all-eight State-EV on one GH200','',
               'Seed 101; 1M updates; 50 checks. Full-5M validation, not independent test results.',
               'No cross-platform checkpoint migration or automatic SPORC contrasts.','',
               '| Accuracy (%) | Macro OVO AUC | Macro OVR AUC | Selected check |',
               '|---:|---:|---:|---:|',
               f"| {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | {m['macro_auc_ovr']:.6f} | {trained['best_check']} |",'',
               '| Check | Accuracy (%) | Macro OVO AUC |','|---:|---:|---:|']
        for row in trained['validation_checks']:
            lines.append(f"| {row['check']} | {100*row['accuracy']:.4f} | {row['macro_auc_ovo']:.6f} |")
        lines+=['','| Signal | Efficiency | QCD rejection |','|---|---:|---:|']
        for signal,targets in m['qcd_rejection'].items():
            for efficiency,row in targets.items():
                value='ZB' if row['rejection'] is None else f"{row['rejection']:.2f}"
                lines.append(f'| {signal} | {100*float(efficiency):g}% | {value} |')
        (directory/'tigris_state_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return result
