"""Full 1M-update CE training, transactional selection and resumable 5M eval."""
from contextlib import ExitStack
import copy
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import protocol as P, runtime as R
from .config import POLICY, recipe, COUNTS, TRAINABLE, TEACHER
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load, capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import optimizer_metadata
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.data import finite_validation_loader
from part_ev_screen.precision import model_precision, audit_precision
from scripts.run_part_ev_screen import classification_metrics, atomic_npz
from .step import attempt
from . import restart as S


def expected_runtime(spec):
    return dict(spec['runtime'],campaign_sha256=spec['content_hash'])


def verify_runtime(spec,gpu):
    if gpu!=spec['runtime']['gpu']: raise ValueError('GPU identity differs from admitted timing')


def initial_state(arm):
    return dict(task=P.task(arm),step=0,validated_checks=0,best_accuracy=-1.,best_check=None,
        best_file=None,best_checkpoint_sha256=None,amp_skipped_steps=0,train_seconds=0.,validation_seconds=0.,
        successful_updates=0,auxiliary_attempts=0,ema_successes=0)


def save(path,s,state,spec):
    if state['step']!=s.attempts or state['successful_updates']!=s.successes:raise ValueError('Not a complete attempt boundary')
    if not R.agree(R.finite_state((s.base.state_dict(),s.opt.state_dict(),s.scaler.state_dict(),
            None if s.teacher is None else s.teacher.state_dict())),s.device,s.world):
        raise FloatingPointError('Refusing to replace complete restart with nonfinite state')
    ids=R.gather((R.parameter_digest(s.base),s.scaler.get_scale(),s.opt.step_counter,s.attempts,
        None if s.teacher is None else R.parameter_digest(s.teacher.model)),s.world)
    if len(set(ids))!=1:raise ValueError('Checkpoint replicas/teacher diverged')
    ranks=R.gather(dict(rank=s.rank,rng=capture_rank_state(s.base),
        generators=None if s.generators is None else s.generators.state_dict()),s.world)
    def write():
        shared=s.state_dict()
        for key in ('rng','generators','rank'):shared.pop(key)
        atomic_torch_save(dict(schema=P.SCHEMA,campaign_sha256=spec['content_hash'],runtime=expected_runtime(spec),
            session=shared,state=copy.deepcopy(state),rank_states=ranks),path)
    R.root_call(write,s.rank,s.world)


def restore(path,s,spec,arm):
    row=trusted_load(path);state=row['state'];shared=row['session']
    if (row.get('schema')!=P.SCHEMA or row.get('campaign_sha256')!=spec['content_hash']
            or row['runtime']!=expected_runtime(spec) or state['task']!=P.task(arm)
            or len(row['rank_states'])!=s.world or not 0<=state['step']<=1000000
            or not 0<=state['validated_checks']<=state['step']//20000
            or state['step']//20000-state['validated_checks']>1
            or shared['attempts']!=state['step'] or shared['last_schedule_label']!=state['step']
            or shared['successes']!=state['step']-state['amp_skipped_steps']
            or state['successful_updates']!=shared['successes'] or state['auxiliary_attempts']!=shared['auxiliary_attempts']
            or state['ema_successes']!=(shared['successes'] if arm=='masked' else 0)
            or shared['auxiliary_attempts']!=(max(0,min(state['step'],899999)-20000) if arm=='masked' else 0)):
        raise ValueError('Wrong/incomplete restart identity, phase or cursor')
    if any(type(state[k]) is not int for k in ('step','validated_checks','amp_skipped_steps')) or not 0<=state['amp_skipped_steps']<=state['step']:
        raise ValueError('Invalid attempt counters')
    if any(not math.isfinite(state[k]) or state[k]<0 for k in ('train_seconds','validation_seconds')) or not R.finite_state(shared):
        raise ValueError('Nonfinite restart')
    if state['best_check'] is None:
        if state['validated_checks'] or state['best_file'] is not None or state['best_checkpoint_sha256'] is not None:
            raise ValueError('Selected metadata without validation')
    else:
        if (type(state['best_check']) is not int or not 1<=state['best_check']<=state['validated_checks']
                or state['best_file']!=f"selected/check_{state['best_check']:02d}.pt" or not 0<=state['best_accuracy']<=1):
            raise ValueError('Invalid selected metadata')
        selected=Path(path).parent/state['best_file']
        if not selected.is_file() or P.sha(selected)!=state['best_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    rank_state=row['rank_states'][s.rank]
    if rank_state['rank']!=s.rank:raise ValueError('Wrong rank restart')
    full=dict(shared,rank=s.rank,rng=rank_state['rng'],generators=rank_state['generators'])
    s.load_state_dict(full,restore_rng=False)
    current=s.state_dict()
    for key in ('model','optimizer','teacher','scaler','generators'):S.compare(current[key],full[key],exact=True)
    return copy.deepcopy(state),rank_state['rng']


def validate(base,iterator,device,rank,world,events,*,audit=False):
    if events<=0 or events%512: raise ValueError('Complete validation batches required')
    base.eval();scores=[];truths=[];precision=None;start=time.monotonic()
    for offset in range(0,events//512,100):
        if R.stopping(device,world): return None
        def chunk():
            nonlocal precision
            with torch.no_grad(),model_precision(base,device,training=False):
                for i in range(offset,min(offset+100,events//512)):
                    values,y=R.tensors(next(iterator),device)
                    with (audit_precision(base,expected_dtype=torch.float32) if audit and i==0 else ExitStack()) as observed:
                        pred=base(*values).float()
                    if len(y)!=512 or not torch.isfinite(pred).all(): raise ValueError('Invalid FP32 validation')
                    if audit and i==0: precision=observed
                    scores.append(pred.cpu().numpy());truths.append(y.cpu().numpy())
        R.root_call(chunk,rank,world)
    def finish():
        start_metrics=time.monotonic();logits,y=np.concatenate(scores),np.concatenate(truths)
        result=classification_metrics(logits,y)
        result.update(seconds=time.monotonic()-start,metric_seconds=time.monotonic()-start_metrics)
        if audit: result['precision']=precision
        return result
    return R.root_call(finish,rank,world)


def audit_training(output,spec,arm):
    from part_ev_state_fullscale.reporting import validate_training
    from part_ev_message_study.controls import runtime_identity
    result=validate_training(Path(output)/'production',dict(spec,parameters=TRAINABLE),P.task(arm),runtime_identity(expected_runtime(spec)))
    state=result['training']
    if not 0<=state['amp_skipped_steps']<.05*state['step']: raise ValueError('Excessive/invalid AMP skips')
    if any(not math.isfinite(state[k]) or state[k]<0 for k in ('train_seconds','validation_seconds')):
        raise ValueError('Invalid timing')
    if (state['successful_updates']!=state['step']-state['amp_skipped_steps']
            or state['auxiliary_attempts']!=(879999 if arm=='masked' else 0)
            or state['ema_successes']!=(state['successful_updates'] if arm=='masked' else 0)):
        raise ValueError('Incomplete EMA/phase counters')
    student=R.fresh_student(arm,torch.device('cpu'),trim=False)
    student.load_state_dict(trusted_load(P.run_directory(output,arm)/'best_model_val.pt')['model'],strict=True)
    return result


def train(output,spec,arm,device,rank,world):
    run=P.run_directory(output,arm)
    with ExitStack() as guard:
        R.root_call(lambda:guard.enter_context(P.file_lock(run/'.training.lock',blocking=False)) and None,rank,world)
        if R.root_call(lambda:(run/'training_complete.json').exists(),rank,world):
            R.root_call(lambda:audit_training(output,spec,arm),rank,world);return
        s=R.fresh(arm,device,world,rank=rank)
        base=s.base.student;opt=s.opt;scaler=s.scaler
        def metadata():
            row=optimizer_metadata(s.base,opt);path=run/'optimizer.json'
            if path.exists() and P.read(path)!=row: raise ValueError('Optimizer contract changed')
            if not path.exists(): P.write(path,row)
        R.root_call(metadata,rank,world)
        state,rng=initial_state(arm),None;last=run/'last.pt'
        if R.root_call(last.exists,rank,world):
            state,rng=R.all_call(lambda:restore(last,s,spec,arm),world)
        resources=spec['resources'];train_loader=val_loader=None
        try:
            train_loader,_=R.all_call(lambda:build_loader('train',spec['inventory'],resources['local_batch'],
                rank=rank,world_size=world,workers=resources['workers'],seed=101),world)
            iterator=iter(train_loader)
            def val_setup():
                nonlocal val_loader
                val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
            R.root_call(val_setup,rank,world)
            val_iterator=iter(val_loader) if rank==0 else None
            if rng is not None:
                R.replay(iterator,state['step'],device,rank,world)
                R.replay(val_iterator,state['validated_checks']*2500,device,rank,world,root_only=True)
                restore_rank_state(s.base,rng)
            else: R.seed_everything(101+rank)
            while True:
                if R.stopping(device,world):
                    save(last,s,state,spec);raise SystemExit(75)
                if state['validated_checks']<state['step']//20000:
                    save(last,s,state,spec)
                    metric=validate(base,val_iterator,device,rank,world,1_280_000)
                    if metric is None: raise SystemExit(75)
                    state['validated_checks']+=1;state['validation_seconds']+=metric['seconds']
                    metric.update(check=state['validated_checks'],step=state['step'],task=P.task(arm),
                                  campaign_sha256=spec['content_hash'])
                    if metric['accuracy']>state['best_accuracy']:
                        state.update(best_accuracy=metric['accuracy'],best_check=metric['check'],
                                     best_file=f"selected/check_{metric['check']:02d}.pt")
                        def write_selected():
                            path=run/state['best_file']
                            atomic_torch_save(dict(model=base.state_dict(),task=P.task(arm),
                                runtime=expected_runtime(spec),campaign_sha256=spec['content_hash'],
                                validation=metric),path)
                            return P.sha(path)
                        state['best_checkpoint_sha256']=R.root_call(write_selected,rank,world)
                    R.root_call(lambda:P.write(run/'validation'/f"check_{metric['check']:02d}.json",metric),rank,world)
                    save(last,s,state,spec)
                    if rank==0: R.log('validation',**metric)
                if state['step']==1_000_000:
                    def finish():
                        if P.sha(run/state['best_file'])!=state['best_checkpoint_sha256']:
                            raise ValueError('Selected candidate changed before completion')
                        selected=trusted_load(run/state['best_file'])
                        atomic_torch_save(selected,run/'best_model_val.pt')
                        P.write(run/'training_complete.json',dict(state,campaign_sha256=spec['content_hash'],
                            runtime=expected_runtime(spec),best_checkpoint_sha256=P.sha(run/'best_model_val.pt')))
                        audit_training(output,spec,arm)
                    R.root_call(finish,rank,world);return
                start=time.monotonic()
                batch=R.all_call(lambda:next(iterator),world);values,truth=R.tensors(batch,device)
                result=attempt(s,values,truth,inspect_latents=(s.attempts+1)%100==0)
                state['step']+=1;state['amp_skipped_steps']+=result['skipped']
                state.update(successful_updates=s.successes,auxiliary_attempts=s.auxiliary_attempts,
                    ema_successes=s.teacher.successes if s.teacher else 0)
                state['train_seconds']+=time.monotonic()-start
                if rank==0 and state['step']%100==0:
                    R.log('training',task=P.task(arm),step=state['step'],loss=result['loss'],
                        global_jets_per_second=state['step']*512/state['train_seconds'],
                        amp_skipped_steps=state['amp_skipped_steps'],ce=result['ce'],auxiliary=result['auxiliary'],
                        coefficient=result['coefficient'],momentum=result['momentum'],eligible=result['eligible'],
                        targets=result['targets'],mask_modes=result['modes'],ema_successes=result['ema_successes'],
                        latent_microbatch_means=result['latents'])
                if state['step']%1000==0:
                    if state['amp_skipped_steps']>=.05*state['step']: raise ValueError('Excessive AMP skips')
                    save(last,s,state,spec)
        finally:
            if train_loader is not None: shutdown_loader(train_loader)
            if val_loader is not None: shutdown_loader(val_loader)


def evaluate(output,spec,arm,device):
    run=P.run_directory(output,arm);trained=audit_training(output,spec,arm)
    checksum=trained['checkpoint_sha256'];directory=run/'full_validation'
    with P.file_lock(directory/'.evaluation.lock',blocking=False):
        selection_path=Path(output)/'production/selection/locked_checkpoints.json'
        with P.file_lock(selection_path.parent/'.selection.lock'):
            selection=P.read(selection_path) if selection_path.exists() else dict(campaign_sha256=spec['content_hash'],checkpoints=[])
            if selection['campaign_sha256']!=spec['content_hash']: raise ValueError('Selection campaign changed')
            entry=dict(task=P.task(arm),checkpoint=str((run/'best_model_val.pt').resolve()),sha256=checksum)
            old=[e for e in selection['checkpoints'] if e['task']==P.task(arm)]
            if old and old!=[entry]: raise ValueError('Selection changed')
            if not old: selection['checkpoints'].append(entry);P.write(selection_path,selection)
        base=R.fresh_student(arm,device,training_world=spec['resources']['gpus'])
        base.load_state_dict(trusted_load(run/'best_model_val.pt')['model'],strict=True);base.eval()
        for module in base.modules():
            if hasattr(module,'_counter'): module._counter=5
        predictions=[];labels=[];records=[]
        for i,record in enumerate(sorted(spec['inventory']['splits']['val']['files'],key=lambda r:r['name'])):
            if R.stopped(): raise SystemExit(75)
            path=directory/f'file_{i:03d}.npz';sidecar=path.with_suffix('.json')
            contract=dict(campaign_sha256=spec['content_hash'],checkpoint_sha256=checksum,file=record,
                          entry_start=0,entry_stop=record['num_entries'])
            if sidecar.exists():
                saved=P.read(sidecar)
                if not path.is_file() or saved['contract']!=contract or saved['sha256']!=P.sha(path):
                    raise ValueError('Committed prediction changed/missing')
            else:
                scores=[];truths=[]
                with torch.no_grad(),model_precision(base,device,training=False):
                    for batch in finite_validation_loader(spec['inventory'],record):
                        if R.stopped(): raise SystemExit(75)
                        values,y=R.tensors(batch,device);pred=base(*values).float().cpu().numpy()
                        if not np.isfinite(pred).all(): raise ValueError('Nonfinite FP32 predictions')
                        scores.append(pred);truths.append(y.cpu().numpy())
                logits,y=np.concatenate(scores),np.concatenate(truths)
                if len(y)!=record['num_entries']: raise ValueError('Incomplete evaluation file')
                atomic_npz(path,logits=logits,labels=y)
                P.write(sidecar,dict(contract=contract,sha256=P.sha(path)))
            with np.load(path,allow_pickle=False) as saved:
                logits,y=saved['logits'],saved['labels']
                if (logits.shape!=(record['num_entries'],10) or y.shape!=(record['num_entries'],)
                        or not np.issubdtype(y.dtype,np.integer) or not np.isfinite(logits).all()
                        or np.any(y<0) or np.any(y>=10)): raise ValueError('Invalid cached predictions')
                predictions.append(logits);labels.append(y)
            records.append(P.read(sidecar));R.log('full_validation_file',arm=arm,completed=i+1,total=50)
        logits,y=np.concatenate(predictions),np.concatenate(labels)
        if len(y)!=5_000_000 or not np.array_equal(np.bincount(y,minlength=10),np.full(10,500000)):
            raise ValueError('Expected full balanced 5M validation')
        metrics=classification_metrics(logits,y,rejection=True)
        P.write(directory/'metrics.json',dict(task=P.task(arm),campaign_sha256=spec['content_hash'],
            checkpoint_sha256=checksum,identity_manifest=records,metrics=metrics))


def report(output,spec):
    from part_ev_message_study.controls import validate_evaluation
    from .status import allocated_cost
    models=[];common_labels=None
    for arm in COUNTS:
        trained=audit_training(output,spec,arm)
        row,labels=validate_evaluation(Path(output)/'production',spec,P.task(arm),trained,recompute=classification_metrics)
        if common_labels is not None and not np.array_equal(common_labels,labels):
            raise ValueError('Paired evaluation labels/order differ')
        common_labels=labels
        models.append(row)
    delta=100*(models[1]['metrics']['accuracy']-models[0]['metrics']['accuracy'])
    result=P.hashed(dict(complete=True,launch_sha256=spec['launch_sha256'],campaign_sha256=spec['content_hash'],
        models=models,accuracy_delta_pp=delta,independent_test=False,platform=spec['platform'],
        benchmark_sha256=spec['benchmark_sha256'],reused_controls=False,
        benchmark_costs=P.checked(Path(output)/'benchmark.json')['models'],
        scheduler_allocations=allocated_cost(output),
        note='Single seed; full validation overlaps selection. Compare only within this platform.'))
    directory=Path(output)/'production/reports'
    P.write(directory/'masked_report.json',result)
    lines=['# Masked predictive training versus matched LW State-EV','',result['note'],'',
        '| Model | Deployed parameters | Trainable parameters | Frozen teacher | Accuracy (%) | Macro OVO AUC | Check |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in zip(COUNTS,models):
        m=row['metrics'];lines.append(f"| {arm} | {COUNTS[arm]:,} | {TRAINABLE[arm]:,} | {TEACHER if arm=='masked' else 0:,} | {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | {row['best_check']} |")
    lines+=['',f'MaskedPredictive minus control: {delta:+.4f} percentage points.','',
        'Fresh controls; no independent-test or multi-seed claim. Raw rejection counts and benchmark evidence are in JSON.','',
        '## Training-time learning curves','',
        '| Check | Control accuracy (%) | MaskedPredictive accuracy (%) | Delta (pp) |','|---:|---:|---:|---:|']
    for a,b in zip(models[0]['validation_checks'],models[1]['validation_checks']):
        lines.append(f"| {a['check']} | {100*a['accuracy']:.4f} | {100*b['accuracy']:.4f} | {100*(b['accuracy']-a['accuracy']):+.4f} |")
    lines+=['','## QCD rejection','',
        'ZB means zero accepted QCD events, not an infinite-precision estimate. Counts are retained in JSON.','',
        '| Signal / efficiency | Control | MaskedPredictive | Change (%) |','|---|---:|---:|---:|']
    for signal,targets in models[0]['metrics']['qcd_rejection'].items():
        for target,entry in targets.items():
            a=entry['rejection'];b=models[1]['metrics']['qcd_rejection'][signal][target]['rejection']
            left='ZB' if a is None else f'{a:.2f}';right='ZB' if b is None else f'{b:.2f}'
            change='--' if a is None or b is None or a<=0 else f'{100*(b/a-1):+.2f}'
            lines.append(f'| {signal} / {100*float(target):g}% | {left} | {right} | {change} |')
    lines+=['','## Recorded cost','',
        '| Model | Train loop (h) | Validation (h) | Train-loop GPU h | AMP skips | Peak allocated GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,row in zip(COUNTS,models):
        t=row['training'];b=next(m for m in result['benchmark_costs'] if m['arm']==arm)
        peak=max(g['peak_allocated'] for g in b['gpus'])/2**30
        h=t['train_seconds']/3600
        lines.append(f"| {arm} | {h:.2f} | {t['validation_seconds']/3600:.2f} | {h*spec['resources']['gpus']:.2f} | {t['amp_skipped_steps']} | {peak:.2f} |")
    lines+=['',
        'Timers exclude queue/replay/startup and some checkpoint/diagnostic overhead; these are not total allocated GPU-hours.',
        'True allocated GPU-hours require scheduler elapsed time multiplied by the allocation GPU count.',
        'Bounded profiles report visible operator FLOPs at 17 particles and 2 events (2 FLOPs/MAC); they are not total model or corpus-training FLOPs.',
        'No architecture-only comparison is made across the one-GH200/native-BN and two-A100/SyncBN platforms.','']
    cost=result['scheduler_allocations']
    if cost['gpu_hours'] is not None:
        lines.append(f"Recorded campaign GPU allocations (benchmark, training, evaluation and retries): {cost['gpu_hours']:.2f} GPU-hours.")
    else:lines.append('Complete scheduler GPU-hour accounting unavailable; no total is inferred from throughput.')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'masked_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
