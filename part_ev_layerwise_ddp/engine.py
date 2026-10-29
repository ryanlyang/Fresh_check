"""Production loop using the measured DDP model/step, with rank-aware recovery."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from . import protocol as P
from part_ev_gpu_scaling import benchmark as B
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load, capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.precision import model_precision, audit_precision
from scripts.run_part_ev_screen import classification_metrics, atomic_npz
from part_ev_screen.data import finite_validation_loader


def progress(stage, **fields):
    print(json.dumps(dict(stage=stage,**fields),allow_nan=False),flush=True)


def root_call(action, rank, world):
    """Propagate rank-zero I/O/validation failures instead of stranding peers."""
    payload = None
    if rank == 0:
        try:
            payload = (True, action())
        except Exception as exc:
            payload = (False, f'{type(exc).__name__}: {exc}')
    if world > 1:
        values = [payload]
        dist.broadcast_object_list(values,src=0)
        payload = values[0]
    if not payload[0]: raise RuntimeError(payload[1])
    return payload[1]


def stopping(device,world):
    return not B.agree(not B.STOP,device,world)


def runtime_record(spec, device, world):
    expected = spec['expected_gpu']
    observed = dict(name=torch.cuda.get_device_name(device),
                    total_memory=torch.cuda.get_device_properties(device).total_memory,
                    nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
    if any(row != expected for row in B.gather(observed,world)):
        raise ValueError('GPU or NCCL settings differ from the successful scaling benchmark')
    return P.expected_runtime(spec)


def initial_state():
    return dict(task=P.TASK,step=0,validated_checks=0,best_accuracy=-1.,best_check=None,
                best_file=None,amp_skipped_steps=0,train_seconds=0.,validation_seconds=0.)


def save(path,base,optimizer,scaler,state,spec,runtime,rank,world):
    identity = (B.parameter_digest(base),scaler.get_scale(),optimizer.step_counter)
    if len(set(B.gather(identity,world))) != 1:
        raise RuntimeError('Model/scaler/Lookahead replicas diverged; checkpoint not committed')
    states = B.gather(capture_rank_state(base),world)
    root_call(lambda: atomic_torch_save(dict(schema=P.SCHEMA,campaign_sha256=spec['content_hash'],
        runtime=runtime,model=base.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        state=copy.deepcopy(state),rank_states=states),path),rank,world)


def restore(path,base,optimizer,scaler,spec,runtime,rank,world):
    saved = trusted_load(path)
    if (saved.get('schema') != P.SCHEMA or saved['campaign_sha256'] != spec['content_hash']
            or saved['runtime'] != runtime or saved['state']['task'] != P.TASK
            or len(saved['rank_states']) != world):
        raise ValueError('Restart checkpoint belongs to another campaign/task/runtime')
    recipe = spec['recipes'][P.TASK['recipe']]
    state = saved['state']
    if (not 0 <= state['step'] <= recipe['total_steps']
            or not 0 <= state['validated_checks'] <= state['step']//recipe['steps_per_epoch']
            or state['step']//recipe['steps_per_epoch']-state['validated_checks'] > 1):
        raise ValueError('Invalid training/validation cursor')
    if state['best_check'] is not None and state['best_file'] != f"selected/check_{state['best_check']:02d}.pt":
        raise ValueError('Invalid selected checkpoint path')
    base.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    return state,saved['rank_states'][rank]


def replay(iterator, count, device, rank, world, *, rank_zero=False):
    """Bound collective waits during potentially long exact data-stream replay."""
    for start in range(0,count,P.POLICY['heartbeat_batches']):
        if stopping(device,world): raise SystemExit(75)  # Keep the previous restart file.
        size = min(P.POLICY['heartbeat_batches'],count-start)
        def advance():
            for _ in range(size): next(iterator)
        if rank_zero: root_call(advance,rank,world)
        else: advance()
        # All ranks arrive every small chunk, including when only rank zero reads.
        if world > 1: dist.barrier()
        if rank == 0 and (start+size)%10000 == 0:
            progress('resume_data_replay',stream='val' if rank_zero else 'train',batches=start+size,total=count)


def validate(base,iterator,device,rank,world,events,*,audit=False,batch_size=512):
    if events <= 0 or events%batch_size: raise ValueError('Incomplete validation budget')
    scores,truths,precision = [],[],None
    base.eval()  # Unwrapped SyncBN in eval does NOT perform training collectives.
    start = time.monotonic()
    for offset in range(0,events//batch_size,P.POLICY['heartbeat_batches']):
        if stopping(device,world): return None
        def chunk():
            nonlocal precision
            with torch.no_grad(),model_precision(base,device,training=False):
                for index in range(offset,min(offset+P.POLICY['heartbeat_batches'],events//batch_size)):
                    values,y = B.tensors(next(iterator),device)
                    with (audit_precision(base,expected_dtype=torch.float32) if audit and index==0 else nullcontext()) as observed:
                        prediction = base(*values).float()
                    if len(y) != batch_size or not torch.isfinite(prediction).all():
                        raise ValueError('Nonfinite/incomplete FP32 validation batch')
                    if observed is not None: precision = observed
                    scores.append(prediction.cpu().numpy());truths.append(y.cpu().numpy())
        root_call(chunk,rank,world)
    def finish():
        values,labels = np.concatenate(scores),np.concatenate(truths)
        result = classification_metrics(values,labels)
        if len(labels) != events: raise ValueError('Incomplete validation population')
        result['seconds'] = time.monotonic()-start
        if audit: result['precision'] = precision
        return result
    return root_call(finish,rank,world)


def fresh(spec,device,world):
    base,model = B.model_pair(device,world)
    optimizer = build_optimizer(base,spec['recipes'][P.TASK['recipe']])
    scaler = torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    return base,model,optimizer,scaler


def preflight(output,spec,rank,world,device,runtime):
    directory = Path(output)/'preflight'
    if root_call(lambda:(directory/'result.json').exists(),rank,world):
        raise ValueError('Preflight evidence already exists; review it or use a fresh campaign')
    base,model,opt,scaler = fresh(spec,device,world)
    recipe = spec['recipes'][P.TASK['recipe']]
    loader,_ = build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
    val_loader = None
    try:
        iterator = iter(loader)
        first = next(iterator)
        B.seed_everything(101+rank)
        state = initial_state()
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(P.POLICY['preflight_updates']):
            values,truth = B.tensors(first if index==0 else next(iterator),device)
            row = B.step(base,model,opt,scaler,values,truth,recipe,index,world,audit=index==0)
            state['step']+=1;state['amp_skipped_steps']+=row['skipped']
        # Keep a real local batch in memory for exact stochastic resume replay.
        values,truth = B.tensors(first,device)
        started=time.monotonic()
        save(directory/'restart_probe.pt',base,opt,scaler,state,spec,runtime,rank,world)
        seconds=max(B.gather(time.monotonic()-started,world))
        expected=[]
        for index in range(state['step'],state['step']+7):
            row=B.step(base,model,opt,scaler,values,truth,recipe,index,world)
            expected.append((row['loss'],row['skipped']))
        parameters={n:p.detach().clone() for n,p in base.named_parameters()}
        slow=[p.clone() for p in opt.slow]
        expected_counter,expected_scale=opt.step_counter,scaler.state_dict()
        state,rng=restore(directory/'restart_probe.pt',base,opt,scaler,spec,runtime,rank,world)
        restore_rank_state(base,rng)
        actual=[]
        for index in range(state['step'],state['step']+7):
            row=B.step(base,model,opt,scaler,values,truth,recipe,index,world)
            actual.append((row['loss'],row['skipped']))
        torch.testing.assert_close(torch.tensor(actual),torch.tensor(expected),rtol=1e-5,atol=1e-6)
        for name,p in base.named_parameters():
            torch.testing.assert_close(p,parameters[name],rtol=1e-5,atol=1e-6)
        for a,b in zip(opt.slow,slow): torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        if opt.step_counter != expected_counter or scaler.state_dict() != expected_scale:
            raise ValueError('Optimizer/scaler restart differs')
        if not B.agree(sum(r[1] for r in actual)==0,device,world):
            raise ValueError('AMP did not settle before checkpoint recovery check')
        if rank==0:
            val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
        metric=validate(base,iter(val_loader) if rank==0 else None,device,rank,world,
                        P.POLICY['preflight_validation_events'],audit=True)
        if metric is None: raise SystemExit(75)
        peaks=B.gather(torch.cuda.max_memory_allocated(device),world)
        result=dict(campaign_sha256=spec['content_hash'],runtime=runtime,
            restart=dict(passed=True,updates_checked=7,rank_count=world),validation=metric,
            checkpoint_seconds=seconds,peak_bytes=peaks,
            passed=all(n < .9*spec['expected_gpu']['total_memory'] for n in peaks))
        result['content_hash']=P.digest(result)
        root_call(lambda:P.write_json(directory/'result.json',result),rank,world)
        if not result['passed']: raise RuntimeError('Preflight lacks memory headroom')
        if rank==0: progress('preflight_complete',**P.recommendation(spec,result,P.read_json(Path(spec['scaling'])/'gpus_2.json')))
    finally:
        shutdown_loader(loader)
        if val_loader is not None: shutdown_loader(val_loader)


def train(output,spec,rank,world,device,runtime):
    run=P.run_directory(output)
    recipe=spec['recipes'][P.TASK['recipe']]
    P.require_approval(output,spec)
    guard=P.file_lock(run/'.training.lock',blocking=False) if rank==0 else nullcontext()
    with guard:
        if root_call(lambda:(run/'training_complete.json').exists(),rank,world):
            root_call(lambda:audit_training(output,spec),rank,world)
            return
        base,model,opt,scaler=fresh(spec,device,world)
        root_call(lambda:P.write_json(run/'optimizer.json',optimizer_metadata(base,opt)),rank,world)
        state,rng=initial_state(),None
        last=run/'last.pt'
        if root_call(last.exists,rank,world):
            state,rng=restore(last,base,opt,scaler,spec,runtime,rank,world)
        train_loader,_=build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
        val_loader=None
        try:
            iterator=iter(train_loader)
            if rank==0: val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
            val_iterator=iter(val_loader) if rank==0 else None
            if rng is not None:
                replay(iterator,state['step'],device,rank,world)
                replay(val_iterator,state['validated_checks']*(recipe['validation_events_per_epoch']//512),device,rank,world,rank_zero=True)
                restore_rank_state(base,rng)
            else:
                B.seed_everything(101+rank)
            while True:
                if stopping(device,world):
                    save(last,base,opt,scaler,state,spec,runtime,rank,world)
                    raise SystemExit(75)
                if state['validated_checks'] < state['step']//recipe['steps_per_epoch']:
                    save(last,base,opt,scaler,state,spec,runtime,rank,world)
                    metric=validate(base,val_iterator,device,rank,world,recipe['validation_events_per_epoch'])
                    if metric is None: raise SystemExit(75)  # Pre-validation restart is intact.
                    state['validated_checks']+=1
                    state['validation_seconds']+=metric['seconds']
                    metric.update(check=state['validated_checks'],step=state['step'])
                    if metric['accuracy'] > state['best_accuracy']:
                        state.update(best_accuracy=metric['accuracy'],best_check=metric['check'],
                                     best_file=f"selected/check_{metric['check']:02d}.pt")
                        # Versioned candidates prevent an interrupted validation commit
                        # from overwriting the best model named by the last restart.
                        root_call(lambda:atomic_torch_save(dict(model=base.state_dict(),task=P.TASK,runtime=runtime,
                            campaign_sha256=spec['content_hash'],validation=metric),run/state['best_file']),rank,world)
                    root_call(lambda:P.write_json(run/'validation'/f"check_{metric['check']:02d}.json",metric),rank,world)
                    save(last,base,opt,scaler,state,spec,runtime,rank,world)
                    if rank==0: progress('validation',**P.TASK,**metric)
                if state['step']==recipe['total_steps']:
                    def finish():
                        selected=trusted_load(run/state['best_file'])
                        atomic_torch_save(selected,run/'best_model_val.pt')
                        P.write_json(run/'training_complete.json',dict(state,campaign_sha256=spec['content_hash'],
                            runtime=runtime,best_checkpoint_sha256=P.file_sha256(run/'best_model_val.pt')))
                        audit_training(output,spec)
                    root_call(finish,rank,world)
                    if rank==0: progress('training_complete',**state)
                    return
                start=time.monotonic()
                values,truth=B.tensors(next(iterator),device)
                row=B.step(base,model,opt,scaler,values,truth,recipe,state['step'],world)
                state['step']+=1;state['amp_skipped_steps']+=row['skipped']
                # Rank-zero wallclock used for report, not a sum of GPU durations.
                state['train_seconds']+=time.monotonic()-start
                if rank==0 and state['step']%100==0:
                    progress('training',**P.TASK,step=state['step'],loss=row['loss'],
                             global_jets_per_second=state['step']*512/state['train_seconds'],
                             amp_skipped_steps=state['amp_skipped_steps'])
                if state['step']%P.POLICY['checkpoint_interval']==0:
                    if len(set(B.gather((state['step'],state['amp_skipped_steps']),world)))!=1:
                        raise RuntimeError('AMP/step counters diverged')
                    if state['amp_skipped_steps'] >= .05*state['step']:
                        raise RuntimeError('Excessive AMP skips; training stopped')
                    save(last,base,opt,scaler,state,spec,runtime,rank,world)
        finally:
            shutdown_loader(train_loader)
            if val_loader is not None: shutdown_loader(val_loader)


def audit_training(output,spec):
    from part_ev_state_fullscale.reporting import validate_training
    from part_ev_message_study.controls import runtime_identity
    expected=P.read_json(Path(output)/'preflight/result.json')['runtime']
    return validate_training(Path(output),spec,P.TASK,runtime_identity(expected))


def evaluate(output,spec,device):
    run=P.run_directory(output)
    trained=audit_training(output,spec)
    checksum=trained['checkpoint_sha256']
    directory=run/'full_validation'
    with P.file_lock(directory/'.evaluation.lock',blocking=False):
        selection=dict(campaign_sha256=spec['content_hash'],checkpoints=[dict(task=P.TASK,
            checkpoint=str((run/'best_model_val.pt').resolve()),sha256=checksum)])
        lock=Path(output)/'selection/locked_checkpoints.json'
        if lock.exists() and P.read_json(lock) != selection:
            raise ValueError('Selected checkpoint changed after full-validation lock')
        if not lock.exists(): P.write_json(lock,selection)
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(B.build_model('state_layerwise',use_amp=False)).to(device).eval()
        checkpoint=trusted_load(run/'best_model_val.pt')
        model.load_state_dict(checkpoint['model'],strict=True)
        for module in model.modules():
            if hasattr(module,'_counter'): module._counter=5
        predictions,labels,records=[],[],[]
        for i,record in enumerate(sorted(spec['inventory']['splits']['val']['files'],key=lambda r:r['name'])):
            if B.STOP: raise SystemExit(75)
            path=directory/f'file_{i:03d}.npz';sidecar=path.with_suffix('.json')
            contract=dict(campaign_sha256=spec['content_hash'],checkpoint_sha256=checksum,file=record,
                          entry_start=0,entry_stop=record['num_entries'])
            if path.exists() and sidecar.exists():
                saved=P.read_json(sidecar)
                if saved['contract'] != contract or saved['sha256'] != P.file_sha256(path):
                    raise ValueError('Committed full-validation prediction changed')
            else:
                scores,truths=[],[]
                with torch.no_grad(),model_precision(model,device,training=False):
                    for batch in finite_validation_loader(spec['inventory'],record):
                        if B.STOP: raise SystemExit(75)
                        values,y=B.tensors(batch,device)
                        logits=model(*values).float().cpu().numpy()
                        if not np.isfinite(logits).all(): raise ValueError('Nonfinite validation logits')
                        scores.append(logits);truths.append(y.cpu().numpy())
                logits,y=np.concatenate(scores),np.concatenate(truths)
                if len(y)!=record['num_entries']: raise ValueError('Incomplete validation file')
                atomic_npz(path,logits=logits,labels=y)
                P.write_json(sidecar,dict(contract=contract,sha256=P.file_sha256(path)))
            with np.load(path,allow_pickle=False) as saved:
                predictions.append(saved['logits']);labels.append(saved['labels'])
            records.append(P.read_json(sidecar))
            progress('full_validation_file',completed=i+1,total=50)
        logits,y=np.concatenate(predictions),np.concatenate(labels)
        if len(y)!=5_000_000 or not np.array_equal(np.bincount(y,minlength=10),np.full(10,500_000)):
            raise ValueError('Expected 5M validation jets, 500k/class')
        metrics=classification_metrics(logits,y,rejection=True)
        P.write_json(directory/'metrics.json',dict(task=P.TASK,campaign_sha256=spec['content_hash'],
            checkpoint_sha256=checksum,identity_manifest=records,metrics=metrics))


def report(output,spec):
    from part_ev_message_study.controls import validate_evaluation
    trained=audit_training(output,spec)
    row,_=validate_evaluation(Path(output),spec,P.TASK,trained,recompute=classification_metrics)
    result=dict(campaign_sha256=spec['content_hash'],model=row,independent_test=False,
                historical_controls_runtime_matched=False,training_gpus=2,
                note='Single seed, full-5M validation overlapping selection. No automatic mixed-runtime contrasts.')
    result['content_hash']=P.digest(result)
    P.write_json(Path(output)/'reports/layerwise_ddp_report.json',result)
    metric=row['metrics']
    lines=['# Layerwise all-eight State-EV: two-A100 full schedule','',result['note'],'',
           '| Seed | Parameters | Accuracy (%) | Macro OVO AUC | Selected check |',
           '|---:|---:|---:|---:|---:|',
           f"| 101 | 2,327,462 | {100*metric['accuracy']:.4f} | {metric['macro_auc_ovo']:.6f} | {row['best_check']} |",'',
           '## Validation learning curve','', '| Check | Step | Accuracy (%) | OVO AUC |', '|---:|---:|---:|---:|']
    lines += [f"| {r['check']} | {r['step']} | {100*r['accuracy']:.4f} | {r['macro_auc_ovo']:.6f} |" for r in trained['validation_checks']]
    lines += ['', '## QCD rejection', '', '| Signal | Efficiency | Rejection |', '|---|---:|---:|']
    for signal,targets in metric['qcd_rejection'].items():
        for efficiency,values in targets.items():
            rejection=values['rejection']
            lines.append(f"| {signal} | {100*float(efficiency):g}% | " + ('ZB' if rejection is None else f'{rejection:.2f}') + ' |')
    lines += ['', 'ZB means zero accepted QCD events; passing counts remain in JSON.',
              'The original shared-State-EV, wider-ParT and baseline jobs/artifacts were not changed.', '']
    (Path(output)/'reports/layerwise_ddp_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
