"""Named parity failures and full-state continuation through real ROOT streams."""
from pathlib import Path
import traceback
import torch
from part_ev_masked_predictive import runtime as R, restart as K, verification as V, step as old
from part_ev_masked_predictive.config import PARITY
from part_ev_masked_speed.step import attempt as speed_attempt
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load, capture_rank_state, restore_rank_state
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.data import batch_fingerprint
from part_ev_screen.precision import model_precision
from .execution import attempt, engine


def fresh(device, *, trim=False):
    s=R.fresh('masked',device,1,trim=trim)
    s.scaler=torch.amp.GradScaler(device.type,init_scale=128,enabled=device.type=='cuda')
    return s


def losses(actual,expected,path):
    for key in ('loss','ce','auxiliary'):
        limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(expected[key]))
        if not abs(actual[key]-expected[key])<=limit:raise ValueError(f'{path}.{key}: restart loss bound')


def states(actual,expected,path):
    for key in ('rng','generators','scaler'):K.compare(actual[key],expected[key],exact=True,path=path+'.'+key)
    return max(K.compare(actual[k],expected[k],path=path+'.'+k) for k in actual if k not in ('rng','generators','scaler'))


def parity(device, *, amp, microbatch=256, batch=512, particles=97):
    s=fresh(device);V.activate(s.base.student)
    fixture=R.fixed_batch(batch,particles);lengths=3+torch.arange(batch)*17%(particles-2);lengths[0]=particles
    fixture[0]['pf_mask'] &= torch.arange(particles)[None,None,:]<lengths[:,None,None]
    x,y=R.tensors(fixture,device);initial=K.capture(s);snapshots=[];rows=[];gradients=[]
    hook=s.opt.register_step_pre_hook(lambda opt,a,k:gradients.append({
        n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in s.base.named_parameters()}))
    try:
        for candidate in (False,True):
            K.restore(s,initial)
            kwargs=dict(schedule_label=100000,diagnostic=True,amp=amp,audit=True)
            if not candidate: row=old.attempt(s,x,y,**kwargs)
            elif microbatch==256: row=attempt(s,x,y,inspect_latents=True,**kwargs)
            else: row=speed_attempt(s,x,y,microbatch=microbatch,**kwargs)
            if row['skipped']:raise ValueError('parity.optimizer: skipped update')
            rows.append(row);snapshots.append(K.capture(s))
    finally:hook.remove()
    if len(gradients)!=2 or gradients[0].keys()!=gradients[1].keys():raise ValueError('parity.gradient coverage')
    worst=0.
    for name,value in gradients[0].items():
        other=gradients[1][name]
        if value is None:
            if other is not None:raise ValueError('parity.unused_gradient.'+name)
        else:
            try:V.close(other,value,gradient=True,amp=amp)
            except AssertionError as exc:raise ValueError('parity.gradient.'+name+': '+str(exc)) from exc
            worst=max(worst,float((other-value).abs().max()))
    for key in ('loss','ce','auxiliary'):V.close(torch.tensor(rows[1][key]),torch.tensor(rows[0][key]),amp=amp)
    for key in ('mask_sha256','eligible','targets','modes','coefficient','momentum','skipped','scale','ema_successes','bn_isolated','rng_isolated'):
        if rows[0][key]!=rows[1][key]:raise ValueError('parity.transaction.'+key)
    error=states(snapshots[1],snapshots[0],'parity')
    for name,_ in s.base.named_buffers():
        K.compare(snapshots[1]['model'][name],snapshots[0]['model'][name],exact=True,path='parity.bn_buffer.'+name)
    return dict(passed=True,amp=amp,microbatch=microbatch,batch=batch,particles=particles,
        gradients=len(gradients[0]),max_gradient_absolute=worst,max_state_absolute=error,
        exact_masks_rng_scaler=True,exact_bn_buffers=True,varied_lengths=True,full_optimizer_teacher=True)


def legacy_diagnostic(device,historical_error):
    trials=[]
    for amp in (False,True):
        try:row=parity(device,amp=amp,microbatch=32)
        except Exception as exc:row=dict(passed=False,amp=amp,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        trials.append(row)
    return dict(collected=True,historical_error=historical_error,trials=trials,
        note='fast32 is excluded from production. Named reproduction only; a passing repeat does not explain the old failure.')


def restart_probe(s,directory,*,transitions=False):
    directory=Path(directory);s.base.student.mod.trimmer.enabled=True;s.base.student.mod.trimmer._counter=5
    initial=K.capture(s);path=directory/'initial.pt';atomic_torch_save(initial,path)
    labels=[19999,20000,20001,39999,40000,899999,900000] if transitions else list(range(100000,100007))
    references=[];trials=[]
    for trial in ('reference','repeat','disk','resumed'):
        K.restore(s,trusted_load(path) if trial in ('disk','resumed') else initial);rows=[];worst=0.
        for i,label in enumerate(labels):
            if trial=='resumed' and i==3:
                atomic_torch_save(K.capture(s),directory/'middle.pt');s=fresh(s.device,trim=True)
                K.restore(s,trusted_load(directory/'middle.pt'))
            x,y=R.tensors(R.fixed_batch(8,17+i%3),s.device)
            row=attempt(s,x,y,schedule_label=label,diagnostic=True,audit=True,inspect_latents=True)
            if row['skipped']:raise ValueError('restart.optimizer: skipped update')
            if i==3:
                rng=capture_rank_state(s.base);flags=[(m,m.training) for m in s.base.modules()]
                s.base.student.eval()
                with torch.no_grad(),model_precision(s.base.student,s.device,training=False):
                    if not torch.isfinite(s.base.student(*x)).all():raise ValueError('Nonfinite probe validation')
                for m,flag in flags:m.training=flag
                restore_rank_state(s.base,rng)
            state=K.capture(s)
            record={k:row[k] for k in ('loss','ce','auxiliary','coefficient','momentum','eligible','targets',
                'skipped','scale','ema_successes','attempt','schedule_label','mask_sha256','bn_isolated')}
            record.update(update=i+1,lookahead=s.opt.step_counter,validation_batches=int(i>=3))
            if trial=='reference':references.append((record,state))
            else:
                expected,saved=references[i];losses(record,expected,'restart.'+trial)
                if any(record[k]!=expected[k] for k in record if k not in ('loss','ce','auxiliary')):
                    raise ValueError('restart.'+trial+': transaction changed')
                worst=max(worst,states(state,saved,'restart.'+trial))
            rows.append(record)
        trials.append(dict(trial=trial,updates=rows,exact_initial_restore=True,max_state_error=worst))
    return dict(passed=True,updates=7,trials=trials,full_state=True,validation_boundaries=1,
        policy=PARITY,schedule='transitions' if transitions else 'active',labels=labels)


def real_stream_resume(output,spec,resources,device):
    """Seven actual auxiliary updates, disk restore, train AND validation replay.

    Labels are diagnostic 100k labels, not a claim of 100k trained updates. The
    stream cursors are actual consumed batches (3 train, 2 validation at save).
    """
    E=engine();opened=[];references=[];input_hashes=[];val_hashes=[];saved_path=Path(output)/'probes/root_midpoint.pt'
    s=fresh(device,trim=True);s.base.student.mod.trimmer._counter=5;V.activate(s.base.student)
    def loader(split):
        value,_=build_loader(split,spec['inventory'],512,workers=resources['workers'],seed=101)
        opened.append(value);return iter(value)
    def close():
        while opened:shutdown_loader(opened.pop())
    def validation(s,iterator):
        fingerprints=[]
        def track():
            for _ in range(2):
                batch=next(iterator);fingerprints.append(batch_fingerprint(batch));yield batch
        metric=E.validate(s.base.student,iter(track()),device,0,1,1024,audit=True)
        if metric is None:raise TimeoutError('Incomplete ROOT restart validation')
        return fingerprints,metric
    try:
        train=loader('train');val=loader('val')
        for i in range(7):
            batch=next(train);input_hashes.append(batch_fingerprint(batch));x,y=R.tensors(batch,device)
            row=attempt(s,x,y,schedule_label=100000+i,diagnostic=True,audit=True,inspect_latents=True)
            if row['skipped']:raise ValueError('ROOT restart update skipped')
            if i in (1,4):val_hashes.append(validation(s,val))
            references.append((row,K.capture(s)))
            if i==2:atomic_torch_save(dict(session=K.capture(s),train_batches=3,val_batches=2),saved_path)
        close();s=fresh(device,trim=True);train=loader('train');val=loader('val')
        saved=trusted_load(saved_path)
        for _ in range(saved['train_batches']):next(train)
        for _ in range(saved['val_batches']):next(val)
        K.restore(s,saved['session'])  # Restore native RNG AFTER dataset replay.
        for i in range(3,7):
            batch=next(train)
            if batch_fingerprint(batch)!=input_hashes[i]:raise ValueError('ROOT input changed after replay')
            x,y=R.tensors(batch,device);row=attempt(s,x,y,schedule_label=100000+i,diagnostic=True,audit=True,inspect_latents=True)
            if i==4:
                fingerprints,metric=validation(s,val);expected_hashes,old_metric=val_hashes[1]
                if fingerprints!=expected_hashes:raise ValueError('Validation stream changed after replay')
                for key in ('accuracy','macro_auc_ovo'):
                    if abs(metric[key]-old_metric[key])>PARITY['restart_absolute']:raise ValueError('Validation continuation changed')
            expected,state=references[i];losses(row,expected,'ROOT.resume')
            for key in ('mask_sha256','eligible','targets','modes','skipped','scale','ema_successes','attempt'):
                if row[key]!=expected[key]:raise ValueError('ROOT.resume.'+key)
            states(K.capture(s),state,'ROOT.resume')
        return dict(passed=True,real_stream=True,full_state=True,exact_roundtrip=True,
            next_masks_equal=True,next_inputs_equal=True,optimizer_teacher_equal=True,rng_restored=True,
            updates=7,train_replayed=3,val_replayed=2,validation_boundaries=2,input_sha256=input_hashes,
            validation_sha256=[v[0] for v in val_hashes],diagnostic_schedule=True)
    finally:close()


def production_checkpoint(output,spec,device):
    E=engine();s=fresh(device);state=E.initial_state('masked')
    x,y=R.tensors(R.fixed_batch(8,17),device)
    for _ in range(3):
        row=attempt(s,x,y,audit=True)
        state.update(step=s.attempts,successful_updates=s.successes,amp_skipped_steps=s.attempts-s.successes,
                     auxiliary_attempts=s.auxiliary_attempts,ema_successes=s.teacher.successes)
    saved=K.capture(s);path=Path(output)/'probes/production_restart/last.pt';E.save(path,s,state,spec)
    s=fresh(device);restored,rng=E.restore(path,s,spec,'masked');restore_rank_state(s.base,rng)
    K.compare(K.capture(s),saved,exact=True,path='production_checkpoint');K.compare(restored,state,exact=True)
    return dict(passed=True,complete_attempts=3,exact_roundtrip=True,production_serializer=True)
