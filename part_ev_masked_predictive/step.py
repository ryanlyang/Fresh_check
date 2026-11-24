"""One complete clean/auxiliary/Ranger/EMA attempt; no partial checkpoint commits."""
from contextlib import nullcontext
import math
import torch
from . import runtime as R,modes,views
from .config import coefficient,momentum,recipe
from part_ev_fullscale.protocol import learning_rate
from part_ev_screen.precision import model_precision,audit_precision
from part_ev_fullscale.checkpoint import capture_rank_state


def no_sync(session,enabled):
    return session.model.no_sync() if enabled and hasattr(session.model,'no_sync') else nullcontext()


def attempt(s,values,truth,*,schedule_label=None,amp=True,audit=False,diagnostic=False,inspect_latents=False):
    u=s.attempts+1 if schedule_label is None else schedule_label
    if schedule_label is not None and not diagnostic:raise ValueError('Schedule override is diagnostic only')
    device=s.device;world=s.world;R.input_health(values,device,world)
    if not R.agree(truth.ndim==1 and len(truth)==len(values[0]) and bool(((truth>=0)&(truth<10)).all()),device,world):
        raise ValueError('Malformed clean labels')
    if len(set(R.gather(len(truth),world)))!=1:raise ValueError('Unequal clean local batches')
    lam=coefficient(u) if s.arm=='masked' else 0.;active=lam>0
    s.base.train();s.opt.zero_grad(set_to_none=True)
    for g in s.opt.param_groups:g['lr']=learning_rate(u-1,recipe())
    with torch.profiler.record_function('masked.bn_snapshot'):
        b0=modes.snapshot(s.base.student) if active else None
    if world>1 and active and len(set(R.gather(modes.fingerprint(b0),world)))!=1:
        raise ValueError('Pre-clean BN buffers disagree')
    expected=torch.float16 if amp and device.type=='cuda' else torch.float32
    precision=None
    with no_sync(s,active),torch.profiler.record_function('masked.clean'):
        with model_precision(s.base.student,device,training=amp),(
                audit_precision(s.base.student,expected_dtype=expected) if audit else nullcontext()) as precision:
            logits,retained,anchor=s.model('clean',values,retain=active)
            ce=torch.nn.functional.cross_entropy(logits,truth)+anchor
        if not R.agree(bool(torch.isfinite(ce)),device,world):raise FloatingPointError('Nonfinite clean CE')
        with torch.profiler.record_function('masked.clean_backward'):s.scaler.scale(ce).backward()
    ce_value=float(ce.detach());del logits,ce,anchor
    b1=modes.snapshot(s.base.student) if active else dict(s.base.student.named_buffers())
    buffers_finite=bool(torch.stack([torch.isfinite(x).all() for x in b1.values()]).all())
    if not R.agree(buffers_finite,device,world):raise FloatingPointError('Nonfinite clean BN buffers')
    if world>1 and len(set(R.gather(modes.fingerprint(b1),world)))!=1:raise ValueError('Post-clean BN buffers disagree')
    local_sum=0.;local_eligible=0;target_count=0;mask_hash=None;mode_counts=[0,0]
    aux_precision=visible_precision=teacher_precision=None;latent_statistics=[]
    native_rng=capture_rank_state(s.base) if audit else None
    if active:
        v=R.all_call(lambda:views.create(retained,s.generators),world)
        if not R.agree(bool(v.healthy),device,world):raise ValueError('Malformed auxiliary geometry/features')
        local_eligible=int(v.eligible.sum());global_eligible=sum(R.gather(local_eligible,world))
        if audit:mask_hash=v.fingerprint()
        if not global_eligible:
            with torch.profiler.record_function('masked.auxiliary_backward'):
                s.scaler.scale(s.model('flush')).backward()
        else:
            slots=math.ceil(len(v.selected)/32)
            with modes.auxiliary(s.base.student),model_precision(s.base.student,device,training=amp):
                for i in range(slots):
                    item=R.all_call(lambda:views.slot(v,32*i,min(32*(i+1),len(v.selected))),world)
                    with no_sync(s,i!=slots-1):
                        if item is None:loss=s.model('flush')
                        else:
                            with torch.profiler.record_function('masked.teacher'),(
                                    audit_precision(s.teacher.model,expected_dtype=expected) if audit else nullcontext()) as teacher_precision:
                                targets=s.teacher.targets(item,amp=amp)
                            with torch.profiler.record_function('masked.visible_predictor'),(
                                    audit_precision(s.base.predictor,expected_dtype=expected) if audit else nullcontext()) as aux_precision,(
                                    audit_precision(s.base.student,expected_dtype=expected) if audit else nullcontext()) as visible_precision:
                                output=s.model('auxiliary',item['visible'],locations=item['locations'],
                                    query_valid=item['query_valid'],targets=targets,buffers=b0,diagnostic=inspect_latents)
                                if inspect_latents:
                                    losses,statistics=output;latent_statistics.append(statistics.detach())
                                else:losses=output
                            loss=lam*world/global_eligible*losses.sum()
                            local_sum+=float(losses.detach().sum());target_count+=int(item['query_valid'].sum())
                            mode_counts[0]+=int((~item['modes']).sum());mode_counts[1]+=int(item['modes'].sum())
                        if not R.agree(bool(torch.isfinite(loss)),device,world):raise FloatingPointError('Nonfinite auxiliary loss')
                        with torch.profiler.record_function('masked.auxiliary_backward'):s.scaler.scale(loss).backward()
                    if item is not None:
                        del targets,losses,output
                        if inspect_latents:del statistics
                    del loss,item
        R.all_call(lambda:modes.assert_equal(dict(s.base.student.named_buffers()),b1),world)
        del v,retained,b0
    else:global_eligible=0
    if audit:
        from .restart import compare
        R.all_call(lambda:compare(capture_rank_state(s.base),native_rng,exact=True),world)
    if not active or not global_eligible:
        if s.arm=='masked':
            for p in s.base.predictor.parameters():p.grad=None
    old=s.scaler.get_scale();s.scaler.unscale_(s.opt)
    gradients=[p.grad for p in s.base.parameters() if p.grad is not None]
    healthy=bool(torch.stack([p.detach().float().norm() for p in gradients]).norm().isfinite())
    if len(set(R.gather(healthy,world)))!=1:raise FloatingPointError('Asymmetric pre-step gradient health')
    if not healthy and not s.scaler.is_enabled():raise FloatingPointError('Nonfinite unscaled gradients')
    with torch.profiler.record_function('masked.optimizer'):
        s.scaler.step(s.opt);s.scaler.update()
    skipped=int(s.scaler.get_scale()<old)
    if len(set(R.gather((skipped,s.scaler.get_scale(),s.opt.step_counter),world)))!=1:
        raise ValueError('Asymmetric scaler/Ranger decision')
    if not R.agree(healthy or bool(skipped),device,world):raise FloatingPointError('Overflow without scaler recovery')
    with torch.profiler.record_function('masked.ema'):
        if s.teacher is not None:s.teacher.commit(s.base.student,u,not skipped)
    s.attempts+=1;s.successes+=1-skipped;s.auxiliary_attempts+=int(active);s.last_schedule_label=u
    sums=R.gather((ce_value,local_sum,local_eligible,target_count,*mode_counts),world)
    ce_global=sum(x[0] for x in sums)/world
    aux=sum(x[1] for x in sums)/max(1,global_eligible)
    latent=None if not latent_statistics else dict(zip(
        ('prediction_norm','target_norm','prediction_variance','target_variance','cosine'),
        torch.stack(latent_statistics).mean(0).cpu().tolist()))
    return dict(loss=ce_global+lam*aux,ce=ce_global,auxiliary=aux,coefficient=lam,momentum=momentum(u),latents=latent,
        eligible=global_eligible,targets=sum(x[3] for x in sums),modes=[sum(x[4] for x in sums),sum(x[5] for x in sums)],
        skipped=skipped,scale=s.scaler.get_scale(),ema_successes=s.teacher.successes if s.teacher else None,
        attempt=s.attempts,schedule_label=u,diagnostic=diagnostic,precision=precision,auxiliary_precision=aux_precision,
        visible_precision=visible_precision,teacher_precision=teacher_precision,
        mask_sha256=mask_hash,bn_isolated=True,rng_isolated=True if audit else None)
