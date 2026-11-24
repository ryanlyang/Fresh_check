"""Small disposable profiles and latent diagnostics, never persistent targets."""
from pathlib import Path
import torch
from . import runtime as R,modes,views
from .config import COUNTS
from part_ev_screen.precision import model_precision
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state
from part_ev_workspace.diagnostics import memory

TRAIN_SCOPES={'masked.bn_snapshot','masked.clean','masked.clean_backward','masked.views','masked.teacher',
              'masked.visible_predictor','masked.auxiliary_backward','masked.optimizer','masked.ema'}


def profile(arm,device):
    s=R.fresh(arm,device,1,trim=False);x,y=R.tensors(R.fixed_batch(2,17),device);records=[]
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    for training in (True,False):
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,
                with_stack=False,with_flops=True) as p:
            if training:
                s.base.train();s.opt.zero_grad(set_to_none=True)
                with torch.profiler.record_function('masked.bn_snapshot'):b0=modes.snapshot(s.base.student)
                with torch.profiler.record_function('masked.clean'),model_precision(s.base.student,device,training=False):
                    logits,kept,anchor=s.base('clean',x,retain=arm=='masked');loss=torch.nn.functional.cross_entropy(logits,y)+anchor
                with torch.profiler.record_function('masked.clean_backward'):loss.backward()
                if arm=='masked':
                    # Two auxiliary rows from the retained two-jet collection;
                    # augmentation draw padding creates no extra encoder forward.
                    kept=tuple(a.repeat(2,1,1) for a in kept);v=views.create(kept,views.Generators(0))
                    item=views.slot(v,0,2)
                    with torch.profiler.record_function('masked.teacher'):target=s.teacher.targets(item,amp=False)
                    with modes.auxiliary(s.base.student),model_precision(s.base.student,device,training=False):
                        with torch.profiler.record_function('masked.visible_predictor'):
                            aux=s.base('auxiliary',item['visible'],locations=item['locations'],query_valid=item['query_valid'],targets=target,buffers=b0).mean()*.1
                        with torch.profiler.record_function('masked.auxiliary_backward'):aux.backward()
                with torch.profiler.record_function('masked.optimizer'):s.opt.step()
                with torch.profiler.record_function('masked.ema'):
                    if s.teacher:s.teacher.commit(s.base.student,100000,True)
            else:
                s.base.student.eval()
                with torch.profiler.record_function('masked.export'),torch.no_grad(),model_precision(s.base.student,device,training=False):
                    exported=R.fresh_student(arm,device,trim=False);exported.load_state_dict(s.base.student.state_dict(),strict=True);exported.eval()
                    expected=s.base.student(*x);actual=exported(*x)
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            R.synchronize(device)
        scopes={e.key:dict(calls=e.count,cpu_us=e.cpu_time_total,device_us=getattr(e,'device_time_total',0.))
            for e in p.key_averages() if e.key.startswith('masked.')}
        required=TRAIN_SCOPES if training and arm=='masked' else (
            {'masked.bn_snapshot','masked.clean','masked.clean_backward','masked.optimizer','masked.ema'} if training else {'masked.export'})
        if not required.issubset(scopes) or any(scopes[k]['cpu_us']+scopes[k]['device_us']<=0 for k in required):
            raise ValueError('Missing bounded profiling scope')
        records.append(dict(training=training,events=2,particles=17,stages=scopes,
            profiler_visible_flops=sum(e.flops or 0 for e in p.key_averages()),
            flops_convention='2 FLOPs/MAC; partial visible operators at small profile shape, not full training FLOPs'))
    return records


def sample(s):
    if s.arm=='control':return dict(events=8,particles=17,deployed_parameters=COUNTS['control'],auxiliary=False)
    rng=capture_rank_state(s.base);flags=[(m,m.training) for m in s.base.modules()]
    original=modes.snapshot(s.base.student)
    try:
        x,_=R.tensors(R.fixed_batch(8,17),s.device)
        v=views.create((x[1],x[2],x[3]),views.Generators(s.rank));item=views.slot(v,0,len(v.selected))
        with torch.no_grad(),modes.auxiliary(s.base.student),model_precision(s.base.student,s.device,training=False):
            target=s.teacher.targets(item,amp=False)
            _,h=torch.func.functional_call(s.base.student,(dict(s.base.student.named_parameters()),original),item['visible'],dict(phase='particles'),strict=True)
            pred,_=s.base.predictor(item['locations'],h,item['visible'][3][:,0],item['query_valid'])
            zero,_=s.base.predictor(item['locations'],torch.zeros_like(h),item['visible'][3][:,0],item['query_valid'])
            p=pred[item['query_valid']];t=target[item['query_valid']]
            result=dict(events=8,particles=17,auxiliary=True,eligible=int(v.eligible.sum()),
                target_variance=float(t.var(0,unbiased=False).mean()),prediction_variance=float(p.var(0,unbiased=False).mean()),
                target_norm=float(t.norm(dim=-1).mean()),prediction_norm=float(p.norm(dim=-1).mean()),
                cosine=float(torch.nn.functional.cosine_similarity(p,t).mean()),zero_kv_max_change=float((pred-zero).abs().max()))
        modes.assert_equal(dict(s.base.student.named_buffers()),original)
        return result
    finally:
        for m,flag in flags:m.training=flag
        restore_rank_state(s.base,rng)
