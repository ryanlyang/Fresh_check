"""Independent math, leakage, native identity and transactional training probes."""
import copy
import math
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from . import runtime as R,modes,views
from .config import PARITY,COUNTS,TRAINABLE,TEACHER,PREDICTOR
from .model import build_model,Trainable
from .predictor import Predictor,normalize,jet_losses
from .step import attempt
from part_ev_state_compare.model import build_model as reference_model,state_branches
from part_ev_screen.precision import model_precision
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state


def activate(student):
    with torch.no_grad():
        for branch in state_branches(student):branch.output.normal_(std=.02)


def dropout_off(model):
    for m in model.modules():
        if isinstance(m,nn.Dropout):m.p=0.
        if isinstance(getattr(m,'dropout',None),float):m.dropout=0.


def close(a,b,*,gradient=False,amp=False,ddp=False):
    if ddp:r=PARITY['ddp_gradient_rtol' if gradient else 'ddp_forward_rtol'];a0=PARITY['ddp_atol']
    elif amp:r=PARITY['amp_gradient_rtol' if gradient else 'amp_forward_rtol'];a0=PARITY['amp_gradient_atol' if gradient else 'amp_forward_atol']
    else:r=PARITY['gradient_rtol' if gradient else 'forward_rtol'];a0=PARITY['gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=r,atol=a0)


def oracle_prediction(module,locations,h,valid,qv):
    def lin(m,x):return F.linear(x,m.weight,m.bias)
    def norm(m,x):return F.layer_norm(x,(128,),m.weight,m.bias,m.eps)
    q=module.mask_vector+lin(module.position[2],F.gelu(lin(module.position[0],locations)))
    q=q.masked_fill(~qv[...,None],0)
    for block in module.blocks:
        a=block.attention;qs=lin(a.query,norm(block.query_norm,q));ks=lin(a.key,norm(block.source_norm,h))
        vs=lin(a.value,norm(block.source_norm,h));rows=[]
        for b in range(len(h)):
            heads=[]
            for head in range(4):
                sl=slice(32*head,32*(head+1))
                weights=(qs[b,:,sl]@ks[b,valid[b],sl].T/(32**.5)).softmax(-1)
                heads.append(weights@vs[b,valid[b],sl])
            rows.append(lin(a.output,torch.cat(heads,-1)))
        q=(q+torch.stack(rows)).masked_fill(~qv[...,None],0)
        q=(q+lin(block.ffn[2],F.gelu(lin(block.ffn[0],norm(block.ffn_norm,q))))).masked_fill(~qv[...,None],0)
    return F.layer_norm(lin(module.output,q),(128,),eps=1e-5).masked_fill(~qv[...,None],0)


def predictor_oracle(device):
    m=Predictor().to(device).double()
    loc=torch.randn(2,3,3,device=device,dtype=torch.float64,requires_grad=True)
    h=torch.randn(2,5,128,device=device,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[1,0,1,1,0],[0,1,0,1,0]],device=device,dtype=torch.bool)
    qv=torch.tensor([[1,1,1],[1,0,0]],device=device,dtype=torch.bool)
    actual,health=m(loc,h,valid,qv);expected=oracle_prediction(m,loc.masked_fill(~qv[...,None],0),h.masked_fill(~valid[...,None],0),valid,qv)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    target=normalize(torch.randn_like(actual));loss=jet_losses(actual,target,qv).mean()
    expected_loss=torch.stack([F.smooth_l1_loss(expected[b,qv[b]],target[b,qv[b]],beta=1.) for b in range(2)]).mean()
    torch.testing.assert_close(loss,expected_loss,rtol=1e-9,atol=1e-10)
    parameters=(loc,h,*m.parameters())
    ga=torch.autograd.grad(loss,parameters,retain_graph=True);gb=torch.autograd.grad(expected_loss,parameters)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    if not bool(health):raise ValueError('Predictor oracle unhealthy')
    return dict(passed=True,fp64=True,head_loop=True,gradients=True,jet_mean=True)


def view_oracle(device):
    x,_=R.tensors(R.fixed_batch(8,17),device)
    counts=torch.tensor([3,4,5,8,11,14,17,17],device=device)
    x[3][:,0]=torch.arange(17,device=device)[None]<counts[:,None]
    retained=(x[1],x[2],x[3]);g=views.Generators(0);before=copy.deepcopy(g.state_dict())
    rng=capture_rank_state(torch.nn.Identity());v=views.create(retained,g)
    from .restart import compare
    compare(capture_rank_state(torch.nn.Identity()),rng,exact=True)
    if len(v.selected.unique())!=4 or int(v.modes.sum())!=2:raise ValueError('Selection/mode population')
    expected=torch.minimum(v.valid.sum(-1)-2,(v.valid.sum(-1)//4).clamp_min(1)).masked_fill(v.valid.sum(-1)<4,0)
    if not torch.equal(v.query_valid.sum(-1),expected):raise ValueError('Mask cardinality')
    # An independent scalar CPU formula, not a call to production rebase.
    p=v.vectors.cpu();f=retained[0][v.selected].cpu();vis=v.visible.cpu();valid=v.valid.cpu()
    expected_f=f.clone()
    for b in range(len(p)):
        ref=p[b,:,vis[b]].sum(-1);jetpt=math.hypot(float(ref[0]),float(ref[1]));energy=float(ref[3])
        eta_vis=math.asinh(float(ref[2])/max(jetpt,1e-8));phi_vis=math.atan2(float(ref[1]),float(ref[0]))
        for i in torch.where(valid[b])[0].tolist():
            px,py,pz,e=[float(a) for a in p[b,:,i]];pt=math.hypot(px,py)
            eta=math.asinh(pz/max(pt,1e-8));phi=math.atan2(py,px) if pt>1e-8 else 0.
            dy=(1 if eta_vis>=0 else -1)*(eta-eta_vis);da=math.atan2(math.sin(phi-phi_vis),math.cos(phi-phi_vis))
            raw={0:math.log(max(pt,1e-8)),1:math.log(max(e,1e-8)),2:math.log(max(pt,1e-8)/max(jetpt,1e-8)),
                3:math.log(max(e,1e-8)/max(energy,1e-8)),4:math.hypot(dy,da),15:dy,16:da}
            for channel,value in raw.items():
                spec=views.transforms()[channel]
                if spec[1] is not None:value=max(spec[3] if len(spec)>3 else -5.,min(spec[4] if len(spec)>4 else 5.,(value-spec[1])*spec[2]))
                expected_f[b,channel,i]=value
    close(v.features.transpose(1,2)[v.valid].cpu(),expected_f.transpose(1,2)[valid])
    g.load_state_dict(before);perm=torch.arange(16,-1,-1,device=device)
    other=views.create(tuple(a[:,:,perm] for a in retained),g)
    if not torch.equal(v.visible[:,perm],other.visible):raise ValueError('Canonical masking permutation')
    close(v.locations,other.locations)
    packed=views.slot(v,0,len(v.selected))
    if packed is not None:
        for row,n in zip(packed['visible'][3][:,0],packed['visible'][2]):
            if not bool(row.any()):raise ValueError('Empty encoded source')
    return dict(passed=True,mask_counts=True,canonical_permutation=True,visible_rebase=True,
                alignment=True,rng_exact=True,yaml_authenticated=True)


def native_identity(device,rank,world):
    R.seed_everything(101);native=reference_model('state_layerwise',trim=False).to(device);activate(native)
    rng=capture_rank_state(native);student=build_model(trim=False).to(device);student.load_state_dict(native.state_dict(),strict=True)
    values,y=R.tensors(R.split_batch(R.fixed_batch(8,9),rank,world),device)
    outputs=[];grads=[];counters={};handles=[]
    for n,m in [('trim',student.mod.trimmer),('embed',student.mod.embed),('pair',student.mod.pair_embed),
                *[(f'block_{i}',m) for i,m in enumerate(student.mod.blocks)],
                *[(f'class_{i}',m) for i,m in enumerate(student.mod.cls_blocks)]]:
        handles.append(m.register_forward_hook(lambda m,a,o,n=n:counters.__setitem__(n,counters.get(n,0)+1)))
    for model in (native,student):
        model.train();restore_rank_state(model,rng)
        with model_precision(model,device,training=False):pred=model(*values);F.cross_entropy(pred,y).backward()
        outputs.append(pred.detach());grads.append({n:p.grad for n,p in model.named_parameters()})
    for handle in handles:handle.remove()
    close(*outputs)
    for n,p in grads[0].items():close(p,grads[1][n],gradient=True)
    if any(v!=1 for v in counters.values()) or len(counters)!=13:raise ValueError('Repeated/missing native modules')
    if native.state_dict().keys()!=student.state_dict().keys():raise ValueError('Student export keys changed')
    if native.mod.trimmer._counter!=student.mod.trimmer._counter:raise ValueError('Trimmer changed')
    return dict(passed=True,native_calls=counters,export_keys=True,gradients=True,nonzero_state=True,
                deployed_parameters=COUNTS['masked'])


def leakage(device):
    R.seed_everything(101);student=build_model(trim=False).to(device);activate(student);student.train()
    raw,_=R.tensors(R.fixed_batch(8,13),device);raw[3][:,:,-3:]=False
    retained=(raw[1],raw[2],raw[3])
    v=views.create(retained,views.Generators(0));b0=modes.snapshot(student)
    original=modes.snapshot(student)
    def output(view):
        item=views.slot(view,0,len(view.selected))
        with modes.auxiliary(student),model_precision(student,device,training=False):
            a=torch.func.functional_call(student,(dict(student.named_parameters()),b0),item['visible'],dict(phase='particles'),strict=True)[1]
        return item,a
    item,expected=output(v);cases=[]
    # Fixed mask and fixed allowed queries. Perturb each disallowed input family.
    for kind in ('hidden_p4','hidden_independent','full_relative','old_padding','postclean_bn','clean_pair_cache'):
        f=retained[0][v.selected].clone();p=v.vectors.clone();valid=v.valid
        hidden=valid&~v.visible
        if kind=='hidden_p4':p=torch.where(hidden[:,None],p*3,p)
        elif kind=='hidden_independent':f[:,5:15]=torch.where(hidden[:,None],f[:,5:15]+7,f[:,5:15])
        elif kind=='full_relative':f[:,[0,1,2,3,4,15,16]]=123.
        elif kind=='old_padding':
            f=f.masked_fill(~valid[:,None],float('nan'));p=p.masked_fill(~valid[:,None],float('nan'))
        elif kind=='postclean_bn':
            for n,b in student.named_buffers():
                if b.is_floating_point():b.add_(17)
        elif kind=='clean_pair_cache':
            student._state_pair_cache.stem=torch.full((1,),float('nan'),device=device)
            student._state_pair_cache.normalized=torch.full((1,),float('nan'),device=device)
        rebased,points,_,health=views.rebase(f,p,valid,v.visible)
        changed=replace(v,features=rebased,points=points,vectors=p,locations=v.locations)
        altered,actual=output(changed)
        for x,y in zip(item['visible'],altered['visible']):close(x,y)
        close(expected,actual)
        with torch.no_grad():
            for n,b in student.named_buffers():b.copy_(original[n])
        cases.append(kind)
    modes.assert_equal(dict(student.named_buffers()),original)
    return dict(passed=True,cases=cases,fixed_masks=True,fixed_queries=True,b0_functional=True)


def learning(device,rank,world,arm='masked'):
    s=R.fresh(arm,device,world,rank=rank,trim=False)
    x,y=R.tensors(R.split_batch(R.fixed_batch(8,17),rank,world),device)
    before={n:p.detach().clone() for n,p in s.base.named_parameters()}
    successes=0;rows=[]
    for i in range(20):
        row=attempt(s,x,y,schedule_label=100000+i,diagnostic=True,audit=i==0)
        successes+=1-row['skipped'];rows.append(row)
        if successes==8:break
    if successes!=8:raise ValueError('Learning did not settle')
    if arm=='masked':
        names=dict(s.base.named_parameters())
        keys=['predictor.mask_vector','predictor.position.0.weight','predictor.output.weight',
              'student.mod.embed.embed.1.weight']
        if any(torch.equal(before[n],names[n]) for n in keys):raise ValueError('Predictor/backbone failed to learn')
        s.teacher.validate_subset(s.base.student)
    return s,dict(passed=True,successful_updates=8,attempts=len(rows),positive_loss=arm=='control' or any(r['auxiliary']>0 for r in rows),
                  ema_successes=s.teacher.successes if s.teacher else 0)


def sequential(device,rank=0,world=1,amp=False):
    """Compare graph-lifetime-only change using exactly the same masks and B0."""
    s=R.fresh('masked',device,1,trim=False);activate(s.base.student)
    # This oracle is rank-local. Actual averaged DDP gradients are checked separately.
    other=copy.deepcopy(s.base);teacher=copy.deepcopy(s.teacher)
    x,y=R.tensors(R.fixed_batch(8,17),device);b0=modes.snapshot(s.base.student)
    rng=capture_rank_state(s.base);losses=[];gradients=[]
    for bundle,separate in ((s.base,True),(other,False)):
        restore_rank_state(bundle,rng);bundle.train();bundle.zero_grad(set_to_none=True)
        with model_precision(bundle.student,device,training=amp):
            logits,retained,anchor=bundle('clean',x,retain=True);clean=F.cross_entropy(logits,y)+anchor
        v=views.create(retained,views.Generators(0));item=views.slot(v,0,len(v.selected))
        if separate:clean.backward()
        targets=teacher.targets(item,amp=amp)
        with modes.auxiliary(bundle.student),model_precision(bundle.student,device,training=amp):
            aux=bundle('auxiliary',item['visible'],locations=item['locations'],query_valid=item['query_valid'],targets=targets,buffers=b0).mean()*.1
            (aux if separate else aux+clean).backward()
        losses.append(float(clean.detach()+aux.detach()));gradients.append({n:p.grad for n,p in bundle.named_parameters()})
    for n,p in gradients[0].items():
        if p is None or not torch.isfinite(p).all():raise ValueError('Missing sequential gradient: '+n)
        close(p,gradients[1][n],gradient=True,amp=amp)
    close(torch.tensor(losses[0]),torch.tensor(losses[1]),amp=amp)
    return dict(passed=True,amp=amp,gradients=len(gradients[0]),preclean_bn=True,simultaneous_oracle=True)


def checkpoint_parity(device,amp=False):
    s=R.fresh('masked',device,1,trim=False);activate(s.base.student)
    other=copy.deepcopy(s.base)
    for state in state_branches(other.student):state.training=False
    x,_=R.tensors(R.fixed_batch(8,97),device)
    v=views.create((x[1],x[2],x[3]),views.Generators(0));item=views.slot(v,0,len(v.selected))
    target=s.teacher.targets(item,amp=amp);b0=modes.snapshot(s.base.student);records=[]
    for bundle in (s.base,other):
        bundle.zero_grad(set_to_none=True)
        values=list(item['visible']);values[1]=values[1].detach().clone().requires_grad_(True)
        with modes.auxiliary(bundle.student),model_precision(bundle.student,device,training=amp):
            loss=bundle('auxiliary',values,locations=item['locations'],query_valid=item['query_valid'],targets=target,buffers=b0).mean()
            loss.backward()
        records.append((loss.detach(),{n:p.grad for n,p in bundle.named_parameters()},values[1].grad))
        if bundle.student._state_pair_cache.stem is not None:raise ValueError('Auxiliary checkpoint leaked pair cache')
    close(records[0][0],records[1][0],amp=amp)
    for n,p in records[0][1].items():
        if p is None or not torch.isfinite(p).all():raise ValueError('Missing checkpointed source gradient')
        close(p,records[1][1][n],gradient=True,amp=amp)
    close(records[0][2],records[1][2],gradient=True,amp=amp)
    return dict(passed=True,amp=amp,particles=97,visible_particles=item['visible'][1].shape[-1],
        checkpoint_backward=True,input_gradients=True,gradients=len(records[0][1]),cache_clear=True)


def auxiliary_gradient(device):
    s=R.fresh('masked',device,1,trim=False);activate(s.base.student)
    x,_=R.tensors(R.fixed_batch(8,17),device)
    v=views.create((x[1],x[2],x[3]),s.generators);item=views.slot(v,0,len(v.selected));b0=modes.snapshot(s.base.student)
    targets=s.teacher.targets(item,amp=False)
    with modes.auxiliary(s.base.student),model_precision(s.base.student,device,training=False):
        loss=s.base('auxiliary',item['visible'],locations=item['locations'],query_valid=item['query_valid'],targets=targets,buffers=b0).mean()
        loss.backward()
    gradient=s.base.student.mod.embed.embed[1].weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max()>0):
        raise ValueError('Auxiliary objective failed to reach backbone')
    if any(p.grad is not None for p in s.teacher.model.parameters()):raise ValueError('Teacher target acquired gradients')
    return dict(passed=True,backbone_gradient_norm=float(gradient.norm()),teacher_detached=True)


def taps(device):
    s=R.fresh('masked',device,1,trim=False);x,_=R.tensors(R.fixed_batch(8,13),device)
    v=views.create((x[1],x[2],x[3]),s.generators);item=views.slot(v,0,len(v.selected));captured={};handles=[]
    for i in (5,7):
        handles.append(s.teacher.model.mod.blocks[i].register_forward_hook(
            lambda m,a,o,i=i:captured.__setitem__(i,o.detach().transpose(0,1).clone())))
    try:actual=s.teacher.targets(item,amp=False)
    finally:
        for h in handles:h.remove()
    expected=normalize((normalize(captured[5])+normalize(captured[7]))*.5)
    expected=expected.gather(1,item['target_ids'][...,None].expand(-1,-1,128))
    close(actual,expected)
    if actual.requires_grad:raise ValueError('Targets are attached')
    return dict(passed=True,taps=[6,8],normalizations=3,parameters=TEACHER,detached=True)


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(to_device(v,device) for v in value)
    return value


def distributed(device,rank,world,*,cpu=False):
    """Public DDP phases versus a direct global ordinary-BN gradient oracle."""
    s=R.fresh('masked',device,world,rank=rank,trim=False,sync_bn=not cpu)
    activate(s.base.student);dropout_off(s.base.student)
    s.teacher.model.load_state_dict({n:v for n,v in s.base.student.state_dict().items() if n in s.teacher.model.state_dict()})
    reference=R.fresh('masked',device,1,trim=False,distributed=False)
    reference.base.load_state_dict(s.base.state_dict());dropout_off(reference.base.student)
    if cpu:
        # Gloo cannot run training SyncBN. A frozen-BN global oracle still
        # proves public DDP accumulation/global-eligible normalization; it is
        # explicitly not accepted as target CUDA/SyncBN evidence.
        for model in (s.base,reference.base):
            for module in model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.training=False
    original=R.fixed_batch(8,13);x,y=R.tensors(R.split_batch(original,rank,world),device)
    global_x,global_y=R.tensors(original,device)
    x=list(x);global_x=list(global_x)
    chosen=int(views.Generators(0).draw(8//world)[0][0])
    global_x[3][chosen,:,3:]=False
    if rank==0:x[3][chosen,:,3:]=False
    x[1]=x[1].detach().requires_grad_(True);global_x[1]=global_x[1].detach().requires_grad_(True)
    b0=modes.snapshot(s.base.student);ref_b0=modes.snapshot(reference.base.student)
    from .step import no_sync
    with no_sync(s,True),model_precision(s.base.student,device,training=False):
        pred,retained,anchor=s.model('clean',x,retain=True);(F.cross_entropy(pred,y)+anchor).backward()
    with model_precision(reference.base.student,device,training=False):
        pred_ref,_,anchor=reference.base('clean',global_x);(F.cross_entropy(pred_ref,global_y)+anchor).backward()
    close(pred,pred_ref[rank*8//world:(rank+1)*8//world],ddp=True)
    close(x[1].grad,global_x[1].grad[rank*8//world:(rank+1)*8//world]*world,gradient=True,ddp=True)
    v=views.create(retained,s.generators);item=views.slot(v,0,len(v.selected));target=s.teacher.targets(item,amp=False)
    count=sum(R.gather(int(v.eligible.sum()),world))
    with modes.auxiliary(s.base.student),model_precision(s.base.student,device,training=False):
        loss=s.model('auxiliary',item['visible'],locations=item['locations'],query_valid=item['query_valid'],targets=target,buffers=b0)
        (loss.sum()*.1*world/count).backward()
    rows=R.gather(to_device(dict(item=item,target=target),torch.device('cpu')),world)
    with modes.auxiliary(reference.base.student),model_precision(reference.base.student,device,training=False):
        ref_loss=0
        for row in rows:
            entry=to_device(row,device);q=entry['item']
            ref_loss=ref_loss+reference.base('auxiliary',q['visible'],locations=q['locations'],query_valid=q['query_valid'],targets=entry['target'],buffers=ref_b0).sum()*.1/count
        ref_loss.backward()
    expected=dict(reference.base.named_parameters())
    for n,p in s.base.named_parameters():close(p.grad,expected[n].grad,gradient=True,ddp=True)
    # Real attempt paths with zero/unequal eligibility, and an empty last slot.
    cases=[]
    for name,n,batch in (('zero',3,8),('one_rank',13,8),('empty_last',13,144)):
        q=R.fresh('masked',device,world,rank=rank,trim=False,sync_bn=not cpu)
        local=R.fixed_batch(batch//world,n) if cpu else R.split_batch(R.fixed_batch(batch,n),rank,world)
        a,b=R.tensors(local,device)
        if name=='one_rank' and rank==0:a[3][:,:,3:]=False
        if name=='empty_last':
            chosen=views.Generators(rank).draw(len(b))[0].to(device)
            a[3][chosen[32:],:,3:]=False
        result=attempt(q,a,b,schedule_label=40000,diagnostic=True,amp=False)
        if len(set(R.gather(R.parameter_digest(q.base),world)))!=1:raise ValueError('DDP replicas diverged')
        if any(p.grad is None for p in q.base.student.parameters()):raise ValueError('Missing student gradients after flush')
        cases.append(dict(case=name,eligible=result['eligible'],passed=True))
    return dict(passed=True,world=world,global_oracle=not cpu,cpu_health_only=cpu,cpu_normalization_oracle=cpu,
                unequal_eligible_counts=True,public_no_sync=True,
                broadcast_buffers=False,find_unused_parameters=False,cases=cases)


def controlled_skip(device,rank,world):
    s=R.fresh('masked',device,world,rank=rank,trim=False)
    if device.type=='cpu':s.scaler=torch.amp.GradScaler('cpu',enabled=True)
    before={n:p.detach().clone() for n,p in s.base.named_parameters()}
    teacher={n:p.detach().clone() for n,p in s.teacher.model.named_parameters()}
    def overflow(g):return torch.full_like(g,float('inf')) if rank==0 else g
    handle=s.base.student.mod.embed.embed[1].weight.register_hook(overflow)
    try:
        x,y=R.tensors(R.split_batch(R.fixed_batch(8,13),rank,world),device)
        row=attempt(s,x,y,schedule_label=40000,diagnostic=True,amp=device.type=='cuda')
    finally:handle.remove()
    if row['skipped']!=1 or s.successes!=0 or s.teacher.successes!=0 or s.teacher.buffer_copies!=1:
        raise ValueError('Skipped optimizer advanced EMA/cursors incorrectly')
    for n,p in s.base.named_parameters():
        if not torch.equal(p,before[n]):raise ValueError('Skipped optimizer changed a parameter')
    for n,p in s.teacher.model.named_parameters():
        if not torch.equal(p,teacher[n]):raise ValueError('Skipped update changed teacher parameters')
    modes.assert_equal(dict(s.teacher.model.named_buffers()),{n:b for n,b in s.base.student.named_buffers()
        if n in dict(s.teacher.model.named_buffers())})
    return dict(passed=True,skipped=1,parameter_ema_skipped=True,bn_copied=True,mask_draws=s.generators.draws)


def rejection(device,rank,world,arm):
    for case in ('empty_input','nonfinite_input'):
        s=R.fresh(arm,device,world,rank=rank,trim=False)
        x,y=R.tensors(R.split_batch(R.fixed_batch(8,7),rank,world),device)
        if rank==0:
            if case=='empty_input':x[3][0]=False
            else:x[1][0,0,0]=float('nan')
        rejected=False
        try:attempt(s,x,y,amp=False)
        except (ValueError,FloatingPointError):rejected=True
        if not R.agree(rejected,device,world):raise ValueError('Input failure not coordinated')
    failed=False
    try:R.all_call(lambda:(_ for _ in ()).throw(OSError('rank-local fixture')) if rank==0 else None,world)
    except RuntimeError:failed=True
    if not R.agree(failed,device,world):raise ValueError('I/O failure not coordinated')
    previous=R.STOP
    try:
        R.STOP=rank==0
        if not R.stopping(device,world):raise ValueError('Stop not coordinated')
    finally:R.STOP=previous
    return dict(passed=True,empty_input=True,nonfinite=True,rank_local_io=True,stop=True)
