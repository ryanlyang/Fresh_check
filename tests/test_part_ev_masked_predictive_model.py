import copy
import pytest
import torch
from torch import nn
from part_ev_masked_predictive import runtime as R,verification as V,modes,views
from part_ev_masked_predictive.model import Trainable,build_model
from part_ev_masked_predictive.predictor import Predictor,jet_losses
from part_ev_masked_predictive.config import COUNTS,TRAINABLE,TEACHER,PREDICTOR,momentum
from part_ev_masked_predictive.step import attempt
from part_ev_masked_predictive.restart import capture,compare
from part_ev_fullscale.checkpoint import capture_rank_state

torch.set_num_threads(1);D=torch.device('cpu')


def test_exact_deployment_training_teacher_counts_native_keys_and_rng():
    torch.manual_seed(101);control=Trainable('control',trim=False);rng=torch.get_rng_state()
    torch.manual_seed(101);candidate=Trainable('masked',trim=False)
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in candidate.parameters())==TRAINABLE['masked']==2626598
    assert sum(p.numel() for p in candidate.predictor.parameters())==PREDICTOR==299136
    assert sum(p.numel() for p in candidate.student.parameters())==COUNTS['masked']==2327462
    for n,p in control.student.state_dict().items():assert torch.equal(p,candidate.student.state_dict()[n])
    session=R.fresh('masked',D,1,trim=False)
    assert sum(p.numel() for p in session.teacher.model.parameters())==TEACHER==1926412
    assert not hasattr(session.teacher.model.mod,'cls_blocks') and not hasattr(session.teacher.model.mod,'fc')
    assert all(not p.requires_grad and p.grad is None for p in session.teacher.model.parameters())
    assert all('teacher' not in n for n,_ in session.base.named_parameters())


def test_independent_predictor_values_loss_and_all_gradients():assert V.predictor_oracle(D)['gradients']


def test_queries_cannot_communicate_and_padding_is_ignored():
    m=Predictor().double();loc=torch.randn(2,4,3,dtype=torch.float64);h=torch.randn(2,7,128,dtype=torch.float64)
    valid=torch.ones(2,7,dtype=torch.bool);valid[:,-2:]=False;qv=torch.ones(2,4,dtype=torch.bool);qv[:,-1]=False
    actual,_=m(loc,h,valid,qv);changed=loc.clone();changed[:,0]+=7
    other,_=m(changed,h,valid,qv);torch.testing.assert_close(actual[:,1:],other[:,1:],rtol=0,atol=0)
    h[~valid]=float('nan');loc[~qv]=float('nan')
    padded,ok=m(loc,h,valid,qv);assert ok;torch.testing.assert_close(padded,actual,rtol=0,atol=0)
    target=torch.randn_like(padded);target[~qv]=float('nan')
    loss=jet_losses(padded,target,qv);assert torch.isfinite(loss).all()
    _,bad=m(torch.nan_to_num(loc),h,torch.zeros_like(valid),qv);assert not bad


def test_native_logits_gradients_calls_and_strict_export():assert V.native_identity(D,0,1)['export_keys']


def test_teacher_tap_alignment_and_no_teacher_gradients():assert V.taps(D)['normalizations']==3


def test_all_hidden_and_historical_feature_leakage_cases():
    row=V.leakage(D);assert row['passed'] and len(row['cases'])==6


def test_sequential_graph_release_matches_simultaneous_oracle():assert V.sequential(D)['simultaneous_oracle']


def test_functional_buffer_checkpoint_recompute_crosses_native_64_query_chunks():
    row=V.checkpoint_parity(D);assert row['visible_particles']>64 and row['input_gradients']


def test_auxiliary_loss_reaches_native_backbone_without_teacher_grad():
    row=V.auxiliary_gradient(D);assert row['teacher_detached'] and row['backbone_gradient_norm']>0


def test_functional_bn_snapshots_keep_clean_references_and_modes_on_failure():
    s=R.fresh('masked',D,1,trim=False);V.activate(s.base.student);student=s.base.student
    x,y=R.tensors(R.fixed_batch(8,13),D);b0=modes.snapshot(student)
    logits,retained,anchor=s.base('clean',x,retain=True);(torch.nn.functional.cross_entropy(logits,y)+anchor).backward()
    b1=modes.snapshot(student);refs={n:b for n,b in student.named_buffers()}
    flags=[m.training for m in student.modules()];counter=student.mod.trimmer._counter
    v=views.create(retained,s.generators);item=views.slot(v,0,4);target=s.teacher.targets(item,amp=False)
    with modes.auxiliary(student):
        assert student.mod.blocks[0].attn.attention.state.training
        assert all(not m.training for m in student.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
        s.base('auxiliary',item['visible'],locations=item['locations'],query_valid=item['query_valid'],targets=target,buffers=b0).sum().backward()
    modes.assert_equal(dict(student.named_buffers()),b1)
    assert all(refs[n] is b for n,b in student.named_buffers())
    assert [m.training for m in student.modules()]==flags and student.mod.trimmer._counter==counter
    with pytest.raises(RuntimeError):
        with modes.auxiliary(student):raise RuntimeError('fixture')
    assert [m.training for m in student.modules()]==flags


def test_warmup_leaves_predictor_radam_moments_absent_but_teacher_tracks_ranger():
    s=R.fresh('masked',D,1,trim=False);x,y=R.tensors(R.fixed_batch(8,9),D)
    for i in range(7):
        old={n:p.detach().clone() for n,p in s.teacher.model.named_parameters()}
        row=attempt(s,x,y,amp=False)
        student=dict(s.base.student.named_parameters())
        for n,p in s.teacher.model.named_parameters():
            expected=old[n].mul_(momentum(i+1)).add_(student[n],alpha=1-momentum(i+1))
            torch.testing.assert_close(p,expected,rtol=0,atol=0)
        assert all(p.grad is None and p not in s.opt.inner.state for p in s.base.predictor.parameters())
    assert s.opt.step_counter==1 and s.teacher.successes==7 and s.generators.draws==0


def test_positive_auxiliary_learning_and_controlled_skip_transaction():
    s,row=V.learning(D,0,1);assert row['successful_updates']==8 and row['positive_loss']
    assert V.controlled_skip(D,0,1)['parameter_ema_skipped']


def test_global_empty_eligibility_flush_and_single_native_rng_advance():
    s=R.fresh('masked',D,1,trim=False);x,y=R.tensors(R.fixed_batch(8,3),D)
    row=attempt(s,x,y,schedule_label=40000,diagnostic=True,amp=False,audit=True)
    assert row['eligible']==0 and row['auxiliary']==0 and row['rng_isolated']
    assert s.generators.draws==s.auxiliary_attempts==1 and s.teacher.buffer_copies==1
    assert all(p.grad is None for p in s.base.predictor.parameters())
    assert s.base.student._state_pair_cache.stem is None
