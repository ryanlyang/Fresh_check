import pytest
import torch
from part_ev_ffn_experts import verification as V,runtime as R,restart as S
from part_ev_ffn_experts import engine as E,diagnostics as D,protocol as P
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state,trusted_load,atomic_torch_save

torch.set_num_threads(1)
DEVICE=torch.device('cpu')


def test_disposable_independent_device_math():assert V.oracle(DEVICE)['passed']


def test_native_identity_ce_gradient_and_call_oracle():assert V.initial_identity(DEVICE,0,1)['passed']


@pytest.mark.parametrize('arm',['control','experts'])
def test_active_checkpoints_global_model_and_failure_coordination(arm):
    assert V.active_parity(arm,DEVICE,0,1)['passed']
    assert V.distributed(arm,DEVICE,0,1)['passed']
    assert V.rejection(arm,DEVICE,0,1)['passed']


def test_bank_global_and_empty_population_oracle():assert V.bank_ddp(DEVICE,0,1)['passed']


def test_real_ranger_all_bank_expert_ce_coverage():assert V.coverage(DEVICE)['passed']


def test_route_aware_seven_update_repeat_disk_and_resume(tmp_path):
    b,m,o,s,learning=V.learning('experts',DEVICE,0,1)
    assert learning['successful_updates']==8
    evidence=S.probe(b,m,o,s,DEVICE,0,1,tmp_path)
    assert evidence['passed']
    for trial in evidence['trials']:
        assert all(r['routes']['minimum_margin']>1 for r in trial['updates'])
        assert all(r['counts']['requested']==r['counts']['executed'] for r in trial['updates'])


@pytest.mark.parametrize('arm',['control','experts'])
def test_real_step_objective_counters_and_full_checkpoint_resume(tmp_path,arm):
    import copy
    b,m,o,s=R.fresh(arm,DEVICE,1,trim=False);state=E.initial_state(arm)
    x,y=R.tensors(R.fixed_batch(8,7),DEVICE)
    for i in range(2):
        result=R.step(b,m,o,s,x,y,i,1,audit=i==0)
        assert result['loss']==result['ce']+(.001*result['balance'] if arm=='experts' else 0.)
        E.record_step(state,result)
    E.audit_counters(state,arm)
    spec=dict(content_hash='cpu_fixture',runtime=dict(platform='cpu_test'))
    before=S.snapshot(b,o,s);E.save(tmp_path/'last.pt',b,o,s,state,spec,0,1)
    R.step(b,m,o,s,x,y,2,1)
    restored,rng=E.restore(tmp_path/'last.pt',b,o,s,spec,arm,0,1);restore_rank_state(b,rng)
    assert restored==state;S.compare(S.snapshot(b,o,s),before,exact=True)
    bad=trusted_load(tmp_path/'last.pt');bad['state']['loss_sums']['total']+=1.
    atomic_torch_save(bad,tmp_path/'bad.pt')
    with pytest.raises(ValueError):E.restore(tmp_path/'bad.pt',b,o,s,spec,arm,0,1)
    if arm=='experts':
        bad=trusted_load(tmp_path/'last.pt');bad['state']['routing_totals']['executed'][0][0]+=1
        atomic_torch_save(bad,tmp_path/'bad.pt')
        with pytest.raises(ValueError):E.restore(tmp_path/'bad.pt',b,o,s,spec,arm,0,1)


def test_diagnostics_preserve_weights_gradients_rng_modes_and_buffers():
    b,_,o,s,learning=V.learning('experts',DEVICE,0,1)
    before=S.snapshot(b,o,s);grads={n:p.grad.clone() for n,p in b.named_parameters()}
    b.train();trim=b.mod.trimmer.enabled
    d=D.sample(b,DEVICE)
    assert len(d['banks'])==6 and set(d['gradient_norms'])=={'router','experts'}
    S.compare(S.snapshot(b,o,s),before,exact=True)
    for n,p in b.named_parameters():torch.testing.assert_close(p.grad,grads[n],rtol=0,atol=0)
    assert b.training and b.mod.trimmer.enabled==trim
    for arm in ('control','experts'):
        profiles=D.profile(arm,DEVICE)
        if arm=='experts':assert all(set(p['stages'])==D.SCOPES for p in profiles)


def test_mixed_precision_router_exceptions_and_fp32_validation(emulate_cuda_autocast):
    from part_ev_ffn_experts.precision import model_precision,audit_precision,precision_ok,modules
    b,_,_,_=R.fresh('experts',DEVICE,1,trim=False);V.activate(b,separated=True)
    x,y=R.tensors(R.fixed_batch(8,7),DEVICE)
    with model_precision(b,torch.device('cuda'),training=True),audit_precision(b,expected_dtype=torch.float16) as observed:
        out=b(*x,return_router_stats=True);ce=torch.nn.functional.cross_entropy(out['logits'],y)
    from part_ev_ffn_experts.losses import balancing
    aux,_=balancing(out['banks'],1,DEVICE)
    (ce+.001*aux).backward()
    assert precision_ok(observed,'torch.float16',modules(b))
    assert all(v=='torch.float32' for n,v in observed['observed_activations'].items() if '.router.' in n)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    with torch.no_grad(),model_precision(b,torch.device('cuda'),training=False),audit_precision(b,expected_dtype=torch.float32) as fp32:
        b.eval();b(*x)
    assert precision_ok(fp32,'torch.float32',modules(b)) and not b.mod.use_amp


from tests.test_part_ev_screen_precision import emulate_cuda_autocast
