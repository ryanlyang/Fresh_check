import copy
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_state_bias import benchmark as C, protocol as P, model as M
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_full_optimizer_rng_resume_and_new_parameter_coverage(tmp_path):
    base,model=C.fresh(torch.device('cpu'),1,trim=True)
    opt=C.build_optimizer(base,P.recipes()['ranger_full_v1'])
    scale=torch.amp.GradScaler('cpu',enabled=False)
    values,y=C.B.tensors(C.B.fixed_batch(3,9),torch.device('cpu'))
    recipe=P.recipes()['ranger_full_v1']; spec=dict(content_hash='test',recipe=recipe)
    for i in range(4): C.B.step(base,model,opt,scale,values,y,recipe,i,1)
    C.save_probe(tmp_path/'probe.pt',base,opt,scale,spec,0,1)
    for i in range(4,11): C.B.step(base,model,opt,scale,values,y,recipe,i,1)
    expected=C.cpu_copy((base.state_dict(),opt.state_dict(),scale.state_dict()))
    C.restore_probe(tmp_path/'probe.pt',base,opt,scale,spec,0,1)
    for i in range(4,11): C.B.step(base,model,opt,scale,values,y,recipe,i,1)
    C.assert_nested((base.state_dict(),opt.state_dict(),scale.state_dict()),expected)
    for key,value in base.state_dict().items():
        torch.testing.assert_close(value,expected[0][key],rtol=0,atol=0)
    heads={id(a.bias_head) for a in M.attentions(base)}
    assert heads.issubset({id(p) for g in opt.param_groups for p in g['params']})
    assert len(opt.slow)==len(list(base.parameters()))
    assert all(a.bias_head.grad.isfinite().all() and a.bias_head.grad.count_nonzero()>0 for a in M.attentions(base))


def test_diagnostics_preserve_buffers_modes_gradients_rng_and_trimming():
    import pickle
    base,model=C.fresh(torch.device('cpu'),1)
    C.active(base)
    opt=C.build_optimizer(base,P.recipes()['ranger_full_v1'])
    scale=torch.amp.GradScaler('cpu',enabled=False)
    values,y=C.B.tensors(C.B.fixed_batch(8,17),torch.device('cpu'))
    C.B.step(base,model,opt,scale,values,y,P.recipes()['ranger_full_v1'],0,1)
    state=C.cpu_copy(base.state_dict())
    gradients={n:p.grad.clone() for n,p in base.named_parameters()}
    rng=C.capture_rank_state(base)
    flags=[m.training for m in base.modules()]
    result=C.diagnostic_sample(base,values,torch.device('cpu'))
    assert len(result['layers'])==8
    for layer in result['layers']:
        assert all(len(layer[k])==8 for k in P.DIAGNOSTIC_HEAD_FIELDS)
        assert max(layer['attention_total_variation_per_head'])>0
    for n,p in base.named_parameters(): torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)
    for n,t in base.state_dict().items(): torch.testing.assert_close(t,state[n],rtol=0,atol=0)
    assert flags==[m.training for m in base.modules()]
    after=C.capture_rank_state(base)
    assert rng['python']==after['python'] and pickle.dumps(rng['numpy'])==pickle.dumps(after['numpy'])
    assert torch.equal(rng['torch'],after['torch'])
    C.assert_nested(rng['trimmers'],after['trimmers'])


def _worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        base,_=C.fresh(torch.device('cpu'),1,trim=False)
        model=torch.nn.parallel.DistributedDataParallel(base,find_unused_parameters=False)
        opt=C.build_optimizer(base,P.recipes()['ranger_full_v1'])
        scale=torch.amp.GradScaler('cpu',enabled=False)
        values,y=C.B.tensors(C.B.split_batch(C.B.fixed_batch(6,7),rank,2),torch.device('cpu'))
        for i in range(7):
            C.B.step(base,model,opt,scale,values,y,P.recipes()['ranger_full_v1'],i,2)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in base.parameters())
        assert len(set(C.B.gather(C.B.parameter_digest(base),2)))==1
        Path(output,f'rank{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally: dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_real_graph_and_lookahead(tmp_path):
    context=mp.spawn(_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('Distributed test timed out')
        assert all(json.loads((tmp_path/f'rank{i}.json').read_text())['passed'] for i in (0,1))
    finally:
        for process in context.processes:
            if process.is_alive(): process.terminate()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU benchmark required on SPORC')
def test_cuda_identity_and_nonzero_correctness():
    assert C.correctness(torch.device('cuda'),0,1)[0]['passed']
