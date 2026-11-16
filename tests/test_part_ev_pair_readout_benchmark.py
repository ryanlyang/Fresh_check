import copy
import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from part_ev_pair_readout import model as M,benchmark as C,protocol as P,pooling
from part_ev_fullscale.checkpoint import seed_everything,capture_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen.precision import model_precision
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_dense_pair_projection_fusion_input_and_all_parameter_gradients():
    """Independent dense softmax, rather than merely using one chunk."""
    seed_everything(177)
    readout=M.PairReadout().double()
    with torch.no_grad():readout.output.weight.normal_(std=.03);readout.output.bias.normal_(std=.02)
    oracle=copy.deepcopy(readout)
    features=torch.randn(2,5,5,32,dtype=torch.float64,requires_grad=True)
    h=torch.randn(2,128,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,0,1,1,1],[0,0,1,0,0]],dtype=torch.bool)
    valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(5,dtype=torch.bool)
    outputs=[];gradients=[]
    for m,dense in ((readout,False),(oracle,True)):
        r=features.detach().clone().requires_grad_();cls=h.detach().clone().requires_grad_()
        if dense:
            u=F.layer_norm(r.masked_fill(~valid[...,None],0),(32,),eps=1e-5)
            keys=m.key(u);values=m.value(u)
            scores=torch.einsum('ar,bijr->baij',m.queries,keys)/4
            summary=pooling.dense_reference(scores.flatten(2),values.flatten(1,2),valid.flatten(1))
        else:
            stats=None
            for start in range(0,5,2):
                scores,values,healthy=m.project(r[:,start:start+2],valid[:,start:start+2])
                row=pooling.statistics(scores,values,valid[:,start:start+2].flatten(1))
                stats=row if stats is None else pooling.merge(stats,row)
            summary,healthy=pooling.finish(stats)
            assert healthy
        out=m.fuse(cls,summary,mask.sum(-1)>=2)[0]
        out.square().sum().backward();outputs.append(out)
        gradients.append([r.grad,cls.grad,*[p.grad for p in m.parameters()]])
    torch.testing.assert_close(*outputs,rtol=1e-10,atol=1e-11)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)


def test_queries_and_fusion_float64_gradcheck():
    seed_everything(101)
    m=M.PairReadout().double()
    with torch.no_grad():m.output.weight.normal_(std=.03)
    features=torch.randn(1,2,2,32,dtype=torch.float64)
    valid=~torch.eye(2,dtype=torch.bool)[None]
    h=torch.randn(1,128,dtype=torch.float64)
    def call(queries,weight):
        scores,values,_=m.project(features,valid)
        # Replace scores explicitly to expose the query argument to gradcheck.
        keys=m.key(M.normalize(features.masked_fill(~valid[...,None],0)))
        scores=torch.einsum('ar,bijr->baij',queries,keys).flatten(2)/4
        summary=pooling.finish(pooling.statistics(scores,values,valid.flatten(1)))[0]
        hidden=F.gelu(m.hidden(torch.cat((h,summary.flatten(1)),-1)))
        return h+F.linear(hidden,weight,m.output.bias)
    assert torch.autograd.gradcheck(call,(m.queries,m.output.weight),fast_mode=True)


def test_isolated_directed_readout_ignores_diagonal_and_padding_and_fp32_reductions():
    seed_everything(101);m=M.PairReadout()
    features=torch.randn(2,5,5,32)
    valid=torch.ones(2,5,5,dtype=torch.bool)&~torch.eye(5,dtype=torch.bool)
    valid[1,-2:]=False;valid[1,:,-2:]=False
    def summaries(r):
        scores,values,healthy=m.project(r,valid)
        assert scores.dtype==values.dtype==torch.float32 and healthy
        return pooling.finish(pooling.statistics(scores,values,valid.flatten(1)))[0]
    with torch.autocast('cpu',dtype=torch.bfloat16):
        expected=summaries(features)
        changed=features.clone();changed[~valid]=float('nan')
        torch.testing.assert_close(summaries(changed),expected,rtol=0,atol=0)
    # Directed pairs both contribute; dropping either orientation changes it.
    scores,values,_=m.project(features,valid)
    upper=valid & torch.triu(torch.ones(5,5,dtype=torch.bool),diagonal=1)
    assert not torch.allclose(pooling.dense_reference(scores,values,upper.flatten(1)),summaries(features))


def test_isolated_diagnostics_preserve_state_rng_gradients_and_flags():
    seed_everything(101);m=M.build_model(trim=True);C.active(m)
    values,y=C.B.tensors(C.B.fixed_batch(8,17),torch.device('cpu'))
    F.cross_entropy(m(*values),y).backward()
    weights=C.cpu_copy(m.state_dict());rng=capture_rank_state(m)
    gradients=[p.grad.clone() for p in m.parameters()]
    flags=[v.training for v in m.modules()]
    row=C.diagnostic_sample(m,values,torch.device('cpu'))
    assert row['healthy'] and row['eligible_jets']==8 and row['mean_pair_count']==272
    assert 0<=row['query0_normalized_entropy']<=1.00001
    assert 1<=row['query1_effective_support']<=272.001
    assert row['disabled_logit_rms_change']>0
    C.assert_rng(capture_rank_state(m),rng)
    for n,w in m.state_dict().items():torch.testing.assert_close(w,weights[n],rtol=0,atol=0)
    for p,g in zip(m.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert flags==[v.training for v in m.modules()]
    with pytest.raises(ValueError,match='isolated'):m.mod.forward_details(values[1],values[2],values[3],diagnostic=True)


def test_eval_batch_independence_and_readout_only_direct_gradient_route():
    seed_everything(101);m=M.build_model(trim=False).eval();C.active(m)
    values,_=C.B.tensors(C.B.fixed_batch(3,9),torch.device('cpu'))
    with torch.no_grad():
        together=m(*values);separate=torch.cat([m(*(v[i:i+1] for v in values)) for i in range(3)])
    torch.testing.assert_close(together,separate,rtol=1e-5,atol=1e-6)
    # Backpropagate only summaries, not the class token or normal particle readout.
    m._state_pair_cache.clear()
    try:
        _,(_,summary,_,_,_,_)=m.mod.forward_details(values[1],values[2],values[3])
        summary.square().sum().backward()
    finally:m._state_pair_cache.clear()
    final=M.state_branches(m)[-1]
    for p in (final.receiver.weight,final.sender.weight,final.edge.weight):
        assert p.grad is not None and p.grad.count_nonzero()
    stem=[p for n,p in m.named_parameters() if 'pair_embed' in n and '.embed.' in n and p.grad is not None]
    assert stem and any(p.grad.count_nonzero() for p in stem)


def test_ranger_activation_and_restart_validation_boundary_cpu(tmp_path,monkeypatch):
    # CPU exercises real model, real Ranger, serialization and RNG, not CUDA AMP/DDP.
    assert C.delayed_activation(torch.device('cpu'),0,1)['passed']
    base,wrapped=C.fresh(torch.device('cpu'),1)
    optimizer=build_optimizer(base,P.recipes()['ranger_full_v1'])
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    values,y=C.B.tensors(C.B.fixed_batch(4,9),torch.device('cpu'))
    for i in range(6):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,1)
    # CUDA sync is the only hardware-specific operation in this smoke test.
    monkeypatch.setattr(C.B,'synchronize',lambda *args:None)
    row,seconds=C.restart_probe(tmp_path,base,wrapped,optimizer,scaler,values,y,
                              dict(content_hash='fixture',recipe=P.recipes()['ranger_full_v1']),0,1)
    assert row==dict(P.RESTART,ranks=1) and seconds>0
    saved=C.trusted_load(tmp_path/'probes/restart.pt')
    assert saved['progress']==dict(attempt=19997,checks=0,fixture_cursor=0)
    assert not any('summary' in k for k in saved['model'])


def test_cpu_correctness_driver_all_paths():
    # FP16 and SyncBN are tested for real only on the queued CUDA benchmark.
    row=C.correctness(torch.device('cpu'),0,1)[0]
    assert row['passed'] and {c['kind'] for c in row['checks']}==P.CHECKS


def _distributed_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        base,_=C.fresh(torch.device('cpu'),1,trim=False)
        wrapped=torch.nn.parallel.DistributedDataParallel(base,find_unused_parameters=False)
        optimizer=build_optimizer(base,P.recipes()['ranger_full_v1'])
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        values,y=C.B.tensors(C.B.split_batch(C.B.fixed_batch(6,9),rank,2),torch.device('cpu'))
        if rank==0:values[-1][:,:,1:]=False
        for i in range(7):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,2)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in base.parameters())
        assert len(set(C.B.gather(C.B.parameter_digest(base),2)))==1
        C.B.synchronize=lambda *args:dist.barrier()
        row,_=C.restart_probe(Path(output),base,wrapped,optimizer,scaler,values,y,
            dict(content_hash='distributed-fixture',recipe=P.recipes()['ranger_full_v1']),rank,2)
        assert row==P.RESTART
        Path(output,f'rank{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_no_pairs_graph_and_restart(tmp_path):
    context=mp.spawn(_distributed_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:pytest.fail('Distributed test timed out')
        assert all(json.loads((tmp_path/f'rank{i}.json').read_text())['passed'] for i in (0,1))
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
