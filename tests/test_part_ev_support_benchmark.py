import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from part_ev_support import model as M,benchmark as C,protocol as P
from part_ev_support.diagnostics import validate_diagnostic
from part_ev_fullscale.checkpoint import seed_everything,capture_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_cpu_correctness_driver_all_paths():
    # CUDA FP16 and SyncBN still require the actual queued benchmark.
    row=C.correctness(torch.device('cpu'),0,1)[0]
    assert row['passed'] and {c['kind'] for c in row['checks']}==P.CHECKS


def test_diagnostics_preserve_state_rng_buffers_gradients_flags():
    seed_everything(101);m=M.build_model(trim=True);C.active(m)
    values,y=C.B.tensors(C.B.fixed_batch(8,17),torch.device('cpu'))
    values[-1][0,:,1:]=False;values[-1][1,:,9:]=False
    F.cross_entropy(m(*values),y).backward()
    weights=C.cpu_copy(m.state_dict());rng=capture_rank_state(m)
    gradients=[p.grad.clone() for p in m.parameters()];flags=[v.training for v in m.modules()]
    row=dict(C.diagnostic_sample(m,values,torch.device('cpu')),population='first_8_validation_probe',
             sample_sha256='a'*64,rank=0)
    validate_diagnostic(row)
    assert row['healthy'] and row['disabled_logit_rms_change']>0 and row['counts_zero_logit_rms_change']>0
    assert m._state_pair_cache.normalized is None
    assert all(b.attn._bias is None for b in m.mod.blocks)
    assert [r['layer'] for r in row['layers']]==list(range(8))
    assert all(r['no_neighbor_receivers']==1 for r in row['layers'])
    assert all(s.observer is None and s.support_mode=='normal' for s in M.state_branches(m))
    C.assert_rng(capture_rank_state(m),rng)
    for n,w in m.state_dict().items():torch.testing.assert_close(w,weights[n],rtol=0,atol=0)
    for p,g in zip(m.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert flags==[v.training for v in m.modules()]
    with pytest.raises(ValueError,match='isolated|Isolated'):
        with M.diagnostic_mode(m):pass


def test_support_only_disable_keeps_trained_state_and_diagnostic_exception_cleanup():
    seed_everything(101);m=M.build_model(trim=False).eval();C.active(m)
    old=M.reference_model('state_layerwise',trim=False).eval()
    old.load_state_dict({n:v for n,v in m.state_dict().items() if not M.support_key(n)})
    values,y=C.B.tensors(C.B.fixed_batch(3,9),torch.device('cpu'))
    with torch.no_grad():
        with M.diagnostic_mode(m,mode='disabled'):
            torch.testing.assert_close(m(*values),old(*values),rtol=0,atol=0)
        def fail(*a):raise RuntimeError('injected diagnostic')
        with pytest.raises(RuntimeError,match='injected'):
            with M.diagnostic_mode(m,observer=fail):m(*values)
    assert all(s.observer is None and s.support_mode=='normal' for s in M.state_branches(m))
    assert m._state_pair_cache.normalized is None
    assert all(b.attn._bias is None for b in m.mod.blocks)
    assert m(*values).isfinite().all()


def test_ranger_activation_and_full_restart_cpu(tmp_path,monkeypatch):
    assert C.delayed_activation(torch.device('cpu'),0,1)['successful_updates']==8
    base,wrapped=C.fresh(torch.device('cpu'),1)
    optimizer=build_optimizer(base,P.recipes()['ranger_full_v1']);scaler=torch.amp.GradScaler('cpu',enabled=False)
    values,y=C.B.tensors(C.B.fixed_batch(4,11),torch.device('cpu'))
    for i in range(6):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,1)
    monkeypatch.setattr(C.B,'synchronize',lambda *a:None)
    row,seconds=C.restart_probe(tmp_path,base,wrapped,optimizer,scaler,values,y,
        dict(content_hash='fixture',recipe=P.recipes()['ranger_full_v1']),0,1)
    assert row==dict(P.RESTART,ranks=1) and seconds>0
    saved=C.trusted_load(tmp_path/'probes/restart.pt')
    assert saved['progress']==dict(attempt=19997,checks=0,fixture_cursor=0,successful_updates=0,skips=0,
                                   best_accuracy=None,best_check=None,best_step=None)
    assert sum(M.support_key(n) for n in saved['model'])==48
    midpoint=C.trusted_load(tmp_path/'probes/selected_restart.pt')
    assert midpoint['progress']['attempt']==20001
    assert midpoint['progress']['best_step']==20000 and midpoint['progress']['best_check']==1
    assert len(midpoint['best_by_rank'])==1 and midpoint['best_by_rank'][0]


def test_bounded_validation_probe_copies_only_first_eight_and_honors_deadline(monkeypatch):
    batches=[C.B.fixed_batch(12,17),C.B.fixed_batch(12,31)];sample=[]
    iterator=C.validation_batches(batches,time.monotonic()+60,sample)
    assert next(iterator) is batches[0]
    assert len(sample)==1 and sample[0][0]['pf_features'].shape==(8,17,17)
    before=C.B.batch_fingerprint(sample[0])
    batches[0][0]['pf_features'].zero_()
    assert C.B.batch_fingerprint(sample[0])==before
    assert next(iterator) is batches[1] and len(sample)==1
    with pytest.raises(StopIteration):next(iterator)
    with pytest.raises(TimeoutError):next(C.validation_batches(batches,time.monotonic()-1,[]))
    monkeypatch.setattr(C.B,'STOP',True)
    with pytest.raises(TimeoutError):next(C.validation_batches(batches,time.monotonic()+60,[]))


def _distributed_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        base,_=C.fresh(torch.device('cpu'),1,trim=False)
        # Trainable hidden biases must NOT leak messages into an empty-support jet.
        with torch.no_grad():
            for b in M.branches(base):b.hidden.bias.fill_(.2)
        wrapped=torch.nn.parallel.DistributedDataParallel(base,find_unused_parameters=False)
        optimizer=build_optimizer(base,P.recipes()['ranger_full_v1']);scaler=torch.amp.GradScaler('cpu',enabled=False)
        values,y=C.B.tensors(C.B.split_batch(C.B.fixed_batch(6,17),rank,2),torch.device('cpu'))
        # Rank zero has no eligible pairs, with a noncontiguous one-particle mask.
        if rank==0:
            values=[v[:,:,:3] for v in values];values[-1][:]=False;values[-1][:,:,1]=True
        for i in range(8):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,2)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in base.parameters())
        assert all(p.grad.count_nonzero() for n,p in base.named_parameters() if M.support_key(n))
        assert len(set(C.B.gather(C.B.parameter_digest(base),2)))==1
        C.B.synchronize=lambda *a:dist.barrier()
        row,_=C.restart_probe(Path(output),base,wrapped,optimizer,scaler,values,y,
            dict(content_hash='distributed-fixture',recipe=P.recipes()['ranger_full_v1']),rank,2)
        assert row==P.RESTART
        Path(output,f'rank{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_unequal_lengths_no_pairs_and_restart(tmp_path):
    context=mp.spawn(_distributed_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:pytest.fail('Distributed test timed out')
        assert all(json.loads((tmp_path/f'rank{i}.json').read_text())['passed'] for i in (0,1))
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
