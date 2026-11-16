import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from part_ev_triangles import model as M,benchmark as C,protocol as P
from part_ev_triangles.diagnostics import NeighborStats,stage_profile,tied_batch
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
    F.cross_entropy(m(*values),y).backward()
    weights=C.cpu_copy(m.state_dict());rng=capture_rank_state(m)
    gradients=[p.grad.clone() for p in m.parameters()];flags=[v.training for v in m.modules()]
    row=C.diagnostic_sample(m,values,torch.device('cpu'))
    assert row['healthy'] and row['disabled_logit_rms_change']>0
    assert m._state_pair_cache.normalized is None
    assert all(b.attn._bias is None for b in m.mod.blocks)
    assert [r['layer'] for r in row['layers']]==[6,7]
    C.assert_rng(capture_rank_state(m),rng)
    for n,w in m.state_dict().items():torch.testing.assert_close(w,weights[n],rtol=0,atol=0)
    for p,g in zip(m.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert flags==[v.training for v in m.modules()]
    with pytest.raises(ValueError,match='isolated'):m.mod.forward_details(values[1],values[2],values[3],diagnostic=True)


def test_histogram_post_trim_not_backward_and_profiling(tmp_path):
    seed_everything(101);m=M.build_model(trim=False);C.active(m)
    stats=NeighborStats(m);stats.phase='train'
    values,y=C.B.tensors(tied_batch(3,17),torch.device('cpu'))
    values[-1][1,:,2:]=False;values[-1][2,:,1:]=False
    F.cross_entropy(m(*values),y).backward()
    rows=stats.report()['train']
    assert rows['receivers']==20 and rows['maximum_support']==16
    assert rows['histogram'][16]==17 and rows['histogram'][1]==2 and rows['histogram'][0]==1
    assert rows['tie_fraction']==17/20
    P.validate_histogram(rows);stats.phase=None;stats.close()
    rng=capture_rank_state(m);weights=C.cpu_copy(m.state_dict())
    for training in (True,False):
        profile=stage_profile(m,m,values,y,torch.device('cpu'),time.monotonic()+60,training=training)
        assert all(v['cpu_us']>0 and v['calls']>0 for v in profile.values())
        assert set(profile)==P.PROFILE_STAGES
        C.assert_rng(capture_rank_state(m),rng)
        for n,w in m.state_dict().items():torch.testing.assert_close(w,weights[n],rtol=0,atol=0)
    assert not list(tmp_path.iterdir())


def test_ranger_activation_and_full_restart_cpu(tmp_path,monkeypatch):
    assert C.delayed_activation(torch.device('cpu'),0,1)['passed']
    base,wrapped=C.fresh(torch.device('cpu'),1)
    optimizer=build_optimizer(base,P.recipes()['ranger_full_v1']);scaler=torch.amp.GradScaler('cpu',enabled=False)
    values,y=C.B.tensors(tied_batch(4,11),torch.device('cpu'))
    for i in range(6):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,1)
    monkeypatch.setattr(C.B,'synchronize',lambda *a:None)
    row,seconds=C.restart_probe(tmp_path,base,wrapped,optimizer,scaler,values,y,
        dict(content_hash='fixture',recipe=P.recipes()['ranger_full_v1']),0,1)
    assert row==dict(P.RESTART,ranks=1) and seconds>0
    saved=C.trusted_load(tmp_path/'probes/restart.pt')
    assert saved['progress']==dict(attempt=19997,checks=0,fixture_cursor=0)
    assert not any('neighbors' in k for k in saved['model'])


def _distributed_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        base,_=C.fresh(torch.device('cpu'),1,trim=False)
        wrapped=torch.nn.parallel.DistributedDataParallel(base,find_unused_parameters=False)
        optimizer=build_optimizer(base,P.recipes()['ranger_full_v1']);scaler=torch.amp.GradScaler('cpu',enabled=False)
        values,y=C.B.tensors(C.B.split_batch(tied_batch(6,11),rank,2),torch.device('cpu'))
        # Rank zero has no triangles; rank one ONLY has tied receivers.
        if rank==0:values=[v[:,:,:2] for v in values]
        for i in range(7):C.B.step(base,wrapped,optimizer,scaler,values,y,P.recipes()['ranger_full_v1'],i,2)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in base.parameters())
        assert len(set(C.B.gather(C.B.parameter_digest(base),2)))==1
        C.B.synchronize=lambda *a:dist.barrier()
        row,_=C.restart_probe(Path(output),base,wrapped,optimizer,scaler,values,y,
            dict(content_hash='distributed-fixture',recipe=P.recipes()['ranger_full_v1']),rank,2)
        assert row==P.RESTART
        Path(output,f'rank{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_unequal_lengths_no_triangles_ties_and_restart(tmp_path):
    context=mp.spawn(_distributed_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=False)
    deadline=time.monotonic()+90
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:pytest.fail('Distributed test timed out')
        assert all(json.loads((tmp_path/f'rank{i}.json').read_text())['passed'] for i in (0,1))
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
