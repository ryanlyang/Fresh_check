import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from part_ev_triangles import model as M,benchmark as C,protocol as P
from part_ev_triangles.diagnostics import NeighborStats,stage_profile,tied_batch,profile_sample,host_memory
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
        assert all(v['cpu_us']>0 and v['calls']>0 for v in profile['stages'].values())
        assert set(profile['stages'])==P.PROFILE_STAGES
        assert profile['policy']==P.PROFILE_POLICY
        assert profile['sample']==dict(source_events=3,source_particles=17,events=2,particles=17,training=training)
        C.assert_rng(capture_rank_state(m),rng)
        for n,w in m.state_dict().items():torch.testing.assert_close(w,weights[n],rtol=0,atol=0)
    assert not list(tmp_path.iterdir())


def test_full_length_input_is_bounded_only_for_profiling(capsys):
    seed_everything(101);m=M.build_model(trim=True);C.active(m)
    values,y=C.B.tensors(tied_batch(8,128),torch.device('cpu'))
    original=C.cpu_copy(values);rng=capture_rank_state(m);weights=C.cpu_copy(m.state_dict())
    flags=[v.training for v in m.modules()];observed=[]
    hook=m.register_forward_pre_hook(lambda _m,args:observed.append((args[0].shape,m.mod.trimmer.enabled)))
    try:
        profile=stage_profile(m,m,values,y,torch.device('cpu'),time.monotonic()+60,label='adversarial_train')
    finally:hook.remove()
    assert observed==[(torch.Size([2,2,32]),False)]
    assert profile['sample']==dict(source_events=8,source_particles=128,events=2,particles=32,training=True)
    assert set(profile['stages'])==P.PROFILE_STAGES
    torch.testing.assert_close(values,original,rtol=0,atol=0)
    C.assert_rng(capture_rank_state(m),rng)
    torch.testing.assert_close(m.state_dict(),weights,rtol=0,atol=0)
    assert m.mod.trimmer.enabled and flags==[v.training for v in m.modules()]
    logs=[json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert [row['phase'] for row in logs]==['start','complete']
    assert all(row['host_memory']['scope']=='process_only_excludes_workers' for row in logs)
    assert all(row['sample']['particles']==32 and row['label']=='adversarial_train' for row in logs)


def test_profile_failure_restores_state_and_logs(capsys):
    m=M.build_model(trim=True)
    values,y=C.B.tensors(tied_batch(4,17),torch.device('cpu'))
    rng=capture_rank_state(m);weights=C.cpu_copy(m.state_dict())
    def broken(*args):
        torch.rand(4)
        m.mod.trimmer._counter+=1
        raise RuntimeError('injected profile failure')
    with pytest.raises(RuntimeError,match='injected'):
        stage_profile(m,broken,values,y,torch.device('cpu'),time.monotonic()+60)
    C.assert_rng(capture_rank_state(m),rng)
    torch.testing.assert_close(m.state_dict(),weights,rtol=0,atol=0)
    assert m.training and m.mod.trimmer.enabled
    logs=[json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert [row['phase'] for row in logs]==['start','failed']


def test_profile_sample_owns_small_storage_and_rejects_empty_jets():
    values,y=C.B.tensors(tied_batch(8,128),torch.device('cpu'))
    bounded,truth,meta=profile_sample(values,y)
    for v in bounded:
        assert v.shape[0]==2 and v.shape[-1]==32 and v.is_contiguous()
        assert v.untyped_storage().nbytes()==v.numel()*v.element_size()
    assert len(truth)==2 and meta['source_events']==8
    assert host_memory()['scope']=='process_only_excludes_workers'
    values[-1][:,:,:32]=False
    with pytest.raises(ValueError,match='empty jet'):profile_sample(values,y)


def test_linux_profile_memory_reports_process_rss_bytes(tmp_path,monkeypatch):
    from part_ev_triangles import diagnostics as D
    status=tmp_path/'status'
    status.write_text('Name:\tpython\n\nVmRSS:\t1024 kB\nVmHWM:\t2048 kB\n')
    monkeypatch.setattr(D,'Path',lambda path:status)
    monkeypatch.setattr(D.sys,'platform','linux')
    memory=D.host_memory()
    assert memory['rss_bytes']==1024**2 and memory['peak_rss_bytes']==2*1024**2
    assert memory['pid']>0 and memory['scope']=='process_only_excludes_workers'


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
        rng=capture_rank_state(base);weights=C.cpu_copy(base.state_dict())
        profile=stage_profile(base,wrapped,values,y,torch.device('cpu'),time.monotonic()+60)
        assert profile['sample']['events']==2
        C.assert_rng(capture_rank_state(base),rng)
        torch.testing.assert_close(base.state_dict(),weights,rtol=0,atol=0)
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
