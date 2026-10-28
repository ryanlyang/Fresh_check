import copy
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_gpu_scaling import protocol as P, benchmark as B
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_state_compare import protocol as C
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('world,batch,workers', [(1,512,2), (2,256,1)])
def test_fixed_exposures_and_resources(world, batch, workers):
    assert P.resources(world)['local_batch'] == batch
    assert P.resources(world)['workers'] == workers
    assert batch*world == 512 and workers*world == 2
    assert C.recipes()['ranger_full_v1']['total_steps'] == 1_000_000
    assert C.recipes()['ranger_full_v1']['epochs'] == 50


@pytest.mark.parametrize('world', [0,3,4,True])
def test_unsupported_world(world):
    with pytest.raises(ValueError): P.resources(world)


def test_fixed_batch_is_identical_partition_not_duplicate_events():
    batch = B.fixed_batch(8, 17)
    shards = [B.split_batch(batch, r, 2) for r in range(2)]
    for m in range(2):
        for key in batch[m]:
            assert torch.equal(torch.cat([s[m][key] for s in shards]), batch[m][key])
    assert B.batch_fingerprint(batch) == B.batch_fingerprint(B.fixed_batch(8,17))
    vectors = batch[0]['pf_vectors']
    assert (vectors[:,3] > vectors[:,:3].square().sum(1).sqrt()).all()
    with pytest.raises(ValueError): B.split_batch(B.fixed_batch(7,4), 0, 2)


def test_step_preserves_existing_single_gpu_recipe_and_gradients():
    from part_ev_state_compare.engine import create_engine
    training_step = create_engine().training_step
    batch = B.fixed_batch(2,7)
    base, model = B.model_pair(torch.device('cpu'), 1, trim=False)
    reference, _ = B.model_pair(torch.device('cpu'), 1, trim=False)
    assert sum(p.numel() for p in base.parameters()) == 2327462
    recipe = C.recipes()['ranger_full_v1']
    opt, refopt = build_optimizer(base,recipe), build_optimizer(reference,recipe)
    scaler = torch.amp.GradScaler('cpu',enabled=False)
    refscale = torch.amp.GradScaler('cpu',enabled=False)
    values, truth = B.tensors(batch, torch.device('cpu'))
    for index in (0,1,2):
        seed_everything(813+index)
        row = B.step(base,model,opt,scaler,values,truth,recipe,index,1,audit=index==0)
        seed_everything(813+index)
        expected = training_step(reference,refopt,refscale,batch,torch.device('cpu'),recipe,index)
        assert row['loss'] == expected['loss']
        assert not row['skipped']
        if index == 0: assert row['precision']['passed']
        for name, parameter in base.named_parameters():
            assert parameter.grad is not None, name  # DDP must not have unused parameters.
        for name, tensor in base.state_dict().items():
            torch.testing.assert_close(tensor,reference.state_dict()[name],rtol=0,atol=0)


def test_syncbn_conversion_preserves_hooks_state_and_eval():
    first, _ = B.model_pair(torch.device('cpu'),1,trim=False,state_test=True)
    second, _ = B.model_pair(torch.device('cpu'),1,trim=False,state_test=True)
    second = torch.nn.SyncBatchNorm.convert_sync_batchnorm(second)
    first.eval(); second.eval()
    assert any(isinstance(m,torch.nn.SyncBatchNorm) for m in second.modules())
    assert not any(type(m) is torch.nn.BatchNorm1d for m in second.modules())
    values,_ = B.tensors(B.fixed_batch(2,7),torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(first(*values),second(*values),rtol=0,atol=0)
    assert second._state_pair_cache.normalized is None


def result(world, rate, **overrides):
    row = dict(completed=True, plan_sha256='plan', world_size=world, numerically_healthy=True,
        memory_headroom_passed=world==2, fixed_global_fingerprint='fixed',
        stream=dict(jets_per_second=rate,skips=2), fixed=dict(jets_per_second=rate),
        gpus=[dict(name='A100',total_memory=40*1024**3,peak_allocated=36*1024**3/world,
                   peak_reserved=38*1024**3/world)]*world)
    row.update(overrides)
    row['content_hash']=P.digest(row)
    return row


def save_results(tmp_path, rows):
    for row in rows: P.write_json(tmp_path/f"gpus_{row['world_size']}.json",row)
    return dict(content_hash='plan',recipe=dict(total_steps=1_000_000))


def test_summary_speedup_cost_memory_failure_not_suppressed(tmp_path):
    plan=save_results(tmp_path,[result(1,1000),result(2,1500)])
    summary=P.summarize(tmp_path,plan)
    assert summary['comparable'] and not summary['authorizes_training']
    assert summary['real_data_scaling']['speedup_2_vs_1']==1.5
    assert summary['real_data_scaling']['gpu_hour_ratio_2_vs_1']==pytest.approx(4/3)
    assert summary['real_data_scaling']['walltime_reduction_percent']==pytest.approx(100/3)
    assert summary['models'][0]['memory_headroom_passed'] is False
    assert summary['models'][0]['training_only_hours']==pytest.approx(512e6/1000/3600)


@pytest.mark.parametrize('second', [None,dict(completed=False,error='OOM'),dict(numerically_healthy=False)])
def test_missing_or_failed_comparison_is_inconclusive(tmp_path,second):
    rows=[result(1,1000)]
    if second is not None: rows.append(result(2,1500,**second))
    summary=P.summarize(tmp_path,save_results(tmp_path,rows))
    assert not summary['comparable'] and 'real_data_scaling' not in summary


def test_mismatched_gpu_or_inputs_are_not_a_speedup(tmp_path):
    different=result(2,1500,gpus=[dict(name='H100',total_memory=80*1024**3,
                                    peak_allocated=20*1024**3,peak_reserved=21*1024**3)]*2)
    plan=save_results(tmp_path,[result(1,1000),different])
    assert not P.summarize(tmp_path,plan)['comparable']
    save_results(tmp_path,[result(2,1500,fixed_global_fingerprint='changed')])
    with pytest.raises(ValueError,match='workloads'): P.summarize(tmp_path,plan)


def test_prepare_is_separate_and_readonly(tmp_path,monkeypatch):
    parent=tmp_path/'campaign'; parent.mkdir()
    (parent/'campaign.json').write_text('{}')
    spec=dict(tasks=C.TASKS,recipes=C.recipes(),content_hash='parent',environment={})
    monkeypatch.setattr(C,'load_campaign',lambda path: copy.deepcopy(spec))
    monkeypatch.setattr(C,'verify_inventory_metadata',lambda _:None)
    before=(parent/'campaign.json').read_bytes()
    for output in (parent,parent/'diagnostic',tmp_path):
        with pytest.raises(ValueError,match='separate'): P.prepare(output,parent)
    plan=P.prepare(tmp_path/'diagnostic',parent)
    assert plan['recipe']==spec['recipes']['ranger_full_v1']
    assert not plan['authorizes_training']
    assert (parent/'campaign.json').read_bytes()==before
    assert list(parent.iterdir())==[parent/'campaign.json']
    with pytest.raises(FileExistsError): P.prepare(tmp_path/'diagnostic',parent)


def _gloo_step_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(91)
        base=torch.nn.Linear(4,10)
        reference=copy.deepcopy(base)
        wrapped=torch.nn.parallel.DistributedDataParallel(base)
        recipe=C.recipes()['ranger_full_v1']
        opt,refopt=build_optimizer(base,recipe),build_optimizer(reference,recipe)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        refscale=torch.amp.GradScaler('cpu',enabled=False)
        x=torch.arange(32,dtype=torch.float32).reshape(8,4)/32
        y=torch.arange(8)
        for index in range(7):  # Cross the Lookahead synchronization step.
            B.step(base,wrapped,opt,scaler,[x[rank*4:(rank+1)*4]],y[rank*4:(rank+1)*4],recipe,index,2)
            B.step(reference,reference,refopt,refscale,[x],y,recipe,index,1)
        for a,b in zip(base.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        checks=B.gather(B.parameter_digest(base),2)
        assert len(set(checks))==1
        assert opt.step_counter==refopt.step_counter==1
        bad=x[rank*4:(rank+1)*4].clone()
        if rank==1: bad.fill_(float('nan'))
        try:
            B.step(base,wrapped,opt,scaler,[bad],y[rank*4:(rank+1)*4],recipe,8,2)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Nonfinite loss on one rank must stop both before backward')
        # Exercise the actual layerwise State-EV graph, hooks and non-reentrant
        # checkpoint backward under DDP too. CPU uses native BN here; the separate
        # CUDA test/probe verifies SyncBN against a global-batch reference.
        seed_everything(101)
        state=B.build_model('state_layerwise',trim=False)
        distributed_state=torch.nn.parallel.DistributedDataParallel(state)
        state_opt=build_optimizer(state,recipe)
        for index in range(3):
            values,truth=B.tensors(B.split_batch(B.fixed_batch(4,7),rank,2),torch.device('cpu'))
            row=B.step(state,distributed_state,state_opt,scaler,values,truth,recipe,index,2,audit=index==0)
            assert not row['skipped']
        assert len(set(B.gather(B.parameter_digest(state),2)))==1
        assert all(p.grad is not None for p in state.parameters())
        Path(output,f'rank_{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_process_cpu_gradient_averaging_and_failure_consensus(tmp_path):
    context=mp.spawn(_gloo_step_worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),
                     nprocs=2,join=False)
    deadline=time.monotonic()+75
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('Distributed test timed out')
        assert all(json.loads((tmp_path/f'rank_{r}.json').read_text())['passed'] for r in range(2))
    finally:
        for process in context.processes:
            if process.is_alive(): process.terminate()


def _cuda_parity_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',init_method=rendezvous,rank=rank,world_size=2)
    try:
        assert all(r['passed'] for r in B.parity(torch.device('cuda',rank),rank,2))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2 or not dist.is_nccl_available(),reason='Two CUDA GPUs required')
def test_actual_state_model_syncbn_ddp_cuda(tmp_path):
    mp.spawn(_cuda_parity_worker,args=((tmp_path/'cuda_rendezvous').as_uri(),),nprocs=2,join=True)
