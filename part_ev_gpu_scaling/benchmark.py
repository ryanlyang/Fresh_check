"""Same-node DDP timing with disposable models and independent rank evidence."""
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import os
import signal
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .protocol import POLICY, resources
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.protocol import learning_rate
from part_ev_screen.data import batch_fingerprint
from part_ev_screen.precision import model_precision, audit_precision
from part_ev_state_compare.model import build_model, COUNTS, state_branches

STOP = False


def stop_signal(*_):
    global STOP
    STOP = True


def gather(value, world):
    if world == 1:
        return [value]
    rows = [None]*world
    dist.all_gather_object(rows, value)
    return rows


def agree(value, device, world):
    flag = torch.tensor(int(bool(value)), device=device)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def synchronize(device, world):
    torch.cuda.synchronize(device)
    if world > 1:
        dist.barrier()


def setup(expected):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if (world != expected or world not in (1, 2) or not 0 <= rank < world
            or local != rank or not torch.cuda.is_available()
            or torch.cuda.device_count() != world):
        raise RuntimeError('Need exactly the requested 1/2 visible GPUs on ONE node, via torchrun')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    torch.set_num_threads(resources(world)['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name in ('SIGTERM', 'SIGUSR1'):
        signal.signal(getattr(signal, name), stop_signal)
    hosts = gather(socket.gethostname(), world)
    if len(set(hosts)) != 1:
        raise RuntimeError('This diagnostic only supports a single node')
    return rank, world, device


def model_pair(device, world, *, trim=True, state_test=False):
    seed_everything(POLICY['seed'])
    base = build_model('state_layerwise', seed=POLICY['seed'], trim=trim, use_amp=False).to(device)
    if state_test:
        with torch.no_grad():
            for state in state_branches(base):
                state.output.normal_(std=.02)
        disable_dropout(base)
    if world == 2:
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        if any(type(m) is torch.nn.BatchNorm1d for m in base.modules()):
            raise RuntimeError('Unsynchronized BatchNorm remained')
        model = DDP(base, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        model = base
    if sum(p.numel() for p in base.parameters()) != COUNTS['state_layerwise']:
        raise ValueError('Model capacity changed')
    return base, model


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        if isinstance(getattr(module, 'dropout', None), float):
            module.dropout = 0.0


def fixed_batch(events=512, particles=128):
    """Physical four-vectors, all-valid masks, fixed synthetic work; no physics result."""
    rng = torch.Generator().manual_seed(719101)
    spatial = torch.randn(events, 3, particles, generator=rng)
    energy = spatial.square().sum(1, keepdim=True).sqrt() + 1
    return (dict(pf_points=torch.zeros(events, 2, particles),
                 pf_features=torch.randn(events, 17, particles, generator=rng),
                 pf_vectors=torch.cat((spatial, energy), 1),
                 pf_mask=torch.ones(events, 1, particles, dtype=torch.bool)),
            {'_label_': torch.arange(events) % 10}, {})


def split_batch(batch, rank, world):
    n = len(batch[1]['_label_'])
    if n % world or not 0 <= rank < world:
        raise ValueError('Global batch must divide exactly across ranks')
    start, stop = rank*n//world, (rank+1)*n//world
    return tuple({k: v[start:stop] for k, v in mapping.items()} for mapping in batch)


def tensors(batch, device):
    return ([batch[0][key].to(device, non_blocking=True) for key in
             ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')],
            batch[1]['_label_'].to(device, non_blocking=True).long())


def parity(device, rank, world):
    """Nonzero-state FP32 SyncBN/DDP gradients vs one global batch, dropout off.

    This checks distributed plumbing, not bitwise-equivalent stochastic training.
    Each rank independently computes the small non-distributed global reference.
    """
    base, model = model_pair(device, world, trim=False, state_test=True)
    reference = build_model('state_layerwise', trim=False, use_amp=False).to(device)
    reference.load_state_dict(base.state_dict())
    disable_dropout(reference)
    batch = fixed_batch(8, 17)
    values, truth = tensors(batch, device)
    reference.train()
    with model_precision(reference, device, training=False):
        expected = reference(*values)
        torch.nn.functional.cross_entropy(expected, truth).backward()
    values, truth = tensors(split_batch(batch, rank, world), device)
    base.train()
    with model_precision(base, device, training=False):
        actual = model(*values)
        torch.nn.functional.cross_entropy(actual, truth).backward()
    passed, error = True, None
    try:
        torch.testing.assert_close(actual, expected[rank*8//world:(rank+1)*8//world],
                                   rtol=5e-4, atol=2e-5)
        expected_params = dict(reference.named_parameters())
        for name, parameter in base.named_parameters():
            if parameter.grad is None or expected_params[name].grad is None:
                raise AssertionError(f'Unused parameter: {name}')
            torch.testing.assert_close(parameter.grad, expected_params[name].grad, rtol=2e-3, atol=2e-5)
    except AssertionError as exc:
        passed, error = False, str(exc)
    evidence = dict(passed=passed, error=error, global_events=8, particles=17,
                    nonzero_state=True, dropout=False, precision='fp32',
                    forward_rtol=5e-4, gradient_rtol=2e-3, atol=2e-5)
    rows = gather(evidence, world)
    del reference, model, base
    torch.cuda.empty_cache()
    if not all(r['passed'] for r in rows):
        raise RuntimeError(f'Distributed plumbing parity failed: {rows}')
    return rows


def step(base, model, optimizer, scaler, values, truth, recipe, index, world, *, audit=False):
    base.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(index, recipe)
    dtype = torch.float16 if truth.device.type == 'cuda' else torch.float32
    with model_precision(base, truth.device, training=True), (
            audit_precision(base, expected_dtype=dtype) if audit else nullcontext()) as precision:
        logits = model(*values)  # Wrapper does not expose .mod: scope AMP on BASE, not DDP.
        loss = torch.nn.functional.cross_entropy(logits, truth)
    if not agree(torch.isfinite(loss).item(), truth.device, world):
        raise FloatingPointError('At least one rank has nonfinite loss')
    scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(base.parameters(), float('inf'))
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < scale
    if not agree(torch.isfinite(norm).item() or skipped, truth.device, world):
        raise FloatingPointError('Nonfinite gradient without scaler recovery')
    return dict(loss=float(loss.detach()), skipped=int(skipped), precision=precision)


def parameter_digest(base):
    h = hashlib.sha256()
    for name, parameter in base.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def measure(base, model, optimizer, scaler, supplier, recipe, device, world,
            warmup, measured, deadline, *, label):
    windows, skipped = [], 0
    precision = None
    for index in range(warmup+measured):
        if not agree(not STOP and time.monotonic() < deadline, device, world):
            raise TimeoutError('Diagnostic stopped before completion; no production artifacts written')
        values, truth = supplier(index)
        last = step(base, model, optimizer, scaler, values, truth, recipe, index, world, audit=index == 0)
        if index == 0:
            precision = last['precision']
        if index+1 == warmup:
            synchronize(device, world)
            start = window_start = time.monotonic()
        if index >= warmup:
            skipped += last['skipped']
            done = index+1-warmup
            if done % POLICY['window_steps'] == 0 or done == measured:
                synchronize(device, world)
                end = time.monotonic()
                count = done % POLICY['window_steps'] or POLICY['window_steps']
                windows.append(dict(steps=count, seconds=end-window_start))
                window_start = end
                if world == 1 or dist.get_rank() == 0:
                    print(dict(stage=label, measured_steps=done, seconds=windows[-1]['seconds']), flush=True)
    seconds = time.monotonic()-start
    local = dict(seconds=seconds, skips=skipped, scaler=scaler.get_scale(), windows=windows,
                 precision=precision, parameters_sha256=parameter_digest(base),
                 lookahead_counter=optimizer.step_counter)
    ranks = gather(local, world)
    if len({(r['skips'], r['scaler'], r['parameters_sha256'], r['lookahead_counter']) for r in ranks}) != 1:
        raise RuntimeError('DDP parameters, AMP updates or optimizer counters diverged')
    seconds = max(r['seconds'] for r in ranks)
    return dict(measured_steps=measured, seconds=seconds, skips=skipped,
                jets_per_second=512*measured/seconds, ranks=ranks,
                healthy=skipped < measured*POLICY['max_skip_fraction'])


def run(plan, parent, rank, world, device):
    deadline = time.monotonic()+POLICY['max_seconds']
    parity_rows = parity(device, rank, world)
    # Bound-method hooks form cycles; discard all parity-only tensors before peaks.
    gc.collect()
    torch.cuda.empty_cache()
    base, model = model_pair(device, world)
    optimizer = build_optimizer(base, plan['recipe'])
    scaler = torch.amp.GradScaler('cuda')
    limits = resources(world)
    loader, _ = build_loader('train', parent['inventory'], limits['local_batch'],
                             rank=rank, world_size=world, workers=limits['workers'], seed=101)
    try:
        iterator = iter(loader)
        start = time.monotonic()
        first = next(iterator)
        cold = time.monotonic()-start
        first_hash = batch_fingerprint(first)
        # Deliberately rank-specific stochastic streams after identical initialization.
        seed_everything(101+rank)
        torch.cuda.reset_peak_memory_stats(device)
        def supplier(index):
            return tensors(first if index == 0 else next(iterator), device)
        stream = measure(base, model, optimizer, scaler, supplier, plan['recipe'], device, world,
                         POLICY['warmup_steps'], POLICY['measured_steps'], deadline, label='real_data')
        stream_peak = torch.cuda.max_memory_allocated(device)
        stream_reserved = torch.cuda.max_memory_reserved(device)
    finally:
        shutdown_loader(loader)
    # Fixed all-128 workload isolates compute/DDP cost from I/O and rank trimming.
    for module in base.modules():
        if type(module).__name__ == 'SequenceTrimmer':
            module.enabled = False
    global_batch = fixed_batch()
    fixed_hash = batch_fingerprint(global_batch)
    values, truth = tensors(split_batch(global_batch, rank, world), device)
    torch.cuda.reset_peak_memory_stats(device)
    fixed = measure(base, model, optimizer, scaler, lambda _: (values, truth), plan['recipe'], device, world,
                    POLICY['fixed_warmup'], POLICY['fixed_steps'], deadline, label='fixed_128')
    # Check optimizer master/slow weights too; don't mislabel a finite loss as a healthy model.
    from scripts.check_part_ev_screen_amp_recovery import finite_state
    finite = agree(finite_state(torch, (base.state_dict(), optimizer.state_dict())), device, world)
    properties = torch.cuda.get_device_properties(device)
    gpu = dict(rank=rank, name=properties.name, total_memory=properties.total_memory,
               peak_allocated=max(stream_peak, torch.cuda.max_memory_allocated(device)),
               peak_reserved=max(stream_reserved, torch.cuda.max_memory_reserved(device)),
               stream_peak_allocated=stream_peak, cold_fetch_seconds=cold,
               first_input_sha256=first_hash)
    gpus = gather(gpu, world)
    return dict(completed=True, world_size=world, plan_sha256=plan['content_hash'],
        authorizes_training=False, numerically_healthy=stream['healthy'] and fixed['healthy'] and finite,
        memory_headroom_passed=all(g['peak_allocated'] < g['total_memory']*POLICY['memory_fraction_limit'] for g in gpus),
        stream=stream, fixed=fixed, fixed_global_fingerprint=fixed_hash, gpus=gpus, parity=parity_rows,
        runtime=dict(hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                     global_batch=512, local_batch=limits['local_batch'], workers_per_rank=limits['workers'],
                     batchnorm='SyncBatchNorm' if world == 2 else 'native', tf32=False,
                     nccl_environment={k: v for k, v in os.environ.items() if k.startswith('NCCL_')}))
