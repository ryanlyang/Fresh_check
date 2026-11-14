"""One-GH200 adapter using the original frozen model, optimizer and checks."""
import copy
import gc
import importlib
from pathlib import Path
import time

import numpy as np
import torch

from . import protocol as P


def backend(spec):
    module = importlib.import_module(P.PACKAGES[spec['model_kind']]+'.benchmark')
    P.verify_imports(Path(spec['donor']['path'])/'source')
    return module


def correctness(C, device):
    rows = copy.deepcopy(C.correctness(device, 0, 1))
    # The reused oracle supports world=1 but its historical label says DDP.
    # Record its ACTUAL native execution, not a claim of testing SyncBN here.
    for row in rows:
        row['execution'] = dict(world_size=1, batchnorm='native', distributed=False)
        for check in row['checks']:
            if check['kind'] == 'active_SyncBN_DDP':
                check['kind'] = 'active_native_global_batch'
    return rows


def assert_rng(C, actual, expected):
    if (actual['python'] != expected['python'] or actual['numpy'][0] != expected['numpy'][0]
            or not np.array_equal(actual['numpy'][1], expected['numpy'][1])
            or actual['numpy'][2:] != expected['numpy'][2:]
            or actual['trimmers'] != expected['trimmers']):
        raise ValueError('Python/NumPy/trimmer restart state diverged')
    torch.testing.assert_close(actual['torch'], expected['torch'], rtol=0, atol=0)
    if len(actual['cuda']) != len(expected['cuda']): raise ValueError('CUDA RNG count changed')
    for a, b in zip(actual['cuda'], expected['cuda']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def restart_probe(C, output, base, optimizer, scaler, values, truth, spec):
    path = Path(output)/'probes/restart.pt'
    if values[0].device.type == 'cuda': torch.cuda.synchronize(values[0].device)
    start = time.monotonic()
    C.save_probe(path, base, optimizer, scaler, spec, 0, 1)
    seconds = time.monotonic()-start

    def updates():
        trace = []
        for index in range(P.POLICY['restart_steps']):
            result = C.B.step(base, base, optimizer, scaler, values, truth, spec['recipe'], index, 1)
            trace.append((result['loss'], result['skipped']))
        return trace

    expected = updates()
    weights = C.cpu_copy((base.state_dict(), optimizer.state_dict(), scaler.state_dict()))
    rng = C.capture_rank_state(base)
    C.restore_probe(path, base, optimizer, scaler, spec, 0, 1)
    actual = updates()
    C.compare(torch.tensor(actual), torch.tensor(expected))
    C.assert_nested((base.state_dict(), optimizer.state_dict(), scaler.state_dict()), weights)
    assert_rng(C, C.capture_rank_state(base), rng)
    if sum(r[1] for r in actual): raise ValueError('AMP did not settle for restart verification')
    return dict(passed=True, updates=7, ranks=1, full_state=True, rng_and_trimmers=True), seconds


def active_parameters(C, base, kind):
    if kind == 'bias': return [a.bias_head for a in C.M.attentions(base)]
    if kind == 'multiply': return list(C.M.gate_parameters(base))
    return [*C.M.readers(base), *C.M.writers_and_gates(base)]


def inference_probe(C, base, device, deadline):
    base.eval()
    for module in base.modules():
        if type(module).__name__ == 'SequenceTrimmer': module.enabled = False
    values, _ = C.B.tensors(C.B.fixed_batch(256, 128), device)
    one = [v[:1] for v in values]

    def timed(inputs, count):
        for _ in range(5): base(*inputs)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        for _ in range(count):
            if C.B.STOP or time.monotonic() >= deadline: raise TimeoutError('Inference budget exhausted')
            if not torch.isfinite(base(*inputs)).all(): raise ValueError('Nonfinite inference')
        torch.cuda.synchronize(device)
        return time.monotonic()-start

    with torch.no_grad(), C.model_precision(base, device, training=False):
        seconds = timed(values, 20)
        latency = timed(one, 100)
    return dict(batch_size=256, particles=128, batches=20, seconds=seconds,
                precision='fp32', batch1_iterations=100, batch1_seconds=latency)


def run(output, spec):
    C = backend(spec)
    B = C.B
    _, world, device = B.setup(1)
    prop = torch.cuda.get_device_properties(device)
    if 'GH200' not in prop.name or prop.total_memory < P.POLICY['minimum_vram_bytes']:
        raise ValueError('Require one GH200 with at least 80 GiB device memory')
    if B.POLICY['window_steps'] != P.POLICY['window'] or B.POLICY['max_skip_fraction'] != P.POLICY['max_skip_fraction']:
        raise ValueError('Frozen measurement helper differs')
    torch.set_num_threads(P.POLICY['torch_threads'])
    runtime = P.runtime()
    runtime.update(world_size=world, torch_threads=torch.get_num_threads(),
                   tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_benchmark=torch.backends.cudnn.benchmark)
    if runtime != P.runtime() or torch.backends.cudnn.allow_tf32:
        raise ValueError('Actual CUDA precision/thread settings differ from the frozen policy')
    deadline = time.monotonic()+P.POLICY['max_seconds']
    checks = correctness(C, device)
    gc.collect(); torch.cuda.empty_cache()
    base, model = C.fresh(device, world)
    if model is not base or any(isinstance(m, torch.nn.SyncBatchNorm) for m in base.modules()):
        raise ValueError('GH200 adapter requires native single-GPU BatchNorm')
    parameters = sum(p.numel() for p in base.parameters())
    if parameters != spec['parameters']: raise ValueError('Model parameter count changed')
    optimizer = C.build_optimizer(base, spec['recipe'])
    P.write(Path(output)/'probes/optimizer.json', C.optimizer_metadata(base, optimizer))
    scaler = torch.amp.GradScaler('cuda')
    loader, _ = C.build_loader('train', spec['inventory'], 512, rank=0, world_size=1, workers=2, seed=101)
    val_loader = None
    try:
        iterator = iter(loader)
        start = time.monotonic(); first = next(iterator); cold = time.monotonic()-start
        C.seed_everything(101)
        torch.cuda.reset_peak_memory_stats(device)
        stream = B.measure(base, model, optimizer, scaler,
            lambda i: B.tensors(first if i == 0 else next(iterator), device), spec['recipe'], device, 1,
            P.POLICY['warmup'], P.POLICY['measured'], deadline, label='evext_gh200_real_data')
        if any(not p.detach().isfinite().all() or not p.detach().count_nonzero()
               for p in active_parameters(C, base, spec['model_kind'])):
            raise ValueError('Extension parameters remained inactive/nonfinite after sustained updates')
        for module in base.modules():
            if type(module).__name__ == 'SequenceTrimmer': module.enabled = False
        values, truth = B.tensors(B.fixed_batch(512, 128), device)
        fixed = B.measure(base, model, optimizer, scaler, lambda i: (values, truth),
            spec['recipe'], device, 1, P.POLICY['fixed_warmup'], P.POLICY['fixed_steps'],
            deadline, label='evext_gh200_fixed128')
        for module in base.modules():
            if type(module).__name__ == 'SequenceTrimmer': module.enabled = True
        restart, seconds = restart_probe(C, output, base, optimizer, scaler, values, truth, spec)
        diagnostic_values, _ = B.tensors(B.fixed_batch(8, 17), device)
        diagnostics = [C.diagnostic_sample(base, diagnostic_values, device)]
        if B.STOP or time.monotonic() >= deadline: raise TimeoutError('Budget exhausted before validation')
        val_loader, _ = C.build_loader('val', spec['inventory'], 512, workers=2, seed=101)
        metric = C.validate(base, iter(val_loader), device, 0, 1, P.POLICY['validation_events'], audit=True)
        if metric is None: raise TimeoutError('Interrupted validation')
        inference = inference_probe(C, base, device, deadline)
        finite = C.finite_state(torch, (base.state_dict(), optimizer.state_dict(), scaler.state_dict()))
        gpu = dict(rank=0, name=prop.name, total_memory=prop.total_memory,
            peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device), cold_fetch_seconds=cold,
            first_input_sha256=B.batch_fingerprint(first))
        healthy = bool(stream['healthy'] and fixed['healthy'] and finite
                       and gpu['peak_allocated'] < P.POLICY['memory_fraction']*prop.total_memory)
        return dict(complete=True, passed=healthy, authorizes_training=False,
            campaign_sha256=spec['content_hash'], parameters=parameters, runtime=runtime,
            correctness=checks, stream=stream, fixed=fixed, restart=restart,
            checkpoint_seconds=seconds, validation=metric, inference=inference,
            diagnostics=diagnostics, gpus=[gpu], finite_state=bool(finite), extension_active=True)
    finally:
        C.shutdown_loader(loader)
        if val_loader is not None: C.shutdown_loader(val_loader)
