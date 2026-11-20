"""Measure the frozen scientific models on one GH200, including native BN."""
import copy
import gc
import importlib
from pathlib import Path
import time

import torch

from . import protocol as P
from .evidence import RENAMES, EXECUTION


def backend(spec):
    module = importlib.import_module(P.PACKAGES[spec['model_kind']]+'.benchmark')
    P.verify_imports(Path(spec['donor']['path'])/'source')
    return module


def correctness(C, device):
    rows = copy.deepcopy(C.correctness(device, 0, 1))
    for row in rows:
        row['execution'] = dict(EXECUTION, device_type=device.type)
        for check in row['checks']:
            check['kind'] = RENAMES.get(check['kind'], check['kind'])
    return rows


def trimming(base, enabled):
    for module in base.modules():
        if type(module).__name__ == 'SequenceTrimmer': module.enabled = enabled


def measure(C, output, spec, device, deadline, *, baseline=False):
    import resource
    B = C.B
    kind = spec['model_kind']
    D = importlib.import_module(P.PACKAGES[kind]+'.diagnostics')
    base, model = C.fresh(device, 1, baseline=baseline)
    if model is not base or any(isinstance(m, torch.nn.SyncBatchNorm) for m in base.modules()):
        raise ValueError('Benchmark requires native single-GPU BatchNorm')
    optimizer = C.build_optimizer(base, spec['recipe'])
    scaler = torch.amp.GradScaler('cuda')
    if not baseline: P.write(Path(output)/'probes/optimizer.json', C.optimizer_metadata(base, optimizer))
    loader, _ = C.build_loader('train', spec['inventory'], 512, rank=0, world_size=1, workers=2, seed=101)
    val_loader = None
    stats = D.NeighborStats(base) if kind == 'triangles' and not baseline else None
    try:
        iterator = iter(loader)
        start = time.monotonic(); first = next(iterator); cold = time.monotonic()-start
        C.seed_everything(101)
        torch.cuda.reset_peak_memory_stats(device)
        if stats: stats.phase = 'train'
        stream = B.measure(base, model, optimizer, scaler,
            lambda i:B.tensors(first if i == 0 else next(iterator), device), spec['recipe'], device, 1,
            P.POLICY['warmup'], P.POLICY['measured'], deadline, label='gh200_'+('baseline' if baseline else kind))
        C.shutdown_loader(loader)
        del iterator
        if stats: stats.phase = None
        trimming(base, False)
        values, truth = B.tensors(B.fixed_batch(512, 128), device)
        fixed = B.measure(base, model, optimizer, scaler, lambda i:(values, truth), spec['recipe'], device, 1,
            P.POLICY['fixed_warmup'], P.POLICY['fixed_steps'], deadline, label='gh200_fixed128')
        result = dict(stream=stream, fixed=fixed, parameters=sum(p.numel() for p in base.parameters()))
        if not baseline:
            if stats:
                stats.phase = 'adversarial'
                tied_values, tied_truth = B.tensors(D.tied_batch(512, 128), device)
                result['adversarial_ties'] = B.measure(base, model, optimizer, scaler,
                    lambda i:(tied_values, tied_truth), spec['recipe'], device, 1, 2, 3, deadline, label='gh200_all_ties128')
                stats.phase = None
                real_values, real_truth = B.tensors(first, device)
                result['profiles'] = [dict(rank=0,
                    real_train=D.stage_profile(base, model, real_values, real_truth, device, deadline, label='real_train'),
                    adversarial_train=D.stage_profile(base, model, tied_values, tied_truth, device, deadline, label='adversarial_train'))]
                del tied_values, tied_truth, real_values, real_truth
            trimming(base, True)
            restart, seconds = C.restart_probe(output, base, model, optimizer, scaler, values, truth, spec, 0, 1)
            gradient_key = {'readout':'readout', 'triangles':'triangle', 'support':'support'}[kind]+'_gradient_norms'
            result[gradient_key] = [D.gradient_norms(base)]
            val_loader, _ = C.build_loader('val', spec['inventory'], 512, workers=2, seed=101)
            sample = []
            def batches():
                for batch in val_loader:
                    if B.STOP or time.monotonic() >= deadline: raise TimeoutError('Validation deadline')
                    if not sample:
                        # Retain only eight CPU examples, not all 128k validation jets.
                        sample.append(tuple({k:v[:8].detach().cpu().clone() for k,v in mapping.items()}
                                            for mapping in batch[:2]))
                    yield batch
            if stats: stats.phase = 'validation'
            metric = C.validate(base, iter(batches()), device, 0, 1, P.POLICY['validation_events'], audit=True)
            if stats: stats.phase = None
            if metric is None: raise TimeoutError('Interrupted validation')
            C.shutdown_loader(val_loader)
            if stats:
                v, y = B.tensors(sample[0], device)
                result['profiles'][0]['real_validation'] = D.stage_profile(base, base, v, y, device, deadline,
                    training=False, label='real_validation')
                del v, y
            if kind == 'support':
                diag_values, _ = B.tensors(sample[0], device)
                diagnostic = dict(C.diagnostic_sample(base, diag_values, device), rank=0,
                    population='first_8_validation_probe', sample_sha256=B.batch_fingerprint(sample[0]))
                result['parameter_inventory_sha256'] = C.P.digest(C.P.parameter_inventory(base))
            else:
                diag_values, _ = B.tensors(B.fixed_batch(8, 17), device)
                diagnostic = dict(C.diagnostic_sample(base, diag_values, device), rank=0)
            del diag_values
            trimming(base, False)
            base.eval()
            with torch.no_grad(), C.model_precision(base, device, training=False):
                if not base(*values).isfinite().all(): raise ValueError('Nonfinite full-128 FP32 validation')
                if stats:
                    tied_values, _ = B.tensors(D.tied_batch(512, 128), device)
                    if not base(*tied_values).isfinite().all(): raise ValueError('Nonfinite tied FP32 validation')
                    del tied_values
            result.update(restart=restart, checkpoint_seconds=seconds, validation=metric,
                          diagnostics=[diagnostic], full128_validation_passed=True)
            if stats:
                result.update(neighbor_statistics=[dict(rank=0, **stats.report())], full128_tied_validation_passed=True)
        # Training uses 512; the inference comparison deliberately uses 256 on both arms.
        inference_values, _ = B.tensors(B.fixed_batch(256, 128), device)
        result['inference'] = C.inference(base, inference_values, device, deadline)
        result['finite_state'] = bool(C.finite_state(torch, (base.state_dict(), optimizer.state_dict(), scaler.state_dict())))
        result['host_peak_rss_bytes'] = [int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)]
        gpu = torch.cuda.get_device_properties(device)
        result['gpus'] = [dict(rank=0, name=gpu.name, total_memory=gpu.total_memory,
            peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device),
            cold_fetch_seconds=cold, first_input_sha256=B.batch_fingerprint(first))]
        return result
    finally:
        if stats: stats.close()
        C.shutdown_loader(loader)
        if val_loader is not None: C.shutdown_loader(val_loader)


def run(output, spec):
    C = backend(spec)
    _, world, device = C.B.setup(1)
    torch.set_num_threads(4)
    gpu = torch.cuda.get_device_properties(device)
    if world != 1 or 'GH200' not in gpu.name or gpu.total_memory < P.POLICY['minimum_vram_bytes']:
        raise ValueError('Require exactly one GH200, at least 80 GiB device memory')
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.backends.cudnn.benchmark or C.B.POLICY['window_steps'] != 1000):
        raise ValueError('Measurement precision/window policy differs')
    deadline = time.monotonic()+P.POLICY['max_seconds']
    checks = correctness(C, device)
    gc.collect(); torch.cuda.empty_cache()
    baseline = measure(C, output, spec, device, deadline, baseline=True)
    gc.collect(); torch.cuda.empty_cache()
    candidate = measure(C, output, spec, device, deadline)
    healthy = all(r['stream']['healthy'] and r['fixed']['healthy'] and r['finite_state']
                  and r['gpus'][0]['peak_allocated'] < .9*gpu.total_memory for r in (baseline, candidate))
    result = dict(candidate, complete=True, passed=bool(healthy), authorizes_training=False,
        campaign_sha256=spec['content_hash'], runtime=P.runtime(), correctness=checks,
        baseline_parameters=baseline['parameters'], baseline_stream=baseline['stream'], baseline_fixed=baseline['fixed'],
        baseline_inference=baseline['inference'], baseline_gpus=baseline['gpus'], baseline_finite_state=baseline['finite_state'],
        measurement_order=['baseline_state_ev', spec['model_kind']])
    if spec['model_kind'] == 'triangles': result['backend'] = C.M.CONFIG['backend']
    return result
