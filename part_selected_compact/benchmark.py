"""Disposable GH200 pilots; no production weights, test split or large caches."""
from collections import Counter
import gc
from pathlib import Path
import time
import torch
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, atomic_torch_save
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen.data import batch_fingerprint as all_input_fingerprint
from part_selected_screen import data
from part_selected_fast.trees import PackedTrees
from part_selected_fast.profiling import MemoryMonitor, host_memory_limit
from part_selected_fullscale.steps import training_step
from scripts.run_part_ev_screen import validate
from . import protocol as P
from .model import build_model, validate_model
from .parity import execution, restart
from .flops import curve


def benchmark(output, index):
    output = Path(output)
    spec = P.load(output)
    if type(index) is not int or index not in (0, 1):
        raise ValueError('Task index must be 0 or 1')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'GH200' not in torch.cuda.get_device_name(0):
        raise ValueError('Require exactly one visible GH200')
    limit = host_memory_limit()
    if limit is None or limit < P.EXECUTION['memory_gib']*2**30:
        raise ValueError('Cannot establish the requested 128 GiB host allocation')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    P.require_space(output, 4*2**30)
    reference, fast_backend, evidence = P.verify_backends(spec)
    normal = P.checked(output/'inputs/normalization.json')
    task = P.TASKS[index]
    arch = task['architecture']
    recipe = P.recipe(arch)

    def factory(architecture, *, optimized=True, fast=True, trim=True):
        seed_everything(101)
        provider = PackedTrees(fast_backend, threads=4, namespace=spec['content_hash']) if fast else None
        return build_model(architecture, normalization=normal, backend=reference,
                           provider=provider, optimized=optimized, trim=trim).to(device)

    def fresh():
        model = factory(arch)
        return model, build_optimizer(model, recipe), torch.amp.GradScaler('cuda', init_scale=32768)

    def loader(split):
        return data.build_loader(split, spec['inventory'], 512, seed=101, workers=2,
                                 config=output/'inputs/selected.yaml')[0]

    train_loader = val_loader = None
    try:
        with MemoryMonitor() as memory:
            train_loader = loader('train')
            first = next(iter(train_loader))
            data.shutdown_loader(train_loader)
            train_loader = None
            if data.batch_fingerprint(first) != spec['input_reference']['fingerprints']['0']:
                raise ValueError('First selected batch differs from native donor stream')
            parity = execution(factory, arch, first, device)
            model = factory(arch)
            restart_result = restart(model, first, device)
            flop_result = curve(model, device)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model, optimizer, scaler = fresh()
            model.mod.trimmer.enabled = False
            confirmed = 0
            for step in range(18):
                row = training_step(model, optimizer, scaler, first, device, recipe, step, full_length=True)
                confirmed = 0 if row['amp_skipped'] else confirmed+1
                if confirmed == 10:
                    break
            if confirmed != 10:
                raise ValueError('Full-length AMP confirmation failed')
            torch.cuda.synchronize(device)
            full_peak = torch.cuda.max_memory_reserved(device)
            del model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            model, optimizer, scaler = fresh()
            warmup, measured = P.BENCHMARK['warmup_steps'], P.BENCHMARK['measured_steps']
            lengths, fingerprints, selected_fingerprints = Counter(), {}, {}
            windows, skips, window_steps = [], 0, 0
            current_length = []
            def observe(_m, _args, result):
                current_length[:] = [result[2].shape[-1]]
            hook = model.mod.trimmer.register_forward_hook(observe)
            startup = time.monotonic()
            train_loader = loader('train')
            iterator = iter(train_loader)
            precision = None
            for step in range(warmup+measured):
                if step == warmup:
                    torch.cuda.synchronize(device)
                    startup_seconds = time.monotonic()-startup
                    window_start = time.monotonic()
                batch = next(iterator)
                if str(step) in spec['input_reference']['fingerprints']:
                    fingerprints[str(step)] = data.batch_fingerprint(batch)
                    selected_fingerprints[str(step)] = all_input_fingerprint(batch)
                    if fingerprints[str(step)] != spec['input_reference']['fingerprints'][str(step)]:
                        raise ValueError('Native stream fingerprint differs at '+str(step))
                row = training_step(model, optimizer, scaler, batch, device, recipe, step, check_precision=step == 0)
                if step == 0:
                    precision = row['precision']
                if step >= warmup:
                    lengths[str(current_length[0])] += 1
                    skips += int(row['amp_skipped'])
                    window_steps += 1
                    if window_steps == 1000 or step == warmup+measured-1:
                        torch.cuda.synchronize(device)
                        windows.append(dict(steps=window_steps, seconds=time.monotonic()-window_start))
                        print(dict(stage='training_window', model=arch, step=step+1, **windows[-1]), flush=True)
                        P.require_space(output, 4*2**30)
                        window_steps, window_start = 0, time.monotonic()
            hook.remove()
            data.shutdown_loader(train_loader)
            train_loader = None
            if skips >= measured*.01:
                raise ValueError('Excessive AMP skips')
            P.require_space(output, 4*2**30)
            torch.cuda.synchronize(device)
            start = time.monotonic()
            probe = output/'benchmarks'/f'{arch}_disposable_restart.pt'
            atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), rng=capture_rank_state(model), disposable=True,
                task=task, campaign_sha256=spec['content_hash']), probe)
            checkpoint_seconds = time.monotonic()-start
            val_loader = loader('val')
            validation = validate(model, iter(val_loader), device, P.BENCHMARK['validation_events'], check_precision=True)
            torch.cuda.synchronize(device)
            if P.runtime_environment() != spec['environment']:
                raise ValueError('Installed environment changed')
            parameters = validate_model(model, arch)['parameters']
            peak = max(full_peak, memory.gpu_reserved, torch.cuda.max_memory_reserved(device))
        row = P.hashed(dict(task=task, campaign_sha256=spec['content_hash'], complete=True, passed=True,
            authorizes_training=False, execution=P.EXECUTION, parameters=parameters, backend_evidence=evidence,
            measured_steps=measured, windows=windows, amp_skips=skips, startup_seconds=startup_seconds,
            length_histogram=dict(lengths), input_fingerprints=fingerprints, selected_input_fingerprints=selected_fingerprints,
            checkpoint_seconds=checkpoint_seconds, checkpoint_bytes=probe.stat().st_size,
            validation=validation, training_precision=precision, full_length_confirmed=confirmed,
            peak_reserved_bytes=peak, host_peak_bytes=memory.peak,
            gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
            gpu=torch.cuda.get_device_name(device), flops=flop_result, parity=parity, restart=restart_result,
            pilot_accuracy_is_not_a_production_result=True))
        P.validate_benchmark(row, spec, task)
        return row
    finally:
        for item in (train_loader, val_loader):
            if item is not None:
                data.shutdown_loader(item)
