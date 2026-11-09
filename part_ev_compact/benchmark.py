"""Disposable GPU evidence only. Never writes production weights or predictions."""
import gc
import time
from collections import Counter
from pathlib import Path
import torch
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, atomic_torch_save
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.data import batch_fingerprint
from scripts.run_part_ev_screen import validate
from part_selected_fullscale.steps import training_step
from . import protocol as P
from .model import build_model
from .parity import execution, restart
from .flops import curve


def fresh(arch, device):
    seed_everything(101)
    model = build_model(arch).to(device)
    return model, build_optimizer(model, P.recipe(arch)), torch.amp.GradScaler('cuda', init_scale=32768)


def benchmark(output, index):
    output = Path(output)
    spec = P.load(output)
    if type(index) is not int or index not in (0, 1): raise ValueError('Task must be 0 or 1')
    task = P.TASKS[index]; arch = task['architecture']; device = torch.device('cuda')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'GH200' not in torch.cuda.get_device_name(0):
        raise ValueError('Benchmark requires exactly one visible GH200')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    P.require_space(output, 4*2**30)
    torch.set_num_threads(4)
    loader = val_loader = None
    try:
        loader, _ = build_loader('train', spec['inventory'], 512, seed=101, workers=2)
        first = next(iter(loader))
        shutdown_loader(loader); loader = None
        parity = execution(first, device) if arch == 'compact' else dict(passed=True, scope='unchanged_native_attention_and_FFN_operations')
        model, optimizer, scaler = fresh(arch, device)
        restart_result = restart(model, first, device)
        # Cost probes use disposable batch=2 and all-valid lengths. Native
        # trimmer is disabled only for these probes, not the measured stream.
        del optimizer, scaler
        flop_result = curve(model, device)
        del model
        gc.collect(); torch.cuda.empty_cache()
        model, optimizer, scaler = fresh(arch, device)
        model.mod.trimmer.enabled = False
        confirmed = 0
        for step in range(18):
            row = training_step(model, optimizer, scaler, first, device, P.recipe(arch), step, full_length=True)
            confirmed = 0 if row['amp_skipped'] else confirmed+1
            if confirmed == 10: break
        if confirmed != 10: raise ValueError('Full-length AMP confirmation failed')
        full_peak = torch.cuda.max_memory_reserved(device)
        del model, optimizer, scaler
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model, optimizer, scaler = fresh(arch, device)
        loader, _ = build_loader('train', spec['inventory'], 512, seed=101, workers=2)
        iterator = iter(loader)
        lengths = Counter(); current_length = []
        def observe(_m, _args, result): current_length[:] = [result[2].shape[-1]]
        hook = model.mod.trimmer.register_forward_hook(observe)
        fingerprints, windows, skips = {}, [], 0
        warmup = P.BENCHMARK['warmup_steps']; measured = P.BENCHMARK['measured_steps']
        startup = time.monotonic(); window_start = None; window_steps = 0; precision = None
        for step in range(warmup+measured):
            if step == warmup:
                torch.cuda.synchronize(device)
                startup_seconds = time.monotonic()-startup
                window_start = time.monotonic()
            batch = next(iterator)
            if str(step) in spec['input_reference']['fingerprints']:
                fingerprints[str(step)] = batch_fingerprint(batch)
                if fingerprints[str(step)] != spec['input_reference']['fingerprints'][str(step)]:
                    raise ValueError(f'Native stream fingerprint mismatch at {step}')
            row = training_step(model, optimizer, scaler, batch, device, P.recipe(arch), step, check_precision=step==0)
            if step == 0: precision = row['precision']
            if step >= warmup:
                lengths[str(current_length[0])] += 1
                skips += int(row['amp_skipped']); window_steps += 1
                if window_steps == 1000 or step == warmup+measured-1:
                    torch.cuda.synchronize(device)
                    windows.append(dict(steps=window_steps, seconds=time.monotonic()-window_start))
                    print(dict(stage='training_window', model=arch, step=step+1, **windows[-1]), flush=True)
                    window_steps=0; window_start=time.monotonic()
        hook.remove()
        shutdown_loader(loader); loader=None
        if skips >= measured*.01: raise ValueError('Excessive AMP skips')
        # A single bounded restart-file probe, retained for forensic evidence.
        torch.cuda.synchronize(device); start=time.monotonic()
        probe = output/'benchmarks'/f'{arch}_disposable_restart.pt'
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                               rng=capture_rank_state(model), disposable=True), probe)
        checkpoint_seconds=time.monotonic()-start
        val_loader, _ = build_loader('val', spec['inventory'], 512, seed=101, workers=2)
        validation = validate(model, iter(val_loader), device, P.BENCHMARK['validation_events'], check_precision=True)
        if P.runtime_environment() != spec['environment']: raise ValueError('Installed environment changed')
        peak = max(full_peak, torch.cuda.max_memory_reserved(device))
        memory = torch.cuda.get_device_properties(device).total_memory
        if peak >= .9*memory: raise ValueError('Insufficient GPU memory headroom')
        return P.hashed(dict(task=task, campaign_sha256=spec['content_hash'], complete=True, passed=True,
            authorizes_training=False, measured_steps=measured, windows=windows, amp_skips=skips,
            startup_seconds=startup_seconds, length_histogram=dict(lengths), input_fingerprints=fingerprints,
            checkpoint_seconds=checkpoint_seconds, checkpoint_bytes=probe.stat().st_size,
            validation=validation, training_precision=precision, full_length_confirmed=confirmed,
            peak_reserved_bytes=peak, gpu_memory_bytes=memory, gpu=torch.cuda.get_device_name(device),
            flops=flop_result, parity=parity, restart=restart_result,
            pilot_accuracy_is_not_a_production_result=True))
    finally:
        for item in (loader, val_loader):
            if item is not None: shutdown_loader(item)
