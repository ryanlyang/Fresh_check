"""Same-GH200, matched ROOT streams; bounded diagnostic rather than production."""
import gc
import hashlib
from pathlib import Path
import time
import torch
from part_ev_masked_predictive import runtime as R
from part_ev_masked_predictive.protocol import hashed, write
from part_ev_screen.data import batch_fingerprint
from part_ev_fullscale.data import build_loader, shutdown_loader
from .config import SCHEMA, VARIANTS, POLICY
from .verification import execute, parity, restart


def measure(session, supplier, variant, *, phase, steps, windows=1, warmup=20):
    timings = []; skipped = 0; fingerprints = hashlib.sha256(); first = None
    if steps<=0 or windows<=0 or steps%windows: raise ValueError('Incomplete timing windows')
    width = steps//windows
    for index in range(warmup+steps):
        if R.stopping(session.device, 1): raise TimeoutError('Speed diagnostic deadline')
        if index == warmup:
            R.synchronize(session.device); start = mark = time.monotonic()
        x, y, fingerprint = supplier(index)
        label = (100000 if phase=='active' else 900000)+index
        row = execute(session, x, y, variant, schedule_label=label, diagnostic=True, audit=index==0)
        if first is None: first = row
        if phase=='active' and row['eligible']<=0: raise ValueError('No active auxiliary work')
        if index >= warmup:
            skipped += row['skipped']; fingerprints.update(fingerprint.encode('ascii'))
            if (index-warmup+1)%width == 0:
                R.synchronize(session.device); now = time.monotonic()
                timings.append(dict(steps=width,seconds=now-mark)); mark = now
    if skipped > POLICY['skip_fraction']*steps: raise ValueError('Excessive timed AMP skips')
    return dict(steps=steps,warmup=warmup,windows=timings,amp_skips=skipped,
        # Conservative maximum window, not a best-case throughput.
        seconds_per_attempt=max(w['seconds']/w['steps'] for w in timings),
        total_seconds=mark-start,stream_sha256=fingerprints.hexdigest(),precision=first['precision'],
        auxiliary_precision=first['auxiliary_precision'],rng_isolated=first['rng_isolated'])


def profile(session, variant, x, y):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,
                               with_stack=False) as profiler:
        execute(session, x, y, variant, schedule_label=100000, diagnostic=True)
        R.synchronize(session.device)
    return {event.key:dict(calls=event.count,cpu_us=event.cpu_time_total,
                          device_us=getattr(event,'device_time_total',0.))
            for event in profiler.key_averages() if event.key.startswith('masked.')}


def run_variant(output, spec, launch, variant, device):
    gates = []
    if variant['optimized']:
        for amp in (False,True):
            gates.append(parity(variant,device,amp=amp,batch=POLICY['parity_batch'],particles=POLICY['parity_particles']))
    restored = restart(variant,device,output/'probes'/variant['name']/'restart.pt')
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    s = R.fresh('masked',device,1,trim=True)
    loader = None
    try:
        loader,_ = build_loader('train',spec['inventory'],512,rank=0,world_size=1,
            workers=launch['resources']['workers'],seed=101)
        iterator = iter(loader)
        def supply(_):
            batch = next(iterator)
            fingerprint = batch_fingerprint(batch)
            x,y = R.tensors(batch,device)
            return x,y,fingerprint
        active = measure(s,supply,variant,phase='active',steps=POLICY['measured'],windows=POLICY['windows'])
        clean = measure(s,supply,variant,phase='clean',steps=POLICY['clean_steps'])
        shutdown_loader(loader); loader = None
        s.base.student.mod.trimmer.enabled = False
        batch = R.fixed_batch(512,128); x,y = R.tensors(batch,device); fingerprint = batch_fingerprint(batch)
        fixed = measure(s,lambda _:(x,y,fingerprint),variant,phase='active',steps=POLICY['fixed_steps'])
        peak = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        if peak >= POLICY['memory_fraction']*total: raise ValueError('Insufficient device headroom at full length')
        if not R.finite_state(s.state_dict()): raise ValueError('Nonfinite full training state')
        stages = profile(s,variant,x,y)
        return dict(**variant,passed=True,parity=gates,restart=restored,active=active,clean=clean,
                    fixed128=fixed,peak_gib=peak/2**30,stages=stages,finite_state=True)
    finally:
        if loader is not None: shutdown_loader(loader)


def run(output, launch, spec, review, gpu, device):
    output = Path(output)
    if (output/'results.json').exists(): raise FileExistsError('Diagnostic already has results; use review')
    R.DEADLINE = min(R.DEADLINE,time.monotonic()+POLICY['max_seconds'])
    records = []
    def save(complete):
        write(output/'results.json',hashed(dict(schema=SCHEMA,launch_sha256=launch['content_hash'],
            donor_files=launch['donor_files'],complete=complete,authorizes_training=False,runtime_gpu=gpu,
            reference_masked=next(m for m in review['models'] if m['arm']=='masked'),variants=records)))
    for variant in VARIANTS:
        R.log('speed_variant_start',variant=variant)
        try:
            result = run_variant(output,spec,launch,variant,device)
        except Exception as exc:
            result = dict(**variant,passed=False,error=f'{type(exc).__name__}: {exc}')
        records.append(result); save(False)
        R.log('speed_variant_complete',name=variant['name'],passed=result['passed'],error=result.get('error'))
        gc.collect(); torch.cuda.empty_cache()
        if not result['passed'] and variant['name']=='baseline32':
            raise ValueError('Baseline failed; candidate comparisons not attempted')
        if R.stopping(device,1): raise TimeoutError('Incomplete speed study; partial evidence retained')
    save(True)
