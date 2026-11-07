"""Disposable same-runtime profiles and full-budget projections; never training admission."""
import copy
import gc
import os
import shutil
from pathlib import Path
import time
import torch
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state, atomic_torch_save, trusted_load
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen import data as native_data
from part_selected_screen import data as selected_data
from part_selected_screen.region import load_backend as reference_backend
from part_selected_screen.model import build_model as ordinary_selected
from part_selected_fast.trees import load_backend, PackedTrees
from part_selected_fast.profiling import MemoryMonitor, host_memory_limit, instrument
from part_selected_fast.engine import host_peak_bytes
from part_selected_tigris.engine import gpu_setup as native_gpu_setup
from scripts.run_part_ev_screen import validate
from .steps import training_step
from . import protocol as P
from .model import build_model, validate_model
from .parity import selected_parity, full_recipe_restart


class Bench:
    def __init__(self, output, index, check_stop):
        self.output, self.index, self.check_stop = Path(output), index, check_stop
        self.storage_check()
        self.spec=P.load_campaign(output); self.task=P.task(index)
        self.recipe=self.spec['recipes'][self.task['recipe']]
        self.selected=index in P.SELECTED
        self.normal=P.checked_json(self.output/'inputs/normalization.json','part_selected_normalization_v1')
        self.old_backend=reference_backend(Path(self.spec['reference']['path'])/'backend') if self.selected else None
        self.backend=load_backend(self.output/'backend') if self.selected else None
        limit=host_memory_limit()
        if limit is None or limit<P.EXECUTION['memory_gib']*2**30:
            raise ValueError('Cannot establish requested host-memory allocation')
        self.device,self.runtime=native_gpu_setup(self,self.spec)
        self.runtime['execution']=P.EXECUTION

    def storage_check(self):
        if shutil.disk_usage(self.output).free<4*2**30:
            raise OSError('Fewer than 4 GiB free; stopping diagnostics without deleting anything')

    def signal(self,*_):
        raise TimeoutError('Diagnostic signal: incomplete, no production admission')

    def model(self, variant='candidate'):
        seed_everything(101)
        provider=(PackedTrees(self.backend,threads=4,namespace=self.spec['content_hash'])
                  if self.selected and variant=='candidate' else None)
        return build_model(self.task['architecture'],normalization=self.normal,backend=self.old_backend,
                           provider=provider,seed=101).to(self.device)

    def loader(self, split):
        if self.selected:
            return selected_data.build_loader(split,self.spec['inventory'],512,seed=101,
                                               config=self.output/'inputs/selected.yaml')[0]
        return native_data.build_loader(split,self.spec['inventory'],512,seed=101)[0]

    def batches(self, loader):
        iterator=iter(loader)
        while True:
            self.check_stop()
            yield next(iterator)

    def parity(self,model,batch):
        if self.selected:
            reference=self.model('reference')
            ordinary=None
            if self.index==6:
                ordinary=ordinary_selected('selected_layerwise_ev',normalization=self.normal,
                                           backend=self.old_backend,seed=101).to(self.device)
            try:
                return selected_parity(model,reference,batch,self.device,ordinary)
            finally:
                del reference,ordinary
                gc.collect(); torch.cuda.empty_cache()
        return dict(passed=True,model=validate_model(model,self.task['architecture']),
                    full_recipe_restart=full_recipe_restart(model,batch,self.device))

    def header(self, schema):
        return dict(schema=schema,campaign_sha256=self.spec['content_hash'],task=self.task,
                    runtime=self.runtime,execution=P.EXECUTION,authorizes_training=False)


def scoped_profile(model, optimizer, scaler, batch, bench):
    """One step; keep only bounded aggregate scopes, never a large Chrome trace."""
    restore=instrument(model)
    handles=[]
    for name,module in model.named_modules():
        if '.state' not in name or name.count('.')>5: continue
        contexts=[]
        def before(module,args,label=name,contexts=contexts):
            scope=torch.profiler.record_function('selected_profile/'+label)
            scope.__enter__(); contexts.append(scope)
        def after(module,args,result,contexts=contexts):
            if contexts: contexts.pop().__exit__(None,None,None)
        handles.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after,always_call=True)))
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
            training_step(model,optimizer,scaler,batch,bench.device,bench.recipe,0)
        return [dict(name=e.key,calls=e.count,cpu_total_us=e.cpu_time_total,
                     device_total_us=getattr(e,'device_time_total',0)) for e in trace.key_averages()
                if e.key.startswith(('selected_profile/','selected_fast/'))]
    finally:
        restore()
        for handle in handles: handle.remove()


def profile(output,index,check_stop):
    if index not in P.SELECTED: raise ValueError('Paired profiles are for selected tasks 2,5,6')
    bench=Bench(output,index,check_stop)
    destination=Path(output)/'profiles'/f'{bench.task["run_id"]}.json'
    with P.file_lock(destination.with_suffix('.lock'),blocking=False):
        if destination.exists():
            old=P.checked_json(destination,'selected_fullscale_profile_v1')
            P.validate_profile(old,bench.spec,index)
            return old
        results=dict(reference=[],candidate=[]); scopes={}; parity=None
        with MemoryMonitor() as memory:
            for trial,order in enumerate((('reference','candidate'),('candidate','reference'),('reference','candidate'))):
                for variant in order:
                    check_stop()
                    model=bench.model(variant)
                    optimizer=build_optimizer(model,bench.recipe); scaler=torch.amp.GradScaler('cuda')
                    loader=bench.loader('train')
                    try:
                        iterator=bench.batches(loader); seed_everything(101)
                        cold=time.monotonic(); first=next(iterator); cold=time.monotonic()-cold
                        if variant=='candidate' and parity is None:
                            parity=bench.parity(model,first)
                        # Reset after diagnostics; all timings start from the same weights and RNG.
                        seed_everything(101)
                        fingerprints={}; skipped=0; loader_seconds=0
                        start=time.monotonic()
                        for step in range(220):
                            check_stop(); fetch=time.monotonic()
                            batch=first if step==0 else next(iterator)
                            if step>=20: loader_seconds+=time.monotonic()-fetch
                            if step in (0,20): fingerprints[str(step)]=selected_data.batch_fingerprint(batch)
                            row=training_step(model,optimizer,scaler,batch,bench.device,bench.recipe,step)
                            if step>=20: skipped+=row['amp_skipped']
                            if step==19: torch.cuda.synchronize(); measured=time.monotonic()
                        torch.cuda.synchronize(); seconds=time.monotonic()-measured
                        results[variant].append(dict(trial=trial,steps=200,seconds=seconds,
                            startup_seconds=measured-start+cold,loader_wait_seconds=loader_seconds,
                            input_fingerprints=fingerprints,amp_skips=skipped))
                        if trial==0:
                            scopes[variant]=scoped_profile(model,optimizer,scaler,first,bench)
                        print(f'{bench.task["run_id"]} {variant} trial={trial} seconds={seconds:.3f}',flush=True)
                    finally:
                        selected_data.shutdown_loader(loader)
                        del model,optimizer,scaler,loader
                        gc.collect(); torch.cuda.empty_cache()
        from scripts.profile_part_selected_fast import summarize
        summary=summarize(results['reference'],results['candidate'])
        expected={k:bench.spec['input_reference']['fingerprints'][k] for k in ('0','20')}
        passed=(parity['passed'] and all(r['input_fingerprints']==expected and r['amp_skips']/200<.01
                for rows in results.values() for r in rows))
        row=P.hashed(dict(**bench.header('selected_fullscale_profile_v1'),complete=True,passed=passed,
            parity=parity,measurements=results,scopes=scopes,host_peak_bytes=max(memory.peak,host_peak_bytes()),
            speedup_established=summary['conservative_speedup']>=1.05,**summary,
            note='Paired timing is measured separately from profiling; scopes overlap. No production authorization.'))
        P.write_json(destination,row)
        if not passed: raise ValueError('Paired input/parity/AMP check failed')
        P.validate_profile(row,bench.spec,index)
        return row


def validate_retry_parity(parity):
    row = parity.get('full_recipe_restart', {})
    expected = dict(passed=True, contract='synchronized_exact_restart_v2',
                    steps=[0,1,2,3,4,5,699999,700000,979999,980000], warmup_steps=10,
                    trials=['uninterrupted','repeat','resumed'], serialized_restart_after=3,
                    optimizer_and_rng_exact=True, exact_roundtrip=True, exact_each_update=True,
                    rtol=0, atol=0)
    if parity.get('passed') is not True or any(row.get(k) != v for k,v in expected.items()):
        raise ValueError('Retry requires the complete synchronized zero-tolerance restart check')


def benchmark(output,index,check_stop, *, retry_root=None, restart_check=None, retry_provenance=None):
    # Explicit sidecar results: the original campaign's measurements are never
    # overwritten or assigned a different campaign/source hash.
    retry = retry_root is not None
    if retry:
        if index in P.SELECTED or restart_check is None or not retry_provenance:
            raise ValueError('Retry only standard arms with explicit check and provenance')
        if Path(retry_root).resolve() == Path(output).resolve():
            raise ValueError('Retry output must differ from the frozen campaign')
    elif restart_check is not None or retry_provenance is not None:
        raise ValueError('A changed restart check requires a separate retry directory')
    bench=Bench(output,index,check_stop)
    directory=Path(retry_root if retry else output)/'benchmarks'; destination=directory/f'{bench.task["run_id"]}.json'
    with P.file_lock(destination.with_suffix('.lock'),blocking=False):
        if destination.exists():
            old=P.checked_json(destination,'selected_fullscale_benchmark_v1')
            P.validate_benchmark(old,bench.spec,index)
            if retry:
                if old.get('retry_provenance') != retry_provenance: raise ValueError('Retry provenance changed')
                validate_retry_parity(old['parity'])
            return old
        if bench.selected:
            profile_path=Path(output)/'profiles'/f'{bench.task["run_id"]}.json'
            paired=P.checked_json(profile_path,'selected_fullscale_profile_v1')
            P.validate_profile(paired,bench.spec,index)
            if paired['runtime']!=bench.runtime: raise ValueError('Profile and benchmark hardware/runtime differ')
        loader=bench.loader('train'); val_loader=None
        with MemoryMonitor() as memory:
            try:
                cold=time.monotonic(); model=bench.model()
                optimizer=build_optimizer(model,bench.recipe); scaler=torch.amp.GradScaler('cuda')
                iterator=bench.batches(loader); seed_everything(101); first=next(iterator)
                cold=time.monotonic()-cold
                if retry:
                    parity=dict(passed=True,model=validate_model(model,bench.task['architecture']),
                                full_recipe_restart=restart_check(model,first,bench.device))
                    validate_retry_parity(parity)
                else:
                    parity=paired['parity'] if bench.selected else bench.parity(model,first)
                seed_everything(101)
                steps=P.BENCHMARK['selected_steps' if bench.selected else 'standard_steps']
                fingerprints={}; windows=[]; skipped=calibration_skips=0; loader_wait=0
                torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
                for step in range(20+steps):
                    check_stop(); fetch=time.monotonic()
                    batch=first if step==0 else next(iterator)
                    if step>=20: loader_wait+=time.monotonic()-fetch
                    if str(step) in bench.spec['input_reference']['fingerprints']:
                        fingerprints[str(step)]=selected_data.batch_fingerprint(batch)
                    row=training_step(model,optimizer,scaler,batch,bench.device,bench.recipe,step,check_precision=step==0)
                    if step==0: precision=row['precision']
                    if step<20: calibration_skips+=row['amp_skipped']
                    else: skipped+=row['amp_skipped']
                    if step==19:
                        torch.cuda.synchronize(); measured=window=time.monotonic(); startup=measured-start+cold
                    if step>=20 and (step-19)%1000==0:
                        bench.storage_check()
                        torch.cuda.synchronize(); stop=time.monotonic()
                        windows.append(dict(steps=1000,seconds=stop-window)); window=stop
                        print(dict(stage='sustained_window',task=bench.task,completed=step-19,window=windows[-1]),flush=True)
                seconds=time.monotonic()-measured
                native_peak=torch.cuda.max_memory_allocated()
                # Worst-length calibration is bounded; it cannot lower the skip
                # threshold by silently extending or discarding failed confirmation.
                model.mod.trimmer.enabled=False
                full_calibration=[]
                for attempt in range(8):
                    check_stop()
                    r=training_step(model,optimizer,scaler,first,bench.device,bench.recipe,0,full_length=True)
                    full_calibration.append(r['amp_skipped'])
                    if not r['amp_skipped']: break
                if full_calibration[-1]: raise ValueError('Full-length AMP calibration exhausted')
                full_skips=0
                for _ in range(10):
                    check_stop()
                    full_skips+=training_step(model,optimizer,scaler,first,bench.device,bench.recipe,0,full_length=True)['amp_skipped']
                full_peak=torch.cuda.max_memory_allocated()
                model.mod.trimmer.enabled=True
                model.mod.trimmer._counter=5
                check_stop(); val_loader=bench.loader('val')
                validation=validate(model,bench.batches(val_loader),bench.device,131072,check_precision=True)
                allocated=max(native_peak,full_peak,torch.cuda.max_memory_allocated())
                reserved=max(memory.gpu_reserved,torch.cuda.max_memory_reserved())
                check_stop(); bench.storage_check(); start=time.monotonic()
                probe=directory/f'{bench.task["run_id"]}_probe.pt'
                payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    rng=capture_rank_state(model),task=bench.task,campaign_sha256=bench.spec['content_hash'],
                    diagnostic_only=True,step=steps+20)
                atomic_torch_save(payload,probe); checkpoint_seconds=time.monotonic()-start
                # Exercise filesystem serialization of restart state, not only RAM bytes.
                saved=trusted_load(probe)
                model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
                scaler.load_state_dict(saved['scaler']); restore_rank_state(model,saved['rng'])
                del saved,payload
                start=time.monotonic(); weights=directory/f'{bench.task["run_id"]}_weights.pt'
                atomic_torch_save(dict(model=model.state_dict(),diagnostic_only=True),weights)
                weight_seconds=time.monotonic()-start
                # Include checkpoint reload temporary allocations as well.
                allocated=max(allocated,torch.cuda.max_memory_allocated())
                reserved=max(reserved,memory.gpu_reserved,torch.cuda.max_memory_reserved())
                row=dict(**bench.header('selected_fullscale_benchmark_v1'),complete=True,passed=True,
                    parity=parity,input_fingerprints=fingerprints,measured_steps=steps,windows=windows,
                    seconds=seconds,jets_per_second=512*steps/seconds,loader_wait_seconds=loader_wait,
                    startup_seconds=startup,amp_skipped_steps=skipped,calibration_skips=calibration_skips,
                    full_length_calibration_skips=full_calibration,full_length_amp_skips=full_skips,
                    full_length_confirmed=10,training_precision=precision,validation=validation,
                    gpu_peak_allocated_bytes=allocated,gpu_peak_reserved_bytes=reserved,
                    host_peak_bytes=max(memory.peak,host_peak_bytes()),checkpoint_seconds=checkpoint_seconds,
                    checkpoint_bytes=probe.stat().st_size,weight_save_seconds=weight_seconds,weight_bytes=weights.stat().st_size)
                if bench.selected: row['profile_sha256']=P.file_sha256(profile_path)
                if retry: row['retry_provenance']=retry_provenance
                try:
                    row['projection']=P.validate_benchmark(row,bench.spec,index)
                except Exception as error:
                    row.update(passed=False,error=str(error)); P.write_json(destination,P.hashed(row)); raise
                P.write_json(destination,P.hashed(row))
                return row
            finally:
                selected_data.shutdown_loader(loader)
                if val_loader is not None: selected_data.shutdown_loader(val_loader)
