#!/usr/bin/env python3
"""Same-GH200 paired speed/profiling trial. Never edits reference artifacts."""
import argparse
import gc
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from part_selected_fast import protocol as P
from part_selected_fast.engine import engine_for, host_peak_bytes
from part_selected_fast.profiling import MemoryMonitor, trace_steps
from part_selected_screen import data
from part_selected_screen.region import load_backend
from part_selected_screen.model import build_model as reference_model
from part_ev_fullscale.model import build_model as standard_model
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_fullscale.optimizer import build_optimizer
from scripts.run_part_ev_screen import training_step, validate

TRIALS=3
WARMUP=20
STEPS=200


def summarize(reference,candidate):
    if len(reference)!=TRIALS or len(candidate)!=TRIALS:
        raise ValueError('Need all paired timing trials')
    a=[r['seconds'] for r in reference]; b=[r['seconds'] for r in candidate]
    if any(not 0<x<float('inf') for x in a+b): raise ValueError('Invalid timing')
    # Conservative non-overlapping observed ranges, not a confidence interval.
    return dict(median_speedup=statistics.median(a)/statistics.median(b),
                conservative_speedup=min(a)/max(b))


def run(output,index):
    spec=P.load_campaign(output); task=spec['tasks'][index]
    recipe=spec['recipes'][task['recipe']]
    engine=engine_for(output)
    device,runtime=engine.gpu_setup(spec)
    directory=output/'profiles'; path=directory/f"{task['run_id']}.json"
    normal=P.checked_json(output/'inputs/normalization.json','part_selected_normalization_v1')
    backend=load_backend(Path(spec['reference']['path'])/'backend')
    with P.file_lock(directory/f"{task['run_id']}.lock",blocking=False):
        if path.exists():
            old=P.checked_json(path,'selected_fast_profile_v1')
            if old.get('campaign_sha256')==spec['content_hash'] and old.get('runtime')==runtime and old.get('passed') is True:
                print('Paired profile already passed',flush=True); return
            raise ValueError('Preserve failed/different profile; create a fresh candidate campaign')
        results={'reference':[],'candidate':[],'standard':[]}
        startup=time.monotonic()
        parity=None
        trace_rows={}
        with MemoryMonitor() as memory:
            # Alternate execution order to reduce warm filesystem/thermal bias.
            orders=[('reference','candidate'),('candidate','reference'),('reference','candidate')]
            for trial,order in enumerate(orders):
                for variant in order:
                    seed_everything(task['seed'])
                    model=(engine.make_model(task,device) if variant=='candidate' else reference_model(task['architecture'],
                        normalization=normal,backend=backend,seed=101,trim=True,use_amp=False).to(device))
                    optimizer=build_optimizer(model,recipe); scaler=torch.amp.GradScaler('cuda')
                    loader,_=data.build_loader('train',spec['inventory'],512,config=output/'inputs/selected.yaml',seed=101)
                    val_loader=None
                    try:
                        iterator=iter(loader); seed_everything(101)
                        cold=time.monotonic(); first=next(iterator); cold=time.monotonic()-cold
                        if variant=='candidate' and parity is None:
                            parity=engine.parity(model,task,first,device)
                        torch.cuda.reset_peak_memory_stats()
                        start=time.monotonic(); fingerprints={}
                        for step in range(WARMUP+STEPS):
                            engine.check_stop()
                            batch=first if step==0 else next(iterator)
                            if step in (0,WARMUP): fingerprints[str(step)]=data.batch_fingerprint(batch)
                            training_step(model,optimizer,scaler,batch,device,recipe,step)
                            if step+1==WARMUP:
                                torch.cuda.synchronize(); measured=time.monotonic()
                        torch.cuda.synchronize(); seconds=time.monotonic()-measured
                        end_to_end=time.monotonic()-start+cold
                        row=dict(trial=trial,seconds=seconds,cold_fetch_seconds=cold,end_to_end_seconds=end_to_end,
                                 steps=STEPS,warmup_steps=WARMUP,peak_allocated=torch.cuda.max_memory_allocated(),
                                 peak_reserved=torch.cuda.max_memory_reserved(),input_fingerprints=fingerprints)
                        if trial==0:
                            # Cold and warm validation reuse of the same bounded
                            # sample is a microbenchmark only, not a projected hit rate.
                            val_loader,_=data.build_loader('val',spec['inventory'],512,config=output/'inputs/selected.yaml',seed=101)
                            vi=iter(val_loader); sample=[next(vi) for _ in range(8)]
                            row['validation_cold']=validate(model,iter(sample),device,4096)
                            row['validation_warm']=validate(model,iter(sample),device,4096)
                            trace_rows[variant]=trace_steps(model,optimizer,scaler,[first],device,recipe,training_step,
                                directory/f"{task['run_id']}_{variant}.trace.json")
                        if variant=='candidate' and model.fast_provider.cache is not None:
                            row['cache']=model.fast_provider.cache.stats()
                        results[variant].append(row)
                        print(f"{task['architecture']} {variant} trial {trial}: {seconds:.3f}s",flush=True)
                    finally:
                        data.shutdown_loader(loader)
                        if val_loader is not None: data.shutdown_loader(val_loader)
                        del model,optimizer,scaler
                        gc.collect(); torch.cuda.empty_cache()
            # A small same-runtime standard throughput reference, not a control rerun.
            arch='part_ev' if task['architecture'].endswith('_ev') else 'part'
            seed_everything(101); model=standard_model(arch,trim=True,use_amp=False).to(device)
            optimizer=build_optimizer(model,recipe); scaler=torch.amp.GradScaler('cuda')
            loader,_=data.build_loader('train',spec['inventory'],512,config=output/'inputs/selected.yaml',seed=101)
            try:
                it=iter(loader); seed_everything(101)
                for step in range(WARMUP+STEPS):
                    engine.check_stop()
                    training_step(model,optimizer,scaler,data.standard_batch(next(it)),device,recipe,step)
                    if step+1==WARMUP: torch.cuda.synchronize(); begin=time.monotonic()
                torch.cuda.synchronize()
                results['standard'].append(dict(architecture=arch,seconds=time.monotonic()-begin,steps=STEPS))
            finally: data.shutdown_loader(loader)
        summary=summarize(results['reference'],results['candidate'])
        standard_seconds=results['standard'][0]['seconds']
        summary['candidate_over_standard_step_time']=statistics.median(r['seconds'] for r in results['candidate'])/standard_seconds
        expected={k:spec['input_reference']['fingerprints'][k] for k in ('0','20')}
        inputs_ok=all(row['input_fingerprints']==expected for variant in ('reference','candidate') for row in results[variant])
        passed=inputs_ok and parity['fast_execution']['passed'] and summary['conservative_speedup']>=P.MIN_SPEEDUP
        record=P.hashed(dict(schema='selected_fast_profile_v1',campaign_sha256=spec['content_hash'],task=task,
            runtime=runtime,execution=spec['execution'],passed=passed,parity=parity,inputs_match=inputs_ok,
            measurements=results,scopes=trace_rows,host_peak_bytes=max(memory.peak,host_peak_bytes()),
            diagnostic_seconds=time.monotonic()-startup,**summary))
        P.write_json(path,record)
        md=['# Selected execution profile','',f"Model: {task['architecture']}",
            f"Median speedup: {summary['median_speedup']:.3f}x",
            f"Conservative observed speedup: {summary['conservative_speedup']:.3f}x",f'Passed: {passed}',
            f"Candidate/standard step time: {summary['candidate_over_standard_step_time']:.3f}x",
            '', 'Same-GH200 diagnostics, not scientific control retraining. Warm validation is a cache microbenchmark only.',
            'Detailed scopes overlap and must not be summed as sequential walltime. See JSON and bounded Chrome traces.']
        path.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
        if not passed: raise RuntimeError('Paired speed/parity gate failed; no production admission')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-index',type=int,required=True)
    args=parser.parse_args(); P.task_directory(args.output,args.task_index)
    run(args.output.resolve(),args.task_index)


if __name__=='__main__': main()
