#!/usr/bin/env python3
"""Prepare/profile/benchmark/review only. There is deliberately no train phase."""
import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_selected_fullscale import protocol as P


def controls_registry(launch):
    """Inventory references without loading weights, rerunning inference or claiming compatibility."""
    result=[]
    for name in ('baseline_control','baseline_evaluation','state_control'):
        root=Path(launch[name]); records={}
        if root.exists():
            for path in (root/'campaign.json',root/'experiment_spec.json',root/'admission.json'):
                if path.is_file(): records[path.name]=P.file_sha256(path)
        result.append(dict(role=name,path=str(root),exists=root.is_dir(),metadata_hashes=records,
            comparison_status='pending_full_recipe_checkpoint_population_runtime_audit',
            reused_metrics=False,training_submitted=False))
    return dict(references=result,note='Read-only locator registry; controls may still be running. No scientific contrast authorized.')


def build_backend_isolated(output):
    """Keep compiler-only setuptools imports out of the checked parent process."""
    from part_selected_fast import trees
    # cpp_extension imports setuptools, which can append _vendor to sys.path.
    # That changes metadata discovery without installing any packages. Use the
    # same interpreter/source in a child, not a weakened/cached environment check.
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    '--phase','build-backend','--output',str(output)],check=True,
                   timeout=P.BENCHMARK['phase_seconds']-P.BENCHMARK['shutdown_margin_seconds'])
    # Authenticate source, ABI, flags and binary hash again in the parent before
    # numerical parity or model construction. No compiler import is needed here.
    return trees.load_backend(output/'backend')


def prepare(output):
    from part_selected_fast import trees
    from part_selected_screen.region import load_backend
    from part_selected_fullscale.model import build_model, validate_model
    from part_ev_fullscale.checkpoint import seed_everything
    with P.file_lock(output/'.prepare.lock',blocking=False):
        if (output/'experiment_spec.json').exists():
            P.load_campaign(output); trees.load_backend(output/'backend'); return
        launch=P.read_json(output/'launch.json'); parent=Path(launch['reference']).resolve(strict=True)
        if output==parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError('New campaign must not overlap a reference')
        old=P.F.reference(parent)
        budget=P.storage_projection()
        free=shutil.disk_usage(output).free
        if free<budget['required_free_bytes']:
            raise OSError(f"Need {budget['required_free_bytes']/2**30:.2f} GiB including reserve; have {free/2**30:.2f}. Nothing deleted.")
        for name in ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json'):
            target=output/name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and P.file_sha256(target)!=old['inputs'][name]:
                raise ValueError('Partial prepared artifact differs: '+name)
            if not target.exists(): shutil.copyfile(parent/name,target)
            if P.file_sha256(target)!=old['inputs'][name]: raise ValueError('Input copy verification failed')
        native=build_backend_isolated(output)
        reference=load_backend(parent/'backend')
        P.write_json(output/'backend/parity.json',trees.backend_parity(native,reference,4))
        normal=P.checked_json(output/'inputs/normalization.json','part_selected_normalization_v1')
        checks=[]
        for task in P.TASKS:
            seed_everything(101)
            model=build_model(task['architecture'],normalization=normal,backend=reference)
            checks.append(dict(task=task,**validate_model(model,task['architecture'])))
            del model
        P.write_json(output/'controls/registry.json',controls_registry(launch))
        P.write_json(output/'prepare_report.json',dict(models=checks,storage=budget,free_bytes=free,
                      authorizes_training=False,note='CPU counts/tree checks only; CUDA admission remains pending.'))
        P.write_json(output/'experiment_spec.json',P.create_spec(output))
        print('Prepared seven-model benchmark campaign. No production training submitted.',flush=True)


def review(output, *, save=False, retry_root=None, retry_provenance=None, retry_validator=None):
    if (retry_root is None) != (retry_provenance is None):
        raise ValueError('Retry review requires its explicit source provenance')
    if retry_root is not None and Path(retry_root).resolve() == Path(output).resolve():
        raise ValueError('Retry report must not overwrite the original campaign report')
    if retry_root is not None and retry_validator is None:
        raise ValueError('Retry review requires the synchronized parity validator')
    spec=P.load_campaign(output,check_environment=False)
    rows=[]; complete=True; healthy=True
    checkpoints={}; weights={}
    for index,task in enumerate(P.TASKS):
        root=Path(retry_root) if retry_root is not None and index not in P.SELECTED else output
        path=root/'benchmarks'/f"{task['run_id']}.json"
        item=dict(task=task,status='pending')
        if retry_root is not None:
            item.update(evidence_path=str(path),reused_original_selected=index in P.SELECTED)
        if not path.exists():
            complete=False; healthy=False
            failures=sorted((root/'failures').glob(f'benchmark_{index}_*.json'))
            if index in P.SELECTED:
                failures+=sorted((output/'failures').glob(f'profile_{index}_*.json'))
            if failures:
                latest=max(failures,key=lambda p:int(p.stem.rsplit('_',1)[1]))
                failure=P.read_json(latest)
                item.update(status='failed',phase=failure['phase'],error=failure['error'],
                            failure_record=str(latest))
        else:
            try:
                result=P.checked_json(path,'selected_fullscale_benchmark_v1')
                if retry_root is not None and index not in P.SELECTED:
                    if result.get('retry_provenance') != retry_provenance:
                        raise ValueError('Retry provenance differs')
                    retry_validator(result['parity'])
                projection=P.validate_benchmark(result,spec,index)
                requested=max(216,projection['recommended_train_hours']) if index in P.SELECTED else 216
                fits=projection['recommended_train_hours'] <= (480 if index in P.SELECTED else 216)
                item.update(status='measured',jets_per_second=result['jets_per_second'],projection=projection,
                    proposed_train_hours=requested,within_planned_ceiling=fits,
                    peak_gpu_gib=result['gpu_peak_allocated_bytes']/2**30,
                    peak_host_gib=result['host_peak_bytes']/2**30)
                if index in P.SELECTED:
                    profile=output/'profiles'/f"{task['run_id']}.json"
                    p=P.checked_json(profile,'selected_fullscale_profile_v1')
                    P.validate_profile(p,spec,index)
                    if P.file_sha256(profile)!=result['profile_sha256'] or p['runtime']!=result['runtime']:
                        raise ValueError('Selected paired evidence differs')
                    item['conservative_speedup']=p['conservative_speedup']
                    item['speedup_established']=p['conservative_speedup']>=1.05
                    healthy &= item['speedup_established']
                    # Near the ceiling a short validation extrapolation is not enough.
                    item['needs_longer_timing']=projection['recommended_train_hours']>=.8*480
                    healthy &= not item['needs_longer_timing']
                healthy &= fits
                checkpoints[task['architecture']]=result['checkpoint_bytes']
                weights[task['architecture']]=result['weight_bytes']
            except (ValueError,KeyError,OSError) as error:
                healthy=False; item.update(status='failed',error=str(error))
        rows.append(item)
    storage=P.storage_projection(checkpoints,weights) if len(checkpoints)==7 else P.storage_projection()
    free=shutil.disk_usage(output).free
    storage.update(free_bytes=free,enough_headroom=free>=storage['required_free_bytes'])
    result=dict(complete=complete,benchmarks_healthy=bool(healthy),storage=storage,models=rows,
        authorizes_training=False,ready_for_resource_review=bool(complete and healthy and storage['enough_headroom']),
        note='Benchmark-only stage. Review runtime/accuracy-equivalence evidence and effective QOS limits before separate production implementation/submission.')
    if retry_root is not None: result['retry_provenance']=retry_provenance
    if save: P.write_json(Path(retry_root or output)/'reports/benchmark_review.json',P.hashed(result))
    print(json.dumps(result,indent=2)); return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','profile','benchmark','review','summary','status','build-backend'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--task-index',type=int)
    args=p.parse_args(argv); args.output=args.output.resolve()
    if args.phase in ('review','summary','status'):
        if not (args.output/'experiment_spec.json').is_file():
            print('Preparation is incomplete; inspect logs/submitted_job_ids.tsv. No training authorized.'); return 0
        review(args.output,save=args.phase=='summary'); return 0
    if args.phase in ('profile','benchmark'): P.task(args.task_index)
    begin=time.monotonic()
    def stop(*_): raise TimeoutError('Five-hour diagnostic incomplete; no production authorization')
    def check_stop():
        if time.monotonic()-begin>P.BENCHMARK['phase_seconds']-P.BENCHMARK['shutdown_margin_seconds']: stop()
    if hasattr(signal,'SIGALRM'):
        signal.signal(signal.SIGALRM,stop); signal.alarm(P.BENCHMARK['phase_seconds']-P.BENCHMARK['shutdown_margin_seconds'])
    for name in ('SIGTERM','SIGUSR1'):
        if hasattr(signal,name): signal.signal(getattr(signal,name),stop)
    try:
        if args.phase=='build-backend':
            # Internal subprocess only; the Slurm launcher has no such job phase.
            from part_selected_fast import trees
            trees.build_backend(args.output/'backend')
        elif args.phase=='prepare': prepare(args.output)
        else:
            from part_selected_fullscale import benchmark as B
            getattr(B,args.phase)(args.output,args.task_index,check_stop)
    except BaseException as error:
        # Never overwrite a previous failure or a successful measurement.
        name=f'{args.phase}_{args.task_index}_{time.time_ns()}.json'
        P.write_json(args.output/'failures'/name,dict(phase=args.phase,task_index=args.task_index,
            complete=False,passed=False,error=f'{type(error).__name__}: {error}',authorizes_training=False))
        raise
    finally:
        if hasattr(signal,'SIGALRM'): signal.alarm(0)
    return 0


if __name__=='__main__': raise SystemExit(main())
