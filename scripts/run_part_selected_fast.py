#!/usr/bin/env python3
"""Isolated selected-relations speed study and admitted training workflow."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_selected_fast import protocol as P
from part_selected_fast import trees


def prepare(output):
    with P.file_lock(output/'.prepare.lock',blocking=False):
        if (output/'experiment_spec.json').exists():
            P.load_campaign(output); trees.load_backend(output/'backend'); return
        launch=P.read_json(output/'launch.json')
        parent=Path(launch['reference']).resolve(strict=True)
        if output==parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError('Output must be disjoint from reference campaign')
        P.execution(launch)
        reference=P.reference(parent)
        if shutil.disk_usage(output).free<6*2**30:
            raise OSError('Need 6 GiB persistent headroom; no data removed')
        from part_selected_screen.reporting import controls
        controls(output,reference)
        for name in ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json'):
            target=output/name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if P.file_sha256(target)!=reference['inputs'][name]: raise ValueError('Prepared input changed')
            else: shutil.copyfile(parent/name,target)
        native=trees.build_backend(output/'backend')
        from part_selected_screen.region import load_backend
        parity=trees.backend_parity(native,load_backend(parent/'backend'),int(launch['tree_threads']))
        P.write_json(output/'backend/parity.json',parity)
        P.write_json(output/'experiment_spec.json',P.create_spec(output))
        print('Prepared new execution. Original campaign/data/environment untouched.',flush=True)


def isolated(module_name, protocol):
    """Reuse reporting/status implementations without rewriting their globals."""
    import importlib
    original=importlib.import_module(module_name)
    spec=importlib.util.spec_from_file_location(module_name+'_fast',original.__file__)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.P=protocol
    return module


def review(output):
    spec=P.load_campaign(output,check_environment=False)
    print('Model                         Profile gain     Sustained hours')
    for task in spec['tasks']:
        profile=output/'profiles'/f"{task['run_id']}.json"
        bench=output/'benchmarks'/f"{task['run_id']}.json"
        p=P.read_json(profile) if profile.exists() else {}
        b=P.read_json(bench) if bench.exists() else {}
        print(f"{task['architecture']:28s} {str(p.get('conservative_speedup','pending')):16s} {b.get('projected_training_hours','pending')}")
    print('Production requires all four new parity/speed/memory gates; results are validation, not test.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','profile','benchmark','gate','train','lock','evaluate','report','review','status'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-index',type=int)
    args=parser.parse_args(argv); args.output=args.output.resolve()
    if args.phase=='prepare': prepare(args.output)
    elif args.phase=='profile':
        P.task_directory(args.output,args.task_index)
        from scripts.profile_part_selected_fast import run
        run(args.output,args.task_index)
    elif args.phase=='review': review(args.output)
    elif args.phase=='status': isolated('scripts.run_part_selected_tigris',P).status(args.output)
    elif args.phase=='report': isolated('part_selected_tigris.reporting',P).report(args.output)
    elif args.phase=='gate':
        rows=P.require_approval(args.output,P.load_campaign(args.output))
        print(f'{len(rows)} models passed fast execution admission. Fresh training permitted.')
    else:
        from part_selected_fast.engine import engine_for
        spec=P.load_campaign(args.output); engine=engine_for(args.output)
        if args.phase=='lock': engine.lock(args.output,spec)
        else:
            P.task_directory(args.output,args.task_index)
            getattr(engine,args.phase)(args)
    return 0


if __name__=='__main__': raise SystemExit(main())
