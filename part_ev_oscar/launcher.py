"""Explicit submission only; no SSH, automatic cancellation, or training phases."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from . import common as C

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path('/oscar/scratch/rlyang/part_ev_test')


def command(args):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True, text=True).stdout.strip()


def snapshot(destination, source=REPO):
    for name in C.TOOLS:
        target = C.child(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.child(source, name), target)
    return {name: C.sha(C.child(destination, name)) for name in C.TOOLS}


def no_active(output):
    ledger = Path(output)/'jobs.json'
    jobs = C.read(ledger) if ledger.exists() else []
    live = {j.split('_')[0] for j in command(['squeue', '--me', '-r', '-h', '-o', '%i']).split()}
    overlap = live.intersection(j['job'] for j in jobs)
    if overlap:
        raise ValueError(f'Campaign jobs still active: {sorted(overlap)}. Nothing cancelled.')


def request(output, plan, phase, key=None, dependencies=()):
    output = Path(output)
    export = phase in ('export-model', 'export-cpu')
    gpu = phase in ('export-model', 'parity', 'evaluate')
    python = plan['donors'][key]['python'] if export else plan['python']
    partition = plan['partition'] if gpu else plan.get('cpu_partition', plan['partition'])
    hours = plan.get('evaluate_hours', 24) if phase == 'evaluate' else 5 if phase == 'metrics' else 2
    if phase == 'report':
        hours = 1
    name = 'evos_'+phase.replace('-','_') + ('_'+key if key else '')
    args = ['sbatch', '--parsable', '--job-name='+name, '--partition='+partition,
            '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=64G',
            f'--time={hours}:00:00', '--signal=B:USR1@300', '--export=ALL',
            '--chdir='+str(output), '--output='+str(output/'logs/%x_%j.out'),
            '--error='+str(output/'logs/%x_%j.err')]
    if plan.get('account'):
        args += ['--account='+plan['account']]
    if gpu:
        args += ['--gres='+plan['gres']]
    if dependencies:
        args += ['--dependency=afterok:'+':'.join(dependencies), '--kill-on-invalid-dep=yes']
    args += [str(output/'source/sbatch/run_part_ev_oscar.sh'), '--python', python,
             '--tool-root', str(output/'source'), '--output', str(output), '--command', phase]
    if key:
        args += ['--model', key]
    return args


def submit(output, plan, phase, key=None, dependencies=()):
    args = request(output, plan, phase, key, dependencies)
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
    job = subprocess.run(args, env=env, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unrecognized sbatch response; inspect queue before retrying')
    path = Path(output)/'jobs.json'
    jobs = C.read(path) if path.exists() else []
    jobs.append(dict(phase=phase, model=key, job=job, request=args))
    C.write(path, jobs)  # commit each successful submission immediately
    print(f'{phase}{" "+key if key else ""} job: {job}', flush=True)
    return job


def valid_resources(args):
    for value in (args.partition, args.cpu_partition, args.account or '', args.gres):
        if value and not re.fullmatch(r'[A-Za-z0-9_:.-]+', value):
            raise ValueError('Invalid scheduler resource value')
    if not re.fullmatch(r'gpu:(?:[A-Za-z0-9_.-]+:)?1', args.gres):
        raise ValueError('Request exactly one GPU per job')


def export_launch(args):
    from . import export as E
    output = args.output.resolve()
    if args.command == 'export-resume':
        plan = C.checked(output/'export_plan.json')
        C.verify_files(output/'source', plan['tools'])
        from .saved_validation import MODE
        if plan.get('export_mode') != MODE:
            raise ValueError('Old GPU export: use a new CPU export directory, not this launcher to resume it')
    else:
        donors = E.describe(args.part, args.state)
        for d in donors.values():
            root = Path(d['root'])
            if output == root or output.is_relative_to(root) or root.is_relative_to(output):
                raise ValueError('Export must be outside both original campaigns')
        from . import saved_validation as V
        references = V.discover(donors, getattr(args, 'validation_results', None))
        for reference in references.values():
            root = Path(reference['root'])
            if output == root or output.is_relative_to(root) or root.is_relative_to(output):
                raise ValueError('Export output must be outside saved validation results')
        if args.gres:
            raise ValueError('CPU export must not request GRES/GPUs')
        partition = args.cpu_partition or args.partition or 'tier3'
        account = args.account or 'reu-aisocial'
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) for x in (partition, account)):
            raise ValueError('Invalid CPU partition/account')
        plan = dict(schema=C.SCHEMA, donors=donors, partition=partition,
                    cpu_partition=partition, account=account, export_mode=V.MODE,
                    validation_results=references)
        print(json.dumps(dict(export_mode=V.MODE, output=str(output), partition=partition,
            resources='Two independent CPU jobs; 4 CPUs, 64 GiB, 2h, zero GPUs each',
            references={k: v['root'] for k, v in references.items()},
            selected_checks={k: d['best_check'] for k, d in donors.items()},
            note='Prediction payload hashes are reverified inside CPU jobs, not on this submit host.'), indent=2))
        if args.command == 'export-plan':
            print('Read-only: no files or jobs changed.')
            return
        if output.exists():
            raise FileExistsError('Use a new export directory or export-resume')
        E.donor_support().refuse_active(donors)
        V.inactive(references, donors)
        output.mkdir(parents=True)
        (output/'logs').mkdir()
        plan['tools'] = snapshot(output/'source')
        plan = C.seal(plan)
        C.write(output/'export_plan.json', plan)
    with C.lock(output/'.submit.lock'):
        no_active(output)
        E.donor_support().refuse_active(plan['donors'])
        from . import saved_validation as V
        V.inactive(plan['validation_results'], plan['donors'])
        for key in C.SELECTED:
            if (output/key/'model.json').exists():
                model = C.checked(output/key/'model.json')
                if (model['key'] != key or model['export_plan_sha256'] != plan['content_hash']
                        or model.get('reference_origin') != V.MODE
                        or model['reference_evaluation'] != plan['validation_results'][key]):
                    raise ValueError('Completed CPU export belongs to a different plan/reference')
                C.verify_files(output/key, model['files'])
                V.verify_portable_provenance(output/key, model)
                continue
            if (output/key).exists():
                raise ValueError('Partial export retained. Use a fresh export directory; nothing deleted.')
            submit(output, plan, 'export-cpu', key)


def init(args):
    scratch = args.scratch_root.resolve(strict=True)
    output = args.output.resolve()
    bundle = args.bundle.resolve(strict=True)
    if (output.exists() or output == scratch or not output.is_relative_to(scratch)
            or output.is_relative_to(bundle) or bundle.is_relative_to(output)):
        raise ValueError('Use a new scratch output directory separate from the bundle')
    if not bundle.is_relative_to(scratch):
        raise ValueError('Put the portable bundle in scratch, not home')
    python = args.python.absolute()
    if not python.is_file() or not python.parent.resolve().is_relative_to(scratch):
        raise ValueError('Use a dedicated Python environment under scratch')
    b = C.load_bundle(bundle)
    valid_resources(args)
    if args.batch_size not in (64, 128, 256, 512) or not 1 <= args.evaluate_hours <= 72:
        raise ValueError('Supported batch sizes: 64/128/256/512; evaluation walltime 1..72h')
    data = args.test_dir.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    if not data.is_relative_to(scratch) or not archive.is_relative_to(scratch):
        raise ValueError('Dataset and archive must be in scratch')
    plan = dict(schema=C.SCHEMA, bundle=str(bundle), bundle_sha256=b['content_hash'],
                python=str(python), test_directory=str(data), archive=str(archive),
                batch_size=args.batch_size, precision='FP32', parity=C.PARITY,
                evaluate_hours=args.evaluate_hours, partition=args.partition,
                cpu_partition=args.cpu_partition, gres=args.gres, account=args.account,
                population='official_test_20M', independent_test=True)
    output.mkdir(parents=True)
    (output/'logs').mkdir()
    plan['tools'] = snapshot(output/'source', bundle/'source')
    C.write(output/'plan.json', C.seal(plan))
    print(f'Initialized {output}. No jobs submitted. Run prepare-submit next.')


def setup_env(args):
    """User-invoked install into a NEW scratch venv; no existing environment modified."""
    scratch = args.scratch_root.resolve(strict=True)
    bundle = C.load_bundle(args.bundle)
    destination = args.venv.absolute()
    if destination.exists() or not destination.parent.resolve().is_relative_to(scratch):
        raise ValueError('Use a NEW venv under scratch (create its parent first)')
    expected = bundle['environment']
    version = command([args.base_python, '-c', 'import platform; print(platform.python_version())'])
    if version.split('.')[:2] != expected['python'].split('.')[:2]:
        raise ValueError(f"Base Python must match donor major.minor ({expected['python']}); got {version}")
    cache = scratch/'cache/pip'
    tmp = scratch/'tmp'
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIP_CACHE_DIR=str(cache), TMPDIR=str(tmp),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_CONFIG_FILE=os.devnull)
    env.pop('PYTHONPATH', None)
    subprocess.run([args.base_python, '-m', 'venv', str(destination)], env=env, check=True)
    python = destination/'bin/python'
    packages = expected['packages']
    cuda = expected['torch_cuda']
    if not cuda or not re.fullmatch(r'\d+\.\d+', cuda):
        raise ValueError('Expected a CUDA-enabled donor environment')
    index = 'https://download.pytorch.org/whl/cu'+cuda.replace('.', '')
    # Both wheel build and scientific dependencies remain pinned to the exporter.
    subprocess.run([str(python), '-m', 'pip', 'install', '--index-url', index,
                    'torch=='+packages['torch']], env=env, check=True)
    subprocess.run([str(python), '-m', 'pip', 'install',
                    *[k+'=='+v for k, v in packages.items()]], env=env, check=True)
    print(f'Environment installed: {python}. Dependencies/source hashes are checked by admission.')


def campaign_submit(args):
    from . import results as R
    output = args.output.resolve(strict=True)
    plan = C.load_plan(output)
    with C.lock(output/'.submit.lock'):
        no_active(output)
        if args.command == 'prepare-submit':
            prepare_job = []
            if not (output/'inventory.json').exists():
                prepare_job = [submit(output, plan, 'prepare')]
            else:
                from .runtime import inventory
                inventory(output, plan)
            for key in C.SELECTED:
                path = output/f'parity_{key}.json'
                if path.exists():
                    row = C.checked(path)
                    if row['plan_sha256'] != plan['content_hash'] or row['passed'] is not True:
                        raise ValueError('Existing failed/different parity; inspect and use a new plan')
                    continue
                submit(output, plan, 'parity', key, prepare_job)
            return
        if not args.reviewed:
            raise ValueError('Read review output, then provide --reviewed to submit test evaluation')
        record = R.admission_record(output, plan)
        path = output/'admission.json'
        if path.exists() and C.checked(path) != record:
            raise ValueError('Admission differs; no overwrite')
        C.write(path, record)
        metrics_jobs = []
        for key in C.SELECTED:
            evaluation = []
            if not (output/key/'inference_complete.json').exists():
                evaluation = [submit(output, plan, 'evaluate', key)]
            # CPU phases revalidate all chunks even if results already exist.
            metrics_jobs.append(submit(output, plan, 'metrics', key, evaluation))
        submit(output, plan, 'report', dependencies=metrics_jobs)


def review(output):
    from .results import admission_record
    output = Path(output)
    plan = C.load_plan(output)
    record = admission_record(output, plan)
    print(json.dumps(record, indent=2))
    for key in C.SELECTED:
        row = C.checked(output/f'parity_{key}.json')
        max_error = max(r['checks']['logits']['maximum_absolute_error'] for r in row['rows'])
        disagreements = sum(r['checks']['logits']['argmax_disagreements'] for r in row['rows'])
        # This deliberately is not an optimistic compute-only production walltime promise.
        print(f"{key}: parity PASS; peak {row['peak_allocated_gib']:.2f} GiB; "
              f"5,120 validation probe events in {row['seconds']:.1f}s; "
              f"max logit error {max_error:.3g}; argmax differences {disagreements}")
    print(f"Planned per-model test walltime: {plan['evaluate_hours']}h; FP32 batch {plan['batch_size']}. "
          'Tiny-probe timing is not a runtime guarantee. Interrupted evaluations are resumable.')
    print('Read-only review. No test jobs submitted.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('export-plan', 'export-submit', 'export-resume', 'export-review',
        'export-model', 'export-cpu', 'setup-env', 'init', 'prepare-submit', 'review', 'evaluate-submit', 'resume',
        'status', 'prepare', 'parity', 'evaluate', 'metrics', 'report'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=('part', 'state'))
    parser.add_argument('--part', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--validation-results', type=Path,
                        help='Completed evrej campaign; otherwise discover a unique matching pair')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--scratch-root', type=Path, default=SCRATCH)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--base-python', default='python3.10')
    parser.add_argument('--venv', type=Path)
    parser.add_argument('--test-dir', type=Path, default=SCRATCH/'data/JetClass/Pythia/test_20M')
    parser.add_argument('--archive', type=Path, default=SCRATCH/'incoming/JetClass_Pythia_test_20M.tar')
    parser.add_argument('--partition')
    parser.add_argument('--cpu-partition')
    parser.add_argument('--gres')
    parser.add_argument('--account')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--evaluate-hours', type=int, default=24)
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if args.command != 'setup-env' and args.output is None:
        parser.error('--output is required')
    if args.command in ('init', 'setup-env') and args.bundle is None:
        parser.error('--bundle is required')
    if args.command == 'init' and args.python is None:
        parser.error('--python is required')
    if args.command == 'setup-env' and args.venv is None:
        parser.error('--venv is required')
    if args.command in ('export-model', 'export-cpu', 'parity', 'evaluate', 'metrics') and args.model is None:
        parser.error('--model is required')
    if args.command in ('export-plan', 'export-submit', 'export-resume'):
        export_launch(args)
    elif args.command == 'export-review':
        from .export import finalize
        finalize(args.output)
    elif args.command == 'setup-env':
        setup_env(args)
    elif args.command == 'init':
        args.partition = args.partition or 'gpu'
        args.cpu_partition = args.cpu_partition or 'batch'
        args.gres = args.gres or 'gpu:nvidia_rtx_a6000:1'
        init(args)
    elif args.command in ('prepare-submit', 'evaluate-submit', 'resume'):
        campaign_submit(args)
    elif args.command == 'review':
        review(args.output)
    elif args.command == 'status':
        plan = C.load_plan(args.output)
        for key in C.SELECTED:
            directory = args.output/key
            print(f"{key}: {len(list(directory.glob('file_*.json')))}/200 chunk sidecars; "
                  f"inference complete={(directory/'inference_complete.json').exists()}; "
                  f"metrics={(directory/'metrics.json').exists()}")
        print('Prediction status is metadata-only; prediction payload hashes were not rechecked.')
        if (args.output/'report.md').exists():
            print((args.output/'report.md').read_text())
    else:
        from . import runtime as R
        R.configure(args.output)
        if args.command in ('export-model', 'export-cpu'):
            from .export import export_model
            export_model(args.output, args.model, cpu=args.command == 'export-cpu')
        elif args.command == 'prepare':
            R.prepare(args.output)
        elif args.command == 'parity':
            R.parity(args.output, args.model)
        elif args.command == 'evaluate':
            R.evaluate(args.output, args.model)
        else:
            from . import results
            if args.command == 'metrics':
                results.metrics(args.output, args.model)
            else:
                results.report(args.output)
