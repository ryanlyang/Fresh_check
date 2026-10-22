#!/usr/bin/env python3
"""Four selected-relation Ranger screens on Tigris GH200; separate from SPORC."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_tigris import protocol as P
from part_selected_tigris.engine import engine_for
from part_selected_screen import data
from part_selected_screen.region import load_backend
from part_selected_screen.normalization import fit
from part_screen_common.contracts import donor_contract, verify_dataset_metadata


def prepare(output):
    with P.file_lock(output/'.prepare.lock', blocking=False):
        if (output/'experiment_spec.json').exists():
            P.load_campaign(output)
            return
        launch = P.read_json(output/'launch.json')
        for name in ('donor', 'layerwise_donor'):
            donor = Path(launch[name]).resolve(strict=True)
            if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
                raise ValueError('Campaign must be disjoint from both read-only donors')
        if shutil.disk_usage(output).free < 6*1024**3:
            raise OSError('Need 6 GiB free for four new runs; nothing deleted')
        source = Path(__file__).resolve().parents[1]
        donor = donor_contract(launch['donor'], P.SHARED, local_source=source)
        environment = P.transfer_environment(donor)
        verify_dataset_metadata(donor['inventory'])
        layerwise = P.layerwise_contract(launch['layerwise_donor'], donor, source)
        from part_selected_screen.reporting import controls
        controls(output, dict(donor=donor, layerwise_donor=layerwise))
        data.write_config(output/'inputs/selected.yaml')
        manifest = output/'backend/backend_manifest.json'
        if not manifest.exists():
            subprocess.run([sys.executable, str(source/'scripts/build_relational_part_tree_backend.py'),
                '--contract', 'relational_ca_tree_v1', '--build-dir', str(output/'backend/build'),
                '--output-dir', str(output/'backend')], check=True)
        backend = load_backend(output/'backend')
        from part_selected_tigris.inputs import prepare_reference
        inputs = prepare_reference(output, donor, environment)
        # Fit the same training-only algorithm against the target native stream.
        # This temporary input contract is NOT the recorded SPORC donor.
        fit(output, dict(inventory=donor['inventory'], input_fingerprints=inputs['fingerprints']), backend)
        P.write_json(output/'experiment_spec.json', P.create_spec(output))
        print('Prepared four new models; four SPORC controls remain read-only (cross-hardware references).', flush=True)


def status(output):
    spec = P.load_campaign(output, check_environment=False)
    print('Cross-hardware references: selected models on GH200; reused controls on SPORC A100.')
    runs = {t['run_id']: P.task_directory(output, i) for i, t in enumerate(P.TASKS)}
    for contract in (spec['donor'], spec['layerwise_donor']):
        runs.update({name: Path(contract['path'])/'runs'/name/'seed_101' for name in contract['controls']})
    history = {name: {r['check']: r['accuracy'] for r in (P.read_json(p)
        for p in sorted((directory/'validation').glob('check_*.json')))} for name, directory in runs.items()}
    for bias, indices in (('Shared bias', (0, 1)), ('Layerwise bias', (2, 3))):
        names = [P.MATCHED[P.TASKS[i]['run_id']] for i in indices] + [P.TASKS[i]['run_id'] for i in indices]
        print('\n'+bias+' - checkpoint validation accuracy')
        print('Check'+''.join(f' {name:>25s}' for name in names))
        for check in range(1, 11):
            values = [f'{100*history[n][check]:.4f}%' if check in history[n] else '--' for n in names]
            print(f'{check:5d}'+''.join(f' {v:>25s}' for v in values))
    print('\nBest accuracy through each matched pair\'s common cutoff:')
    for new, old in P.MATCHED.items():
        shared = set(history[new]) & set(history[old])
        if not shared:
            print(new+': no shared completed check yet')
            continue
        cutoff = max(shared)
        a, b = (max(v for k, v in history[name].items() if k <= cutoff) for name in (new, old))
        print(f'{new}: check {cutoff}, selected={100*a:.4f}%, control={100*b:.4f}%, delta={100*(a-b):+.4f} pp')
    print('-- means not recorded. Checks are 20k updates, not full dataset epochs.')


def review(output):
    spec = P.load_campaign(output, check_environment=False)
    print('Model                       Passed   Train hours*   Peak GPU GiB')
    for task in spec['tasks']:
        path = output/'benchmarks'/f"{task['run_id']}.json"
        if not path.exists():
            print(f"{task['architecture']:27s} not recorded")
            continue
        row = P.read_json(path)
        peak = max(row[k] for k in ('native_peak_bytes', 'full_length_peak_bytes', 'validation_peak_bytes'))/2**30
        print(f"{task['architecture']:27s} {str(row['passed']):6s} {row['projected_training_hours']:12.2f} {peak:14.2f}")
    print('*Projection, not a guarantee. Admission requires 30% time margin and all four checks.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'benchmark', 'gate', 'train', 'lock', 'evaluate', 'report', 'status', 'review'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'status':
        status(args.output)
    elif args.phase == 'review':
        review(args.output)
    elif args.phase == 'report':
        from part_selected_tigris.reporting import report
        report(args.output)
    elif args.phase == 'gate':
        review(args.output)
        rows = P.require_approval(args.output, P.load_campaign(args.output))
        for row in rows:
            print(f"{row['task']['run_id']}: projected train {row['projected_training_hours']:.2f} h")
        print('All four benchmarks passed; queued training may proceed from fresh initialization.')
    else:
        spec = P.load_campaign(args.output)
        engine = engine_for(args.output)
        if args.phase == 'lock':
            engine.lock(args.output, spec)
        else:
            P.task_directory(args.output, args.task_index)
            getattr(engine, args.phase)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
