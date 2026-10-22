#!/usr/bin/env python3
"""Selected shared/layerwise ParT and ParT-EV on the reusable Ranger screen."""
import argparse
import functools
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from part_selected_screen import protocol as P, data
from part_selected_screen.model import build_model, CONFIGS
from part_selected_screen.region import load_backend
from part_selected_screen.normalization import fit
from part_selected_screen.memory import execution_parity
from part_ev_fullscale.model import build_model as reference_model
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_screen_common.contracts import donor_contract, verify_dataset_metadata


def initial_parity(model, task, batch, device):
    """Compare unchanged weights, NOT falsely demand equal selected logits."""
    rng = capture_rank_state(model)
    try:
        seed_everything(task['seed'])
        reference = reference_model('part_ev' if CONFIGS[task['architecture']][1] else 'part').to(device)
        count = 0
        for name in ('embed', 'cls_blocks', 'norm', 'fc'):
            a, b = getattr(model.mod, name).state_dict(), getattr(reference.mod, name).state_dict()
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                count += 1
        torch.testing.assert_close(model.mod.cls_token, reference.mod.cls_token, rtol=0, atol=0)
        for a, b in zip(model.mod.blocks, reference.mod.blocks):
            for key, value in a.state_dict().items():
                other_key = key.replace('attn.attention.', 'attn.', 1) if key.startswith('attn.attention.') else key
                torch.testing.assert_close(value, b.state_dict()[other_key], rtol=0, atol=0)
                count += 1
        del reference
        result = dict(passed=True, kind='unchanged_backbone_weights_exact_not_equal_logits', tensors=count)
        if batch is not None:
            result['memory_execution'] = execution_parity(model, batch, device)
        return result
    finally:
        restore_rank_state(model, rng)


def diagnostics(model, batch, device):
    pair = model.mod.pair_embed
    return dict(relation_parameter_norms={k: float(torch.cat([p.detach().float().flatten()
        for p in getattr(pair.relations, k).parameters()]).norm()) for k in ('pt', 'track_node', 'track_pair', 'region')})


def engine_for(output):
    """Isolated copy of the common engine with explicit selected data adapters.

    Do not mutate common.engine's globals or any donor source. Its exact step,
    validation, optimizer, checkpoint and metric code remains the shared code.
    This separate module namespace changes only loader/fingerprint bindings.
    """
    import part_screen_common.engine as common
    module_spec = importlib.util.spec_from_file_location('part_screen_common._selected_data_engine', common.__file__)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config = output/'inputs/selected.yaml'
    module.build_loader = functools.partial(data.build_loader, config=config)
    module.finite_validation_loader = functools.partial(data.finite_validation_loader, config=config)
    module.batch_fingerprint = data.batch_fingerprint
    backend = load_backend(output/'backend')
    normal = P.checked_json(output/'inputs/normalization.json', 'part_selected_normalization_v1')
    factory = functools.partial(build_model, normalization=normal, backend=backend)
    return module.ScreenEngine(P, factory, initial_parity, diagnostics)


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
        P.require_environment(donor['environment'])
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
        fit(output, donor, backend)
        P.write_json(output/'experiment_spec.json', P.create_spec(output))
        print('Prepared four new models; four verified controls remain read-only.', flush=True)


def status(output):
    spec = P.load_campaign(output, check_environment=False)
    runs = {t['run_id']: P.task_directory(output, i) for i, t in enumerate(P.TASKS)}
    for contract in (spec['donor'], spec['layerwise_donor']):
        runs.update({name: Path(contract['path'])/'runs'/name/'seed_101' for name in contract['controls']})
    history = {name: {r['check']: r['accuracy'] for r in (P.read_json(p)
        for p in sorted((directory/'validation').glob('check_*.json')))} for name, directory in runs.items()}
    for bias, indices in (('Shared bias', (0, 1)), ('Layerwise bias', (2, 3))):
        names = [P.MATCHED[P.TASKS[i]['run_id']] for i in indices] + [P.TASKS[i]['run_id'] for i in indices]
        print('\n'+bias+' — checkpoint validation accuracy')
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'benchmark', 'gate', 'train', 'lock', 'evaluate', 'report', 'status'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'status':
        status(args.output)
    elif args.phase == 'report':
        from part_selected_screen.reporting import report
        report(args.output)
    elif args.phase == 'gate':
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
