#!/usr/bin/env python3
"""Three new seed-101 full-budget runs and two read-only existing controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from part_ev_state_compare import protocol as P
from part_ev_state_compare.model import build_model, validate_capacity_model, nonzero_equivalence
from part_ev_state_compare.engine import create_engine, evaluate, lock_finalists
from part_ev_state_fullscale import model as old_model
from part_layerwise_screen.model import build_model as layerwise_reference
from part_ev_message_study.parity import initial_parity as isolated_parity, ParityError
from part_ev_message_study import controls as C
from part_ev_fullscale.data import inspect_dataset

D = create_engine()
_import = importlib.util.spec_from_file_location('_compare_full_driver',
    Path(__file__).with_name('run_part_ev_state_fullscale.py'))
probe = importlib.util.module_from_spec(_import)
_import.loader.exec_module(probe)
probe.P, probe.D, probe.gpu_setup = P, D, D.gpu_setup


def initial_parity(candidate, task, batch, device):
    arch = task['architecture']
    def reference(_architecture, **kwargs):
        if arch == 'part_state_matched':
            return build_model('part_state_matched', seed=task['seed'], **kwargs)
        if arch == 'state_layerwise':
            return layerwise_reference('part_ev_layerwise', seed=task['seed'], **kwargs)
        return old_model.build_model('part_ev', **kwargs)
    row = isolated_parity(D, candidate, task, batch, device, reference)
    if arch == 'part_state_matched':
        row.update(reference_kind='independent_width601_ParT', capacity_control=validate_capacity_model(candidate))
    else:
        row['reference_kind'] = ('layerwise_EV_zero_state_branch' if arch == 'state_layerwise'
                                 else 'ordinary_EV_zero_state_branch')
        rng = D.capture_rank_state(candidate)
        try:
            D.seed_everything(task['seed'] + 991)
            row['state_gradients'] = nonzero_equivalence(arch, device)
        except Exception as error:
            row.update(passed=False, error=str(error))
            raise ParityError(str(error), row) from error
        finally:
            D.restore_rank_state(candidate, rng)
    return row


probe.initial_parity = initial_parity


def prepare(args):
    if platform.machine() != 'x86_64' or torch.version.cuda is None:
        raise RuntimeError('Prepare in the unchanged SPORC CUDA-enabled environment')
    with P.file_lock(args.output / '.prepare.lock', blocking=False):
        if (args.output / 'campaign.json').exists():
            P.load_campaign(args.output)
            return
        launch = P.read_json(args.output / 'launch.json')
        P.control_contract(launch['controls'])
        if shutil.disk_usage(args.output).free < 20*1024**3:
            raise OSError('Keep 20 GiB free for three trainings, five full-validation outputs and headroom')
        inventory = inspect_dataset(launch['data_root'], check_entries=True, splits=('train', 'val'))
        contents = C.verify_data_contents(inventory)
        contents['provenance'] = 'staging_receipts_and_actual_ROOT_bytes_verified_now'
        for task in P.TASKS:
            D.seed_everything(task['seed'])
            candidate = build_model(task['architecture'], seed=task['seed'])
            D.progress('model_contract', **task, parameters=sum(p.numel() for p in candidate.parameters()))
            del candidate
        spec = P.create_spec(inventory, launch, contents)
        P.write_json(args.output / 'data_contents.json', contents)
        P.write_json(args.output / 'campaign.json', spec)
        D.progress('prepared', new_trainings=3, reused_controls=2, validation_checks=50)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare','benchmark','review','train','lock','evaluate','report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--approve', action='store_true')
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase in ('benchmark','train','evaluate'):
        P.task_directory(args.output, args.task_index)
    if args.phase == 'train' and args.task_index not in P.TRAIN_INDICES:
        raise ValueError('Reused controls must never be retrained here')
    if args.phase == 'benchmark' and args.task_index not in P.BENCHMARK_INDICES:
        raise ValueError('Benchmark only the three new architectures: tasks 0, 1, 2')
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'benchmark':
        probe.benchmark(args)
    else:
        spec = P.load_campaign(args.output)
        if args.phase == 'review':
            summary = P.approve(args.output, spec, reviewed=args.reviewed) if args.approve else P.benchmark_summary(args.output, spec)
            print(json.dumps(probe.review_view(summary), indent=2))
        else:
            P.verify_inventory_metadata(spec)
            P.require_approval(args.output, spec)
            if args.phase == 'train':
                D.train(args)
            elif args.phase == 'lock':
                lock_finalists(args.output, spec)
            elif args.phase == 'evaluate':
                evaluate(args, D)
            else:
                from part_ev_state_compare.reporting import report
                report(args.output, spec, D)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
