#!/usr/bin/env python3
"""Bounded isolated benchmark worker; deliberately has no production phase."""
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Match the approved selected/recovery startup (part_ev_fullscale.data imports
# NumPy before Torch). On Tigris, importing Torch first selected its bundled
# libgomp instead of the approved environment's libomp. Keep this at module
# scope so spawned loader workers have the same order as the main process.
# This does not preload/replace libraries or relax verify_backends: the exact
# approved library paths and hashes still have to match on every GPU worker.
import numpy as _numpy_before_torch
import torch as _torch_after_numpy

from part_selected_compact import protocol as P
from scripts.submit_part_selected_compact import verify_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'benchmark', 'summary', 'review'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-index', type=int, choices=(0, 1))
    a = parser.parse_args(argv)
    output = a.output.resolve(strict=True)
    verify_snapshot(output)
    launch = P.read_json(output/'launch.json')
    if launch['output'] != str(output) or launch['authorizes_training'] is not False:
        raise ValueError('Frozen output/admission identity changed')
    if a.phase == 'benchmark' and a.task_index is None:
        parser.error('--task-index required')
    def stop(*_):
        raise TimeoutError('Bounded benchmark stopped; no training authorized')
    if hasattr(signal, 'SIGALRM'):
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, stop)
        signal.alarm(1500 if a.phase in ('summary', 'review') else 17400)
    try:
        if a.phase == 'prepare':
            with P.file_lock(output/'.prepare.lock', blocking=False):
                P.prepare(output)
            print('Prepared independent selected compact benchmarks. No production training authorized.')
        elif a.phase == 'benchmark':
            from part_selected_compact.benchmark import benchmark
            name = P.TASKS[a.task_index]['architecture']
            destination = output/'benchmarks'/f'{name}.json'
            with P.file_lock(output/f'.benchmark_{name}.lock', blocking=False):
                if destination.exists():
                    raise FileExistsError('Benchmark evidence exists; use a fresh campaign for retries')
                row = benchmark(output, a.task_index)
                P.write_json(destination, row)
            print(json.dumps(dict(model=name, passed=True, authorizes_training=False)))
        else:
            required = ['campaign.json', 'benchmarks/compact.json', 'benchmarks/matched.json']
            missing = [n for n in required if not (output/n).is_file()]
            if missing:
                row = dict(passed=False, complete=False, authorizes_training=False, missing=missing,
                           note='Inspect prepare/benchmark logs. Slurm completion alone is not admission.')
                if a.phase == 'summary':
                    P.write_json(output/'reports/incomplete.json', P.hashed(row))
                print(json.dumps(row, indent=2))
                return 1
            row = P.summarize(output)
            if a.phase == 'summary':
                P.write_json(output/'reports/benchmark_summary.json', P.hashed(row))
            print(json.dumps(row, indent=2))
        return 0
    except Exception as exc:
        if a.phase != 'review':
            P.write_json(output/'diagnostics'/f'{a.phase}_{a.task_index}_failure.json',
                         dict(passed=False, authorizes_training=False, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
