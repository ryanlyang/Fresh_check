#!/usr/bin/env python3
"""Independent Tigris benchmark worker. No production or cancellation phases."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_extensions_tigris import protocol as P


def claim_phase(output, phase):
    """An intentional retry gets a new directory, never an overwritten attempt."""
    if phase not in ('prepare', 'benchmark'):
        return
    with (Path(output)/(phase+'_started.json')).open('x', encoding='utf-8') as stream:
        json.dump(dict(phase=phase, pid=os.getpid(), authorizes_training=False), stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('inspect', 'prepare', 'benchmark', 'summary', 'review'), required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--model', choices=tuple(P.PACKAGES))
    a = p.parse_args(argv)
    if a.phase == 'inspect':
        if a.benchmark is None or a.model is None: p.error('inspect needs benchmark and model')
        print(json.dumps(P.inspect(a.benchmark, a.model), allow_nan=False))
        return 0
    if a.output is None: p.error('output required')
    P.launch_config(a.output)
    if a.phase == 'review':
        result = P.summary(a.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1

    claim_phase(a.output, a.phase)

    def timeout(*_):
        raise TimeoutError('Diagnostic allocation budget exhausted; no production authorized')

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(P.POLICY['max_seconds'] if a.phase == 'benchmark' else P.MINUTES[a.phase]*60-120)
    try:
        if a.phase == 'prepare':
            P.prepare(a.output)
            print('Independent GH200 benchmark prepared; SPORC campaigns unchanged.', flush=True)
            return 0
        if a.phase == 'benchmark':
            spec = P.load(a.output)
            P.check_storage(spec['launch'])
            from part_ev_extensions_tigris.benchmark import run
            result = run(a.output, spec)
            P.write(a.output/'benchmark.json', P.hashed(result))
        result = P.summary(a.output)
        if a.phase == 'summary': P.write(a.output/'summary.json', result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except BaseException as exc:
        P.write(a.output/(a.phase+'_error.json'), dict(
            authorizes_training=False, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        raise
    finally:
        if hasattr(signal, 'SIGALRM'): signal.alarm(0)


if __name__ == '__main__': raise SystemExit(main())
