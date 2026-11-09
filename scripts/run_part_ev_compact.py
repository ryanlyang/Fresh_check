#!/usr/bin/env python3
"""Bounded CPU preparation, two GPU benchmarks, and fail-closed review."""
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_compact import protocol as P
from scripts.submit_part_ev_compact import verify_snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','benchmark','summary','review'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-index',type=int,choices=(0,1))
    args=parser.parse_args(argv)
    output=args.output.resolve(strict=True)
    verify_snapshot(output)
    if args.phase=='benchmark' and args.task_index is None: parser.error('--task-index required')
    def stop(_signal,_frame): raise TimeoutError('Bounded benchmark stopped; no training authorized')
    if hasattr(signal,'SIGALRM'):
        for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGUSR1): signal.signal(sig,stop)
        signal.alarm(1500 if args.phase in ('summary','review') else 17400)
    try:
        if args.phase=='prepare':
            P.prepare(output)
            print('Prepared independent compact campaign; datasets and donor are read-only.')
        elif args.phase=='benchmark':
            from part_ev_compact.benchmark import benchmark
            name=P.TASKS[args.task_index]['architecture']
            destination=output/'benchmarks'/f'{name}.json'
            with P.file_lock(output/f'.benchmark_{name}.lock',blocking=False):
                if destination.exists(): raise ValueError('Benchmark evidence already exists; not rerunning')
                row=benchmark(output,args.task_index)
                P.write_json(destination,row)
            print(json.dumps(dict(model=name,passed=True,authorizes_training=False)))
        else:
            row=P.summarize(output)
            if args.phase=='summary': P.write_json(output/'reports/benchmark_summary.json',P.hashed(row))
            print(json.dumps(row,indent=2))
        return 0
    except Exception as exc:
        if args.phase!='review':
            P.write_json(output/'diagnostics'/f'{args.phase}_{args.task_index}_failure.json',
                         dict(passed=False,authorizes_training=False,error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if hasattr(signal,'SIGALRM'): signal.alarm(0)


if __name__=='__main__': raise SystemExit(main())
