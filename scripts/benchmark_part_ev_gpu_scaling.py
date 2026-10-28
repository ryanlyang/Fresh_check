#!/usr/bin/env python3
"""Prepare/review a separate paired GPU diagnostic; no production training mode."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_gpu_scaling import protocol as P


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'benchmark', 'review'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--gpus', type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        if not args.campaign:
            parser.error('--campaign is required for prepare')
        P.prepare(args.output, args.campaign)
        print('Diagnostic prepared. Existing campaigns unchanged.')
        return 0
    plan, parent = P.load(args.output, check_environment=args.phase == 'benchmark')
    if args.phase == 'review':
        print(json.dumps(P.summarize(args.output, plan), indent=2, allow_nan=False))
        return 0
    if args.gpus is None:
        parser.error('--gpus is required for benchmark')
    P.C.verify_inventory_metadata(parent)
    from part_ev_gpu_scaling import benchmark as B
    rank = int(os.environ.get('RANK', '0'))
    path = args.output/f'gpus_{args.gpus}.json'
    # Torchrun stops peer processes if any rank fails. Never overwrite previous evidence.
    if path.exists():
        raise FileExistsError('Use a fresh diagnostic directory to retry')
    try:
        rank, world, device = B.setup(args.gpus)
        row = B.run(plan, parent, rank, world, device)
        if rank == 0:
            row['content_hash'] = P.digest(row)
            P.write_json(path, row)
        if not row['numerically_healthy']:
            raise RuntimeError('Benchmark completed but numerical checks failed; inspect JSON')
    except BaseException as exc:
        failure = dict(completed=False, authorizes_training=False, world_size=args.gpus,
                       plan_sha256=plan['content_hash'], rank=rank,
                       error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        failure['content_hash'] = P.digest(failure)
        P.write_json(args.output/f'gpus_{args.gpus}_rank_{rank}_error.json', failure)
        if rank == 0 and not path.exists():
            P.write_json(path, failure)
        raise
    finally:
        if B.dist.is_initialized():
            B.dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
