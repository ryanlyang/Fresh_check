#!/usr/bin/env python3
"""Bounded compatibility check or explicitly admitted original production retry."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_compiler_recovery import protocol as P


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('inspect', 'check', 'review', 'train', 'evaluate', 'report'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--task-index', type=int, choices=P.INDICES)
    a = parser.parse_args(argv)
    if a.phase == 'inspect':
        if a.campaign is None:
            parser.error('inspect requires campaign')
        _, plan = P.activate(a.campaign)
        print(json.dumps(dict(production_sha256=plan['content_hash'], artifacts=P.evidence(plan))))
        return
    if a.output is None:
        parser.error('output required')
    a.output = a.output.resolve(strict=True)
    row, original, plan = P.load(a.output)
    campaign = Path(row['campaign'])
    if a.phase == 'check':
        from part_ev_fullscale.protocol import file_lock
        from part_selected_compiler_recovery.backend import verify
        with file_lock(a.output/'.check.lock', blocking=False):
            if (a.output/'compatibility.json').exists():
                raise FileExistsError('Compatibility evidence is immutable; use a new recovery directory')
            # No training admission is published if any verification raises.
            _, _, evidence = verify(plan)
            result = P.hashed(dict(schema=P.SCHEMA, recovery_sha256=row['content_hash'],
                indices=list(P.INDICES), passed=True, authorizes_recovery=True, evidence=evidence,
                host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                note='Existing binaries unchanged. Workers must reverify all evidence; no jobs submitted by this check.'))
            P.write(a.output/'compatibility.json', result)
        print(json.dumps(result, indent=2))
        return
    approved = P.approval(a.output, row)
    if a.phase == 'review':
        print(json.dumps(dict(passed=True, campaign=str(campaign), indices=list(P.INDICES),
            compatibility_sha256=approved['content_hash'], compiler=approved['evidence']['compiler'],
            train_hours={str(i): plan['train_hours'][i] for i in P.INDICES},
            evaluate_hours={str(i): plan['evaluate_hours'][i] for i in P.INDICES},
            note='Read-only. Submit requires --reviewed; every worker rechecks compiler, libraries, ABI and parity.'), indent=2))
        return
    from part_selected_production import engine as original_engine
    if a.phase == 'report':
        from part_selected_production.reporting import report
        print(json.dumps(report(campaign, plan), indent=2))
        return
    if a.task_index is None:
        parser.error('task index required')
    # Allocation deadline established before checking backends; checking consumes
    # walltime, never extends the original shutdown margin.
    deadline = P.original_driver(plan).deadline()
    from part_selected_compiler_recovery.backend import verify
    reference, fast, evidence = verify(plan, expected=approved['evidence'])
    from part_selected_compiler_recovery.engine import create_engine
    worker = create_engine(campaign, plan, a.task_index, reference, fast, deadline=deadline)
    started = time.time()
    session = a.output/'sessions'/f"{a.phase}_{a.task_index}_{os.environ.get('SLURM_JOB_ID')}_{time.time_ns()}.json"
    record = dict(phase=a.phase, index=a.task_index, start=started, job_id=os.environ.get('SLURM_JOB_ID'),
        compatibility_sha256=approved['content_hash'], evidence=evidence, production_sha256=plan['content_hash'])
    P.write(session, dict(record, status='started'))
    completed = False
    try:
        if a.phase == 'train':
            original_engine.train(campaign, plan, a.task_index, worker)
        else:
            a.output = campaign  # Original evaluator writes to the same authenticated run.
            worker.evaluate(a)
        completed = True
    finally:
        P.write(session, dict(record, status='completed' if completed else 'interrupted_or_failed',
                             elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    main()
