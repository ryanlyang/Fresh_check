#!/usr/bin/env python3
"""Verified CPU controller. This is the only new worker allowed to submit jobs."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_benchmark_followup import protocol as F
from part_ev_state_production import protocol as P
from scripts import submit_part_ev_state_production as S


def run(output):
    output=F.canonical(output)
    arm=F.load(output)
    production=Path(arm['production'])
    # File lock is deliberately stdlib/Linux, before importing benchmark science.
    import fcntl
    with (output/'.dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=output/'result.json'
        if result.exists():
            prior=P.read(result)
            if prior['status']=='submitted':
                print(json.dumps(prior));return prior  # A scheduler retry cannot submit twice.
            if prior['status'] in ('dispatch_started','needs_reconciliation'):
                raise ValueError('Previous submission is uncertain; inspect journals before recovery')
        if production.exists():
            raise FileExistsError('Production output already exists; use explicit production recovery')
        args=SimpleNamespace(mode='submit',model=arm['model_kind'],benchmark=Path(arm['benchmark']),
                             output=production,reviewed=True)
        try:
            # Separate clean interpreter: no live-checkout scientific modules.
            evidence=S.inspect(args)
            P.write(output/'admission.json',evidence)
            F.admit(arm,evidence)
            # Pin checks again immediately before handing off. The production
            # launcher rechecks evidence, aggregate storage and live Slurm limits.
            F.load(output)
            P.write(result,dict(status='dispatch_started',production=str(production)))
            S.launch(args,repo=Path(arm['repo']),frozen_source=dict(path=str(output/'source'),
                manifest=arm['source'],commit=arm['commit']),expected_evidence=evidence)
            jobs=(production/'submitted_job_ids.tsv').read_text().splitlines()
            if [line.split()[0] for line in jobs]!=['train','evaluate','report']:
                raise ValueError('Production chain ledger is incomplete; reconcile before retrying')
            row=dict(status='submitted',production=str(production),jobs=jobs,
                     train_hours=evidence['train_hours'],evaluate_hours=evidence['evaluate_hours'])
            P.write(result,row);print(json.dumps(row,indent=2));return row
        except BaseException as exc:
            # No output means the production launcher could not have reached
            # sbatch. Any output (including a freeze-only failure) is reconciled
            # explicitly, never guessed safe or automatically resubmitted.
            P.write(result,dict(status='needs_reconciliation' if production.exists() else 'blocked',
                production=str(production),error=f'{type(exc).__name__}: {exc}'))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    run(args.output)


if __name__=='__main__':main()
