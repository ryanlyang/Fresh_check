#!/usr/bin/env python3
"""Submit only multiscale using an existing, unchanged benchmark snapshot.

This controller is deliberately outside source/: old training/evaluation workers
need no upgrade. The original scientific admission remains unchanged (including
both benchmark arms); candidate_dispatch.json records the narrower execution
scope. No control or paired report is submitted or represented as completed.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys


ARM = 'multiscale'
KEYS = {'train_multiscale', 'evaluate_multiscale'}
SCHEMA = 'multiscale_candidate_dispatch_v1'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}


def frozen_modules(output):
    source = output / 'source'
    if not source.is_dir():
        raise ValueError('Existing frozen campaign source required')
    sys.path.insert(0, str(source))
    p = importlib.import_module('part_ev_multiscale.protocol')
    s = importlib.import_module('part_ev_multiscale.submission')
    for module in (p, s):
        if Path(module.__file__).resolve().parents[1] != source:
            raise ValueError('Controller must import the frozen campaign, not live training code')
    # Includes the source inventory, host, interpreter and immutable launch checks.
    p.load_launch(output, worker=True)
    return p, s


def receipt(p, launch, plan):
    return p.hashed(dict(schema=SCHEMA, launch_sha256=launch['content_hash'],
        production_sha256=plan['content_hash'], arms=[ARM],
        phases=sorted(KEYS), paired_report=False,
        controller_sha256=p.sha(Path(__file__))))


def owner_path(output, launch, s):
    return output.parent / (launch['platform'] + '_' + s.science_hash() + '.production-owner.json')


def verify_existing(output, launch, plan, p, s):
    """A paired/partial/uncertain chain is never permission to start another one."""
    rows = s.journal(output, 'production')
    expected_owner = dict(output=str(output), campaign_sha256=plan['content_hash'])
    owner = owner_path(output, launch, s)
    if not rows or {r['phase'] for r in rows} != KEYS:
        raise ValueError('Not a complete candidate-only chain; reconcile partial/paired submissions first')
    if not owner.is_file() or p.read(owner) != expected_owner:
        raise ValueError('Candidate production owner differs')
    for name in ('production.json', 'production/campaign.json'):
        if p.checked(output / name) != plan:
            raise ValueError('Existing production admission differs')
    saved = p.checked(output / 'candidate_dispatch.json')
    expected = receipt(p, launch, plan)
    # Keep the original controller hash as provenance across later controller
    # updates. It is not part of the unchanged scientific worker identity.
    for key in ('schema', 'launch_sha256', 'production_sha256', 'arms', 'phases', 'paired_report'):
        if saved.get(key) != expected[key]:
            raise ValueError('Candidate-only execution scope differs')
    return rows


def requests(output, launch, plan, s, *, training=True):
    phases = ('train', 'evaluate') if training else ('evaluate',)
    return [s.request(output, launch, phase, arm=ARM, plan=plan) for phase in phases]


def preview(output, p, s):
    launch = p.load_launch(output, worker=True)
    plan = p.production_spec(output)  # Revalidate raw benchmark, environment, data and ceilings.
    model = next(m for m in plan['models'] if m['arm'] == ARM)
    required = p.check_storage(launch, production=True)
    return dict(mode='candidate_only', campaign=str(output), model=model,
        arm=ARM, control_submitted=False, paired_report_submitted=False,
        required_free_gib=required / 2**30, requests=requests(output, launch, plan, s),
        note='Read-only. Two jobs: multiscale train -> afterok evaluation. '
             'Unchanged benchmark/source/recipe; conservative original paired storage budget retained.')


def dispatch(output, p, s):
    launch = p.load_launch(output, worker=True)
    with s.locks(launch['repo'], output.parent):
        plan = p.production_spec(output)
        owner = owner_path(output, launch, s)
        rows = s.journal(output, 'production')  # Unknown sbatch response fails closed.
        if rows:
            rows = verify_existing(output, launch, plan, p, s)
            return dict(status='already_submitted', jobs={r['phase']: r['job'] for r in rows},
                        note='No jobs added. Use resume-candidate only for an audited terminal attempt.')
        # A crash after admission but before the journal still requires explicit
        # reconciliation; never overwrite an existing campaign or training state.
        if (owner.exists() or any((output / n).exists() for n in
                ('production.json', 'production', 'candidate_dispatch.json', 'production_job_ids.tsv'))
                or list(output.glob('resume_*_submissions.json'))):
            raise ValueError('Existing production artifacts/owner; reconcile before submitting')
        p.check_storage(launch, production=True)
        s.preflight(output, launch, requests(output, launch, plan, s))
        p.write(output / 'candidate_dispatch.json', receipt(p, launch, plan))
        p.write(output / 'production.json', plan)
        p.write(output / 'production/campaign.json', plan)
        p.write(owner, dict(output=str(output), campaign_sha256=plan['content_hash']))
        train = s.submit_one(output, launch, 'train', arm=ARM, prefix='production', plan=plan)
        evaluate = s.submit_one(output, launch, 'evaluate', arm=ARM, prefix='production', plan=plan,
                               dependency='afterok:' + train)
        return dict(status='submitted', jobs=dict(train_multiscale=train, evaluate_multiscale=evaluate),
                    production=str(output / 'production'), control_submitted=False,
                    paired_report_submitted=False)


def resume(output, p, s):
    """Same frozen workers and restart audit, without requiring a control/report."""
    launch = p.load_launch(output, worker=True)
    with s.locks(launch['repo'], output.parent):
        plan = p.load_production(output, worker=True)
        rows = verify_existing(output, launch, plan, p, s)
        latest = {r['phase']: r['job'] for r in rows}
        attempts = sorted(output.glob('resume_[0-9][0-9][0-9]_submissions.json'))
        for path in attempts:
            rows = s.journal(output, path.name.removesuffix('_submissions.json'))
            if {r['phase'] for r in rows} not in (KEYS, {'evaluate_multiscale'}):
                raise ValueError('Partial or noncandidate retry; reconcile before submitting')
            latest.update({r['phase']: r['job'] for r in rows})
        states = s.job_states(set(latest.values()))
        if set(states) != set(latest.values()) or any(v not in TERMINAL for v in states.values()):
            raise ValueError('Previous candidate jobs active or accounting incomplete; no retry submitted')
        metrics = p.run_directory(output, ARM) / 'full_validation/metrics.json'
        if metrics.exists():
            raise ValueError('Final evaluation metrics already exist; audit them instead of resubmitting')
        complete = s.checkpoint_for_retry(output, plan, ARM)
        p.check_storage(launch, production=True)
        s.preflight(output, launch, requests(output, launch, plan, s, training=not complete))
        if len(attempts) >= 999:
            raise ValueError('Retry journal exhausted')
        prefix = f'resume_{len(attempts)+1:03d}'
        jobs = {}
        dependency = None
        if not complete:
            jobs['train_multiscale'] = s.submit_one(output, launch, 'train', arm=ARM, prefix=prefix, plan=plan)
            dependency = 'afterok:' + jobs['train_multiscale']
        jobs['evaluate_multiscale'] = s.submit_one(output, launch, 'evaluate', arm=ARM,
            prefix=prefix, plan=plan, dependency=dependency)
        return dict(status='recovery_submitted', jobs=jobs, restart='latest_complete_state',
                    control_submitted=False, paired_report_submitted=False)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('plan', 'dispatch', 'resume'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if args.mode != 'plan' and not args.reviewed:
        parser.error('Explicit --reviewed authority required')
    if not args.output.is_absolute() or args.output != args.output.resolve(strict=True):
        parser.error('Use the canonical absolute existing campaign path')
    return args


def main(args):
    p, s = frozen_modules(args.output)
    action = dict(plan=preview, dispatch=dispatch, resume=resume)[args.mode]
    print(json.dumps(action(args.output, p, s), indent=2))


if __name__ == '__main__':
    main(parse())
