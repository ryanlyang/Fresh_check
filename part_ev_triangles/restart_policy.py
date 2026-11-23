"""SPORC-only restart admission; no change to scientific forward/gradient parity."""
import math

METADATA = dict(passed=True, updates=7, ranks=2, full_state=True, validation_boundaries=1,
                repeatability=True, fixture_cursor_replay=True, real_stream_resume_verified=False)

POLICY = dict(
    version='triangle_sporc_restart_v2', updates=7, resume_after=3, ranks=2,
    initial_attempt=19997, fixture_cursor_replay=True, real_stream_resume_verified=False,
    exact_restoration=True, exact_rng_counters_scaler_schedule=True,
    trials=['reference', 'repeat', 'disk_start', 'resume_after_3'],
    # Fixed before the new benchmark, not fitted to each new allocation.
    # Observed SPORC losses differed by <=6.87e-5; state by <=7.12e-6.
    loss=dict(atol=1e-4, rtol=1e-5, absolute_cap=2e-4),
    state=dict(atol=1e-5, rtol=1e-4, absolute_cap=1e-4),
)


def validate_comparison(row, *, exact=False):
    if (row.get('passed') is not True or row.get('finite') is not True
            or row.get('exact_discrete') is not True
            or type(row.get('leaves')) is not int or row['leaves'] < 1
            or type(row.get('floating_leaves')) is not int
            or not 0 <= row['floating_leaves'] <= row['leaves']):
        raise ValueError('Missing complete restart state comparison')
    for field in ('max_abs_error', 'max_budget_fraction'):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Nonfinite restart comparison')
    if row['max_budget_fraction'] > 1 or row['max_abs_error'] > POLICY['state']['absolute_cap']:
        raise ValueError('Restart trajectory exceeds its fixed numerical bounds')
    if exact and (row.get('exact') is not True or row['max_abs_error'] != 0
                  or row['max_budget_fraction'] != 0):
        raise ValueError('Checkpoint restoration must be exact')


def validate_evidence(row):
    """Recheck every rank/trial, not merely a top-level passed flag."""
    if (row.get('policy') != POLICY or row.get('passed') is not True
            or any(type(row.get(k)) is not type(v) or row[k] != v for k, v in METADATA.items())):
        raise ValueError('Missing versioned SPORC Triangle restart policy')
    ranks = row.get('rank_evidence', [])
    if (len(ranks) != 2 or {r.get('rank') for r in ranks} != {0, 1}
            or row.get('ranks_agree_on_update_schedule') is not True):
        raise ValueError('Missing two-rank restart agreement')
    schedules = []
    for rank in ranks:
        if rank.get('passed') is not True or set(rank.get('trials', {})) != set(POLICY['trials']):
            raise ValueError('Missing uninterrupted/restart trial')
        validate_comparison(rank['checkpoint_roundtrip'], exact=True)
        reference = None
        for name in POLICY['trials']:
            trial = rank['trials'][name]
            validate_comparison(trial['initial_restore'], exact=True)
            if name == 'resume_after_3':
                validate_comparison(trial['mid_deserialization'], exact=True)
                validate_comparison(trial['mid_restore'], exact=True)
            if trial.get('progress') != dict(attempt=20004, checks=1, fixture_cursor=7):
                raise ValueError('Restart validation/fixture progress changed')
            updates = trial['updates']
            if len(updates) != POLICY['updates']:
                raise ValueError('Incomplete restart update sequence')
            schedule = []
            for i, update in enumerate(updates):
                if (any(type(update.get(k)) is not int for k in
                        ('attempt', 'fixture_cursor', 'skipped', 'lookahead_before', 'lookahead_after'))
                        or update.get('attempt') != POLICY['initial_attempt']+i
                        or update.get('fixture_cursor') != i or update.get('skipped') != 0
                        or not 0 <= update['lookahead_before'] < 6
                        or update['lookahead_after'] != (update['lookahead_before']+1) % 6
                        or update['scale_before'] != update['scale_after']):
                    raise ValueError('Restart update/counter/AMP schedule changed')
                if i and (update['lookahead_before'] != updates[i-1]['lookahead_after']
                          or update['scale_before'] != updates[i-1]['scale_after']):
                    raise ValueError('Restart counter/scale continuity changed')
                for key in ('loss', 'lr', 'scale_before', 'scale_after'):
                    if type(update.get(key)) not in (int, float) or not math.isfinite(update[key]):
                        raise ValueError('Nonfinite restart loss/schedule')
                if update['lr'] <= 0 or update['scale_before'] <= 0:
                    raise ValueError('Invalid restart LR/scale')
                schedule.append({k: v for k, v in update.items() if k != 'loss'})
            if not any(u['lookahead_after'] == 0 for u in updates):
                raise ValueError('Restart did not cross Lookahead boundary')
            if reference is None:
                reference = updates
                reference_schedule = schedule
            elif schedule != reference_schedule:
                raise ValueError('Repeat/restart update schedules differ')
            if name != 'reference':
                comparisons = trial.get('comparisons', [])
                if len(comparisons) != POLICY['updates']:
                    raise ValueError('Missing per-update full-state comparisons')
                for comparison in comparisons:
                    validate_comparison(comparison)
                for a, b in zip(reference, updates):
                    budget = min(POLICY['loss']['absolute_cap'],
                                 POLICY['loss']['atol']+POLICY['loss']['rtol']*abs(a['loss']))
                    if abs(a['loss']-b['loss']) > budget:
                        raise ValueError('Restart loss exceeds fixed bounds')
        schedules.append(reference_schedule)
    if schedules[0] != schedules[1]:
        raise ValueError('Rank restart schedules differ')

