"""Separate, frozen submission chains for an existing approved DDP campaign."""
from pathlib import Path

from part_ev_layerwise_ddp import protocol as P
from part_ev_message_study.controls import checked_hash

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(walltime_seconds=72*3600, shutdown_margin_seconds=3600,
              default_segments=4, max_segments=8, dependency='afterok',
              restart='latest_complete_training_state_not_best_weights',
              planned_stop='monotonic_deadline_verified_progress_and_handoff',
              genuine_failure='nonzero_exit_blocks_successors',
              replay='unchanged_exact_stream_replay_counted_in_segment_walltime',
              final_gate='require_full_step_and_validation_budget')


def source_manifest():
    rows = P.source_manifest()
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT/'part_ev_ddp_segments').glob('*.py')]
    files += ['scripts/run_part_ev_ddp_segments.py', 'sbatch/submit_part_ev_ddp_segments.sh',
              'sbatch/run_part_ev_ddp_segments.sh']
    rows.update({name: P.file_sha256(ROOT/name) for name in files})
    return rows


def snapshot(campaign, spec):
    """Small hash-bound cursor, not a new copy of a model or the dataset."""
    from part_ev_fullscale.checkpoint import trusted_load
    run = P.run_directory(campaign)
    last = run/'last.pt'
    if not last.exists():
        if (run/'training_complete.json').exists():
            raise ValueError('Training completion without a restart checkpoint')
        return dict(step=0, validated_checks=0, checkpoint_sha256=None, complete=False)
    saved = trusted_load(last)
    state = saved['state']
    recipe = spec['recipes'][P.TASK['recipe']]
    if (saved.get('schema') != P.SCHEMA or saved['campaign_sha256'] != spec['content_hash']
            or saved['runtime'] != P.expected_runtime(spec) or state['task'] != P.TASK
            or len(saved['rank_states']) != 2
            or not 0 <= state['step'] <= recipe['total_steps']
            or not 0 <= state['validated_checks'] <= state['step']//recipe['steps_per_epoch']
            or state['step']//recipe['steps_per_epoch']-state['validated_checks'] > 1):
        raise ValueError('Restart checkpoint identity/progress is invalid')
    if state['best_check'] is not None:
        relative = f"selected/check_{state['best_check']:02d}.pt"
        if state['best_file'] != relative or not (run/relative).is_file():
            raise ValueError('Committed best candidate is missing or inconsistent')
    complete = (run/'training_complete.json').exists()
    if complete:
        from part_ev_layerwise_ddp.engine import audit_training
        audit_training(campaign, spec)
        finished = P.read_json(run/'training_complete.json')
        if (state['step'] != recipe['total_steps'] or state['validated_checks'] != recipe['epochs']
                or any(state[key] != finished[key] for key in
                       ('step','validated_checks','best_check','best_accuracy','amp_skipped_steps'))):
            raise ValueError('Latest checkpoint does not match completed training')
    return dict(step=state['step'], validated_checks=state['validated_checks'],
                checkpoint_sha256=P.file_sha256(last), complete=complete)


def review(campaign, *, check_environment=True):
    campaign = Path(campaign).resolve(strict=True)
    spec = P.load(campaign, check_environment=check_environment)
    estimate = P.require_approval(campaign, spec)
    return spec, dict(campaign=str(campaign), task=spec['task'], policy=POLICY,
                     original_estimate=estimate, checkpoint=snapshot(campaign, spec),
                     note='No new GPU preflight. Existing passed DDP preflight is retained. '
                          'Restart data replay consumes allocation time; four slots are not a guarantee.')


def prepare(chain, campaign, segments):
    chain, campaign = Path(chain).resolve(), Path(campaign).resolve(strict=True)
    if type(segments) is not int or not 1 <= segments <= POLICY['max_segments']:
        raise ValueError('Choose 1..8 sequential segments')
    if chain.parent != campaign/'segment_chains':
        raise ValueError('Chain must be a new direct child of campaign/segment_chains')
    if (chain/'plan.json').exists():
        raise FileExistsError('Use a new immutable chain directory')
    spec, reviewed = review(campaign)
    # A crash after finalization but before a receipt may leave a fully complete
    # run. A fresh one-slot recovery chain can audit/no-op, then evaluate it.
    plan = dict(schema='part_ev_ddp_segments_v1', campaign=str(campaign),
                campaign_sha256=spec['content_hash'], policy=POLICY, segments=segments,
                source=source_manifest(), initial=reviewed['checkpoint'],
                approval_sha256=P.file_sha256(campaign/'production_approval.json'),
                preflight_sha256=P.file_sha256(campaign/'preflight/result.json'),
                evaluate_hours=reviewed['original_estimate']['evaluate_hours'])
    plan['content_hash'] = P.digest(plan)
    P.write_json(chain/'plan.json', plan)
    return plan


def load(chain, *, check_environment=True):
    chain = Path(chain).resolve(strict=True)
    plan = checked_hash(P.read_json(chain/'plan.json'))
    campaign = Path(plan['campaign']).resolve(strict=True)
    if (chain.parent != campaign/'segment_chains' or plan['schema'] != 'part_ev_ddp_segments_v1'
            or plan['policy'] != POLICY or plan['source'] != source_manifest()
            or type(plan['segments']) is not int or not 1 <= plan['segments'] <= POLICY['max_segments']):
        raise ValueError('Segment policy/source changed')
    spec = P.load(campaign, check_environment=check_environment)
    estimate = P.require_approval(campaign, spec)
    if (spec['content_hash'] != plan['campaign_sha256']
            or P.file_sha256(campaign/'production_approval.json') != plan['approval_sha256']
            or P.file_sha256(campaign/'preflight/result.json') != plan['preflight_sha256']
            or plan['evaluate_hours'] != estimate['evaluate_hours']):
        raise ValueError('Original campaign/preflight approval changed')
    return plan, spec


def receipt_path(chain, index):
    return Path(chain)/'receipts'/f'segment_{index:02d}.json'


def previous_snapshot(chain, plan, index):
    if type(index) is not int or not 0 <= index < plan['segments']:
        raise ValueError('Segment index outside frozen chain')
    if receipt_path(chain, index).exists():
        raise FileExistsError('Segment already committed; retries use a new chain')
    if index == 0:
        return plan['initial']
    row = checked_hash(P.read_json(receipt_path(chain, index-1)))
    if (row['chain_sha256'] != plan['content_hash'] or row['index'] != index-1
            or row['status'] not in ('paused', 'complete', 'already_complete')
            or row['after']['complete'] != (row['status'] != 'paused')):
        raise ValueError('Previous segment lacks a valid handoff')
    return row['after']


def completion_gate(chain, plan, spec):
    row = checked_hash(P.read_json(receipt_path(chain, plan['segments']-1)))
    current = snapshot(Path(plan['campaign']), spec)
    if (row['chain_sha256'] != plan['content_hash'] or row['index'] != plan['segments']-1
            or row['status'] not in ('complete','already_complete')
            or not current['complete'] or current != row['after']):
        raise ValueError('Full training is not complete. Latest progress is retained; '
                         'submit another chain after the current jobs are terminal.')
    return current
