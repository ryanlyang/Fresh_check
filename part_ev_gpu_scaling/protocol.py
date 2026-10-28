"""Frozen diagnostic identity and conservative scaling summaries."""
import math
from pathlib import Path

from part_ev_fullscale.protocol import digest, file_sha256, read_json, write_json
from part_ev_message_study.controls import checked_hash
from part_ev_state_compare import protocol as C

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(global_batch=512, seed=101, warmup_steps=20, measured_steps=4200,
              window_steps=1000, total_workers=2, torch_threads_total=4,
              fixed_warmup=10, fixed_steps=40, fixed_particles=128,
              max_seconds=14400, memory_fraction_limit=.90, max_skip_fraction=.05,
              gradient_clipping=None, world_sizes=[1, 2],
              distributed='single_node_DDP_one_process_per_GPU_SyncBatchNorm',
              trimming='native_per_rank_for_stream_disabled_for_fixed_workload',
              fixed_data='same_seeded_global_synthetic_512x128_batch_split_across_ranks',
              authorizes_training=False)


def resources(world):
    if type(world) is not int or world not in (1, 2):
        raise ValueError('Only one or two GPUs are supported')
    return dict(local_batch=POLICY['global_batch']//world,
                workers=POLICY['total_workers']//world,
                threads=POLICY['torch_threads_total']//world)


def source_manifest():
    files = list((ROOT/'part_ev_gpu_scaling').glob('*.py'))
    files += [ROOT/p for p in ('scripts/benchmark_part_ev_gpu_scaling.py',
                               'sbatch/submit_part_ev_gpu_scaling.sh',
                               'sbatch/run_part_ev_gpu_scaling.sh')]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(files)}


def prepare(output, campaign):
    output, campaign = Path(output).resolve(), Path(campaign).resolve(strict=True)
    if output == campaign or output.is_relative_to(campaign) or campaign.is_relative_to(output):
        raise ValueError('Use a separate diagnostic directory, not a production campaign')
    if (output/'plan.json').exists():
        raise FileExistsError('Diagnostic is immutable; choose a new output')
    parent = C.load_campaign(campaign)  # No approval required for the failed memory probe.
    C.verify_inventory_metadata(parent)
    task = parent['tasks'][0]
    if task['architecture'] != 'state_layerwise' or task['seed'] != 101:
        raise ValueError('Expected layerwise all-eight State-EV, seed 101')
    plan = dict(schema='part_ev_gpu_scaling_v1', policy=POLICY, task=task,
                recipe=parent['recipes'][task['recipe']], source=source_manifest(),
                parent=str(campaign), parent_sha256=parent['content_hash'],
                parent_file_sha256=file_sha256(campaign/'campaign.json'),
                environment=parent['environment'], authorizes_training=False)
    plan['content_hash'] = digest(plan)
    write_json(output/'plan.json', plan)
    return plan


def load(output, *, check_environment=True):
    plan = checked_hash(read_json(Path(output)/'plan.json'))
    if plan['policy'] != POLICY or plan['source'] != source_manifest():
        raise ValueError('Diagnostic policy/source changed')
    parent = C.load_campaign(Path(plan['parent']), check_environment=check_environment)
    if (parent['content_hash'] != plan['parent_sha256']
            or file_sha256(Path(plan['parent'])/'campaign.json') != plan['parent_file_sha256']
            or parent['tasks'][0] != plan['task']
            or parent['recipes'][plan['task']['recipe']] != plan['recipe']):
        raise ValueError('Parent campaign changed')
    return plan, parent


def comparison(one, two, section):
    a, b = one[section], two[section]
    for row in (a, b):
        if not math.isfinite(row['jets_per_second']) or row['jets_per_second'] <= 0:
            raise ValueError('Invalid throughput')
    speedup = b['jets_per_second']/a['jets_per_second']
    return dict(speedup_2_vs_1=speedup, walltime_reduction_percent=100*(1-1/speedup),
                gpu_hour_ratio_2_vs_1=2/speedup, parallel_efficiency=speedup/2)


def summarize(output, plan):
    rows, missing = {}, []
    for world in (1, 2):
        path = Path(output)/f'gpus_{world}.json'
        if not path.exists():
            missing.append(world)
            continue
        row = checked_hash(read_json(path))
        if row['plan_sha256'] != plan['content_hash'] or row['world_size'] != world:
            raise ValueError('Result belongs to another diagnostic')
        rows[world] = row
    warnings = ['Diagnostic only: no training authorization or accuracy claim.',
                'Stream ranks have disjoint file shards; order/trimming differ across GPU counts.',
                'Training-only projections exclude validation, checkpoints, restarts and queue delays.']
    complete = not missing and all(r.get('completed') is True for r in rows.values())
    comparable = complete and all(r.get('numerically_healthy') is True for r in rows.values())
    if comparable:
        identities = [[(g['name'], g['total_memory']) for g in r['gpus']] for r in rows.values()]
        comparable = len(set(sum(identities, []))) == 1
        if not comparable:
            warnings.append('Different GPU models/memory sizes: scaling comparison withheld.')
        if rows[1]['fixed_global_fingerprint'] != rows[2]['fixed_global_fingerprint']:
            raise ValueError('Fixed workloads differ')
    result = dict(complete=complete, comparable=comparable, missing=missing,
                  authorizes_training=False, warnings=warnings, models=[])
    for world, row in rows.items():
        if not row.get('completed'):
            result['models'].append(dict(gpus=world, error=row.get('error'), completed=False))
            continue
        result['models'].append(dict(gpus=world, local_batch=resources(world)['local_batch'],
            training_jets_per_second=row['stream']['jets_per_second'],
            fixed_128_jets_per_second=row['fixed']['jets_per_second'],
            peak_allocated_gib=max(g['peak_allocated'] for g in row['gpus'])/1024**3,
            peak_reserved_gib=max(g['peak_reserved'] for g in row['gpus'])/1024**3,
            memory_headroom_passed=row['memory_headroom_passed'],
            amp_skips=row['stream']['skips'], numerically_healthy=row['numerically_healthy'],
            training_only_hours=plan['recipe']['total_steps']*512/row['stream']['jets_per_second']/3600))
    if comparable:
        result['real_data_scaling'] = comparison(rows[1], rows[2], 'stream')
        result['fixed_128_scaling'] = comparison(rows[1], rows[2], 'fixed')
    return result
