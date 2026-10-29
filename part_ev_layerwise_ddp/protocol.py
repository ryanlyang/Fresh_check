"""Admit only the measured two-A100 runtime; never amend old approvals."""
import copy
import math
from pathlib import Path

from part_ev_gpu_scaling import protocol as G
from part_ev_fullscale.protocol import digest, file_sha256, file_lock, read_json, write_json
from part_ev_message_study.controls import checked_hash

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'part_ev_layerwise_ddp_v1'
TASK = copy.deepcopy(G.C.TASKS[0])
POLICY = dict(world_size=2, local_batch=256, global_batch=512, train_workers_per_rank=1,
              validation_workers=2, validation_batch=512, batchnorm='SyncBatchNorm',
              checkpoint_interval=1000, heartbeat_batches=100, preflight_updates=24,
              preflight_validation_events=128000, planning_margin=1.30,
              restart='rank_rng_trimmers_and_exact_stream_replay',
              selection='first_highest_accuracy_of_50_checks',
              evaluation='single_GPU_FP32_official_validation_5M',
              historical_controls='not_runtime_matched_no_automatic_contrasts')


def source_manifest():
    rows = G.C.source_manifest(ROOT)
    rows.update(G.source_manifest())
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'part_ev_layerwise_ddp').glob('*.py')]
    names += ['scripts/run_part_ev_layerwise_ddp.py', 'sbatch/submit_part_ev_layerwise_ddp.sh',
              'sbatch/run_part_ev_layerwise_ddp.sh']
    rows.update({p: file_sha256(ROOT/p) for p in names})
    return rows


def validate_scaling(row, plan):
    if (row.get('completed') is not True or row.get('numerically_healthy') is not True
            or row.get('memory_headroom_passed') is not True or row.get('world_size') != 2
            or row.get('plan_sha256') != plan['content_hash']):
        raise ValueError('Need the completed, healthy two-GPU memory/throughput result')
    for section, count in (('stream',4200), ('fixed',40)):
        part = row[section]
        if (part['measured_steps'] != count or part['healthy'] is not True
                or not 0 <= part['skips'] < count*.05
                or not math.isfinite(part['seconds']) or part['seconds'] <= 0
                or not math.isclose(part['jets_per_second'],512*count/part['seconds'],rel_tol=1e-10)
                or len(part['ranks']) != 2):
            raise ValueError('Incomplete scaling measurement')
        identities = set()
        for rank in part['ranks']:
            if (sum(w['steps'] for w in rank['windows']) != count
                    or any(w['steps'] <= 0 or not math.isfinite(w['seconds']) or w['seconds'] <= 0 for w in rank['windows'])
                    or rank['precision']['passed'] is not True
                    or rank['precision']['expected_activation_dtype'] != 'torch.float16'):
                raise ValueError('Missing timing/AMP evidence')
            identities.add((rank['parameters_sha256'],rank['scaler'],rank['skips'],rank['lookahead_counter']))
        if len(identities) != 1:
            raise ValueError('Distributed replicas diverged in scaling check')
    if len(row['parity']) != 2 or any(p['passed'] is not True or p['nonzero_state'] is not True for p in row['parity']):
        raise ValueError('Missing nonzero-state distributed parity')
    if (len(row['gpus']) != 2 or {g['rank'] for g in row['gpus']} != {0,1}
            or len({(g['name'],g['total_memory']) for g in row['gpus']}) != 1
            or any(not 0 < g['peak_allocated'] < .9*g['total_memory'] for g in row['gpus'])):
        raise ValueError('Two matching GPUs with memory headroom required')
    runtime = row['runtime']
    if (runtime['global_batch'],runtime['local_batch'],runtime['workers_per_rank'],runtime['batchnorm'],runtime['tf32']) != (512,256,1,'SyncBatchNorm',False):
        raise ValueError('Scaling runtime differs from production')
    return dict(name=row['gpus'][0]['name'], total_memory=row['gpus'][0]['total_memory'],
                nccl_environment=runtime['nccl_environment'])


def prepare(output, scaling):
    output, scaling = Path(output).resolve(), Path(scaling).resolve(strict=True)
    plan, parent = G.load(scaling)
    for forbidden in (scaling, Path(plan['parent'])):
        if output == forbidden or output.is_relative_to(forbidden) or forbidden.is_relative_to(output):
            raise ValueError('Use a new, separate production directory')
    if (output/'campaign.json').exists():
        raise FileExistsError('Campaign already prepared; use review/train modes')
    G.C.verify_inventory_metadata(parent)
    row = checked_hash(read_json(scaling/'gpus_2.json'))
    expected_gpu = validate_scaling(row, plan)
    spec = dict(schema=SCHEMA, task=TASK, tasks=[TASK], policy=POLICY,
                recipes={TASK['recipe']: plan['recipe']}, parameters={'state_layerwise':2327462},
                source=source_manifest(), environment=parent['environment'], inventory=parent['inventory'],
                scaling=str(scaling), scaling_sha256=file_sha256(scaling/'gpus_2.json'),
                scaling_plan_sha256=file_sha256(scaling/'plan.json'), expected_gpu=expected_gpu)
    spec['content_hash'] = digest(spec)
    write_json(output/'campaign.json', spec)
    return spec


def load(output, *, check_environment=True):
    spec = checked_hash(read_json(Path(output)/'campaign.json'))
    if (spec['schema'] != SCHEMA or spec['policy'] != POLICY or spec['task'] != TASK
            or spec['tasks'] != [TASK] or spec['source'] != source_manifest()
            or spec['parameters'] != {'state_layerwise':2327462}):
        raise ValueError('Frozen production contract changed')
    scaling = Path(spec['scaling'])
    plan, parent = G.load(scaling,check_environment=check_environment)
    row = checked_hash(read_json(scaling/'gpus_2.json'))
    if (file_sha256(scaling/'gpus_2.json') != spec['scaling_sha256']
            or file_sha256(scaling/'plan.json') != spec['scaling_plan_sha256']
            or spec['expected_gpu'] != validate_scaling(row,plan)
            or spec['inventory'] != parent['inventory']
            or spec['environment'] != parent['environment']
            or spec['recipes'] != {TASK['recipe']:plan['recipe']}):
        raise ValueError('Scaling evidence or dataset contract changed')
    return spec


def run_directory(output):
    return Path(output)/'runs'/TASK['run_id']/f"seed_{TASK['seed']}"


def expected_runtime(spec):
    expected = spec['expected_gpu']
    return dict(campaign_sha256=spec['content_hash'], world_size=2, local_batch_size=256,
                workers_per_rank=1, batchnorm='SyncBatchNorm', gpu=expected['name'],
                gpu_memory_bytes=expected['total_memory'], tf32=False, cudnn_benchmark=False,
                nccl_environment=expected['nccl_environment'])


def recommendation(spec, preflight, scaling):
    recipe = spec['recipes'][TASK['recipe']]
    per_step = max(w['seconds']/w['steps'] for r in scaling['stream']['ranks'] for w in r['windows'])
    val = preflight['validation']
    if (val['events'] != POLICY['preflight_validation_events'] or not math.isfinite(val['seconds']) or val['seconds'] <= 0
            or not math.isfinite(preflight['checkpoint_seconds']) or preflight['checkpoint_seconds'] <= 0):
        raise ValueError('Invalid validation/checkpoint timing')
    train = per_step*recipe['total_steps']
    validation = val['seconds']*recipe['epochs']*recipe['validation_events_per_epoch']/val['events']
    writes = recipe['total_steps']//POLICY['checkpoint_interval']+3*recipe['epochs']+2
    total = train+validation+writes*preflight['checkpoint_seconds']
    return dict(train_hours=math.ceil(POLICY['planning_margin']*total/3600),
                evaluate_hours=max(5,math.ceil(POLICY['planning_margin']*val['seconds']*5_000_000/val['events']/3600)),
                projected_train_compute_hours=train/3600, projected_validation_hours=validation/3600,
                note='Includes 50 validations and checkpoint writes, plus 30% margin; excludes queue/replay.')


def require_preflight(output, spec):
    result = checked_hash(read_json(Path(output)/'preflight/result.json'))
    if (result.get('passed') is not True or result['campaign_sha256'] != spec['content_hash']
            or result['runtime'] != expected_runtime(spec)
            or result['restart'] != dict(passed=True,updates_checked=7,rank_count=2)
            or result['validation']['precision']['passed'] is not True
            or result['validation']['precision']['expected_activation_dtype'] != 'torch.float32'
            or len(result['peak_bytes']) != 2
            or any(not 0 < n < .9*spec['expected_gpu']['total_memory'] for n in result['peak_bytes'])):
        raise ValueError('Production restart/FP32-validation preflight must pass')
    estimate = recommendation(spec,result,read_json(Path(spec['scaling'])/'gpus_2.json'))
    return result, estimate


def approve(output, spec, reviewed):
    if not reviewed:
        raise ValueError('Review preflight, then pass --reviewed')
    _, estimate = require_preflight(output,spec)
    row = dict(campaign_sha256=spec['content_hash'],preflight_sha256=file_sha256(Path(output)/'preflight/result.json'))
    path = Path(output)/'production_approval.json'
    if path.exists() and read_json(path) != row:
        raise ValueError('Approved evidence changed')
    if not path.exists(): write_json(path,row)
    return estimate


def require_approval(output, spec):
    row = read_json(Path(output)/'production_approval.json')
    expected = dict(campaign_sha256=spec['content_hash'],preflight_sha256=file_sha256(Path(output)/'preflight/result.json'))
    if row != expected: raise ValueError('Production approval changed')
    return require_preflight(output,spec)[1]
