"""Native-GH200 evidence gates. Never impersonate two-rank A100 evidence."""
import importlib
import math

from . import protocol as P


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_correctness(spec, rows):
    from part_ev_message_study.parity import validate_report
    kind = spec['model_kind']
    required = {'zero_identity', 'active_chunk', 'active_native_global_batch'}
    if kind != 'bias': required.add('unchanged_attention_and_ev')
    if kind == 'memory': required |= {'dense_recurrence_and_history', 'boundaries_masks_lifetime'}
    if len(rows) != 1: raise ValueError('Require one native-GPU correctness record')
    row = rows[0]
    if (row['rank'] != 0 or row['passed'] is not True or row['tolerances'] != spec['parity']
            or row['execution'] != dict(world_size=1, batchnorm='native', distributed=False)
            or len(row['checks']) != len(required) or {c['kind'] for c in row['checks']} != required
            or any(c['passed'] is not True for c in row['checks'])):
        raise ValueError('Missing/invalid native-GPU correctness evidence')
    validate_report(row['initialization'])
    if row['initialization']['device_type'] != 'cuda':
        raise ValueError('CPU parity does not admit a GH200 benchmark')
    if kind != 'bias':
        delayed = row['delayed_activation']
        flags = ('first_success_gate_gradients_zero', 'first_success_output_gradients_nonzero',
                 'all_gate_matrices_active', 'all_gate_gradients_nonzero') if kind == 'multiply' else (
                 'first_success_writer_gradients_zero', 'first_success_reader_gradients_nonzero',
                 'all_writers_and_gates_updated', 'all_memory_gradients_nonzero')
        if (delayed['passed'] is not True or any(delayed.get(k) is not True for k in flags)
                or delayed['successful_updates'] != 3 or not 3 <= delayed['attempts'] <= 20
                or delayed['skips'] != delayed['attempts']-3):
            raise ValueError('Delayed learning path missing/invalid')
    if kind == 'memory':
        history = next(c for c in row['checks'] if c['kind'] == 'dense_recurrence_and_history')
        boundary = next(c for c in row['checks'] if c['kind'] == 'boundaries_masks_lifetime')
        if (history.get('layers') != 8 or history.get('boundaries') != 7
                or history.get('final_reader_first_writer_gradient') is not True
                or history.get('earlier_observation_changes_later_read') is not True
                or boundary.get('boundaries') != 7 or boundary.get('memory_precision') != 'fp32'
                or boundary.get('overflow_rejected') is not True
                or boundary.get('padding_and_permutation') is not True):
            raise ValueError('Memory history/boundary safety gate missing')


def check_diagnostics(spec, samples):
    science = importlib.import_module(P.PACKAGES[spec['model_kind']]+'.protocol')
    if len(samples) != 1: raise ValueError('Missing native-GPU diagnostics')
    sample = samples[0]
    if (sample['events'] != 8 or sample['particles'] != 17 or sample['precision'] != 'fp32'
            or sample['dropout'] is not False or sample['optimizer_steps'] != 0
            or len(sample['layers']) != 8):
        raise ValueError('Incomplete fixed-sample diagnostics')
    for index, layer in enumerate(sample['layers']):
        if spec['model_kind'] == 'bias':
            for key in science.DIAGNOSTIC_HEAD_FIELDS:
                values = layer[key]
                if not isinstance(values, list) or len(values) != 8 or any(not nonnegative(v) for v in values):
                    raise ValueError('Invalid per-head bias diagnostics')
            if any(not nonnegative(layer[k]) for k in (
                    'new_to_ev_norm', 'correction_to_native_norm', 'state_output_parameter_norm')):
                raise ValueError('Invalid state diagnostics')
        else:
            for key in science.DIAGNOSTIC_FIELDS:
                value = layer[key]
                if not nonnegative(value) and not (key == 'gate_mean' and nonnegative(-value)):
                    raise ValueError('Invalid extension diagnostics')
            if spec['model_kind'] == 'multiply':
                if abs(layer['gate_mean']) > 1 or layer['gate_abs_max'] > 1 or layer['gate_saturation_fraction'] > 1:
                    raise ValueError('Modulation bounds violated')
            elif (layer['layer'] != index or any(layer[k] > 1 for k in (
                    'memory_rms', 'candidate_rms', 'write_mean', 'write_saturation_fraction'))):
                raise ValueError('Memory bounds/layer order violated')


def summarize(spec, row):
    from part_ev_message_study.protocol import precision_ok
    if row.get('campaign_sha256') != spec['content_hash']:
        raise ValueError('Benchmark belongs to another campaign')
    if row.get('complete') is not True or row.get('passed') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    if (row.get('authorizes_training') is not False or row['runtime'] != P.runtime()
            or row['parameters'] != spec['parameters'] or row['finite_state'] is not True
            or row['extension_active'] is not True):
        raise ValueError('Runtime, parameter count or finite/active state differs')
    for name, steps in (('stream', P.POLICY['measured']), ('fixed', P.POLICY['fixed_steps'])):
        result = row[name]
        if (result['measured_steps'] != steps or result['healthy'] is not True
                or not positive(result['seconds']) or len(result['ranks']) != 1
                or not 0 <= result['skips'] < steps*P.POLICY['max_skip_fraction']
                or not math.isclose(result['jets_per_second'], 512*steps/result['seconds'], rel_tol=1e-10)):
            raise ValueError('Incomplete timing/AMP evidence')
        rank = result['ranks'][0]
        windows = [min(P.POLICY['window'], steps-start) for start in range(0, steps, P.POLICY['window'])]
        if (not precision_ok(rank['precision'], 'torch.float16')
                or [w['steps'] for w in rank['windows']] != windows
                or any(not positive(w['seconds']) for w in rank['windows'])
                or rank['skips'] != result['skips'] or not positive(rank['scaler'])
                or not rank['parameters_sha256']):
            raise ValueError('Missing measured windows/precision')
    check_correctness(spec, row['correctness'])
    check_diagnostics(spec, row['diagnostics'])
    if row['restart'] != dict(passed=True, updates=7, ranks=1, full_state=True, rng_and_trimmers=True):
        raise ValueError('Full-state restart/RNG gate missing')
    if len(row['gpus']) != 1: raise ValueError('Require one GH200')
    gpu = row['gpus'][0]
    if (gpu['rank'] != 0 or 'GH200' not in gpu['name']
            or gpu['total_memory'] < P.POLICY['minimum_vram_bytes']
            or not 0 < gpu['peak_allocated'] < P.POLICY['memory_fraction']*gpu['total_memory']
            or not gpu['peak_allocated'] <= gpu['peak_reserved'] <= gpu['total_memory']
            or not nonnegative(gpu['cold_fetch_seconds'])):
        raise ValueError('One GH200 with 10% allocation headroom required')
    val, inference = row['validation'], row['inference']
    if (val['events'] != P.POLICY['validation_events'] or not positive(val['seconds'])
            or not precision_ok(val['precision'], 'torch.float32')
            or not positive(row['checkpoint_seconds'])):
        raise ValueError('FP32 validation/checkpoint timing missing')
    if (inference['batch_size'] != 256 or inference['particles'] != 128
            or inference['batches'] != 20 or inference['precision'] != 'fp32'
            or not positive(inference['seconds']) or inference['batch1_iterations'] != 100
            or not positive(inference['batch1_seconds'])):
        raise ValueError('FP32 inference/latency evidence missing')
    recipe = spec['recipe']
    train = max(w['seconds']/w['steps'] for w in row['stream']['ranks'][0]['windows'])*recipe['total_steps']
    validation = val['seconds']*recipe['epochs']*recipe['validation_events_per_epoch']/val['events']
    checkpoint = (recipe['total_steps']//1000+3*recipe['epochs']+2)*row['checkpoint_seconds']
    cold = gpu['cold_fetch_seconds']
    hours = math.ceil(P.POLICY['planning_margin']*(train+validation+checkpoint+cold)/3600)
    return dict(passed=True, complete=True, authorizes_training=False, model=spec['task'],
        parameters=spec['parameters'], gpu=gpu['name'], runtime=row['runtime'],
        train_jets_per_second=row['stream']['jets_per_second'],
        validation_end_to_end_jets_per_second=val['events']/val['seconds'],
        fixed128_inference_jets_per_second=256*20/inference['seconds'],
        fixed128_batch1_latency_ms=1000*inference['batch1_seconds']/100,
        peak_allocated_gib=gpu['peak_allocated']/2**30, peak_reserved_gib=gpu['peak_reserved']/2**30,
        projected_train_hours=train/3600, projected_validation_hours=validation/3600,
        projected_checkpoint_hours=checkpoint/3600, cold_fetch_seconds=cold,
        recommended_train_hours=hours,
        recommended_evaluate_hours=max(1, math.ceil(1.3*val['seconds']*5e6/val['events']/3600)),
        within_20_day_training_ceiling=hours <= 480,
        note='Diagnostic only. Full 1M-update/50-check projection +30% margin; queue, replay and '
             'future I/O contention excluded. Native single-GPU BatchNorm; not bitwise equivalent '
             'to SPORC DDP. No production authorized, submitted or cancelled.')
