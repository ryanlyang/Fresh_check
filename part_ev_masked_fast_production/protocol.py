"""Candidate-only admission bound to the original benchmark and speed study."""
from pathlib import Path
import math
import shutil
from part_ev_masked_predictive import protocol as O, config as C
from part_ev_masked_speed import protocol as D
from .config import SCHEMA, RESOURCES, SCIENCE, TASK, POLICY, VARIANT, execution_hash

read, write, sha, hashed, checked, digest = O.read, O.write, O.sha, O.hashed, O.checked, O.digest


def task(arm):
    if arm!='masked': raise ValueError('Candidate only; no control training')
    return dict(TASK)


def run_directory(output, arm='masked'):
    return Path(output)/'production/runs'/task(arm)['run_id']/'seed_101'


def inspect_speed(speed, source):
    speed, source = Path(speed), Path(source)
    launch = D.load(speed); summary = D.summarize(speed)
    if not summary['passed']: raise ValueError('Speed study did not pass')
    choice = next(m for m in summary['models'] if m['name']=='fast256')
    if choice['passed'] is not True or choice['within_480_hours'] is not True:
        raise ValueError('Passed fast256 under 480 hours required')
    original, spec, review, files = D.donor(launch['reference'],source)
    manifest = read(speed/'source_manifest.json')
    # The very same measured kernel is imported by production, not edited/copied.
    for name in ('part_ev_masked_speed/step.py','part_ev_masked_speed/verification.py',
                 'part_ev_masked_speed/protocol.py','part_ev_masked_speed/config.py'):
        if sha(source/name)!=manifest[name]: raise ValueError('Speed-study implementation changed: '+name)
    identities = {n:sha(speed/n) for n in ('launch.json','source_manifest.json','results.json')}
    return dict(speed=str(speed),speed_files=identities,reference=launch['reference'],donor_files=files,
        original_launch=original,choice=choice,gpu=review['runtime']['gpu'],environment=spec['environment'],
        gradient_count=len(spec['training_inventories']['masked']))


def load(output, *, worker=False, environment=False):
    output = Path(output).resolve(strict=True); row = checked(output/'launch.json')
    if (row.get('schema')!=SCHEMA or row.get('execution_sha256')!=execution_hash()
            or row.get('output')!=str(output) or row.get('platform')!='tigris'
            or row.get('resources')!=RESOURCES or row.get('science')!=SCIENCE or row.get('task')!=TASK
            or row.get('policy')!=POLICY or row.get('source_sha256')!=O.verify_snapshot(output)
            or output.parent!=Path(row['repo'])/'checkpoints/part_ev_masked_fast_production'
            or type(row.get('auto_production')) is not bool
            or not 1<=row.get('max_train_hours',0)<=480 or not 1<=row.get('max_evaluate_hours',0)<=24):
        raise ValueError('Fast256 frozen contract changed')
    for root, files in ((row['speed'],row['speed_files']), (row['reference'],row['donor_files'])):
        for name, expected in files.items():
            if sha(Path(root)/name)!=expected: raise ValueError('Referenced evidence changed: '+str(Path(root)/name))
    if environment:
        if O.runtime_environment()!=row['environment']: raise ValueError('Use the admitted unchanged environment')
        O.load(row['reference'])  # Data/config identities; no old production ceiling.
    if worker:
        O.verify_host(row)
        if Path(__file__).resolve().parents[1]!=output/'source': raise ValueError('Frozen worker source required')
    return row


def check_storage(output):
    output = Path(output); root = output
    while not root.exists(): root = root.parent
    needed = POLICY['reserve_bytes']+max(0,POLICY['budget_bytes']-O.used_bytes(output))
    if shutil.disk_usage(root).free<needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB local headroom including 4 GiB reserve; nothing deleted')
    return needed


def validate_evidence(launch, row):
    from part_ev_masked_predictive.evidence import timing, precision_ok, positive, restart
    if (row.get('schema')!=SCHEMA or row.get('launch_sha256')!=launch['content_hash']
            or row.get('complete') is not True or row.get('authorizes_training') is not False
            or row.get('gpu')!=launch['gpu'] or row.get('policy')!=POLICY):
        raise ValueError('Incomplete or wrong fast256 preflight')
    gates = row['parity']
    if len(gates)!=4 or [(g['repeat'],g['amp']) for g in gates]!=[(i,a) for i in range(2) for a in (False,True)]:
        raise ValueError('Both precisions and two independent repetitions required')
    for g in gates:
        if (any(g.get(k) is not True for k in ('passed','exact_masks_rng_scaler','exact_bn_buffers',
                                              'full_optimizer_teacher','varied_lengths'))
                or g['microbatch']!=256 or g['batch']!=512 or g['particles']!=97 or g['gradients']!=launch['gradient_count']
                or not math.isfinite(g['max_gradient_absolute']) or g['max_gradient_absolute']<0
                or not 0<=g['max_state_absolute']<=C.PARITY['restart_absolute']):
            raise ValueError('Production full-step parity failed')
    if row.get('legacy_fast32',{}).get('collected') is not True:
        raise ValueError('Missing named fast32 diagnostic evidence')
    if [t.get('amp') for t in row['legacy_fast32'].get('trials',[])]!=[False,True]:
        raise ValueError('Incomplete named fast32 diagnostic')
    for name, transitions in (('restart',False),('transitions',True)):
        restart(row[name],'masked',transitions)
    replay = row['real_stream_resume']
    if (any(replay.get(k) is not True for k in ('passed','real_stream','full_state','exact_roundtrip',
            'next_masks_equal','next_inputs_equal','optimizer_teacher_equal','rng_restored'))
            or replay.get('updates')!=7 or replay.get('train_replayed')!=3 or replay.get('val_replayed')!=2
            or replay.get('validation_boundaries')!=2 or len(replay.get('input_sha256',[]))!=7):
        raise ValueError('Actual auxiliary ROOT-stream restart not verified')
    if (any(row['production_checkpoint'].get(k) is not True for k in ('passed','exact_roundtrip','production_serializer'))
            or row['production_checkpoint'].get('complete_attempts')!=3):
        raise ValueError('Production checkpoint transaction not verified')
    def phase(name, kind, n, windows):
        evidence = row[name]
        if evidence['microbatch']!=256: raise ValueError('Wrong measured auxiliary batch')
        # Only schema metadata is normalized for the original shared validator;
        # execution is checked above and is bound to the immutable 256 kernel.
        return timing(dict(evidence,microbatch=32),1,'masked',kind,n,windows)
    active = phase('stream','active',4200,POLICY['windows'])
    clean = phase('clean','clean',200,[200]); phase('fixed','active',40,[40])
    val = row['validation']; io = row['checkpoint']
    if val['events']!=128000 or not positive(val['seconds']) or not precision_ok(val['precision'],'torch.float32'):
        raise ValueError('FP32 validation incomplete')
    if row['finite_state'] is not True or not 0<row['peak_allocated']<.9*row['gpu']['total_memory']:
        raise ValueError('Nonfinite state/device headroom failure')
    if not all(positive(io[k]) for k in ('restart_seconds','transfer_seconds','load_seconds','weights_seconds',
                                       'prediction_write_seconds','prediction_load_seconds')):
        raise ValueError('Incomplete checkpoint/prediction I/O timing')
    if io['student_only'] is not True or io['full_state_includes_transfer'] is not True or io['prediction_events']!=100000:
        raise ValueError('Incorrect I/O scope')
    for name, batch, steps in (('throughput',256,20),('latency',1,100),('full_validation',512,3)):
        item=row['inference'][name]
        if (item['batch']!=batch or item['steps']!=steps or item['particles']!=128 or not positive(item['seconds'])
                or not precision_ok(item['precision'],'torch.float32')): raise ValueError('Student-only export/inference failed')
    if not positive(row['metric_5m_seconds']) or not positive(row['cold_fetch_seconds']):
        raise ValueError('Missing metric/startup timings')
    compute = 879999*active+120001*clean
    validation = 500*val['seconds']; writes = 1102*io['restart_seconds']+51*io['weights_seconds']
    measured_train = math.ceil(1.3*(compute+validation+writes+row['cold_fetch_seconds'])/3600)
    evaluate = math.ceil(1.3*(5e6/128000*val['seconds']+row['metric_5m_seconds']+io['load_seconds']+
                             50*(io['prediction_write_seconds']+io['prediction_load_seconds']))/3600)
    return dict(passed=True,complete=True,authorizes_training=False,model=TASK['run_id'],
        train_hours=max(launch['choice']['provisional_train_hours'],measured_train),evaluate_hours=max(2,evaluate),
        compute_hours=compute/3600,validation_hours=validation/3600,checkpoint_hours=writes/3600,
        active_seconds_per_attempt=active,clean_seconds_per_attempt=clean,peak_gib=row['peak_allocated']/2**30,
        legacy_fast32=row['legacy_fast32'],note='Candidate-only; fresh seed 101, 1M attempts/50 checks. '
        '+30% margin and prior speed-study walltime floor; excludes queue/replay. Explicit dispatch required.')


def review(output):
    launch=load(output);path=Path(output)/'preflight.json'
    if not path.exists(): return dict(passed=False,complete=False,authorizes_training=False,missing=['preflight.json'])
    return validate_evidence(launch,checked(path))


def production_spec(output):
    output=Path(output);launch=load(output,environment=True);reviewed=review(output)
    if (not reviewed['passed'] or reviewed['train_hours']>launch['max_train_hours']
            or reviewed['evaluate_hours']>launch['max_evaluate_hours']):
        raise ValueError('Preflight not admitted within reviewed walltime ceilings; no training submitted')
    original=O.load(launch['reference']);old=O.checked(Path(launch['reference'])/'benchmark.json')
    runtime=dict(old['runtime'],resources=RESOURCES,execution_sha256=execution_hash(),gpu=launch['gpu'])
    return hashed(dict(schema=SCHEMA,launch_sha256=launch['content_hash'],execution_sha256=execution_hash(),
        benchmark_sha256=sha(output/'preflight.json'),speed_results_sha256=launch['speed_files']['results.json'],
        source_sha256=launch['source_sha256'],models=[dict(reviewed,arm='masked')],tasks=[TASK],
        parameters={'masked':C.COUNTS['masked']},trainable_parameters={'masked':C.TRAINABLE['masked']},teacher_parameters=C.TEACHER,
        recipes={'ranger_full_v1':C.recipe()},inventory=original['inventory'],data_config_sha256=original['data_config_sha256'],
        data_contents_sha256=original['data_contents_sha256'],environment=launch['environment'],science=SCIENCE,
        platform='tigris',resources=RESOURCES,runtime=runtime,authorizes_training=True))


def load_production(output):
    expected=production_spec(output)
    if any(checked(Path(output)/n)!=expected for n in ('production.json','production/campaign.json')):
        raise ValueError('Production admission changed')
    return expected
