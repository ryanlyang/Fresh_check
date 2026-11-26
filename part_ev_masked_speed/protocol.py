"""Independent diagnostic identities; never authorize or submit production."""
from pathlib import Path
import math
import shutil
from part_ev_masked_predictive import protocol as P
from .config import SCHEMA, VARIANTS, POLICY, REFERENCE_PREFIXES, REFERENCE_FILES


def donor(root, source):
    root, source = Path(root), Path(source)
    launch = P.load_launch(root)
    if launch['platform']!='tigris': raise ValueError('This speed study requires a Tigris donor')
    spec = P.load(root)  # Environment/data/evidence checks, but not the 480-hour production ceiling.
    review = P.review(root)
    if not review['passed']: raise ValueError('Passed original masked benchmark required')
    manifest = P.read(root/'source_manifest.json')
    required = {k:v for k,v in manifest.items() if (
        k.startswith(REFERENCE_PREFIXES) and k.endswith(('.py','.yaml','.yml'))) or k in REFERENCE_FILES}
    if 'part_ev_masked_predictive/step.py' not in required:
        raise ValueError('Incomplete reference source manifest')
    for name, expected in required.items():
        if not (source/name).is_file() or P.sha(source/name)!=expected:
            raise ValueError('Reference implementation differs from donor: '+name)
    identities = {n:P.sha(root/n) for n in ('launch.json','source_manifest.json','campaign.json',
                                          'data_contents.json','benchmark.json')}
    return launch, spec, review, identities


def load(output, *, worker=False):
    output = Path(output).resolve(strict=True); row = P.checked(output/'launch.json')
    if (row.get('schema')!=SCHEMA or row.get('variants')!=VARIANTS or row.get('policy')!=POLICY
            or row.get('output')!=str(output) or row.get('platform')!='tigris'
            or row.get('resources')!=P.PLATFORMS['tigris'] or row.get('authorizes_training') is not False
            or output.parent!=Path(row['repo'])/'checkpoints/part_ev_masked_speed'
            or row.get('source_sha256')!=P.verify_snapshot(output)):
        raise ValueError('Speed diagnostic snapshot/contract changed')
    if worker:
        P.verify_host(row)
        if Path(__file__).resolve().parents[1]!=output/'source':
            raise ValueError('Worker must use the new frozen diagnostic snapshot')
    return row


def check_storage(output):
    root = Path(output)
    used = P.used_bytes(root) if root.exists() else 0
    while not root.exists(): root = root.parent
    needed = POLICY['reserve_bytes']+max(0,POLICY['output_budget']-used)
    if shutil.disk_usage(root).free < needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB diagnostic headroom; nothing deleted')
    return needed


def validate_variant(item, gpu):
    """Recheck raw gates and timing coverage, not just a saved passed flag."""
    variant = next((v for v in VARIANTS if v['name']==item.get('name')), None)
    if variant is None or any(item.get(k)!=v for k,v in variant.items()):
        raise ValueError('Unknown/changed execution variant')
    if type(item.get('passed')) is not bool: raise ValueError('Missing variant status')
    if not item['passed']: return
    restart = item['restart']
    if (any(restart.get(k) is not True for k in ('passed','exact_roundtrip','full_state','lookahead_crossed'))
            or restart.get('updates')!=7
            or restart.get('schedule_boundaries')!=[19999,20000,20001,39999,40000,899999,900000]):
        raise ValueError('Incomplete restart evidence')
    if variant['optimized']:
        if [r.get('amp') for r in item['parity']]!=[False,True]: raise ValueError('Both precision parity gates required')
        for gate in item['parity']:
            if (any(gate.get(k) is not True for k in ('passed','exact_masks_rng_scaler','exact_bn_buffers',
                    'varied_lengths','full_optimizer_teacher')) or gate.get('batch')!=POLICY['parity_batch']
                    or gate.get('particles')!=POLICY['parity_particles'] or gate.get('gradients',0)<=0
                    or not 0<=gate.get('max_state_absolute',float('inf'))<=P.PARITY['restart_absolute']
                    or not math.isfinite(gate.get('max_gradient_absolute',float('nan')))):
                raise ValueError('Incomplete full-step parity evidence')
    if (item.get('finite_state') is not True or not math.isfinite(item['peak_gib'])
            or not 0<item['peak_gib']*2**30<POLICY['memory_fraction']*gpu['total_memory']):
        raise ValueError('Nonfinite state or insufficient device headroom')
    required = {'masked.clean','masked.teacher','masked.visible_predictor','masked.auxiliary_backward',
                'masked.optimizer','masked.ema'}
    if not required.issubset(item['stages']): raise ValueError('Incomplete phase profile')
    for name in required:
        if item['stages'][name].get('calls',0)<=0: raise ValueError('Unexecuted profile phase')
    for phase,n,windows,warmup in (('active',POLICY['measured'],POLICY['windows'],POLICY['warmup']),
            ('clean',POLICY['clean_steps'],1,POLICY['warmup']),('fixed128',POLICY['fixed_steps'],1,POLICY['warmup'])):
        timing = item[phase]; seconds = timing['seconds_per_attempt']; blocks = timing['windows']
        if (timing.get('steps')!=n or timing.get('warmup')!=warmup or len(blocks)!=windows
                or not 0<=timing.get('amp_skips',-1)<=POLICY['skip_fraction']*n
                or not math.isfinite(seconds) or seconds<=0 or timing.get('rng_isolated') is not True
                or len(timing.get('stream_sha256',''))!=64): raise ValueError('Invalid diagnostic timing')
        if any(b['steps']!=n//windows or not math.isfinite(b['seconds']) or b['seconds']<=0 for b in blocks):
            raise ValueError('Incomplete timing windows')
        if not math.isclose(seconds,max(b['seconds']/b['steps'] for b in blocks),rel_tol=1e-12):
            raise ValueError('Timing projection differs from raw windows')
        if not math.isclose(timing['total_seconds'],sum(b['seconds'] for b in blocks),rel_tol=1e-12):
            raise ValueError('Total differs from raw timing windows')


def summarize(output):
    output = Path(output); launch = load(output)
    path = output/'results.json'
    if not path.is_file():
        return dict(passed=False,complete=False,authorizes_training=False,missing=['results.json'])
    row = P.checked(path)
    if (row.get('schema')!=SCHEMA or row.get('launch_sha256')!=launch['content_hash']
            or row.get('authorizes_training') is not False):
        raise ValueError('Wrong diagnostic results')
    if row.get('donor_files')!=launch['donor_files']:
        raise ValueError('Donor evidence changed')
    for name,expected in launch['donor_files'].items():
        if P.sha(Path(launch['reference'])/name)!=expected: raise ValueError('Donor evidence changed')
    if (row.get('runtime_gpu')!=launch['reference_runtime_gpu']
            or row.get('reference_masked')!=launch['reference_masked']):
        raise ValueError('Donor projection/hardware changed')
    records = row['variants']; baseline = next((r for r in records if r['name']=='baseline32' and r['passed']),None)
    if len({r['name'] for r in records})!=len(records): raise ValueError('Duplicate execution variants')
    summaries = []
    for item in records:
        validate_variant(item,row['runtime_gpu'])
        if not item['passed']:
            summaries.append(dict(name=item['name'],passed=False,error=item.get('error'))); continue
        if baseline is not None and any(item[p]['stream_sha256']!=baseline[p]['stream_sha256'] for p in ('active','clean')):
            raise ValueError('Not a matched real-data stream')
        hours = (879999*item['active']['seconds_per_attempt']+120001*item['clean']['seconds_per_attempt'])/3600
        # Deliberately retain the historical validation/I/O costs, labeled below.
        other = row['reference_masked']['validation_hours']+row['reference_masked']['write_hours']
        requested = math.ceil(POLICY['margin']*(hours+other))
        summaries.append(dict(name=item['name'],passed=True,
            active_speedup_vs_baseline=None if baseline is None else baseline['active']['seconds_per_attempt']/item['active']['seconds_per_attempt'],
            active_seconds=item['active']['seconds_per_attempt'],peak_gib=item['peak_gib'],
            provisional_train_hours=requested,within_480_hours=requested<=480))
    complete = row.get('complete') is True and [r['name'] for r in records]==[v['name'] for v in VARIANTS]
    faster = [s for s in summaries if s['passed'] and s['name']!='baseline32'
              and s['active_speedup_vs_baseline'] is not None and s['active_speedup_vs_baseline']>1]
    return dict(complete=complete,passed=complete and baseline is not None and any(
        s['passed'] and s['name']!='baseline32' for s in summaries),authorizes_training=False,models=summaries,
        fastest_validated_candidate=None if not faster else min(faster,key=lambda s:s['provisional_train_hours'])['name'],
        note='Diagnostic only. Fresh disposable weights; no accuracy claim. Short matched timing windows; '
             'projections reuse historical validation/I/O and exclude queue/replay. '
             'A winning path needs full production admission, not automatic dispatch.')
