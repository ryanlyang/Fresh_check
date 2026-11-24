"""Immutable scientific identities; local headroom gate, aggregate advisory."""
import copy
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig
from part_ev_fullscale.protocol import digest, file_sha256 as sha, read_json as read, write_json as write, file_lock
from .config import SCHEMA, SCIENCE, COUNTS, TRAINABLE, TEACHER, TASKS, PLATFORMS, POLICY, PARITY, STORAGE_POLICY, recipe, science_hash, campaign_budget


def hashed(value):
    if 'content_hash' in value: raise ValueError('Already hashed')
    return dict(value, content_hash=digest(value))


def checked(path):
    row = read(path)
    if row.get('content_hash') != digest({k:v for k,v in row.items() if k!='content_hash'}):
        raise ValueError('Artifact checksum mismatch: '+str(path))
    return row


def inventory(source):
    source = Path(source)
    result={}
    for path in source.rglob('*'):
        if path.is_symlink(): raise ValueError('Source symlinks forbidden')
        if path.is_file(): result[path.relative_to(source).as_posix()] = sha(path)
    if not result: raise ValueError('Empty frozen source')
    return result


def verify_snapshot(output):
    output=Path(output)
    expected=read(output/'source_manifest.json')
    if inventory(output/'source') != expected: raise ValueError('Frozen source inventory/content changed')
    return digest(expected)


def runtime_environment():
    from part_ev_fullscale.protocol import environment
    row=environment()
    roots=sorted({sysconfig.get_path(k) for k in ('purelib','platlib')})
    row['installed_distributions']=dict(sorted((d.metadata['Name'].lower(),d.version)
        for d in importlib.metadata.distributions(path=roots) if d.metadata['Name']))
    return row


def verify_host(launch):
    p=PLATFORMS[launch['platform']]
    if platform.system()!='Linux' or platform.machine()!=p['machine']:
        raise ValueError('Use the requested Linux submission platform: '+launch['platform'])
    if Path(sys.executable).resolve()!=Path(launch['python']).resolve():
        raise ValueError('Use the frozen platform interpreter')


def load_launch(output, *, worker=False):
    output=Path(output).resolve(strict=True)
    row=checked(output/'launch.json')
    if (row.get('schema')!=SCHEMA or row.get('science_sha256')!=science_hash()
            or row.get('output')!=str(output) or row.get('platform') not in PLATFORMS
            or row.get('resources')!=PLATFORMS[row['platform']]
            or row.get('tasks')!=TASKS or row.get('policy')!=POLICY
            or row.get('source_sha256')!=verify_snapshot(output)
            or output.parent!=Path(row['repo'])/'checkpoints/part_ev_masked_predictive'
            or type(row.get('auto_production')) is not bool
            or row.get('storage_policy',STORAGE_POLICY)!=STORAGE_POLICY
            or not 1<=row['max_train_hours']<=480 or not 1<=row['max_evaluate_hours']<=24):
        raise ValueError('MaskedPredictive launch/location/science contract changed')
    if worker:
        verify_host(row)
        if Path(__file__).resolve().parents[1]!=output/'source':
            raise ValueError('Worker imported live checkout, not frozen campaign source')
    return row


def verify_data(spec):
    from part_ev_message_study.protocol import verify_inventory_metadata
    from part_ev_fullscale.data import DATA_CONFIG
    if spec['data_config_sha256']!=sha(DATA_CONFIG): raise ValueError('Preprocessing YAML changed')
    if set(spec['inventory']['splits'])!={'train','val'}: raise ValueError('No test split permitted')
    verify_inventory_metadata(spec)


def prepare(output):
    from part_ev_fullscale.data import inspect_dataset, DATA_CONFIG
    from part_ev_message_study.controls import verify_data_contents
    from .model import build_model, Trainable, inventory as model_inventory
    from .teacher import Teacher
    from .views import transforms
    launch=load_launch(output,worker=True);output=Path(output)
    if (output/'campaign.json').exists(): raise FileExistsError('Already prepared; use a fresh campaign')
    check_storage(launch)
    env=runtime_environment()
    if env!=launch['environment']: raise ValueError('Interpreter environment changed since submission')
    if env.get('torch_cuda') is None:raise ValueError('CUDA-enabled preparation interpreter required')
    transforms()
    data=inspect_dataset(launch['data'],include_test=False,check_entries=True)
    contents=verify_data_contents(data)  # Actual bytes, not merely names/mtime.
    inventories={};training_inventories={};teacher_inventory=None
    import torch
    for arm in COUNTS:
        torch.manual_seed(101); model=build_model(arm)
        inventories[arm]=model_inventory(model)
        del model
        torch.manual_seed(101);bundle=Trainable(arm)
        training_inventories[arm]=model_inventory(bundle)
        if arm=='masked':teacher_inventory=model_inventory(Teacher(bundle.student).model)
        del bundle
    spec=hashed(dict(schema=SCHEMA,launch_sha256=launch['content_hash'],science=SCIENCE,
        science_sha256=science_hash(),tasks=TASKS,parameters=COUNTS,policy=POLICY,parity=PARITY,
        recipes={'ranger_full_v1':recipe()},environment=env,source_sha256=launch['source_sha256'],
        inventory=data,data_config_sha256=sha(DATA_CONFIG),data_contents_sha256=digest(contents),
        model_inventories=inventories,training_inventories=training_inventories,teacher_inventory=teacher_inventory,
        trainable_parameters=TRAINABLE,teacher_parameters=TEACHER,authorizes_training=False))
    write(output/'data_contents.json',contents);write(output/'campaign.json',spec)
    return spec


def load(output, *, environment=True, worker=False):
    output=Path(output);launch=load_launch(output,worker=worker)
    row=checked(output/'campaign.json')
    fixed=dict(schema=SCHEMA,science=SCIENCE,science_sha256=science_hash(),tasks=TASKS,
        parameters=COUNTS,trainable_parameters=TRAINABLE,teacher_parameters=TEACHER,policy=POLICY,parity=PARITY,recipes={'ranger_full_v1':recipe()},
        source_sha256=launch['source_sha256'],launch_sha256=launch['content_hash'],authorizes_training=False)
    if any(row.get(k)!=v for k,v in fixed.items()): raise ValueError('MaskedPredictive campaign contract changed')
    if row['data_contents_sha256']!=digest(read(output/'data_contents.json')):
        raise ValueError('Data content evidence changed')
    verify_data(row)
    if environment and row['environment']!=runtime_environment():
        raise ValueError('Use the unchanged preparation environment')
    return row


def used_bytes(root):
    total=0
    for p in Path(root).rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_size
        except FileNotFoundError:
            # A sibling arm may have just committed/replaced an atomic file.
            # Not subtracting that vanished file is conservative for headroom.
            continue
    return total


def required_storage(repo, output, requested):
    # This existing helper accounts for registered RTS/extension/SPORC budgets;
    # never mutate its protocol globals or write into those campaigns.
    from part_ev_multiscale.protocol import required_storage as shared
    repo,output=Path(repo),Path(output)
    required=shared(repo,output,0)
    for path in (repo/'checkpoints/part_ev_masked_predictive').glob('*/launch.json'):
        if path.parent==output: continue
        other=checked(path)
        report=path.parent/'production/reports/masked_report.json'
        if report.exists():
            done=checked(report)
            if done.get('launch_sha256')!=other['content_hash'] or done.get('complete') is not True:
                raise ValueError('Unrecognized completed reservation')
            continue
        reserved=other['reservation_bytes']
        if (path.parent/'production.json').exists(): reserved=campaign_budget()
        required+=max(0,reserved-used_bytes(path.parent))
    return required+max(0,requested-used_bytes(output))


def check_storage(launch, *, production=False):
    """Block only on this campaign's remaining budget plus the 4-GiB reserve.

    Other registered campaigns' speculative future writes are advisory, as
    explicitly requested by the user. Failed/stale/corrupt foreign reservations
    must not block an otherwise adequately provisioned masked campaign.
    """
    budget=campaign_budget() if production or launch['auto_production'] else POLICY['benchmark_bytes']
    needed=POLICY['reserve_bytes']+max(0,budget-used_bytes(launch['output']))
    path=Path(launch['output'])
    while not path.exists(): path=path.parent
    available=shutil.disk_usage(path).free
    if available<needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB for this campaign\'s remaining output budget '
                      f'plus the shared 4-GiB reserve; available {available/2**30:.2f} GiB. '
                      'Other campaigns are not a hard gate; nothing deleted')
    try:
        aggregate=required_storage(launch['repo'],launch['output'],budget)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'Storage advisory: other-campaign reservations could not be calculated '
              f'({type(exc).__name__}: {exc}). Proceeding: this campaign requires '
              f'{needed/2**30:.2f} GiB; available {available/2**30:.2f} GiB. Monitor free space.',
              file=sys.stderr,flush=True)
    else:
        if aggregate>available:
            print(f'Storage advisory: aggregate reservations estimate {aggregate/2**30:.2f} GiB, '
                  f'but this campaign requires only {needed/2**30:.2f} GiB including reserve; '
                  f'available {available/2**30:.2f} GiB. Proceeding as requested. '
                  'Other campaigns may consume space; monitor free space.',file=sys.stderr,flush=True)
    return needed


def review(output):
    output=Path(output)
    missing=[n for n in ('campaign.json','benchmark.json') if not (output/n).is_file()]
    if missing: return dict(passed=False,complete=False,authorizes_training=False,missing=missing)
    from .evidence import validate
    spec=load(output,environment=False)
    row=checked(output/'benchmark.json')
    return validate(spec,row,load_launch(output))


def production_spec(output):
    launch=load_launch(output);spec=load(output)
    summary=review(output)
    if not summary['passed']: raise ValueError('Benchmark not admitted')
    if (any(m['train_hours']>launch['max_train_hours'] for m in summary['models'])
            or any(m['evaluate_hours']>launch['max_evaluate_hours'] for m in summary['models'])):
        raise ValueError('Measured recommendation exceeds reviewed ceiling; no jobs submitted')
    return hashed(dict(schema=SCHEMA,launch_sha256=launch['content_hash'],
        campaign_sha256=spec['content_hash'],benchmark_sha256=sha(Path(output)/'benchmark.json'),
        models=summary['models'],tasks=TASKS,parameters=COUNTS,trainable_parameters=TRAINABLE,teacher_parameters=TEACHER,recipes=spec['recipes'],
        inventory=spec['inventory'],data_config_sha256=spec['data_config_sha256'],
        data_contents_sha256=spec['data_contents_sha256'],environment=spec['environment'],
        source_sha256=spec['source_sha256'],platform=launch['platform'],
        resources=launch['resources'],runtime=summary['runtime'],authorizes_training=True))


def load_production(output, *, worker=False):
    load(output,worker=worker)
    saved=checked(Path(output)/'production.json')
    if saved!=production_spec(output): raise ValueError('Production admission/evidence changed')
    return saved


def task(arm):
    return copy.deepcopy(next(t for t in TASKS if t['architecture']==arm))


def run_directory(output, arm):
    return Path(output)/'production/runs'/task(arm)['run_id']/'seed_101'
