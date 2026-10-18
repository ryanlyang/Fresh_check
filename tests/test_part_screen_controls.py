import copy
from pathlib import Path
import numpy as np
import pytest
from part_ev_fullscale.protocol import write_json, read_json, file_sha256, digest, CLASS_NAMES
from part_ev_screen import protocol as S
from part_ev_screen.precision import POLICY
from part_screen_common import contracts as C, controls as R
from scripts.run_part_ev_screen import classification_metrics


def precision(dtype):
    return dict(passed=True, parameter_dtypes=['torch.float32'], expected_activation_dtype=dtype,
                observed_activations={'linear': dtype})


@pytest.fixture
def donor(tmp_path):
    root = tmp_path / 'donor'
    sources = {}
    for name in C.REQUIRED_SOURCE:
        source = root / 'source' / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('fixture')
        sources[name] = file_sha256(source)
    inventory = dict(splits={})
    for split, count in [('train', 1000), ('val', 50)]:
        names = sorted(f'{label}_{i:03d}.root' for label in C.CLASSES for i in C.SPLITS[split][1])
        inventory['splits'][split] = dict(files=[dict(name=name, path=f'/fixture/{split}/{name}',
             num_entries=100000, size_bytes=10, mtime_ns=100) for name in names])
    inventory['inventory_sha256'] = digest(inventory)
    sha = sources['particle_transformer/data/JetClass/JetClass_full.yaml']
    spec = C.hashed(dict(schema=C.DONOR_SCHEMA, tasks=S.TASKS, recipes=S.recipes(), precision=POLICY,
        source=sources, parameters=S.COUNTS, data_config_sha256=sha,
        inventory=inventory, environment={'fixture': True}))
    write_json(root / 'campaign.json', spec)
    runtime = dict(campaign_sha256=spec['content_hash'], gpu='NVIDIA A100 fixture', gpu_memory_bytes=10000,
        batch_size=512, workers=2, world_size=1, batchnorm='native', precision=POLICY,
        tf32=False, cudnn_benchmark=False)
    for name in ['PART_RANGER', 'EV_RANGER']:
        write_json(root / 'benchmarks' / f'{name}.json', dict(task=C.CONTROL_TASKS[name],
            campaign_sha256=spec['content_hash'], runtime=runtime, passed=True,
            initial_parity={'passed':True}, input_fingerprints={'0':'fixture'},
            training_precision=precision('torch.float16'), validation={'precision':precision('torch.float32')}))
    return root


def make_selected(tmp_path):
    run = tmp_path / 'run'
    task = dict(run_id='fixture', architecture='fixed', seed=101, recipe='ranger_short_v1')
    recipe = dict(S.recipes()['ranger_short_v1'], total_steps=4, epochs=2, steps_per_epoch=2,
                  validation_events_per_epoch=512)
    runtime = {'fixture':True}
    run.mkdir()
    (run / 'best_model_val.pt').write_bytes(b'checkpoint fixture')
    sha = file_sha256(run / 'best_model_val.pt')
    for i in (1,2):
        write_json(run / 'validation' / f'check_{i:02d}.json', dict(check=i,step=2*i,events=512,accuracy=.8))
    write_json(run / 'training_complete.json', dict(task=task,step=4,validated_checks=2,
        campaign_sha256='campaign',runtime=runtime,best_check=1,best_accuracy=.8,best_checkpoint_sha256=sha))
    write_json(run / 'optimizer.json', dict(optimizer='Ranger',groups=[dict(name='group_0',
        parameters=1,parameter_tensors=1,parameter_names=['fixture'],lr=recipe['start_lr'],
        betas=recipe['betas'],eps=recipe['eps'],weight_decay=0.)],parameters=1))
    return run, task, recipe, runtime, sha


def prediction_fixture(run, task, sha):
    directory = run / 'full_validation'
    directory.mkdir(parents=True)
    files, records, all_logits, all_labels = [], [], [], []
    for i in range(2):
        labels = np.tile(np.arange(10), 5)
        logits = np.random.default_rng(i).normal(size=(50,10)).astype(np.float32)
        logits[np.arange(50),labels] += 2
        file = dict(name=f'source_{i}.root', path=f'/fixture/{i}',num_entries=50)
        files.append(file)
        path = directory / f'file_{i:03d}.npz'
        np.savez(path,logits=logits,labels=labels)
        sidecar = dict(contract=dict(campaign_sha256='campaign',checkpoint_sha256=sha,
            file=file,entry_start=0,entry_stop=50),sha256=file_sha256(path))
        write_json(path.with_suffix('.json'),sidecar)
        records.append(sidecar)
        all_logits.append(logits); all_labels.append(labels)
    metrics = classification_metrics(np.concatenate(all_logits),np.concatenate(all_labels),rejection=True)
    write_json(directory / 'metrics.json',dict(task=task,campaign_sha256='campaign',checkpoint_sha256=sha,
        identity_manifest=records,metrics=metrics))
    return files


def test_donor_contract_pending_and_read_only_refresh(donor,tmp_path):
    before = {str(p):file_sha256(p) for p in donor.rglob('*') if p.is_file()}
    contract = C.donor_contract(donor,['PART_RANGER','EV_RANGER'])
    assert contract['runtime'].get('campaign_sha256') is None
    output = tmp_path / 'grouped'
    result = R.refresh_controls(output,contract)
    assert not result['ready']
    assert all(r['status']=='pending_training' for r in result['controls'].values())
    assert R.refresh_controls(output,contract) == result
    assert before == {str(p):file_sha256(p) for p in donor.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='separate'):
        R.refresh_controls(donor / 'bad',contract)


@pytest.mark.parametrize('mutation',['seed','recipe','precision','source','inventory','runtime'])
def test_incompatible_donor_rejected(donor,mutation):
    spec = read_json(donor/'campaign.json')
    if mutation=='seed': spec['tasks'][0]['seed']=202
    elif mutation=='recipe': spec['recipes']['ranger_short_v1']['total_steps']=4
    elif mutation=='precision': spec['precision']['training']='fp32'
    elif mutation=='source':
        next((donor/'source').rglob('*.yaml')).write_text('changed')
    elif mutation=='inventory': spec['inventory']['splits']['val']['files'][0]['num_entries']=1
    else:
        path=donor/'benchmarks/PART_RANGER.json'
        row=read_json(path); row['runtime']['gpu']='GH200'; write_json(path,row)
    spec.pop('content_hash');write_json(donor/'campaign.json',C.hashed(spec))
    with pytest.raises(ValueError): C.donor_contract(donor,['PART_RANGER'])


def test_selected_training_first_tie_and_mutations(tmp_path):
    run,task,recipe,runtime,sha=make_selected(tmp_path)
    row=R.selected_training(run,task,'campaign',recipe,runtime)
    assert row['best_check']==1 and row['checkpoint_sha256']==sha
    completion=read_json(run/'training_complete.json')
    completion['best_check']=2; write_json(run/'training_complete.json',completion)
    with pytest.raises(ValueError,match='Selection'): R.selected_training(run,task,'campaign',recipe,runtime)
    completion['best_check']=1; write_json(run/'training_complete.json',completion)
    (run/'best_model_val.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'): R.selected_training(run,task,'campaign',recipe,runtime)


def test_prediction_validation_tamper_and_missing_coverage(tmp_path):
    run,task,recipe,runtime,sha=make_selected(tmp_path)
    files=prediction_fixture(run,task,sha)
    result=R.validate_predictions(run,task,'campaign',sha,files,population=100)
    assert result['population']==100 and len(result['labels_by_file'])==2
    path=run/'full_validation/metrics.json'
    original=read_json(path); changed=copy.deepcopy(original);changed['metrics']['accuracy']=.999
    write_json(path,changed)
    with pytest.raises(ValueError,match='Metrics disagree'): R.validate_predictions(run,task,'campaign',sha,files,population=100)
    write_json(path,original)
    with pytest.raises(ValueError,match='coverage'): R.validate_predictions(run,task,'campaign',sha,files,population=200)
    with (run/'full_validation/file_000.npz').open('ab') as h: h.write(b'tamper')
    with pytest.raises(ValueError,match='checksum'): R.validate_predictions(run,task,'campaign',sha,files,population=100)


def test_source_path_escape_rejected(donor):
    with pytest.raises(ValueError,match='Unsafe'): C.child(donor,'../outside')
    with pytest.raises(ValueError,match='Unsafe'): C.child(donor,donor.resolve())


def test_reviewed_donor_recovery_is_reused_not_new_model_admission(donor):
    spec = read_json(donor / 'campaign.json')
    path = donor / 'benchmarks/PART_RANGER.json'
    original = read_json(path)
    original['passed'] = False  # the donor's bounded recovery approved this probe
    write_json(path, original)
    recovery = donor / 'diagnostics/recovery_fixture'
    write_json(recovery / 'recovery.json', {'fixture':True})
    hashes = {}
    for suffix in ('.json', '.steps.jsonl'):
        target = recovery / ('PART_RANGER' + suffix)
        target.write_text('{}\n')
        hashes[target.name] = file_sha256(target)
    approval_dir = donor / 'production_approval'
    approval_dir.mkdir()
    source = {}
    for key,name in (('driver_sha256','run_part_ev_screen_after_recovery.py'),
                     ('helper_sha256','check_part_ev_screen_amp_recovery.py'),('launch_sha256','launch.env')):
        target = approval_dir / name; target.write_text('fixture\n')
        source[key] = file_sha256(target)
    approval = C.hashed(dict(schema='part_ev_screen_reviewed_recovery_approval_v1',
        campaign=str(donor.resolve()),campaign_sha256=spec['content_hash'],
        recovery=str(recovery.resolve()),recovery_plan_sha256=file_sha256(recovery/'recovery.json'),
        recovery_results_sha256=hashes,authorizes_training=True,reviewed=True,
        basis='original_memory_precision_parity_throughput_plus_bounded_amp_recovery',
        initialization='fresh_original_seed_model_optimizer_and_grad_scaler',
        runtime=original['runtime'],summary=dict(passed=True,models=[original]),**source))
    write_json(approval_dir / 'approval.json', approval)
    assert C.donor_contract(donor,['PART_RANGER'])['evidence']['production_approval/approval.json']
    (recovery/'PART_RANGER.steps.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='recovery evidence changed'):
        C.donor_contract(donor,['PART_RANGER'])
