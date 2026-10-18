import copy
import pytest
from part_group_screen import protocol as P
from part_group_screen.config import TASKS, CONFIG
from part_ev_screen.precision import POLICY
from part_screen_common.contracts import hashed, runtime_identity
from part_ev_fullscale.protocol import write_json, read_json, file_sha256
from tests.test_part_screen_controls import precision, donor


def benchmark_fixture(tmp_path):
    runtime=dict(campaign_sha256='new',gpu='NVIDIA A100',gpu_memory_bytes=1000,precision=POLICY)
    spec=dict(content_hash='new',tasks=TASKS,benchmark=P.BENCHMARK,
              donor=dict(runtime=runtime_identity(runtime),input_fingerprints={'0':'fingerprint'}))
    for task in TASKS:
        row=dict(passed=True,task=task,campaign_sha256='new',runtime=runtime,
            initial_parity={'passed':True},input_fingerprints={'0':'fingerprint'},
            training_precision=precision('torch.float16'),validation=dict(precision=precision('torch.float32'),seconds=2,events=128000),
            measured_steps=4200,amp_skipped_steps=0,full_length_amp_skips=0,native_peak_bytes=100,
            full_length_peak_bytes=200,validation_peak_bytes=100,projected_training_hours=5.)
        write_json(tmp_path/'benchmarks'/f"{task['run_id']}.json",row)
    return spec,runtime


def test_review_approval_and_mutated_evidence(tmp_path):
    spec,runtime=benchmark_fixture(tmp_path)
    summary=P.benchmark_summary(tmp_path,spec,runtime)
    assert summary['recommended_train_hours']==7
    P.approve(tmp_path,spec)
    assert P.require_approval(tmp_path,spec,runtime)==summary
    path=tmp_path/'benchmarks'/f"{TASKS[0]['run_id']}.json"
    row=read_json(path);row['projected_training_hours']=6.;write_json(path,row)
    with pytest.raises(ValueError,match='admission changed'):P.require_approval(tmp_path,spec,runtime)


@pytest.mark.parametrize('change',['failed','runtime','fingerprint','precision','skips','memory','missing'])
def test_bad_benchmark_never_admits(tmp_path,change):
    spec,runtime=benchmark_fixture(tmp_path)
    path=tmp_path/'benchmarks'/f"{TASKS[0]['run_id']}.json"
    row=read_json(path)
    if change=='failed':row['passed']=False
    elif change=='runtime':row['runtime']['gpu']='GH200'
    elif change=='fingerprint':row['input_fingerprints']['0']='different'
    elif change=='precision':row['training_precision']['observed_activations']['linear']='torch.float32'
    elif change=='skips':row['full_length_amp_skips']=1
    elif change=='memory':row['full_length_peak_bytes']=950
    else:row.pop('initial_parity')
    write_json(path,row)
    with pytest.raises(ValueError):P.approve(tmp_path,spec)
    assert not (tmp_path/'production_approval/approval.json').exists()


def test_load_rejects_nonproduction_recipe(tmp_path,monkeypatch):
    source={'fixture.py':'hash'}
    spec=hashed(dict(schema=P.SCHEMA,tasks=TASKS,model=CONFIG,precision=POLICY,
        recipes={'ranger_short_v1':P.recipes()['ranger_short_v1']},benchmark=P.BENCHMARK,source=source,environment={}))
    write_json(tmp_path/'experiment_spec.json',spec)
    monkeypatch.setattr(P,'source_manifest',lambda _:source)
    assert P.load_campaign(tmp_path,check_environment=False)==spec
    spec['recipes']['ranger_short_v1']['total_steps']=4
    spec.pop('content_hash');write_json(tmp_path/'experiment_spec.json',hashed(spec))
    with pytest.raises(ValueError,match='protocol differs'):P.load_campaign(tmp_path,check_environment=False)


def test_source_manifest_includes_all_new_runtime_dependencies():
    from pathlib import Path
    paths=P.source_manifest(Path(__file__).resolve().parents[1])
    for name in ('part_screen_common/engine.py','part_screen_common/controls.py','part_group_screen/grouping.py',
                 'scripts/run_part_ev_screen.py','scripts/run_part_group_screen.py','sbatch/run_part_group_screen_sporc.sh'):
        assert name in paths


def test_prepare_refuses_donor_destination_before_lock_or_write(tmp_path):
    from types import SimpleNamespace
    from scripts.run_part_group_screen import prepare
    write_json(tmp_path/'launch.json',dict(donor=str(tmp_path.resolve())))
    before = {str(p):file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='separate from donor'):
        prepare(SimpleNamespace(output=tmp_path))
    assert before == {str(p):file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
