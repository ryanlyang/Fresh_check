import copy
import numpy as np
import pytest
from part_group_screen import reporting as R
from part_screen_common.controls import selected_training, validate_predictions
from part_ev_fullscale.protocol import read_json, write_json
from tests.test_part_screen_controls import make_selected, prediction_fixture
from part_screen_common.contracts import hashed


def test_matched_contrasts_are_grouping_not_optimizer_claims():
    values=dict(PART_RANGER=.8,EV_RANGER=.81,PART_GROUP_FIXED_RANGER=.82,
        PART_GROUP_LEARNED_RANGER=.83,PART_GROUP_HYBRID_RANGER=.84,PART_GROUP_RANDOM_RANGER=.805)
    effects=R.contrasts(values)
    assert effects['fixed_minus_random']==pytest.approx(.015)
    assert effects['hybrid_minus_learned']==pytest.approx(.01)
    assert effects['PART_GROUP_HYBRID_RANGER_minus_PART_RANGER']==pytest.approx(.04)
    assert effects['PART_GROUP_HYBRID_RANGER_minus_EV_RANGER']==pytest.approx(.03)


def test_paired_join_by_file_identity(tmp_path):
    run,task,recipe,runtime,sha=make_selected(tmp_path)
    files=prediction_fixture(run,task,sha)
    result=validate_predictions(run,task,'campaign',sha,files,population=100)
    reordered=dict(result,predictions=list(reversed(result['predictions'])))
    paired=R.paired_accuracy(reordered,result)
    assert paired==dict(wins=0,losses=0,ties=100,events=100,accuracy_difference=0.)


def test_pending_controls_write_progress_not_final_report(tmp_path,monkeypatch):
    monkeypatch.setattr(R.P,'load_campaign',lambda _:dict(donor={}))
    monkeypatch.setattr(R.P,'require_approval',lambda *a: {})
    monkeypatch.setattr(R,'refresh_controls',lambda *a:dict(ready=False,controls={'PART_RANGER':{'status':'pending_training'}}))
    assert R.report(tmp_path) is False
    assert read_json(tmp_path/'reports/progress.json')['complete'] is False
    assert not (tmp_path/'reports/grouped_screen_report.json').exists()


def test_complete_report_validates_predictions_locks_and_plots(tmp_path, monkeypatch):
    names = ['PART_RANGER', 'PART_GROUP_FIXED_RANGER', 'PART_GROUP_LEARNED_RANGER',
             'PART_GROUP_HYBRID_RANGER', 'PART_GROUP_RANDOM_RANGER']
    output, donor = tmp_path / 'grouped', tmp_path / 'donor'
    output.mkdir(); donor.mkdir()
    paths, tasks, records, parameters = [], [], {}, {}
    benchmark = dict(jets_per_second=100., full_length_peak_bytes=1024**3,
                     validation=dict(events=100, inference_seconds=.1, seconds=.2))
    for i, name in enumerate(names):
        parent = tmp_path / f'fixture_{i}'
        parent.mkdir()
        run, task, recipe, runtime, sha = make_selected(parent)
        task = dict(task, run_id=name, architecture=('part', 'fixed', 'learned', 'hybrid', 'random')[i])
        files = prediction_fixture(run, task, sha)
        complete = read_json(run / 'training_complete.json')
        complete.update(task=task, train_seconds=100., amp_skipped_steps=0)
        write_json(run / 'training_complete.json', complete)
        for check in (1, 2):
            path = run / 'validation' / f'check_{check:02d}.json'
            row = read_json(path); row.update(macro_auc_ovo=.8, macro_auc_ovr=.8)
            write_json(path, row)
        training = selected_training(run, task, 'campaign', recipe, runtime)
        evaluation = validate_predictions(run, task, 'campaign', sha, files, population=100)
        records[name] = dict(task=task, status='verified', training=training, evaluation=evaluation)
        root = donor if i == 0 else output
        write_json(root / 'benchmarks' / f'{name}.json', benchmark)
        if i:
            tasks.append(task); paths.append(run); parameters[task['architecture']] = 1
            for check in (1, 2):
                diagnostic = dict(check=check, mean_assignment_entropy=.4, effective_groups=[2., 2.],
                    correction_to_backbone_norm=.01, gamma=.001, angular_width=[[.1,.2],[.1,.2]],
                    mean_slot_similarity=.5, occupancy=[[1.,1.],[1.,1.]], pt_share=[[.6,.4],[.5,.5]],
                    examples=dict(assignments=[[[1.,0.],[0.,1.]]],valid=[[True,True]]))
                write_json(run / 'diagnostics' / f'check_{check:02d}.json', diagnostic)
    spec = dict(content_hash='campaign', tasks=tasks, donor=dict(path=str(donor),runtime=runtime),
        inventory={'splits': {'val': {'files': files}}}, recipes={'ranger_short_v1':recipe}, parameters=parameters)
    registry = hashed(dict(ready=True, controls={'PART_RANGER':records['PART_RANGER']}))
    lock = dict(campaign_sha256='campaign', checkpoints=[dict(task=t,sha256=records[t['run_id']]['training']['checkpoint_sha256']) for t in tasks])
    write_json(output / 'selection/locked_checkpoints.json', lock)
    monkeypatch.setattr(R.P, 'load_campaign', lambda _:spec)
    monkeypatch.setattr(R.P, 'require_approval', lambda *a:{})
    monkeypatch.setattr(R.P, 'task_directory', lambda _,i:paths[i])
    monkeypatch.setattr(R, 'refresh_controls', lambda *a:registry)
    monkeypatch.setattr(R, 'validate_predictions', lambda *a:validate_predictions(*a,population=100))
    assert R.report(output)
    result = read_json(output / 'reports/grouped_screen_report.json')
    assert result['complete'] and len(result['models']) == 5
    assert result['models']['PART_RANGER']['reused']
    assert result['contrasts']['accuracy']['fixed_minus_random'] == 0
    assert (output / 'reports/learning_curves.pdf').is_file()
    assert len(list((output / 'reports/grouping_diagnostics').glob('*.pdf'))) == 8
    original_lock = read_json(output / 'comparison_lock.json')
    assert R.report(output)
    assert read_json(output / 'comparison_lock.json') == original_lock
    with (paths[0] / 'full_validation/file_000.npz').open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        R.report(output)
