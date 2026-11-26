from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from part_ev_masked_fast_production import protocol as P, submission as F
from part_ev_masked_fast_production.config import RESOURCES
from scripts import submit_part_ev_masked_fast_production as CLI


def test_resource_requests_one_long_candidate_and_no_short_job_signals(tmp_path):
    launch=dict(account='test');plan=dict(models=[dict(train_hours=255,evaluate_hours=2)])
    for phase in ('preflight','train','evaluate'):
        row=F.request(tmp_path,launch,phase,plan=plan)
        assert '--gres=gpu:gh200:1' in row and '--mem=128G' in row
        assert not any('array' in s for s in row)
    assert '--time=15300' in F.request(tmp_path,launch,'train',plan=plan)
    for phase in ('summary','dispatch','report'):
        row=F.request(tmp_path,launch,phase)
        assert not any(s.startswith(('--signal','--gres')) for s in row)
    with pytest.raises(ValueError):F.request(tmp_path,launch,'control',plan=plan)


@pytest.fixture
def dispatch_fixture(tmp_path,monkeypatch):
    output=tmp_path/'checkpoints/part_ev_masked_fast_production/campaign';output.mkdir(parents=True)
    old=tmp_path/'old_campaign';old.mkdir();(old/'keep').write_text('untouched')
    launch=dict(repo=str(tmp_path),python='/env/python',platform='tigris',resources=RESOURCES,account='test')
    plan=P.hashed(dict(models=[dict(arm='masked',train_hours=255,evaluate_hours=2)]))
    monkeypatch.setattr(P,'load',lambda *a,**k:launch)
    monkeypatch.setattr(P,'production_spec',lambda *a:plan)
    monkeypatch.setattr(P,'load_production',lambda *a:plan)
    monkeypatch.setattr(P,'check_storage',lambda *a:1)
    monkeypatch.setattr(F.S,'locks',lambda *a:nullcontext())
    calls=[];previews=[];flags=dict(ambiguous=False,fail_preflight=False)
    def command(argv,**kwargs):
        calls.append(argv)
        return 'unknown' if flags['ambiguous'] else str(100+len(calls))
    def preflight(*args):
        if flags['fail_preflight']:raise ValueError('partition ceiling')
        previews.extend(args[-1])
    monkeypatch.setattr(F.S,'command',command);monkeypatch.setattr(F.S,'preflight',preflight)
    return output,plan,calls,previews,flags,old


def test_candidate_chain_exact_dependencies_and_idempotence(dispatch_fixture):
    output,plan,calls,previews,_,old=dispatch_fixture
    result=F.dispatch(output)
    assert result['jobs']==dict(train='101',evaluate='102',report='103') and not result['control_submitted']
    assert '--dependency=afterok:101' in calls[1] and '--dependency=afterok:102' in calls[2]
    assert not any('control' in s for c in calls for s in c)
    assert len(calls)==len(previews)==3 and (old/'keep').read_text()=='untouched'
    assert F.dispatch(output)['status']=='already_submitted' and len(calls)==3
    assert P.checked(output/'production/campaign.json')==plan


def test_failure_before_submission_does_not_make_production(dispatch_fixture):
    output,_,calls,_,flags,_=dispatch_fixture;flags['fail_preflight']=True
    with pytest.raises(ValueError):F.dispatch(output)
    assert not calls and not (output/'production.json').exists() and not F.owner_path(output).exists()


def test_ambiguous_partial_submission_blocks_duplicate(dispatch_fixture):
    output,_,calls,_,flags,_=dispatch_fixture;flags['ambiguous']=True
    with pytest.raises(ValueError):F.dispatch(output)
    flags['ambiguous']=False
    with pytest.raises(ValueError):F.dispatch(output)
    assert len(calls)==1


def test_second_output_cannot_duplicate_same_execution(dispatch_fixture):
    output,_,calls,_,_,_=dispatch_fixture;F.dispatch(output)
    second=output.parent/'second';second.mkdir()
    with pytest.raises(ValueError,match='Existing owner'):F.dispatch(second)
    assert len(calls)==3


def test_resume_refuses_running_or_missing_accounting(dispatch_fixture,monkeypatch):
    output,_,calls,_,_,_=dispatch_fixture;F.dispatch(output)
    for states in ({'101':'RUNNING','102':'PENDING','103':'PENDING'},{'101':'COMPLETED'}):
        monkeypatch.setattr(F.S,'job_states',lambda *a,s=states:s)
        with pytest.raises(ValueError,match='Prior jobs'):F.resume(output)
    assert len(calls)==3


def test_reviewed_scope_and_explicit_path_required(tmp_path,monkeypatch):
    monkeypatch.setattr(CLI,'ROOT',tmp_path);output=tmp_path/'checkpoints/part_ev_masked_fast_production/new'
    speed=tmp_path/'speed';speed.mkdir()
    for mode in ('pipeline','dispatch','resume'):
        with pytest.raises(SystemExit):CLI.parse(['--mode',mode,'--output',str(output),'--from-speed',str(speed)])
    args=CLI.parse(['--mode','pipeline','--output',str(output),'--from-speed',str(speed),'--reviewed'])
    assert args.max_train_hours==480
    with pytest.raises(SystemExit):CLI.parse(['--mode','pipeline','--output',str(output),'--from-speed',str(speed),
                                            '--reviewed','--max-train-hours','745'])


@pytest.mark.parametrize('mode,following,dependency',[('pipeline','dispatch','afterok'),('preflight','summary','afterany')])
def test_initial_chain_does_not_submit_training_before_qualification(tmp_path,monkeypatch,mode,following,dependency):
    monkeypatch.setattr(CLI,'ROOT',tmp_path)
    output=tmp_path/'checkpoints/part_ev_masked_fast_production/new'
    source=tmp_path/'completed_speed';source.mkdir()
    original=dict(python='/env/python',account='test')
    evidence=dict(original_launch=original,choice=dict(provisional_train_hours=255))
    monkeypatch.setattr(CLI.D,'load',lambda *a:original)
    monkeypatch.setattr(P,'check_storage',lambda *a:8*2**30)
    monkeypatch.setattr(CLI.S,'locks',lambda *a:nullcontext())
    calls=[];previews=[];proposals=[]
    def command(argv,**kwargs):
        if argv[0]=='sbatch':
            calls.append(argv);return str(200+len(calls))
        assert '--phase' in argv and 'inspect' in argv
        return json.dumps(evidence)
    def freeze(repo,where,proposal):
        proposals.append(copy.deepcopy(proposal))
        for name in ('part_ev_masked_fast_production/protocol.py','part_ev_masked_fast_production/execution.py',
                     'scripts/run_part_ev_masked_fast_production.py','sbatch/run_part_ev_masked_fast_production.sh'):
            path=where/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen')
        return P.hashed(proposal)
    monkeypatch.setattr(CLI.S,'command',command)
    monkeypatch.setattr(CLI.S,'preflight',lambda *a:previews.extend(a[-1]))
    monkeypatch.setattr(CLI.S,'freeze',freeze)
    args=SimpleNamespace(mode='plan',output=output,from_speed=source,max_train_hours=480,max_evaluate_hours=24)
    CLI.launch(args)
    assert not output.exists() and not calls and not proposals and not previews
    args.mode=mode;CLI.launch(args)
    assert [call[-1] for call in calls]==['preflight',following]
    assert '--dependency='+dependency+':201' in calls[1]
    assert len(previews)==2 and all(str(tmp_path/'sbatch/run_part_ev_masked_fast_production.sh')==r[-3] for r in previews)
    assert proposals[0]['auto_production']==(mode=='pipeline')
    assert proposals[0]['resources']['microbatch']==256 and proposals[0]['max_train_hours']==480
    assert not (output/'production').exists() and not (output/'production.json').exists()
    assert {r['phase'] for r in CLI.S.journal(output,'pipeline')}=={'preflight',following}
    with pytest.raises(FileExistsError):CLI.launch(args)
    assert len(calls)==2
