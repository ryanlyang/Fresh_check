import pytest
from part_ev_compact import protocol as P
from part_ev_compact import flops as F


def reports(tmp, monkeypatch):
    spec={'content_hash':'campaign','input_reference':{'fingerprints':{k:k for k in ('0','20','1020','2020','4020')}}}
    monkeypatch.setattr(P,'load',lambda _:spec)
    for task in P.TASKS:
        compact=task['architecture']=='compact'
        coeff=[2370048,11361792,85760] if compact else [2370048,11372544,39168]
        forward=[790016,3787264,27904] if compact else [790016,3790848,13056]
        samples={str(n):{kind:{'per_jet':sum(c*n**i for i,c in enumerate(cs))}
                    for kind,cs in [('training',coeff),('forward',forward)]}
                 for n in (1,17,32,64,65,96,127,128)}
        row=dict(task=task,campaign_sha256='campaign',passed=True,complete=True,authorizes_training=False,
             measured_steps=4200,input_fingerprints=spec['input_reference']['fingerprints'],
             parity={'passed':True},restart={'passed':True},flops=dict(convention=F.CONVENTION,
                 samples=samples,coefficients={'training':list(map(str,coeff)),'forward':list(map(str,forward))}),
             length_histogram={'128':4200},amp_skips=0,full_length_confirmed=10,
             training_precision={'passed':True},validation={'precision':{'passed':True},'events':131072,'seconds':100},
             peak_reserved_bytes=40*2**30,gpu_memory_bytes=96*2**30,gpu='NVIDIA GH200',
             windows=[dict(steps=1000,seconds=200)]*4+[dict(steps=200,seconds=40)],
             checkpoint_seconds=.1,startup_seconds=100)
        P.write_json(tmp/'benchmarks'/f'{task["architecture"]}.json',P.hashed(row))
    return tmp


def mutate(root,field,value):
    path=root/'benchmarks/compact.json'
    row=P.read_json(path);row.pop('content_hash')
    target=row
    for key in field[:-1]:target=target[key]
    target[field[-1]]=value
    P.write_json(path,P.hashed(row))


def test_review_freezes_fair_budget_without_authorizing_training(tmp_path,monkeypatch):
    root=reports(tmp_path,monkeypatch)
    result=P.summarize(root)
    assert result['passed'] and result['authorizes_training'] is False
    assert result['budget']['matched_steps']==1362850
    assert result['recipes']['matched']['epochs']==50
    assert result['recipes']['matched']['total_steps']==1362850
    assert len(result['evidence'])==2


@pytest.mark.parametrize('field,value',[
    (['passed'],False), (['complete'],False), (['authorizes_training'],True),
    (['measured_steps'],100), (['amp_skips'],50), (['full_length_confirmed'],9),
    (['parity','passed'],False), (['restart','passed'],False),
    (['validation','events'],100), (['validation','seconds'],0),
    (['training_precision','passed'],False), (['peak_reserved_bytes'],95*2**30),
    (['input_fingerprints','4020'],'wrong'), (['length_histogram'],{'127':4200}),
    (['flops','coefficients','training'],['0','0','1']),
    (['windows'],[dict(steps=1000,seconds=200)]),
    (['gpu'],'CPU')])
def test_review_rejects_incomplete_or_mismatched_evidence(tmp_path,monkeypatch,field,value):
    root=reports(tmp_path,monkeypatch);mutate(root,field,value)
    with pytest.raises((ValueError,KeyError)):P.summarize(root)


def test_missing_report_and_hash_mutation(tmp_path,monkeypatch):
    root=reports(tmp_path,monkeypatch)
    path=root/'benchmarks/compact.json';path.rename(path.with_suffix('.missing'))
    with pytest.raises(FileNotFoundError):P.summarize(root)
    path.with_suffix('.missing').rename(path)
    row=P.read_json(path);row['passed']=False;P.write_json(path,row)
    with pytest.raises(ValueError,match='hash'):P.summarize(root)
