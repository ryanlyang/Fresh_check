import copy
import getpass
from pathlib import Path
import pytest
from part_ev_state_capacity.migration import replacement_status, Slurm, includes_index


class Scheduler:
    def __init__(self,root):
        self.calls=[]
        self.split=True
        self.rows={}
        common=dict(UserId=getpass.getuser()+'(12)',Command=str(root/'source/sbatch/run_part_ev_state_fullscale.sh'))
        for index in range(7):
            self.rows[f'100_{index}']=dict(common,JobId=str(200+index) if index<3 else '100',
                ArrayJobId='100',ArrayTaskId=str(index) if index<3 else '3-6',
                JobState='RUNNING' if index<3 else 'PENDING',JobName='evfull_train')
        self.rows['101']=dict(common,JobId='101',ArrayJobId='101',ArrayTaskId='0-6',
                              JobState='PENDING',JobName='evfull_evaluate')
        self.rows['102']=dict(common,JobId='102',JobState='PENDING',JobName='evfull_report')

    def job(self,job): return copy.deepcopy(self.rows.get(job))
    def terminal_state(self,job): raise ValueError('Unknown terminal job')
    def call(self,*args):
        self.calls.append(args)
        if args[:2]==('scontrol','hold'):
            if self.split:
                row=self.rows[args[2]]
                row.update(JobId=str(900+int(args[2].split('_')[1])),ArrayTaskId=args[2].split('_')[1])
        else:
            assert args[0]=='scancel'
            for row in self.rows.values():
                if row['JobId']==args[1]: row['JobState']='CANCELLED'
        return ''


def setup(tmp_path):
    root=tmp_path/'old'
    contract=dict(path=str(root),jobs=dict(train='100',evaluate='101',report='102'),runtime={})
    return root,contract,Scheduler(root)


def test_plan_readonly_and_apply_only_three_singletons(tmp_path):
    root,c,s=setup(tmp_path)
    before=copy.deepcopy(s.rows)
    plan=replacement_status(root,{},c,s)
    assert plan['kept_dependencies']==['100_0','100_1','100_2','100_6']
    assert s.calls==[]
    result=replacement_status(root,{},c,s,apply=True)
    assert result['applied'] and not result['old_campaign_files_modified']
    assert [a for a in s.calls if a[0]=='scancel']==[
        ('scancel','903'),('scancel','904'),('scancel','905'),('scancel','101'),('scancel','102')]
    assert all(s.rows[f'100_{i}']==before[f'100_{i}'] for i in (0,1,2,6))
    assert not root.exists()
    s.calls.clear()
    assert replacement_status(root,{},c,s,require_switched=True)['kept_dependencies']==plan['kept_dependencies']
    replacement_status(root,{},c,s,apply=True)
    assert not s.calls


@pytest.mark.parametrize('changed',['running','complete','owner','command','name','wrong_array','wrong_index','downstream_running'])
def test_bad_state_fail_before_any_mutation(tmp_path,changed):
    root,c,s=setup(tmp_path)
    row=s.rows['100_5']
    if changed=='running': row['JobState']='RUNNING'
    if changed=='complete': row['JobState']='COMPLETED'
    if changed=='owner': row['UserId']='someone_else(4)'
    if changed=='command': row['Command']='/different/script'
    if changed=='name': row['JobName']='unrelated'
    if changed=='wrong_array': row['ArrayJobId']='other'
    if changed=='wrong_index': row['ArrayTaskId']='0-2'
    if changed=='downstream_running': s.rows['101']['JobState']='RUNNING'
    with pytest.raises(ValueError): replacement_status(root,{},c,s,apply=True)
    assert not s.calls


def test_compressed_array_never_cancelled(tmp_path):
    root,c,s=setup(tmp_path); s.split=False
    with pytest.raises(ValueError): replacement_status(root,{},c,s,apply=True)
    assert s.calls==[('scontrol','hold','100_3')]


def test_require_switch_before_production(tmp_path):
    root,c,s=setup(tmp_path)
    with pytest.raises(ValueError,match='explicit switch'): replacement_status(root,{},c,s,require_switched=True)
    assert not s.calls


def test_finished_control_needs_fifty_check_artifacts(tmp_path,monkeypatch):
    from part_ev_state_fullscale import reporting
    root,c,s=setup(tmp_path); s.rows['100_0']['JobState']='COMPLETED'
    source={'tasks':list(range(7))}
    calls=[]
    monkeypatch.setattr(reporting,'validate_training',lambda *a:calls.append(a))
    plan=replacement_status(root,source,c,s)
    assert plan['kept_dependencies']==['100_1','100_2','100_6']
    assert calls==[(root,source,0,{})]
    s.rows['100_0']['JobState']='FAILED'
    with pytest.raises(ValueError,match='recover'): replacement_status(root,source,c,s)


def test_range_parser_bounded():
    assert includes_index('3-6%3',6) and includes_index('[0-2,6%3]',1)
    assert not includes_index('0-10000000000',4) and not includes_index('0-2',6)


def test_automatic_dependency_cancellation_during_switch(tmp_path):
    root,c,s=setup(tmp_path)
    call=s.call
    def auto_cancel(*args):
        result=call(*args)
        if args==('scancel','903'):
            s.rows['101']['JobState']='CANCELLED'
            s.rows['102']['JobState']='CANCELLED'
        return result
    s.call=auto_cancel
    assert replacement_status(root,{},c,s,apply=True)['applied']
    assert [a for a in s.calls if a[0]=='scancel']==[('scancel','903'),('scancel','904'),('scancel','905')]


def test_accounting_compressed_cancelled_array(monkeypatch):
    scheduler=Slurm()
    monkeypatch.setattr(scheduler,'call',lambda *a:'101_[0-6%3]|CANCELLED by 12|\n')
    assert scheduler.terminal_state('101_4')=='CANCELLED'
    with pytest.raises(ValueError): scheduler.terminal_state('100_4')
    with pytest.raises(ValueError): scheduler.terminal_state('101_7')
