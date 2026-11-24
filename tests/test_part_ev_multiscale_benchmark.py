"""Disposable CPU I/O/replay wiring; never substitutes for CUDA admission."""
import torch
from part_ev_multiscale import benchmark as B,runtime as R
from part_ev_multiscale.config import PLATFORMS
from part_ev_fullscale.checkpoint import capture_rank_state
from part_ev_multiscale.restart import compare

torch.set_num_threads(1)
D=torch.device('cpu')


def test_prediction_and_full_state_checkpoint_io_are_measured_and_roundtrip(tmp_path):
    base,_,opt,scaler=R.fresh('multiscale',D,1,trim=False)
    row=B.checkpoints(base,opt,scaler,tmp_path,'multiscale',D,0,1)
    assert row['prediction_events']==100000
    assert all(v>0 for k,v in row.items() if k.endswith('_seconds'))
    assert (tmp_path/'probes/multiscale/io_predictions.npz').is_file()


def test_cursor_replay_checks_bounded_real_input_mapping_without_model_updates(monkeypatch):
    closed=[];created=[]
    def loader(split,*args,**kwargs):
        batches=[R.fixed_batch(8,9+i) for i in range(3)]
        created.append((split,kwargs))
        return batches,None
    monkeypatch.setattr(B,'build_loader',loader)
    monkeypatch.setattr(B,'shutdown_loader',lambda x:closed.append(len(x)))
    owner=torch.nn.Identity();rng=capture_rank_state(owner)
    result=B.stream_replay(dict(inventory={}),dict(resources=PLATFORMS['tigris']),0,1,D)
    assert result['passed'] and result['model_updates']==0 and result['rng_restored']
    assert [s for s,_ in created]==['train','train','val','val'] and closed==[3]*4
    assert all(len(r['geometry_sha256'])==64 and r['mapping_scope']=='two_pretrim_events'
               for r in result['streams'].values())
    compare(capture_rank_state(owner),rng,exact=True)
