"""Real two-process CPU coordination; not SyncBN/NCCL or GPU-fit evidence."""
from datetime import timedelta
import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,directory):
    from part_ev_multiscale import runtime as R
    torch.set_num_threads(1)
    root=Path(directory)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),
        rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        result=R.root_call(lambda:(time.sleep(.05),'root payload')[1],rank,2)
        assert result=='root payload'
        rejected=False
        try:R.all_call(lambda:(_ for _ in ()).throw(OSError('rank-local fixture')) if rank==1 else None,2)
        except RuntimeError:rejected=True
        assert rejected
        values,_=R.tensors(R.split_batch(R.fixed_batch(8,7),rank,2),torch.device('cpu'))
        if rank==1:values[-1][0]=False
        rejected=False
        try:R.input_health(values,torch.device('cpu'),2)
        except ValueError:rejected=True
        assert rejected
        R.STOP=rank==0
        assert R.stopping(torch.device('cpu'),2)
        R.STOP=False
        assert R.gather(rank,2)==[0,1]
        (root/f'rank_{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='CPU Gloo unavailable')
def test_two_process_failure_stop_and_root_work_coordination(tmp_path):
    context=mp.spawn(worker,args=(str(tmp_path),),nprocs=2,join=False)
    deadline=time.monotonic()+55
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('Bounded CPU coordination test timed out')
        assert all(json.loads((tmp_path/f'rank_{r}.json').read_text())['passed'] for r in range(2))
    finally:
        for p in context.processes:
            if p.is_alive():p.terminate()
        for p in context.processes:p.join(timeout=5)

