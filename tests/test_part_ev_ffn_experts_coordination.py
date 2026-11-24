"""Two actual CPU/Gloo ranks prove the dropless/global-gradient contract."""
from datetime import timedelta
import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,directory):
    from part_ev_ffn_experts import runtime as R,verification as V
    root=Path(directory);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=2,
        timeout=timedelta(seconds=30))
    try:
        proof=V.bank_ddp(torch.device('cpu'),rank,2)
        assert proof['passed'] and proof['local_empty'] and proof['global_empty']
        assert R.root_call(lambda:(time.sleep(.05),42)[1],rank,2)==42
        caught=False
        try:R.all_call(lambda:(_ for _ in ()).throw(OSError('fixture')) if rank==1 else None,2)
        except RuntimeError:caught=True
        assert caught
        R.STOP=rank==0;assert R.stopping(torch.device('cpu'),2);R.STOP=False
        (root/f'rank_{rank}.json').write_text(json.dumps(proof))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='CPU Gloo unavailable')
def test_two_process_unequal_tokens_empty_experts_global_aux_and_stop(tmp_path):
    context=mp.spawn(worker,args=(str(tmp_path),),nprocs=2,join=False)
    deadline=time.monotonic()+55
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('Bounded experts Gloo test timed out')
        for i in range(2):
            p=json.loads((tmp_path/f'rank_{i}.json').read_text())
            assert p['shared_parameter_gradients'] and p['input_gradient_world_factor']
    finally:
        for p in context.processes:
            if p.is_alive():p.terminate()
        for p in context.processes:p.join(timeout=5)
