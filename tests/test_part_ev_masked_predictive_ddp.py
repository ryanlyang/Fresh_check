"""Two real CPU DDP processes; health evidence only, not CUDA/SyncBN admission."""
from datetime import timedelta
import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,directory):
    from part_ev_masked_predictive.verification import distributed
    torch.set_num_threads(1);root=Path(directory)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        result=distributed(torch.device('cpu'),rank,2,cpu=True)
        (root/f'rank_{rank}.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='CPU Gloo unavailable')
def test_actual_two_process_public_ddp_phases_and_empty_final_flush(tmp_path):
    context=mp.spawn(worker,args=(str(tmp_path),),nprocs=2,join=False);deadline=time.monotonic()+100
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise TimeoutError('DDP test exceeded bounded timeout')
        for r in range(2):
            result=json.loads((tmp_path/f'rank_{r}.json').read_text());assert result['passed'] and result['cpu_health_only']
    finally:
        for p in context.processes:
            if p.is_alive():p.terminate()
        for p in context.processes:p.join(timeout=5)
