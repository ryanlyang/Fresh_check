"""Experts-aware CE plus global balancing; explicit same-node runtime."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math
import os
import signal
import socket
import subprocess
import time
import torch
import torch.distributed as dist
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.protocol import learning_rate
from .precision import model_precision,audit_precision
from part_ev_gpu_scaling.benchmark import gather, agree, tensors, fixed_batch, split_batch, parameter_digest
from .config import POLICY,SCIENCE,recipe
from .model import build_model
from .losses import balancing,local_health

STOP=False
DEADLINE=float('inf')


def stop_signal(*_):
    global STOP
    STOP=True


def stopped():
    marker=os.environ.get('EVEX_STOP_FILE')
    return STOP or time.monotonic()>=DEADLINE or bool(marker and __import__('pathlib').Path(marker).exists())


def stopping(device,world):
    return not agree(not stopped(),device,world)


def log(stage,**fields):
    print(json.dumps(dict(stage=stage,**fields),allow_nan=False),flush=True)


def root_call(action,rank,world):
    if world==1:
        try:payload=(True,action())
        except Exception as e:payload=(False,f'{type(e).__name__}: {e}')
    else:
        # Root-only validation/5M CPU metrics may outlast a process-group timeout.
        # Peers must exchange bounded heartbeats, not park inside one NCCL
        # broadcast until all root work finishes. The action is rank-local and
        # owns its own precision/no-grad contexts; no model work runs on peers.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(action) if rank==0 else None
            while True:
                packet=None
                if rank==0 and future.done():
                    try:packet=(True,future.result())
                    except BaseException as e:packet=(False,f'{type(e).__name__}: {e}')
                rows=[packet];dist.broadcast_object_list(rows,src=0);payload=rows[0]
                if payload is not None:break
                time.sleep(.2)
    if not payload[0]: raise RuntimeError(payload[1])
    return payload[1]


def all_call(action,world):
    """Coordinate rank-local loader/I/O errors before the next model collective."""
    result=None;error=None
    try: result=action()
    except Exception as e: error=f'{type(e).__name__}: {e}'
    errors=gather(error,world)
    if any(e is not None for e in errors): raise RuntimeError(str(errors))
    return result


def setup(launch, *, training=True):
    r=launch['resources'];world=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    expected=r['gpus'] if training else 1
    if (world!=expected or not 0<=rank<world or local!=rank or not torch.cuda.is_available()
            or torch.cuda.device_count()!=expected):
        raise RuntimeError('Exactly the requested visible same-node GPUs are required')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    if len(set(gather(socket.gethostname(),world)))!=1: raise ValueError('Same-node GPUs required')
    torch.set_num_threads(r['threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name in ('SIGTERM','SIGUSR1'): signal.signal(getattr(signal,name),stop_signal)
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True)
    drivers=set(driver.strip().splitlines())
    if len(drivers)!=1:raise ValueError('Cannot establish one NVIDIA driver identity')
    gpu=dict(name=torch.cuda.get_device_name(device),total_memory=torch.cuda.get_device_properties(device).total_memory,
             capability=list(torch.cuda.get_device_capability(device)),driver=next(iter(drivers)))
    gpus=gather(gpu,world)
    if len({(g['name'],g['total_memory']) for g in gpus})!=1: raise ValueError('Mismatched GPUs')
    token='A100' if launch['platform']=='sporc' else 'GH200'
    if token not in gpu['name'] or (token=='GH200' and gpu['total_memory']<80*2**30):
        raise ValueError('Wrong GPU type/memory')
    return rank,world,device,gpu


def fresh(arm,device,world,*,trim=True,checkpointed=True,training_world=None):
    seed_everything(101)
    base=build_model(arm,trim=trim,checkpointed=checkpointed).to(device)
    if (world if training_world is None else training_world)==2:
        base=torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
    if world==2:
        model=torch.nn.parallel.DistributedDataParallel(base,device_ids=[device.index],
            output_device=device.index,find_unused_parameters=False,gradient_as_bucket_view=True)
    else: model=base
    opt=build_optimizer(base,recipe())
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    return base,model,opt,scaler


def finite_state(value):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_state(v) for v in value.values())
    if isinstance(value,(tuple,list)): return all(finite_state(v) for v in value)
    if isinstance(value,float): return math.isfinite(value)
    return True


def input_health(values,device,world):
    ok=False
    try:
        mask=values[3].bool()
        shape=(mask.ndim==3 and mask.shape[1]==1 and 0<mask.shape[2]<=128)
        if shape:
            healthy=mask.any(-1).all()
            for v,c in zip(values[:3],(2,17,4)):
                shape=shape and v.shape==(mask.shape[0],c,mask.shape[2])
                healthy=healthy & torch.isfinite(v.masked_fill(~mask.expand_as(v),0)).all()
            ok=shape and bool(healthy)
    except (RuntimeError,IndexError,TypeError):ok=False
    if not agree(ok,device,world): raise ValueError('Malformed/nonfinite/empty input on at least one rank')


def step(base,model,opt,scaler,values,truth,index,world,*,audit=False,amp=True,route_details=False):
    device=truth.device;input_health(values,device,world)
    if not agree(truth.ndim==1 and len(truth)==len(values[0]) and bool(((truth>=0)&(truth<10)).all()),device,world):
        raise ValueError('Malformed labels')
    base.train();opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=learning_rate(index,recipe())
    expected=torch.float16 if amp and device.type=='cuda' else torch.float32
    with model_precision(base,device,training=amp), (
            audit_precision(base,expected_dtype=expected) if audit else nullcontext()) as precision:
        result=model(*values,return_router_stats=True,return_route_details=route_details)
        logits=result['logits'];ce=torch.nn.functional.cross_entropy(logits,truth)
    candidate=base.experts_architecture=='experts'
    healthy=result['health'] & torch.isfinite(logits).all()
    if candidate:healthy=healthy & local_health(result['banks'],device)
    elif result['banks']:healthy=healthy&False
    if not agree(bool(healthy),device,world):raise FloatingPointError('Invalid experts output/route statistics')
    routing=None;aux=None
    if candidate:
        auxiliary,routing=balancing(result['banks'],world,device)
        aux=routing['balance'];loss=ce+SCIENCE['alpha']*auxiliary
    else:loss=ce  # The control has no fake auxiliary term or statistics collective.
    if not agree(bool(torch.isfinite(loss)),device,world): raise FloatingPointError('Nonfinite loss/health')
    old=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
    norms=[p.grad.detach().float().norm() for p in base.parameters() if p.grad is not None]
    healthy=bool(torch.stack(norms).norm().isfinite())
    if not scaler.is_enabled() and not agree(healthy,device,world):
        raise FloatingPointError('Nonfinite unscaled gradient')
    scaler.step(opt);scaler.update();skipped=int(scaler.get_scale()<old)
    if not agree(healthy or bool(skipped),device,world): raise FloatingPointError('Gradient overflow without recovery')
    states=gather((skipped,scaler.get_scale(),opt.step_counter,float(ce.detach())),world)
    if len({s[:3] for s in states})!=1: raise ValueError('Scaler/Lookahead ranks diverged')
    # Reuse the existing step-schedule gather for global-mean CE logging. The
    # backward CE remains the original local mean; DDP averages its gradients.
    global_ce=sum(s[3] for s in states)/world
    balance_value=None if aux is None else float(aux)
    record=dict(loss=global_ce+(SCIENCE['alpha']*balance_value if candidate else 0.),
        ce=global_ce,balance=balance_value,
        alpha=SCIENCE['alpha'] if candidate else 0.,skipped=skipped,precision=precision,
        lr=opt.param_groups[0]['lr'],routing=None)
    if routing is not None:
        record['routing']={k:v.detach().cpu().tolist() for k,v in routing.items()
            if k in ('requested','executed','primary','secondary','tokens','probability_sum')}
    if route_details:
        import hashlib
        h=hashlib.sha256();margins=[]
        for b in result['banks']:
            for key in ('ids','positions'):
                value=b['details'][key].detach().cpu().contiguous()
                h.update(str((key,tuple(value.shape),str(value.dtype))).encode());h.update(value.numpy().tobytes())
            for key in ('requested','executed'):
                h.update(b[key].detach().cpu().numpy().tobytes())
            margins.append(float(b['min_margin']))
        record['routes']=dict(sha256=h.hexdigest(),minimum_margin=min(margins) if margins else None)
    return record


def synchronize(device,world=1):
    if device.type=='cuda': torch.cuda.synchronize(device)
    if world>1: dist.barrier()


def replay(iterator,count,device,rank,world,*,root_only=False):
    for start in range(0,count,100):
        if stopping(device,world): raise SystemExit(75)
        n=min(100,count-start)
        def advance():
            for _ in range(n): next(iterator)
        if root_only: root_call(advance,rank,world)
        else: all_call(advance,world)
        if rank==0 and (start+n)%10000==0: log('replay',batches=start+n,total=count,root_only=root_only)
