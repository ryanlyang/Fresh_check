"""Transactional full-budget training; reuse authenticated numerical kernels."""
import importlib.util
import math
from pathlib import Path
import shutil
import signal
import time
import types
from . import protocol as P


def initial_state(task):
    return dict(task=task, step=0, validated_checks=0, best_accuracy=-1., best_check=None,
        best_file=None, best_sha256=None, amp_skipped_steps=0, clipped_steps=0,
        train_seconds=0., validation_seconds=0., length_histogram={}, input_fingerprints={})


def validate_state(state, task, recipe):
    step, checks = state.get('step'), state.get('validated_checks')
    if (state.get('task') != task or type(step) is not int or type(checks) is not int
            or not 0 <= step <= recipe['total_steps'] or not 0 <= checks <= recipe['epochs']
            or checks > step//recipe['steps_per_epoch'] or step > (checks+1)*recipe['steps_per_epoch']):
        raise ValueError('Invalid training/validation stream cursor')
    if checks:
        if (state.get('best_check') not in range(1,checks+1)
                or state.get('best_file') != f"selected/check_{state['best_check']:02d}.pt"):
            raise ValueError('Invalid selected-checkpoint pointer')
    elif state.get('best_file') is not None or state.get('best_check') is not None:
        raise ValueError('Selected checkpoint before validation')
    hist = state.get('length_histogram', {})
    if (sum(hist.values()) != step or any(str(int(n)) != n or not 1 <= int(n) <= 128
            or type(count) is not int or count <= 0 for n,count in hist.items())):
        raise ValueError('Invalid realized-compute histogram/cursor')
    if not 0 <= state['amp_skipped_steps'] <= step: raise ValueError('Invalid AMP-skip count')


def compute_accounting(state, curve, batch=512):
    from part_ev_compact.flops import mean_cost
    steps = state['step']
    cost = mean_cost(curve['coefficients']['training'], state['length_histogram'],
                    singleton_cost=curve['samples']['1']['training']['per_jet']) if steps else 0
    return dict(convention=curve['convention'], attempted_updates=steps,
        successful_updates=steps-state['amp_skipped_steps'], exposures=steps*batch,
        length_histogram=state['length_histogram'], dense_training_flops=float(cost*steps*batch),
        includes_amp_skipped_forward_backward=True, includes_validation=False,
        note='Cost-curve accounting over observed lengths; excludes optimizer, non-dense operations and replay.')


def audit(output, plan, index):
    from part_screen_common.controls import selected_training
    task = plan['science']['tasks'][index]; arch = task['architecture']
    row = selected_training(P.run_directory(output,task),task,plan['content_hash'],
        plan['science']['recipes'][task['recipe']],plan['runtime_by_architecture'][arch])
    validate_state(row['complete'],task,plan['science']['recipes'][task['recipe']])
    if row['optimizer']['parameters'] != plan['science']['parameters'][arch]:
        raise ValueError('Optimizer parameter count differs')
    if row['complete']['input_fingerprints'] != plan['benchmark_spec']['input_reference']['fingerprints']:
        raise ValueError('Production native stream fingerprints differ')
    expected = compute_accounting(row['complete'],plan['flop_curves'][arch])
    if row['complete'].get('compute') != expected: raise ValueError('Realized compute audit differs')
    return row


def gpu_setup(plan, index):
    import platform
    import torch
    task = plan['science']['tasks'][index]
    expected = plan['runtime_by_architecture'][task['architecture']]
    if (platform.machine() != 'aarch64' or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != expected['gpu']
            or torch.cuda.get_device_properties(0).total_memory != expected['gpu_memory_bytes']):
        raise ValueError('Require one GH200 matching the passed benchmark')
    torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda:0'), dict(expected)


def create_engine(output, plan, index, *, deadline=float('inf')):
    import torch
    import part_screen_common.engine as common
    from part_ev_compact.model import build_model
    from part_ev_fullscale.data import build_loader
    from part_ev_fullscale.protocol import file_lock
    from part_selected_fullscale.steps import training_step
    from scripts.run_part_ev_screen import validate
    task = plan['science']['tasks'][index]
    device, runtime = gpu_setup(plan,index)
    spec = dict(plan['science'],content_hash=plan['content_hash'],authorizes_training=True,schema=P.SCHEMA)
    # A private evaluator namespace; historical module globals remain unchanged.
    module_spec = importlib.util.spec_from_file_location('part_screen_common._compact_production',common.__file__)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    def approval(out,current,actual):
        if Path(out).resolve() != Path(output).resolve() or current != spec or actual != runtime:
            raise ValueError('Production evaluation identity changed')
    adapter = types.SimpleNamespace(SCHEMA=P.SCHEMA,read_json=P.read,write_json=P.write,
        file_sha256=P.sha,file_lock=file_lock,load_campaign=lambda _:spec,
        task_directory=lambda out,i:P.run_directory(out,spec['tasks'][i]),require_approval=approval)
    class Engine(module.ScreenEngine):
        def gpu_setup(self,_): return device,runtime
        def check_stop(self):
            if self.stopping or time.monotonic() >= deadline: raise SystemExit(75)
        def make_model(self,task,device):
            from part_ev_fullscale.checkpoint import seed_everything
            seed_everything(task['seed'])
            model = build_model(task['architecture'],seed=task['seed'],trim=True,use_amp=False).to(device)
            if sum(p.numel() for p in model.parameters()) != spec['parameters'][task['architecture']]:
                raise ValueError('Model parameter count differs')
            return model
        def loader(self,split):
            return build_loader(split,spec['inventory'],512,seed=task['seed'],workers=2)[0]
        def observe(self,model):
            def length(_m,_args,result): self.last_length = result[2].shape[-1]
            return model.mod.trimmer.register_forward_hook(length)
        def lock(self,out,_):
            selected = audit(out,plan,index)
            row = dict(task=task,checkpoint=selected['checkpoint'],sha256=selected['checkpoint_sha256'])
            path = P.run_directory(out,task)/'selection.json'
            if path.exists() and P.read(path) != row: raise ValueError('Selected checkpoint lock changed')
            if not path.exists(): P.write(path,row)
            return dict(checkpoints={index:row})
    engine = Engine(adapter,None,None,None)
    engine.spec,engine.task,engine.recipe = spec,task,spec['recipes'][task['recipe']]
    engine.device,engine.runtime,engine.torch = device,runtime,torch
    engine.training_step = training_step
    engine.validate = types.FunctionType(validate.__code__,dict(validate.__globals__,check_stop=engine.check_stop),
        validate.__name__,validate.__defaults__,validate.__closure__)
    engine.validate.__kwdefaults__ = validate.__kwdefaults__
    for name in ('SIGTERM','SIGUSR1'):
        if hasattr(signal,name): signal.signal(getattr(signal,name),engine.signal)
    return engine


def finish(output,plan,index,state,engine):
    from part_ev_fullscale.checkpoint import trusted_load,atomic_torch_save
    validate_state(state,engine.task,engine.recipe)
    if state['step'] != engine.recipe['total_steps'] or state['validated_checks'] != engine.recipe['epochs']:
        raise ValueError('Incomplete training cannot authorize evaluation')
    run = P.run_directory(output,engine.task)
    candidate = run/state['best_file']
    if P.sha(candidate) != state['best_sha256']: raise ValueError('Selected checkpoint changed')
    atomic_torch_save(trusted_load(candidate),run/'best_model_val.pt')
    P.write(run/'training_complete.json',dict(state,campaign_sha256=plan['content_hash'],runtime=engine.runtime,
        best_checkpoint_sha256=P.sha(run/'best_model_val.pt'),
        compute=compute_accounting(state,plan['flop_curves'][engine.task['architecture']])))
    audit(output,plan,index)


def train_body(output,plan,index,engine):
    from part_ev_fullscale.optimizer import build_optimizer,optimizer_metadata
    from part_ev_fullscale.checkpoint import seed_everything,restore_rank_state,atomic_torch_save
    from part_ev_screen.data import shutdown_loader,batch_fingerprint
    from scripts.run_part_ev_screen import progress
    task,recipe,spec = engine.task,engine.recipe,engine.spec
    run = P.run_directory(output,task)
    model = engine.make_model(task,engine.device)
    optimizer = build_optimizer(model,recipe)
    scaler = engine.torch.amp.GradScaler(engine.device.type,enabled=engine.device.type=='cuda',
                                       init_scale=P.POLICY['initial_amp_scale'])
    metadata = optimizer_metadata(model,optimizer)
    if (run/'optimizer.json').exists() and P.read(run/'optimizer.json') != metadata:
        raise ValueError('Optimizer contract changed')
    if not (run/'optimizer.json').exists(): P.write(run/'optimizer.json',metadata)
    last = run/'last.pt'; state,rng = initial_state(task),None
    if last.exists(): state,rng = engine.load(last,model,optimizer,scaler,spec,engine.runtime,task)
    validate_state(state,task,recipe)
    expected_fingerprints = plan['benchmark_spec']['input_reference']['fingerprints']
    if state['input_fingerprints'] != {k:v for k,v in expected_fingerprints.items() if int(k)<state['step']}:
        raise ValueError('Restart native stream evidence differs')
    if state['best_file'] and P.sha(run/state['best_file']) != state['best_sha256']:
        raise ValueError('Selected checkpoint referenced by restart changed')
    if state['step']==recipe['total_steps'] and state['validated_checks']==recipe['epochs']:
        finish(output,plan,index,state,engine); return
    def save():
        if shutil.disk_usage(output).free < 4*2**30:
            raise OSError('Keep 4 GiB free; previous committed checkpoint retained')
        validate_state(state,task,recipe)
        engine.save(last,model,optimizer,scaler,state,spec,engine.runtime)
    loaders=[]; hook=engine.observe(model)
    try:
        loaders.append(engine.loader('train')); loaders.append(engine.loader('val'))
        train,val = (iter(loader) for loader in loaders)
        if rng is not None:
            # If interrupted here, never overwrite last with pre-restoration RNG.
            engine.replay(train,state['step'])
            engine.replay(val,state['validated_checks']*(recipe['validation_events_per_epoch']//512))
            restore_rank_state(model,rng)
        else: seed_everything(task['seed'])
        while True:
            try: engine.check_stop()
            except SystemExit:
                save(); progress('checkpointed_for_resume',**task,step=state['step'],checks=state['validated_checks'])
                raise
            if state['step']//recipe['steps_per_epoch'] > state['validated_checks']:
                save()  # A partial validation must leave this cursor/RNG intact.
                metric = engine.validate(model,val,engine.device,recipe['validation_events_per_epoch'],check_precision=True)
                if (not math.isfinite(metric['accuracy']) or not 0 <= metric['accuracy'] <= 1
                        or metric['events'] != recipe['validation_events_per_epoch'] or not metric['precision']['passed']):
                    raise ValueError('Invalid validation metric/precision')
                state['validated_checks'] += 1; state['validation_seconds'] += metric['seconds']
                metric.update(check=state['validated_checks'],step=state['step'])
                if metric['accuracy'] > state['best_accuracy']:
                    candidate = f"selected/check_{metric['check']:02d}.pt"
                    atomic_torch_save(dict(model=model.state_dict(),task=task,runtime=engine.runtime,
                        campaign_sha256=spec['content_hash'],validation=metric),run/candidate)
                    state.update(best_accuracy=metric['accuracy'],best_check=metric['check'],
                        best_file=candidate,best_sha256=P.sha(run/candidate))
                P.write(run/'validation'/f"check_{metric['check']:02d}.json",metric)
                save()  # Commit selected pointer, all RNG, and val cursor together.
                progress('validation',**task,**metric)
            if state['step']==recipe['total_steps']:
                finish(output,plan,index,state,engine)
                progress('training_complete',**task,step=state['step']); return
            started=time.monotonic(); batch=next(train); key=str(state['step'])
            if key in expected_fingerprints:
                actual=batch_fingerprint(batch)
                if actual != expected_fingerprints[key]: raise ValueError('Native input fingerprint differs at '+key)
                state['input_fingerprints'][key]=actual
            engine.last_length=None
            row=engine.training_step(model,optimizer,scaler,batch,engine.device,recipe,state['step'],
                                     check_precision=state['step']==0)
            if state['step']==0 and not row['precision']['passed']: raise ValueError('Training precision differs')
            length=engine.last_length
            if type(length) is not int or not 1 <= length <= 128: raise ValueError('Missing trimmer length')
            state['length_histogram'][str(length)] = state['length_histogram'].get(str(length),0)+1
            state['train_seconds'] += time.monotonic()-started; state['step'] += 1
            state['amp_skipped_steps'] += int(row['amp_skipped']); state['clipped_steps'] += int(row['clipped'])
            if state['step']%100==0:
                progress('training',**task,step=state['step'],**row,validated_checks=state['validated_checks'],
                    jets_per_second=state['step']*512/state['train_seconds'],amp_skipped_steps=state['amp_skipped_steps'])
            if state['step']%P.POLICY['checkpoint_interval']==0:
                if state['amp_skipped_steps'] >= .05*state['step']: raise RuntimeError('Excessive AMP skips')
                save()
    finally:
        hook.remove()
        for loader in loaders: shutdown_loader(loader)


def train(output,plan,index,engine):
    from part_ev_fullscale.protocol import file_lock
    run=P.run_directory(output,engine.task)
    with file_lock(run/'.train.lock',blocking=False):
        if (run/'training_complete.json').exists(): audit(output,plan,index); return
        train_body(output,plan,index,engine)
