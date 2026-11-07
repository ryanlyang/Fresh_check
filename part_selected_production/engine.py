"""Full-budget single-GH200 training and restartable full-validation evaluation."""
import functools
import importlib.util
from pathlib import Path
import shutil
import time
import types

from . import protocol as P


def initial_state(task):
    return dict(task=task, step=0, validated_checks=0, best_accuracy=-1., best_check=None,
                best_file=None, best_sha256=None, amp_skipped_steps=0, clipped_steps=0,
                train_seconds=0., validation_seconds=0.)


def validate_state(state, task, recipe):
    step, checks = state.get('step'), state.get('validated_checks')
    if (state.get('task') != task or type(step) is not int or type(checks) is not int or
            not 0 <= step <= recipe['total_steps'] or not 0 <= checks <= recipe['epochs'] or
            checks > step // recipe['steps_per_epoch'] or
            step > (checks + 1) * recipe['steps_per_epoch']):
        raise ValueError('Invalid training/validation stream cursor')
    if checks and (state.get('best_check') not in range(1, checks + 1) or
                   state.get('best_file') != f"selected/check_{state['best_check']:02d}.pt"):
        raise ValueError('Invalid committed best-checkpoint pointer')
    if checks == 0 and (state.get('best_file') is not None or state.get('best_check') is not None):
        raise ValueError('Selected checkpoint cannot precede validation')


def audit(output, plan, index):
    from part_screen_common.controls import selected_training
    from part_screen_common.contracts import runtime_identity
    task = plan['science']['tasks'][index]
    benchmark = P.read(Path(plan['review']['models'][index]['evidence_path']))
    return selected_training(P.run_directory(output, task), task, plan['content_hash'],
                             plan['science']['recipes'][task['recipe']],
                             runtime_identity(benchmark['runtime']))


def create_engine(output, plan, index, *, deadline=float('inf')):
    # All these imports resolve to the authenticated original benchmark source.
    from part_selected_fullscale.benchmark import Bench
    from part_selected_fullscale.steps import training_step
    import part_screen_common.engine as common
    from part_ev_screen import data as native_data
    from part_selected_screen import data as selected_data
    from scripts.run_part_ev_screen import validate
    import torch

    class Runtime(Bench):
        stopping = False

        def signal(self, *_):
            self.stopping = True

        def stop(self):
            if self.stopping or time.monotonic() >= deadline:
                raise SystemExit(75)

    runtime = Runtime(Path(plan['benchmark']), index, lambda: None)
    runtime.check_stop = runtime.stop
    expected = P.read(Path(plan['review']['models'][index]['evidence_path']))['runtime']
    if runtime.runtime != expected:
        raise ValueError('GPU/execution runtime differs from passed benchmark')
    spec = dict(plan['science'], content_hash=plan['content_hash'], authorizes_training=True)
    spec['schema'] = P.SCHEMA
    recipe = spec['recipes'][runtime.task['recipe']]
    if (recipe['total_steps'], recipe['epochs'], recipe['global_batch_size']) != (1000000, 50, 512):
        raise ValueError('Production requires the full matched recipe')

    # Private copy of the common evaluator; never monkeypatch historical modules.
    mod_spec = importlib.util.spec_from_file_location('part_screen_common._seven_production', common.__file__)
    module = importlib.util.module_from_spec(mod_spec); mod_spec.loader.exec_module(module)
    data = selected_data if index in (2, 5, 6) else native_data
    config = dict(config=Path(plan['benchmark'])/'inputs/selected.yaml') if index in (2, 5, 6) else {}
    module.finite_validation_loader = functools.partial(data.finite_validation_loader, **config)
    from part_ev_fullscale.protocol import file_lock, file_sha256, read_json, write_json
    def approval(out, current, actual):
        if (Path(out).resolve() != Path(output).resolve() or current != spec or actual != expected):
            raise ValueError('Production admission/runtime identity changed')
    adapter = types.SimpleNamespace(SCHEMA=P.SCHEMA, read_json=read_json, write_json=write_json,
        file_sha256=file_sha256, file_lock=file_lock, load_campaign=lambda _: spec,
        task_directory=lambda out, i: P.run_directory(out, spec['tasks'][i]),
        require_approval=approval)  # Evidence/source checks are done once by worker/load, before optimizer imports.

    class Engine(module.ScreenEngine):
        def gpu_setup(self, _):
            return runtime.device, runtime.runtime

        def check_stop(self):
            runtime.stop()

        def make_model(self, task, device):
            return runtime.model()

        def lock(self, out, _):
            # Per-task evaluation can start as soon as THIS training finishes.
            row = audit(out, plan, index)
            record = dict(task=runtime.task, checkpoint=row['checkpoint'], sha256=row['checkpoint_sha256'])
            path = P.run_directory(out, runtime.task)/'selection.json'
            if path.exists() and P.read(path) != record:
                raise ValueError('Selected checkpoint changed after evaluation lock')
            if not path.exists(): P.write(path, record)
            return dict(checkpoints={index: record})

    engine = Engine(adapter, None, None, None)
    engine.runtime_controller = runtime
    engine.spec, engine.task, engine.recipe = spec, runtime.task, recipe
    engine.device, engine.runtime = runtime.device, runtime.runtime
    engine.loader = runtime.loader
    engine.training_step = training_step
    # Validation uses the exact original implementation, with a private stop
    # callback so a timeout leaves the PRE-validation restart intact.
    globals_copy = dict(validate.__globals__, check_stop=engine.check_stop)
    engine.validate = types.FunctionType(validate.__code__, globals_copy, validate.__name__,
                                        validate.__defaults__, validate.__closure__)
    engine.validate.__kwdefaults__ = validate.__kwdefaults__
    engine.torch = torch
    return engine


def finish(output, plan, index, state, engine):
    from part_ev_fullscale.checkpoint import trusted_load, atomic_torch_save
    run = P.run_directory(output, engine.task)
    validate_state(state, engine.task, engine.recipe)
    if state['step'] != engine.recipe['total_steps'] or state['validated_checks'] != engine.recipe['epochs']:
        raise ValueError('Cannot finish an incomplete training budget')
    candidate = run / state['best_file']
    if P.sha(candidate) != state['best_sha256']:
        raise ValueError('Committed selected checkpoint changed')
    atomic_torch_save(trusted_load(candidate), run/'best_model_val.pt')
    P.write(run/'training_complete.json', dict(state, campaign_sha256=plan['content_hash'],
        runtime=engine.runtime, best_checkpoint_sha256=P.sha(run/'best_model_val.pt')))
    audit(output, plan, index)


def train_body(output, plan, index, engine):
    from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata
    from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state, atomic_torch_save
    from part_ev_screen.data import shutdown_loader
    from scripts.run_part_ev_screen import progress
    task, recipe, spec = engine.task, engine.recipe, engine.spec
    run = P.run_directory(output, task)
    model = engine.make_model(task, engine.device)
    optimizer = build_optimizer(model, recipe)
    scaler = engine.torch.amp.GradScaler(engine.device.type, enabled=engine.device.type == 'cuda')
    audit_path = run/'optimizer.json'
    metadata = optimizer_metadata(model, optimizer)
    if audit_path.exists() and P.read(audit_path) != metadata:
        raise ValueError('Optimizer contract changed')
    if not audit_path.exists(): P.write(audit_path, metadata)
    state, rng = initial_state(task), None
    last = run/'last.pt'
    if last.exists():
        state, rng = engine.load(last, model, optimizer, scaler, spec, engine.runtime, task)
    validate_state(state, task, recipe)
    if state['best_file'] and P.sha(run/state['best_file']) != state['best_sha256']:
        raise ValueError('Selected checkpoint referenced by restart changed')
    if state['step'] == recipe['total_steps'] and state['validated_checks'] == recipe['epochs']:
        finish(output, plan, index, state, engine)
        return

    def save():
        if shutil.disk_usage(output).free < 4*2**30:
            raise OSError('Keep 4 GiB free; previous committed restart retained')
        engine.save(last, model, optimizer, scaler, state, spec, engine.runtime)

    loaders = []
    try:
        loaders.append(engine.loader('train')); loaders.append(engine.loader('val'))
        train, val = (iter(loader) for loader in loaders)
        if rng is not None:
            engine.replay(train, state['step'])
            engine.replay(val, state['validated_checks'] * (recipe['validation_events_per_epoch']//512))
            restore_rank_state(model, rng)
        else:
            seed_everything(task['seed'])
        while True:
            try:
                engine.check_stop()
            except SystemExit:
                save()
                progress('checkpointed_for_resume', **task, step=state['step'], checks=state['validated_checks'])
                raise
            if state['step']//recipe['steps_per_epoch'] > state['validated_checks']:
                save()
                # Never catch a partial validation and commit its stream cursor.
                metric = engine.validate(model, val, engine.device, recipe['validation_events_per_epoch'])
                state['validated_checks'] += 1
                state['validation_seconds'] += metric['seconds']
                metric.update(check=state['validated_checks'], step=state['step'])
                if metric['accuracy'] > state['best_accuracy']:
                    candidate = f"selected/check_{metric['check']:02d}.pt"
                    atomic_torch_save(dict(model=model.state_dict(), task=task, runtime=engine.runtime,
                        campaign_sha256=spec['content_hash'], validation=metric), run/candidate)
                    state.update(best_accuracy=metric['accuracy'], best_check=metric['check'],
                                 best_file=candidate, best_sha256=P.sha(run/candidate))
                P.write(run/'validation'/f"check_{metric['check']:02d}.json", metric)
                save()  # Commits best pointer, RNG and validation cursor together.
                progress('validation', **task, **metric)
            if state['step'] == recipe['total_steps']:
                finish(output, plan, index, state, engine)
                progress('training_complete', **task, step=state['step'])
                return
            started = time.monotonic()
            row = engine.training_step(model, optimizer, scaler, next(train), engine.device, recipe, state['step'])
            state['train_seconds'] += time.monotonic()-started
            state['step'] += 1
            state['amp_skipped_steps'] += int(row['amp_skipped'])
            state['clipped_steps'] += int(row['clipped'])
            if state['step'] % 100 == 0:
                progress('training', **task, step=state['step'], **row,
                         jets_per_second=state['step']*512/state['train_seconds'])
            if state['step'] % P.POLICY['checkpoint_interval'] == 0:
                if state['amp_skipped_steps'] >= .05*state['step']:
                    raise RuntimeError('Excessive AMP skips; preserve restart and investigate')
                save()
    finally:
        for loader in loaders: shutdown_loader(loader)


def train(output, plan, index, engine):
    from part_ev_fullscale.protocol import file_lock
    run = P.run_directory(output, engine.task)
    with file_lock(run/'.train.lock', blocking=False):
        if (run/'training_complete.json').exists():
            audit(output, plan, index)
            return
        train_body(output, plan, index, engine)
