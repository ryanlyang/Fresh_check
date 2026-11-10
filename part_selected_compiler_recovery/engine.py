"""Private backend-loader injection; original models, steps and checkpoints reused."""
import functools
import importlib.util
from pathlib import Path
import time
import types


def admitted_bench(base, plan, reference, fast):
    """Do not monkeypatch Bench, the reference loader, sys.modules, or manifests."""
    old_directory = (Path(plan['science']['reference']['path'])/'backend').resolve()
    fast_directory = (Path(plan['benchmark'])/'backend').resolve()

    def loader(expected, module):
        def load(path):
            if Path(path).resolve() != expected:
                raise ValueError('Recovery backend requested for a different directory')
            return module
        return load

    init = base.__init__
    private_init = types.FunctionType(init.__code__, dict(init.__globals__,
        reference_backend=loader(old_directory, reference), load_backend=loader(fast_directory, fast)),
        init.__name__, init.__defaults__, init.__closure__)
    private_init.__kwdefaults__ = init.__kwdefaults__
    return type('AdmittedBench', (base,), {'__init__': private_init})


def create_engine(output, plan, index, reference, fast, *, deadline=float('inf')):
    # Mirrors the original production composition, with only the private Bench
    # constructor's two already-verified loader bindings substituted. Every
    # scientific method, training loop, checkpoint and metric implementation
    # continues to come from the authenticated original snapshot.
    from part_selected_production import protocol as P
    from part_selected_production.engine import audit
    from part_selected_fullscale.benchmark import Bench
    from part_selected_fullscale.steps import training_step
    import part_screen_common.engine as common
    from part_selected_screen import data
    from scripts.run_part_ev_screen import validate
    import torch
    if index not in (2, 5, 6):
        raise ValueError('Recovery cannot run non-selected tasks')
    base = admitted_bench(Bench, plan, reference, fast)

    class Runtime(base):
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
    spec = dict(plan['science'], content_hash=plan['content_hash'], authorizes_training=True, schema=P.SCHEMA)
    recipe = spec['recipes'][runtime.task['recipe']]
    if (recipe['total_steps'], recipe['epochs'], recipe['global_batch_size']) != (1000000, 50, 512):
        raise ValueError('Production requires the full matched recipe')
    mod_spec = importlib.util.spec_from_file_location('part_screen_common._compiler_recovery', common.__file__)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    module.finite_validation_loader = functools.partial(data.finite_validation_loader,
        config=Path(plan['benchmark'])/'inputs/selected.yaml')
    from part_ev_fullscale.protocol import file_lock, file_sha256, read_json, write_json

    def approval(out, current, actual):
        if Path(out).resolve() != Path(output).resolve() or current != spec or actual != expected:
            raise ValueError('Production admission/runtime identity changed')

    adapter = types.SimpleNamespace(SCHEMA=P.SCHEMA, read_json=read_json, write_json=write_json,
        file_sha256=file_sha256, file_lock=file_lock, load_campaign=lambda _: spec,
        task_directory=lambda out, i: P.run_directory(out, spec['tasks'][i]), require_approval=approval)

    class Engine(module.ScreenEngine):
        def gpu_setup(self, _):
            return runtime.device, runtime.runtime

        def check_stop(self):
            runtime.stop()

        def make_model(self, task, device):
            return runtime.model()

        def lock(self, out, _):
            row = audit(out, plan, index)
            record = dict(task=runtime.task, checkpoint=row['checkpoint'], sha256=row['checkpoint_sha256'])
            path = P.run_directory(out, runtime.task)/'selection.json'
            if path.exists() and P.read(path) != record:
                raise ValueError('Selected checkpoint changed after evaluation lock')
            if not path.exists():
                P.write(path, record)
            return dict(checkpoints={index: record})

    engine = Engine(adapter, None, None, None)
    engine.runtime_controller = runtime
    engine.spec, engine.task, engine.recipe = spec, runtime.task, recipe
    engine.device, engine.runtime = runtime.device, runtime.runtime
    engine.loader = runtime.loader
    engine.training_step = training_step
    engine.validate = types.FunctionType(validate.__code__, dict(validate.__globals__, check_stop=engine.check_stop),
        validate.__name__, validate.__defaults__, validate.__closure__)
    engine.validate.__kwdefaults__ = validate.__kwdefaults__
    engine.torch = torch
    return engine
