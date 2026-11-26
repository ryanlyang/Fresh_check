"""Use the byte-identical benchmarked fast256 kernel with optional latent logging."""
import types
import torch
from part_ev_masked_speed.step import attempt as fast_attempt


def attempt(session, values, truth, *, inspect_latents=False, **kwargs):
    if 'microbatch' in kwargs: raise ValueError('Production auxiliary batch is fixed at 256')
    statistics = []
    def before(module, args, named):
        if args[0]=='auxiliary': return args, dict(named, diagnostic=True)
    def after(module, args, named, output):
        if args[0]=='auxiliary':
            loss, stats = output
            statistics.append(stats.detach())
            return loss
    handles = []
    try:
        if inspect_latents:
            handles.append(session.base.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(session.base.register_forward_hook(after, with_kwargs=True))
        row = fast_attempt(session, values, truth, microbatch=256, **kwargs)
    finally:
        for handle in handles: handle.remove()
    if statistics:
        row['latents'] = dict(zip(('prediction_norm','target_norm','prediction_variance','target_variance','cosine'),
                                 torch.stack(statistics).mean(0).cpu().tolist()))
    return row


def private_module(original, **overrides):
    """Rebind local function globals; never patch the imported historical module.

    Engine transactions/data/selection logic is reused exactly. Rebinding every
    local function makes calls such as train -> save -> restore use this private
    namespace, without affecting any other campaign in this process.
    """
    module = types.ModuleType(original.__name__+'_fast256_private')
    module.__dict__.update(original.__dict__)
    for key, value in original.__dict__.items():
        if isinstance(value, types.FunctionType) and value.__module__==original.__name__:
            function = types.FunctionType(value.__code__, module.__dict__, value.__name__,
                                          value.__defaults__, value.__closure__)
            function.__kwdefaults__ = value.__kwdefaults__
            module.__dict__[key] = function
    module.__dict__.update(overrides)
    return module


def engine():
    from part_ev_masked_predictive import engine as old, protocol as original
    from . import protocol
    facade = types.SimpleNamespace(**dict(original.__dict__, SCHEMA=protocol.SCHEMA,
                                         task=protocol.task, run_directory=protocol.run_directory))
    return private_module(old, P=facade, attempt=attempt)
