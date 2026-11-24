"""Path-aware native/expert FP16 versus intentional FP32-router audit."""
from contextlib import contextmanager
import torch
from torch import nn
from part_ev_screen.precision import model_precision


def modules(base):
    return {name:('router' if '.specialists.router.' in name else 'native_expert')
        for name,m in base.named_modules() if isinstance(m,(nn.Linear,nn.Conv1d))
        and not isinstance(m,nn.modules.linear.NonDynamicallyQuantizableLinear)}


@contextmanager
def audit_precision(base,*,expected_dtype):
    dtypes=sorted({str(p.dtype) for p in base.parameters()})
    if dtypes!=['torch.float32']:raise ValueError('Experts require FP32 master parameters')
    paths=modules(base)
    expected={n:'torch.float32' if family=='router' else str(expected_dtype) for n,family in paths.items()}
    result=dict(passed=False,parameter_dtypes=dtypes,expected_activation_dtype=str(expected_dtype),
        expected_by_path=expected,observed_activations={},router_paths=sorted(n for n,f in paths.items() if f=='router'))
    handles=[]
    def hook(name):
        def observe(m,inputs,out):
            actual=str(out.dtype);result['observed_activations'][name]=actual
            if actual!=expected[name]:raise RuntimeError('Expert precision audit failed at '+name+': '+actual)
        return observe
    try:
        for n,m in base.named_modules():
            if n in paths:handles.append(m.register_forward_hook(hook(n)))
        yield result
        if set(result['observed_activations'])!=set(paths):raise RuntimeError('Missing native/expert precision paths')
        result['passed']=True
    finally:
        for h in handles:h.remove()


def precision_ok(row,dtype,expected_paths):
    expected={n:'torch.float32' if f=='router' else dtype for n,f in expected_paths.items()}
    return (row.get('passed') is True and row.get('parameter_dtypes')==['torch.float32']
        and row.get('expected_activation_dtype')==dtype and row.get('expected_by_path')==expected
        and row.get('observed_activations')==expected
        and row.get('router_paths')==sorted(n for n,f in expected_paths.items() if f=='router'))
