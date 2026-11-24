"""Scoped auxiliary BN-eval arithmetic without mutating registered B1 buffers."""
from contextlib import contextmanager
import torch
from torch import nn


def snapshot(student):return {n:b.detach().clone() for n,b in student.named_buffers()}


@contextmanager
def auxiliary(student):
    flags=[(m,m.training) for m in student.modules()]
    probabilities=[];trim=student.mod.trimmer.enabled;amp=student.mod.use_amp
    try:
        student.mod.trimmer.enabled=False
        for module in student.modules():
            if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.Dropout)):module.training=False
            if isinstance(getattr(module,'dropout',None),float):
                probabilities.append((module,'dropout',module.dropout));module.dropout=0.
            if isinstance(module,nn.Dropout):
                probabilities.append((module,'p',module.p));module.p=0.
        yield
    finally:
        for m,attribute,value in probabilities:setattr(m,attribute,value)
        for m,flag in flags:m.training=flag
        student.mod.trimmer.enabled=trim;student.mod.use_amp=amp


def assert_equal(actual,expected):
    if actual.keys()!=expected.keys() or any(not torch.equal(actual[n],expected[n]) for n in actual):
        raise ValueError('Auxiliary phase mutated clean BN buffers')


def fingerprint(buffers):
    import hashlib
    raw=torch.cat([x.detach().contiguous().reshape(-1).view(torch.uint8) for x in buffers.values()])
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()
