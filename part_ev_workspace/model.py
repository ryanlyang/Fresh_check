"""Full native LW State-EV plus four pure persistent-workspace exchanges."""
import hashlib
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_ev_state_compare.model import build_model as reference_model
from .attention import Attention, StableNorm, reduction_dtype
from .config import COUNTS, SCIENCE


@contextmanager
def isolated(seed, component):
    value = int.from_bytes(hashlib.sha256(f'workspace-v1:{seed}:{component}'.encode()).digest()[:8], 'little')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(value & ((1<<63)-1))
        yield


def initialize(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight); nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight); nn.init.zeros_(child.bias)


def zero_output(module):
    nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)


class WorkspaceStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.read_query, self.read_kv = StableNorm(), StableNorm()
        self.mix_norm, self.ffn_norm = StableNorm(), StableNorm()
        self.write_query, self.write_kv = StableNorm(), StableNorm()
        self.read, self.mix, self.write = Attention(), Attention(), Attention()
        self.ffn = nn.Sequential(nn.Linear(128,512), nn.GELU(), nn.Linear(512,128))
        initialize(self); zero_output(self.write.output)

    def forward(self, hidden, workspace, valid):
        safe = hidden.masked_fill(~valid[...,None], 0)
        message, a = self.read(self.read_query(workspace), self.read_kv(safe), valid)
        z1 = workspace + message.to(workspace.dtype)
        norm = self.mix_norm(z1)
        message, b = self.mix(norm, norm)
        z2 = z1 + message.to(z1.dtype)
        z3 = z2 + self.ffn(self.ffn_norm(z2)).to(z2.dtype)
        delta, c = self.write(self.write_query(safe), self.write_kv(z3))
        correction = delta.to(hidden.dtype).masked_fill(~valid[...,None],0)
        out = hidden + correction
        health = a & b & c & torch.isfinite(z3).all() & torch.isfinite(out).all()
        return out, z3, health


class ClassRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm, self.workspace_norm = StableNorm(), StableNorm()
        self.attention = Attention()
        initialize(self); zero_output(self.attention.output)

    def forward(self, cls, workspace):
        delta, healthy = self.attention(self.query_norm(cls[:,None]), self.workspace_norm(workspace))
        fused = cls + delta[:,0].to(cls.dtype)
        return fused, healthy & torch.isfinite(fused).all()


class GlobalWorkspace(nn.Module):
    def __init__(self, seed=101):
        super().__init__()
        with isolated(seed, 'initial_tokens'):
            self.initial_tokens = nn.Parameter(torch.empty(1,8,128))
            nn.init.normal_(self.initial_tokens, std=.02)
        self.stages = nn.ModuleList()
        for i in range(4):
            with isolated(seed, f'stage_{i}'): self.stages.append(WorkspaceStage())
        with isolated(seed, 'class_read'): self.class_read = ClassRead()

    def initial(self, batch):
        return self.initial_tokens.expand(batch,-1,-1)


class WorkspaceTransformer(ParticleTransformer):
    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        if uu is not None or uu_idx is not None or self.for_inference or self.fc is None:
            raise ValueError('Workspace requires standard-relations ten-logit classification')
        try:
            with torch.no_grad():
                x,v,mask,_ = self.trimmer(x,v,mask,None)
                valid = mask[:,0].bool()
                padding = ~valid
            with torch.autocast(x.device.type, enabled=self.use_amp):
                hidden = self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                bias = self.pair_embed(v, None)
                workspace = self.global_workspace.initial(x.shape[0])
                health = valid.any(-1).all()
                for i, block in enumerate(self.blocks):
                    hidden = block(hidden, padding_mask=padding, attn_mask=bias)
                    if (i+1) in SCIENCE['stages_after']:
                        stage = self.global_workspace.stages[(i+1)//2-1]
                        args = (hidden.transpose(0,1), workspace, valid)
                        if self.training and torch.is_grad_enabled() and self.workspace_checkpointed:
                            h,workspace,ok = checkpoint(stage,*args,use_reentrant=False,preserve_rng_state=False)
                        else: h,workspace,ok = stage(*args)
                        hidden = h.transpose(0,1); health = health & ok
                cls = self.cls_token.expand(1,hidden.shape[1],-1)
                for block in self.cls_blocks: cls = block(hidden,x_cls=cls,padding_mask=padding)
                cls = self.norm(cls).squeeze(0)
                fused,ok = self.global_workspace.class_read(cls,workspace)
                logits = self.fc(fused)
                return torch.where(health & ok,logits,torch.full_like(logits,float('nan')))
        finally:
            self.pair_embed.clear()


def build_model(architecture='workspace', *, seed=101, trim=True, use_amp=False, checkpointed=True):
    if architecture not in COUNTS: raise ValueError('Unknown workspace arm')
    model = reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if architecture == 'workspace':
        if type(model.mod) is not ParticleTransformer: raise ValueError('Pinned native type changed')
        model.mod.global_workspace = GlobalWorkspace(seed)
        model.mod.workspace_checkpointed = checkpointed
        model.mod.__class__ = WorkspaceTransformer
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError('Workspace parameter inventory changed')
    return model


def inventory(model):
    return {name:dict(shape=list(p.shape),parameters=p.numel()) for name,p in model.named_parameters()}
