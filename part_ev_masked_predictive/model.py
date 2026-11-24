"""Explicit native-compatible clean/particle forwards; original student keys."""
from contextlib import contextmanager
import hashlib
import torch
from torch import nn
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_ev_fullscale.model import ParticleTransformerWrapper
from part_ev_state_compare.model import build_model as native_model
from .config import COUNTS,TRAINABLE


@contextmanager
def isolated(component,seed=101):
    value=int.from_bytes(hashlib.sha256(f'masked-v1:{seed}:{component}'.encode()).digest()[:8],'little')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(value&((1<<63)-1));yield


class MaskedTransformer(ParticleTransformer):
    def forward(self,x,v=None,mask=None,uu=None,uu_idx=None,*,phase='classify',retain=False):
        if phase not in ('classify','particles') or uu is not None or uu_idx is not None:
            raise ValueError('Standard particle input required')
        try:
            with torch.no_grad():
                x,v,mask,_=self.trimmer(x,v,mask,None);padding=~mask[:,0]
                kept=(x.detach(),v.detach(),mask.detach()) if retain else None
            with torch.autocast(x.device.type,dtype=torch.float16,enabled=self.use_amp):
                h=self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                bias=self.pair_embed(v,None);h6=None
                for i,b in enumerate(self.blocks):
                    h=b(h,x_cls=None,padding_mask=padding,attn_mask=bias)
                    if i==5:h6=h
                if phase=='particles':return h6.transpose(0,1),h.transpose(0,1)
                cls=self.cls_token.expand(1,h.shape[1],-1)
                for b in self.cls_blocks:cls=b(h,x_cls=cls,padding_mask=padding)
                logits=self.fc(self.norm(cls).squeeze(0))
                return (logits,kept) if retain else logits
        finally:self.pair_embed.clear()


class MaskedStudent(ParticleTransformerWrapper):
    def forward(self,points,features,vectors,mask,*,phase='classify',retain=False):
        try:return self.mod(features,v=vectors,mask=mask,phase=phase,retain=retain)
        finally:self.mod.pair_embed.clear()


def build_model(architecture='masked',*,seed=101,trim=True,use_amp=False):
    if architecture not in COUNTS:raise ValueError('Unknown masked arm')
    model=native_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    model.mod.__class__=MaskedTransformer;model.__class__=MaskedStudent
    if sum(p.numel() for p in model.parameters())!=COUNTS[architecture]:raise ValueError('Student count changed')
    return model


class Trainable(nn.Module):
    """Only student/predictor enter DDP. Every trainable phase includes anchors."""
    def __init__(self,arm='masked',*,trim=True):
        super().__init__();self.arm=arm;self.student=build_model(arm,trim=trim)
        if arm=='masked':
            from .predictor import Predictor
            with isolated('predictor'):self.predictor=Predictor()
        if sum(p.numel() for p in self.parameters())!=TRAINABLE[arm]:raise ValueError('Trainable count changed')

    def anchor(self):return torch.stack([p.reshape(-1)[0] for p in self.parameters()]).sum()*0

    def forward(self,phase,values=None,*,locations=None,query_valid=None,targets=None,buffers=None,retain=False,diagnostic=False):
        anchor=self.anchor()
        if phase=='clean':
            logits,retained=self.student(*values,retain=True) if retain else (self.student(*values),None)
            return logits,retained,anchor
        if phase=='flush':return anchor
        if phase!='auxiliary' or self.arm!='masked':raise ValueError('Invalid trainable phase')
        from torch.func import functional_call
        from .predictor import jet_losses
        params=dict(self.student.named_parameters())
        _,h=functional_call(self.student,(params,buffers),tuple(values),dict(phase='particles'),strict=True)
        prediction,ok=self.predictor(locations,h,values[3][:,0].bool(),query_valid)
        losses=jet_losses(prediction,targets,query_valid)+anchor
        result=torch.where(ok,losses,torch.full_like(losses,float('nan')))
        if not diagnostic:return result
        with torch.no_grad():
            p=prediction[query_valid].float();t=targets[query_valid].float()
            statistics=torch.stack((p.norm(dim=-1).mean(),t.norm(dim=-1).mean(),p.var(0,unbiased=False).mean(),
                t.var(0,unbiased=False).mean(),torch.nn.functional.cosine_similarity(p,t).mean()))
        return result,statistics


def inventory(model):return {n:dict(shape=list(p.shape),parameters=p.numel()) for n,p in model.named_parameters()}
