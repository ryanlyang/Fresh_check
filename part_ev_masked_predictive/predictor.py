"""Two cross-attention predictor blocks; targets never attend one another."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import Attention,StableNorm,reduction_dtype


def normalize(x):
    with torch.autocast(x.device.type,enabled=False):return F.layer_norm(x.to(reduction_dtype(x)),(128,),eps=1e-5)


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.query_norm=StableNorm();self.source_norm=StableNorm();self.ffn_norm=StableNorm()
        self.attention=Attention(heads=4)
        self.ffn=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))

    def forward(self,q,h,valid,qv):
        delta,ok=self.attention(self.query_norm(q),self.source_norm(h),valid,query_valid=qv)
        q=(q+delta.to(q.dtype)).masked_fill(~qv[...,None],0)
        q=(q+self.ffn(self.ffn_norm(q)).to(q.dtype)).masked_fill(~qv[...,None],0)
        return q,ok&torch.isfinite(q).all()


class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.mask_vector=nn.Parameter(torch.empty(128))
        self.position=nn.Sequential(nn.Linear(3,128),nn.GELU(),nn.Linear(128,128))
        self.blocks=nn.ModuleList([Block(),Block()]);self.output=nn.Linear(128,128)
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
        nn.init.normal_(self.mask_vector,std=.02)

    def forward(self,locations,h,valid,qv):
        h=h.masked_fill(~valid[...,None],0)
        locations=locations.masked_fill(~qv[...,None],0)
        q=(self.mask_vector+self.position(locations).to(self.mask_vector.dtype)).masked_fill(~qv[...,None],0)
        ok=torch.isfinite(q).all()
        for block in self.blocks:q,health=block(q,h,valid,qv);ok=ok&health
        result=normalize(self.output(q)).masked_fill(~qv[...,None],0)
        return result,ok&torch.isfinite(result).all()


def jet_losses(prediction,target,valid):
    with torch.autocast(prediction.device.type,enabled=False):
        p=prediction.to(reduction_dtype(prediction)).masked_fill(~valid[...,None],0)
        t=target.detach().masked_fill(~valid[...,None],0)
        loss=F.smooth_l1_loss(p,t,beta=1.,reduction='none').mean(-1)
        return loss.masked_fill(~valid,0).sum(-1)/valid.sum(-1).clamp_min(1)
