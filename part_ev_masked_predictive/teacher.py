"""Detached particle-only teacher, successful-step EMA and clean buffer copy."""
import copy
import torch
from .config import TEACHER,momentum
from .predictor import normalize
from .modes import snapshot
from part_ev_screen.precision import model_precision

EXCLUDED=('mod.cls_blocks.','mod.norm.','mod.fc.')


def particle_key(name):return name!='mod.cls_token' and not name.startswith(EXCLUDED)


class Teacher:
    def __init__(self,student):
        self.model=copy.deepcopy(student)
        for name in ('cls_blocks','norm','fc','cls_token'):delattr(self.model.mod,name)
        self.model.mod.trimmer.enabled=False;self.model.eval();self.model.requires_grad_(False)
        self.successes=0;self.buffer_copies=0
        expected={n:p for n,p in student.named_parameters() if particle_key(n)}
        if (set(expected)!=dict(self.model.named_parameters()).keys()
                or sum(p.numel() for p in self.model.parameters())!=TEACHER):
            raise ValueError('Teacher subset/count changed')
        self.validate_subset(student)

    def validate_subset(self,student):
        source=dict(student.named_parameters());target=dict(self.model.named_parameters())
        if set(target)!={n for n in source if particle_key(n)}:raise ValueError('Teacher keys changed')
        if any(p.requires_grad or p.grad is not None for p in target.values()):raise ValueError('Teacher acquired gradients')

    @torch.no_grad()
    def targets(self,slot,amp=True):
        self.model.eval()
        device=slot['teacher'][1].device
        with model_precision(self.model,device,training=amp):a,b=self.model(*slot['teacher'],phase='particles')
        target=normalize((normalize(a)+normalize(b))*.5)
        return target.gather(1,slot['target_ids'][...,None].expand(-1,-1,128)).detach()

    @torch.no_grad()
    def commit(self,student,u,success):
        self.validate_subset(student);parameters=dict(student.named_parameters())
        if success:
            m=momentum(u)
            for n,p in self.model.named_parameters():p.mul_(m).add_(parameters[n],alpha=1-m)
            self.successes+=1
        buffers=dict(student.named_buffers())
        for n,b in self.model.named_buffers():b.copy_(buffers[n])
        self.buffer_copies+=1

    def state_dict(self):return dict(schema='masked_particle_teacher_v1',model=self.model.state_dict(),
        successes=self.successes,buffer_copies=self.buffer_copies,keys=list(self.model.state_dict()))

    def load_state_dict(self,row):
        if (row['schema']!='masked_particle_teacher_v1' or row['keys']!=list(self.model.state_dict())
                or any(type(row[k]) is not int or row[k]<0 for k in ('successes','buffer_copies'))
                or row['successes']>row['buffer_copies']):raise ValueError('Invalid teacher state')
        self.model.load_state_dict(row['model'],strict=True);self.successes=row['successes'];self.buffer_copies=row['buffer_copies']
