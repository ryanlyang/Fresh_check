"""Explicit ownership: DDP trainable bundle; frozen teacher and rank RNG outside."""
import torch
from .model import Trainable
from .teacher import Teacher
from .views import Generators
from .config import recipe
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.checkpoint import seed_everything,capture_rank_state,restore_rank_state


class Session:
    def __init__(self,arm,device,world,rank=0,*,trim=True,distributed=True,sync_bn=True):
        seed_everything(101);self.arm=arm;self.device=device;self.world=world;self.rank=rank
        self.base=Trainable(arm,trim=trim).to(device)
        if world==2 and sync_bn:self.base=torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.base)
        if world>1 and distributed:
            kwargs=dict(broadcast_buffers=False,find_unused_parameters=False,static_graph=False,gradient_as_bucket_view=True)
            if device.type=='cuda':kwargs.update(device_ids=[device.index],output_device=device.index)
            self.model=torch.nn.parallel.DistributedDataParallel(self.base,**kwargs)
        else:self.model=self.base
        self.opt=build_optimizer(self.base,recipe())
        self.scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
        self.teacher=Teacher(self.base.student) if arm=='masked' else None
        self.generators=Generators(rank) if arm=='masked' else None
        self.attempts=self.successes=self.auxiliary_attempts=0;self.last_schedule_label=0

    def state_dict(self):
        return dict(schema='masked_complete_attempt_v1',arm=self.arm,world=self.world,rank=self.rank,
            model=self.base.state_dict(),optimizer=self.opt.state_dict(),scaler=self.scaler.state_dict(),
            teacher=None if self.teacher is None else self.teacher.state_dict(),
            generators=None if self.generators is None else self.generators.state_dict(),
            attempts=self.attempts,successes=self.successes,auxiliary_attempts=self.auxiliary_attempts,
            last_schedule_label=self.last_schedule_label,rng=capture_rank_state(self.base))

    def load_state_dict(self,row,*,restore_rng=True):
        if (row['schema']!='masked_complete_attempt_v1' or row['arm']!=self.arm or row['world']!=self.world
                or row['rank']!=self.rank or any(type(row[k]) is not int or row[k]<0 for k in
                ('attempts','successes','auxiliary_attempts','last_schedule_label'))
                or row['successes']>row['attempts'] or row['auxiliary_attempts']>row['attempts']):
            raise ValueError('Wrong/incomplete full attempt state')
        self.base.load_state_dict(row['model'],strict=True);self.opt.load_state_dict(row['optimizer']);self.scaler.load_state_dict(row['scaler'])
        if self.arm=='masked':
            self.teacher.load_state_dict(row['teacher']);self.generators.load_state_dict(row['generators'])
            if (self.teacher.successes!=row['successes'] or self.teacher.buffer_copies!=row['attempts']
                    or self.generators.draws!=row['auxiliary_attempts']):raise ValueError('EMA/generator cursors disagree')
        elif row['teacher'] is not None or row['generators'] is not None:raise ValueError('Control acquired auxiliary state')
        for k in ('attempts','successes','auxiliary_attempts','last_schedule_label'):setattr(self,k,row[k])
        if restore_rng:restore_rank_state(self.base,row['rng'])
