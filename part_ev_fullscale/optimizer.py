"""Reference Ranger arithmetic with complete, modern checkpoint semantics.

Weaver 0.4.17 Lookahead.state_dict omits slow weights and its step counter.
The wrapper below retains the reference RAdam implementation and Lookahead
update, but saves all state and initializes the current torch Optimizer API.
"""
from __future__ import annotations

import copy
import torch


class Ranger(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        from weaver.utils.nn.optimizer.radam import RAdam
        params = list(params)
        super().__init__(params, dict(lr=lr))
        self.inner = RAdam(params, lr=lr, betas=(0.95, 0.999), eps=1e-5, weight_decay=0)
        self.param_groups = self.inner.param_groups
        self.slow = [p.detach().clone() for group in self.param_groups for p in group["params"]]
        self.step_counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.inner.step(closure)
        self.step_counter += 1
        if self.step_counter == 6:
            self.step_counter = 0
            for p, slow in zip(self.parameters(), self.slow):
                p.mul_(0.5).add_(slow, alpha=0.5)
                slow.copy_(p)
        return loss

    def parameters(self):
        return [p for group in self.param_groups for p in group["params"]]

    def state_dict(self):
        return {"schema": "full_ranger_v1", "inner": self.inner.state_dict(),
                "slow": self.slow, "step_counter": self.step_counter}

    def load_state_dict(self, state):
        if state.get("schema") != "full_ranger_v1" or len(state["slow"]) != len(self.parameters()):
            raise ValueError("Incomplete Ranger checkpoint; slow weights are required")
        if not 0 <= state["step_counter"] < 6:
            raise ValueError("Invalid Lookahead counter")
        self.inner.load_state_dict(copy.deepcopy(state["inner"]))
        self.param_groups = self.inner.param_groups
        self.slow = [v.detach().to(device=p.device, dtype=p.dtype).clone()
                     for p, v in zip(self.parameters(), state["slow"])]
        self.step_counter = state["step_counter"]
