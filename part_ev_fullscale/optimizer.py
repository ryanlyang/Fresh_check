"""Reference Ranger arithmetic with complete, modern checkpoint semantics.

Weaver 0.4.17 Lookahead.state_dict omits slow weights and its step counter.
The wrapper below retains the reference RAdam implementation and Lookahead
update, but saves all state and initializes the current torch Optimizer API.
"""
from __future__ import annotations

import copy
import torch


ADAMW_WEIGHT_DECAY_POLICY = "exclude_bias_normalization_and_cls_token_include_edge_projection"


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


def adamw_parameter_groups(model, *, weight_decay):
    """Partition trainable FP32 parameters once; any exclusion wins for aliases.

Matrix/convolution weights and the three-dimensional EV projections decay.
Biases, normalization parameters, scalar/vector scales, and the model's explicit
``no_weight_decay()`` names do not. Registration order determines the stable
parameter ordering required by optimizer checkpoints.
"""
    explicit = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    norm_types = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
    )
    if hasattr(torch.nn, "RMSNorm"):
        norm_types += (torch.nn.RMSNorm,)
    entries, all_names = {}, set()
    # Traverse aliases deliberately: a tied weight also registered as a bias
    # or normalization parameter must not accidentally receive weight decay.
    for module_name, module in model.named_modules(remove_duplicate=False):
        for local_name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            name = f"{module_name}.{local_name}" if module_name else local_name
            all_names.add(name)
            if not parameter.requires_grad:
                continue
            if parameter.dtype != torch.float32:
                raise ValueError("AdamW requires FP32 master parameters; found {} for {}".format(
                    parameter.dtype, name))
            excluded = (
                parameter.ndim <= 1 or local_name.endswith("bias") or local_name.startswith("bias_")
                or isinstance(module, norm_types) or name in explicit
            )
            record = entries.setdefault(id(parameter), {"parameter": parameter, "excluded": False})
            record["excluded"] = record["excluded"] or excluded
    if explicit - all_names:
        raise ValueError("Unknown no_weight_decay parameter names: {}".format(sorted(explicit - all_names)))
    if not entries:
        raise ValueError("AdamW requires at least one trainable parameter")
    groups = []
    for excluded, name in ((False, "decay"), (True, "no_decay")):
        params = [record["parameter"] for record in entries.values() if record["excluded"] == excluded]
        if params:
            groups.append({"params": params, "weight_decay": 0.0 if excluded else float(weight_decay),
                           "group_name": name})
    return groups


def build_optimizer(model, recipe):
    """Construct the optimizer selected by the already-frozen campaign recipe."""
    name = recipe["optimizer"]
    if name == "Weaver_RAdam_plus_Lookahead":
        # Preserve the historical ungrouped parameter sequence and arithmetic.
        return Ranger(model.parameters())
    if name == "AdamW":
        if recipe.get("weight_decay_policy") != ADAMW_WEIGHT_DECAY_POLICY:
            raise ValueError("AdamW weight_decay_policy differs from the implemented policy")
        return torch.optim.AdamW(
            adamw_parameter_groups(model, weight_decay=recipe["weight_decay"]),
            lr=recipe["start_lr"], betas=tuple(recipe["betas"]), eps=recipe["eps"],
            weight_decay=recipe["weight_decay"],
        )
    raise ValueError("Unknown optimizer in campaign recipe: {}".format(name))


def optimizer_metadata(model, optimizer):
    """JSON-safe parameter-group audit without touching gradients or state."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        groups.append({
            "name": group.get("group_name", f"group_{index}"),
            "parameter_names": [names[id(parameter)] for parameter in parameters],
            "parameter_tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
            "weight_decay": group["weight_decay"], "lr": group["lr"],
            "betas": list(group["betas"]), "eps": group["eps"],
        })
    return {"optimizer": type(optimizer).__name__, "groups": groups,
            "parameters": sum(group["parameters"] for group in groups)}
