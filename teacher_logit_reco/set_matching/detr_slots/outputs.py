"""Loss-facing output contract for DETR/free-slot reconstructors."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping


DETR_SLOT_OUTPUT_STEP = "detr_free_slot_step3_output_contract"
DETR_SLOT_OUTPUT_CONTRACT = "tokens_loss_features_core_outputs_existence_logits_slot_mask_aux_v2"


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("DETR slot output objects require PyTorch")
    return torch


def _as_float_tensor(value, *, name: str, device=None, dtype=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype or torch.float32)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return tensor


def _as_bool_tensor(value, *, name: str, device=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device) if device is not None else value
    else:
        tensor = torch.as_tensor(value, device=device)
    return tensor.bool()


def _scalar_float(value) -> float:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            raise ValueError("expected scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)


def _jsonable_aux(mapping: Mapping[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        name = str(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, torch.Tensor) and int(value.numel()) == 1:
            out[name] = _scalar_float(value)
    return out


@dataclass
class DetrSlotOutput:
    """Validated DETR/free-slot reconstruction output.

    Shapes are deliberately strict:

    ```text
    tokens:           [B, K, F]
    existence_logits: [B, K]
    slot_mask:        [B, K]
    ```

    `slot_mask` marks which learned slots are valid candidates.  It is not the
    same thing as predicted existence; existence is learned through
    `existence_logits`.
    """

    tokens: Any
    existence_logits: Any
    slot_mask: Any
    aux: Mapping[str, Any] | None = field(default_factory=dict)
    loss_features: Any | None = None
    core_outputs: Any | None = None
    aux_outputs: Any | None = None

    def __post_init__(self) -> None:
        tokens = _as_float_tensor(self.tokens, name="tokens")
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [batch, slots, features], got {tuple(tokens.shape)}")
        if int(tokens.shape[0]) <= 0 or int(tokens.shape[1]) <= 0 or int(tokens.shape[2]) <= 0:
            raise ValueError(f"tokens dimensions must all be positive, got {tuple(tokens.shape)}")
        existence_logits = _as_float_tensor(
            self.existence_logits,
            name="existence_logits",
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if existence_logits.ndim != 2:
            raise ValueError(
                f"existence_logits must have shape [batch, slots], got {tuple(existence_logits.shape)}"
            )
        expected = tuple(tokens.shape[:2])
        if tuple(existence_logits.shape) != expected:
            raise ValueError(f"existence_logits shape {tuple(existence_logits.shape)} does not match {expected}")
        slot_mask = _as_bool_tensor(self.slot_mask, name="slot_mask", device=tokens.device)
        if slot_mask.ndim != 2:
            raise ValueError(f"slot_mask must have shape [batch, slots], got {tuple(slot_mask.shape)}")
        if tuple(slot_mask.shape) != expected:
            raise ValueError(f"slot_mask shape {tuple(slot_mask.shape)} does not match {expected}")
        aux = dict(self.aux or {})
        loss_features = tokens
        if self.loss_features is not None:
            loss_features = _as_float_tensor(
                self.loss_features,
                name="loss_features",
                device=tokens.device,
                dtype=tokens.dtype,
            )
            if loss_features.ndim != 3:
                raise ValueError(
                    f"loss_features must have shape [batch, slots, features], got {tuple(loss_features.shape)}"
                )
            if tuple(loss_features.shape[:2]) != expected:
                raise ValueError(f"loss_features leading shape {tuple(loss_features.shape[:2])} does not match {expected}")
            if int(loss_features.shape[2]) != int(tokens.shape[2]):
                raise ValueError(
                    f"loss_features feature dim {int(loss_features.shape[2])} does not match tokens {int(tokens.shape[2])}"
                )
        core_outputs = None
        if self.core_outputs is not None:
            core_outputs = _as_float_tensor(
                self.core_outputs,
                name="core_outputs",
                device=tokens.device,
                dtype=tokens.dtype,
            )
            if core_outputs.ndim != 3 or int(core_outputs.shape[-1]) != 4:
                raise ValueError(f"core_outputs must have shape [batch, slots, 4], got {tuple(core_outputs.shape)}")
            if tuple(core_outputs.shape[:2]) != expected:
                raise ValueError(f"core_outputs leading shape {tuple(core_outputs.shape[:2])} does not match {expected}")
        aux_outputs = None
        if self.aux_outputs is not None:
            aux_outputs = _as_float_tensor(
                self.aux_outputs,
                name="aux_outputs",
                device=tokens.device,
                dtype=tokens.dtype,
            )
            if aux_outputs.ndim != 3:
                raise ValueError(f"aux_outputs must have shape [batch, slots, aux], got {tuple(aux_outputs.shape)}")
            if tuple(aux_outputs.shape[:2]) != expected:
                raise ValueError(f"aux_outputs leading shape {tuple(aux_outputs.shape[:2])} does not match {expected}")
            if int(aux_outputs.shape[2]) < 0:
                raise ValueError(f"aux_outputs last dimension must be non-negative, got {tuple(aux_outputs.shape)}")

        self.tokens = tokens
        self.existence_logits = existence_logits
        self.slot_mask = slot_mask
        self.aux = aux
        self.loss_features = loss_features
        self.core_outputs = core_outputs
        self.aux_outputs = aux_outputs

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.tokens.shape[2])

    @property
    def device(self):
        return self.tokens.device

    @property
    def dtype(self):
        return self.tokens.dtype

    @property
    def candidate_mask(self):
        """Alias used by the existing set-matching loss vocabulary."""

        return self.slot_mask

    @property
    def predicted_features(self):
        """Cache/export-compatible raw particle tokens."""

        return self.tokens

    @property
    def export_features(self):
        """Hard-sanitized raw tokens intended for cache/export."""

        return self.tokens

    @property
    def loss_predicted_features(self):
        """Smooth loss-facing particle features used by ``to_loss_kwargs``."""

        return self.loss_features

    @property
    def candidate_weights(self):
        """Compatibility with the existing reconstructed-view cache writer."""

        return self.masked_existence_probabilities()

    @property
    def diagnostics(self) -> dict[str, float]:
        """Compatibility with the existing set-matching train/cache loops."""

        return self.detached_float_diagnostics()

    def existence_probabilities(self):
        torch = require_torch()
        return torch.sigmoid(self.existence_logits)

    def masked_existence_probabilities(self):
        return self.existence_probabilities() * self.slot_mask.to(dtype=self.dtype)

    def active_slot_counts(self):
        return self.slot_mask.to(dtype=self.dtype).sum(dim=1)

    def expected_particle_counts(self):
        return self.masked_existence_probabilities().sum(dim=1)

    def to_loss_kwargs(
        self,
        *,
        offline_features=None,
        offline_mask=None,
        hlt_features=None,
        hlt_mask=None,
        include_aux_logits: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "predicted_features": self.loss_predicted_features,
            "existence_logits": self.existence_logits,
            "candidate_mask": self.slot_mask,
        }
        if bool(include_aux_logits) and self.aux_outputs is not None:
            payload["predicted_aux_logits"] = self.aux_outputs
        if offline_features is not None:
            payload["offline_features"] = offline_features
        if offline_mask is not None:
            payload["offline_mask"] = offline_mask
        if hlt_features is not None:
            payload["hlt_features"] = hlt_features
        if hlt_mask is not None:
            payload["hlt_mask"] = hlt_mask
        return payload

    def shape_report(self) -> dict[str, Any]:
        return {
            "tokens_shape": list(self.tokens.shape),
            "existence_logits_shape": list(self.existence_logits.shape),
            "slot_mask_shape": list(self.slot_mask.shape),
            "loss_features_shape": list(self.loss_features.shape),
            "core_outputs_shape": None if self.core_outputs is None else list(self.core_outputs.shape),
            "aux_outputs_shape": None if self.aux_outputs is None else list(self.aux_outputs.shape),
            "batch_size": self.batch_size,
            "num_slots": self.num_slots,
            "feature_dim": self.feature_dim,
            "contract": DETR_SLOT_OUTPUT_CONTRACT,
        }

    def detached_float_diagnostics(self) -> dict[str, float]:
        torch = require_torch()
        probs = self.existence_probabilities().detach()
        mask = self.slot_mask.detach()
        active_probs = probs[mask]
        if int(active_probs.numel()) == 0:
            active_mean = probs.sum() * 0.0
            active_max = probs.sum() * 0.0
        else:
            active_mean = active_probs.mean()
            active_max = active_probs.max()
        counts = self.active_slot_counts().detach()
        expected = self.expected_particle_counts().detach()
        diagnostics = {
            "slot_count": float(self.num_slots),
            "feature_dim": float(self.feature_dim),
            "candidate_slot_fraction": _scalar_float(mask.float().mean()),
            "candidate_slot_count_mean": _scalar_float(counts.mean()),
            "candidate_slot_count_min": _scalar_float(counts.min()) if counts.numel() else 0.0,
            "candidate_slot_count_max": _scalar_float(counts.max()) if counts.numel() else 0.0,
            "existence_probability_mean": _scalar_float(probs.mean()),
            "existence_probability_active_mean": _scalar_float(active_mean),
            "existence_probability_active_max": _scalar_float(active_max),
            "expected_particle_count_mean": _scalar_float(expected.mean()),
            "loss_features_are_distinct_from_export": float(self.loss_features is not self.tokens),
        }
        if self.core_outputs is not None:
            core = self.core_outputs.detach()
            diagnostics["core_output_abs_mean"] = _scalar_float(core.abs().mean())
            diagnostics["core_output_abs_max"] = _scalar_float(core.abs().max())
        if self.aux_outputs is not None:
            aux_outputs = self.aux_outputs.detach()
            diagnostics["aux_output_abs_mean"] = _scalar_float(aux_outputs.abs().mean())
            diagnostics["aux_output_abs_max"] = _scalar_float(aux_outputs.abs().max())
        for key, value in _jsonable_aux(self.aux).items():
            if isinstance(value, (int, float, bool)):
                diagnostics[f"aux_{key}"] = float(value)
        if not torch.isfinite(torch.as_tensor(list(diagnostics.values()), dtype=torch.float32)).all():
            raise FloatingPointError("DETR slot output diagnostics contain non-finite values")
        return diagnostics


def validate_detr_slot_output(output: DetrSlotOutput) -> DetrSlotOutput:
    """Return a validated output, constructing one if needed."""

    if isinstance(output, DetrSlotOutput):
        return DetrSlotOutput(
            tokens=output.tokens,
            existence_logits=output.existence_logits,
            slot_mask=output.slot_mask,
            aux=output.aux,
            loss_features=output.loss_features,
            core_outputs=output.core_outputs,
            aux_outputs=output.aux_outputs,
        )
    raise TypeError(f"Expected DetrSlotOutput, got {type(output).__name__}")


def detr_slot_output_from_tensors(
    tokens,
    existence_logits,
    slot_mask,
    *,
    aux: Mapping[str, Any] | None = None,
    loss_features=None,
    core_outputs=None,
    aux_outputs=None,
) -> DetrSlotOutput:
    return DetrSlotOutput(
        tokens=tokens,
        existence_logits=existence_logits,
        slot_mask=slot_mask,
        aux=dict(aux or {}),
        loss_features=loss_features,
        core_outputs=core_outputs,
        aux_outputs=aux_outputs,
    )
