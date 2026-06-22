"""Shared DETR/free-slot decoder and prediction heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Mapping

from .features import DetrSlotFeatureConfig, decode_slot_outputs_to_loss_features, decode_slot_outputs_to_raw_tokens
from .outputs import DetrSlotOutput


DETR_SLOT_DECODER_STEP = "detr_free_slot_step4_decoder_heads"


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("DETR slot decoder utilities require PyTorch")
    return torch


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


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


def _as_bool_mask(value, *, name: str, expected_shape: tuple[int, int], device=None):
    torch = require_torch()
    if value is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    tensor = tensor.bool()
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def _make_mlp(torch, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(input_dim)),
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
    )


def _safe_memory_mask(memory_tokens, memory_mask):
    """Ensure every batch row has at least one valid memory token."""

    torch = require_torch()
    if bool(memory_mask.all(dim=1).all().detach().cpu().item()):
        return memory_tokens, memory_mask, 0
    safe_mask = memory_mask.clone()
    safe_memory = memory_tokens.clone()
    empty_rows = ~safe_mask.any(dim=1)
    forced_count = int(empty_rows.sum().detach().cpu().item())
    if forced_count > 0:
        safe_mask[empty_rows, 0] = True
        safe_memory[empty_rows, 0, :] = 0.0
    return safe_memory, safe_mask, forced_count


@dataclass(frozen=True)
class LearnedSlotQueryConfig:
    """Configuration for learned DETR reconstruction queries."""

    num_slots: int = 160
    embed_dim: int = 128
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if int(self.num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        if float(self.init_std) <= 0.0:
            raise ValueError("init_std must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetrSlotDecoderConfig:
    """Configuration for the shared free-slot Transformer decoder."""

    num_slots: int = 160
    embed_dim: int = 128
    memory_dim: int = 128
    context_dim: int | None = None
    condition_queries_on_context: bool = True
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    activation: str = "gelu"
    norm_first: bool = True

    def __post_init__(self) -> None:
        for field_name in ("num_slots", "embed_dim", "memory_dim", "num_layers", "num_heads"):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.context_dim is not None and int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive when provided")
        if int(self.embed_dim) % int(self.num_heads) != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def dim_feedforward(self) -> int:
        return max(int(round(float(self.mlp_ratio) * int(self.embed_dim))), int(self.embed_dim))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dim_feedforward"] = int(self.dim_feedforward)
        return payload


@dataclass(frozen=True)
class DetrPredictionHeadsConfig:
    """Configuration for converting slot embeddings into raw particle tokens."""

    embed_dim: int = 128
    hidden_dim: int = 256
    context_dim: int | None = None
    condition_on_context: bool = True
    feature_config: DetrSlotFeatureConfig = field(default_factory=DetrSlotFeatureConfig)
    dropout: float = 0.05
    existence_bias: float = -2.0
    core_output_scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.feature_config, Mapping):
            object.__setattr__(self, "feature_config", DetrSlotFeatureConfig(**dict(self.feature_config)))
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.context_dim is not None and int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive when provided")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.core_output_scale) <= 0.0:
            raise ValueError("core_output_scale must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.feature_config.feature_dim)

    @property
    def aux_dim(self) -> int:
        return int(self.feature_config.aux_dim)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_config"] = self.feature_config.to_dict()
        payload["feature_dim"] = int(self.feature_dim)
        payload["aux_dim"] = int(self.aux_dim)
        return payload


class LearnedSlotQueries(_ModuleBase):
    """Learned reconstruction slot queries repeated for each batch row."""

    def __init__(self, config: LearnedSlotQueryConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, LearnedSlotQueryConfig) else LearnedSlotQueryConfig(**dict(config or {}))
        self.query_embeddings = torch.nn.Parameter(torch.empty(int(self.config.num_slots), int(self.config.embed_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        torch.nn.init.normal_(self.query_embeddings, mean=0.0, std=float(self.config.init_std))

    def forward(self, batch_size: int, *, device=None, dtype=None):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        queries = self.query_embeddings
        if device is not None or dtype is not None:
            queries = queries.to(device=device if device is not None else queries.device, dtype=dtype if dtype is not None else queries.dtype)
        return queries.unsqueeze(0).expand(int(batch_size), -1, -1)


class DetrSlotDecoder(_ModuleBase):
    """Shared Transformer decoder that cross-attends learned slots to HLT memory."""

    def __init__(self, config: DetrSlotDecoderConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, DetrSlotDecoderConfig) else DetrSlotDecoderConfig(**dict(config or {}))
        self.queries = LearnedSlotQueries(
            LearnedSlotQueryConfig(num_slots=int(self.config.num_slots), embed_dim=int(self.config.embed_dim))
        )
        if int(self.config.memory_dim) == int(self.config.embed_dim):
            self.memory_projection = torch.nn.Identity()
        else:
            self.memory_projection = torch.nn.Linear(int(self.config.memory_dim), int(self.config.embed_dim))
        self.context_projection = None
        if bool(self.config.condition_queries_on_context) and self.config.context_dim is not None:
            self.context_projection = torch.nn.Linear(int(self.config.context_dim), int(self.config.embed_dim))
        layer = torch.nn.TransformerDecoderLayer(
            d_model=int(self.config.embed_dim),
            nhead=int(self.config.num_heads),
            dim_feedforward=int(self.config.dim_feedforward),
            dropout=float(self.config.dropout),
            activation=str(self.config.activation),
            batch_first=True,
            norm_first=bool(self.config.norm_first),
        )
        self.decoder = torch.nn.TransformerDecoder(layer, num_layers=int(self.config.num_layers))
        self.output_norm = torch.nn.LayerNorm(int(self.config.embed_dim))
        self.last_diagnostics: dict[str, float] = {}

    def forward(self, memory_tokens, memory_mask=None, global_context=None):
        memory = _as_float_tensor(memory_tokens, name="memory_tokens")
        if memory.ndim != 3:
            raise ValueError(f"memory_tokens must have shape [batch, memory, dim], got {tuple(memory.shape)}")
        if int(memory.shape[0]) <= 0 or int(memory.shape[1]) <= 0:
            raise ValueError(f"memory_tokens batch and memory dimensions must be positive, got {tuple(memory.shape)}")
        if int(memory.shape[2]) != int(self.config.memory_dim):
            raise ValueError(f"memory dim {int(memory.shape[2])} != configured memory_dim {int(self.config.memory_dim)}")
        mask = _as_bool_mask(
            memory_mask,
            name="memory_mask",
            expected_shape=(int(memory.shape[0]), int(memory.shape[1])),
            device=memory.device,
        )
        memory, mask, forced_count = _safe_memory_mask(memory, mask)
        memory = self.memory_projection(memory)
        queries = self.queries(int(memory.shape[0]), device=memory.device, dtype=memory.dtype)
        context_was_used = False
        if global_context is not None:
            context = _as_float_tensor(
                global_context,
                name="global_context",
                device=memory.device,
                dtype=memory.dtype,
            )
            if context.ndim != 2:
                raise ValueError(f"global_context must have shape [batch, dim], got {tuple(context.shape)}")
            if int(context.shape[0]) != int(memory.shape[0]):
                raise ValueError(
                    f"global_context batch size {int(context.shape[0])} != memory batch size {int(memory.shape[0])}"
                )
            if self.context_projection is not None:
                if int(context.shape[1]) != int(self.config.context_dim):
                    raise ValueError(
                        f"global_context dim {int(context.shape[1])} != configured context_dim {int(self.config.context_dim)}"
                    )
                queries = queries + self.context_projection(context).unsqueeze(1)
                context_was_used = True
            elif bool(self.config.condition_queries_on_context) and int(context.shape[1]) == int(self.config.embed_dim):
                queries = queries + context.unsqueeze(1)
                context_was_used = True
            elif bool(self.config.condition_queries_on_context):
                raise ValueError(
                    "global_context was provided but decoder context_dim is unset and context dim "
                    f"{int(context.shape[1])} does not equal embed_dim {int(self.config.embed_dim)}"
                )
        slot_embeddings = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~mask,
        )
        slot_embeddings = self.output_norm(slot_embeddings)
        if not require_torch().isfinite(slot_embeddings).all():
            raise FloatingPointError("DETR slot decoder produced non-finite slot embeddings")
        self.last_diagnostics = {
            "forced_nonempty_memory_rows": float(forced_count),
            "memory_valid_fraction": float(mask.float().mean().detach().cpu().item()),
            "num_slots": float(self.config.num_slots),
            "embed_dim": float(self.config.embed_dim),
            "context_conditioned_queries": float(context_was_used),
        }
        return slot_embeddings


class DetrPredictionHeads(_ModuleBase):
    """Prediction heads that map slot embeddings to ``DetrSlotOutput``."""

    def __init__(self, config: DetrPredictionHeadsConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        if isinstance(config, DetrPredictionHeadsConfig):
            self.config = config
        else:
            payload = dict(config or {})
            feature_config = payload.get("feature_config")
            if isinstance(feature_config, Mapping):
                payload["feature_config"] = DetrSlotFeatureConfig(**dict(feature_config))
            self.config = DetrPredictionHeadsConfig(**payload)
        self.core_head = _make_mlp(
            torch,
            int(self.config.embed_dim),
            int(self.config.hidden_dim),
            4,
            float(self.config.dropout),
        )
        self.aux_head = None
        if int(self.config.aux_dim) > 0:
            self.aux_head = _make_mlp(
                torch,
                int(self.config.embed_dim),
                int(self.config.hidden_dim),
                int(self.config.aux_dim),
                float(self.config.dropout),
            )
        self.existence_head = _make_mlp(
            torch,
            int(self.config.embed_dim),
            int(self.config.hidden_dim),
            1,
            float(self.config.dropout),
        )
        self.context_projection = None
        if bool(self.config.condition_on_context) and self.config.context_dim is not None:
            self.context_projection = torch.nn.Linear(int(self.config.context_dim), int(self.config.embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch = require_torch()
        final = self.existence_head[-1]
        if isinstance(final, torch.nn.Linear):
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.constant_(final.bias, float(self.config.existence_bias))

    def forward(
        self,
        slot_embeddings,
        slot_mask=None,
        *,
        global_context=None,
        aux: Mapping[str, Any] | None = None,
    ) -> DetrSlotOutput:
        embeddings = _as_float_tensor(slot_embeddings, name="slot_embeddings")
        if embeddings.ndim != 3:
            raise ValueError(f"slot_embeddings must have shape [batch, slots, dim], got {tuple(embeddings.shape)}")
        if int(embeddings.shape[-1]) != int(self.config.embed_dim):
            raise ValueError(
                f"slot embedding dim {int(embeddings.shape[-1])} != configured embed_dim {int(self.config.embed_dim)}"
            )
        mask = _as_bool_mask(
            slot_mask,
            name="slot_mask",
            expected_shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
            device=embeddings.device,
        )
        context_was_used = False
        if global_context is not None:
            context = _as_float_tensor(
                global_context,
                name="global_context",
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            if context.ndim != 2:
                raise ValueError(f"global_context must have shape [batch, dim], got {tuple(context.shape)}")
            if int(context.shape[0]) != int(embeddings.shape[0]):
                raise ValueError(
                    f"global_context batch size {int(context.shape[0])} != slot batch size {int(embeddings.shape[0])}"
                )
            if self.context_projection is not None:
                if int(context.shape[1]) != int(self.config.context_dim):
                    raise ValueError(
                        f"global_context dim {int(context.shape[1])} != configured context_dim {int(self.config.context_dim)}"
                    )
                embeddings = embeddings + self.context_projection(context).unsqueeze(1)
                context_was_used = True
            elif bool(self.config.condition_on_context) and int(context.shape[1]) == int(self.config.embed_dim):
                embeddings = embeddings + context.unsqueeze(1)
                context_was_used = True
            elif bool(self.config.condition_on_context):
                raise ValueError(
                    "global_context was provided but heads context_dim is unset and context dim "
                    f"{int(context.shape[1])} does not equal embed_dim {int(self.config.embed_dim)}"
                )
        core_outputs = self.core_head(embeddings) * float(self.config.core_output_scale)
        aux_outputs = None if self.aux_head is None else self.aux_head(embeddings)
        tokens = decode_slot_outputs_to_raw_tokens(
            core_outputs,
            aux_outputs,
            config=self.config.feature_config,
        )
        loss_features = decode_slot_outputs_to_loss_features(
            core_outputs,
            aux_outputs,
            config=self.config.feature_config,
        )
        existence_logits = self.existence_head(embeddings).squeeze(-1)
        output_aux = {
            "decoder_step": DETR_SLOT_DECODER_STEP,
            "feature_dim": float(self.config.feature_dim),
            "aux_dim": float(self.config.aux_dim),
            "existence_bias": float(self.config.existence_bias),
            "context_conditioned_heads": float(context_was_used),
        }
        output_aux.update(dict(aux or {}))
        return DetrSlotOutput(
            tokens=tokens,
            existence_logits=existence_logits,
            slot_mask=mask,
            aux=output_aux,
            loss_features=loss_features,
            core_outputs=core_outputs,
            aux_outputs=aux_outputs,
        )


def build_detr_slot_decoder_and_heads(
    *,
    decoder_config: DetrSlotDecoderConfig | Mapping[str, Any] | None = None,
    heads_config: DetrPredictionHeadsConfig | Mapping[str, Any] | None = None,
) -> tuple[DetrSlotDecoder, DetrPredictionHeads]:
    decoder = DetrSlotDecoder(decoder_config)
    heads_payload = dict(heads_config or {}) if not isinstance(heads_config, DetrPredictionHeadsConfig) else heads_config
    if not isinstance(heads_payload, DetrPredictionHeadsConfig):
        heads_payload.setdefault("embed_dim", int(decoder.config.embed_dim))
        if decoder.config.context_dim is not None:
            heads_payload.setdefault("context_dim", int(decoder.config.context_dim))
    heads = DetrPredictionHeads(heads_payload)
    return decoder, heads
