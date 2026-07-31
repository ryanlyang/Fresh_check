"""Parity-safe Weaver Particle Transformer taps for HOSD.

The adapter intentionally executes Weaver's own forward.  Temporary hooks
observe (or, for a declared later feedback graph, replace) block outputs.
Consequently the disabled path does not duplicate trimming, embedding, pair
encoding, class attention, autocast, or RNG-sensitive behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from teacher_logit_reco.relational_part.model import RelationalParticleTransformer

from .contracts import SPLIT_FORWARD_CONTRACT, with_content_hash

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


TAP_BLOCKS = {"TAP_EARLY": 2, "TAP_MID": 4, "TAP_LATE": 8}


def split_forward_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": SPLIT_FORWARD_CONTRACT,
            "schema_version": 1,
            "block_numbering": "one_based",
            "taps": dict(TAP_BLOCKS),
            "execution": "temporary_hooks_around_unmodified_weaver_forward",
            "disabled_path": {
                "hooks_registered": False,
                "state_dictionary_changed": False,
                "rng_calls_added": False,
            },
            "tap_state_layout": "batch_particle_channel",
            "tap_mask_layout": "batch_particle_boolean",
            "feedback_insertion": "post_block_4_only_when_explicitly_supplied",
            "authoritative_parity": {
                "precision": "FP32_mixed_precision_disabled",
                "logits": {"absolute": 1e-6, "relative": 1e-5},
                "gradients": {"absolute": 2e-6, "relative": 2e-5},
                "masks_state_keys_state_shapes": "exact",
            },
        }
    )


@dataclass(frozen=True)
class SplitForwardResult:
    logits: Any
    states: Mapping[str, Any]
    masks: Mapping[str, Any]


class WeaverSplitForwardAdapter:
    """Observe exact post-block states without becoming part of the model."""

    def __init__(self, model: RelationalParticleTransformer) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD split forward")
        if not isinstance(model, RelationalParticleTransformer):
            raise TypeError("split forward requires exact H_BASE wrapper")
        blocks = getattr(model.mod, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 8:
            raise RuntimeError("H_BASE must expose exactly eight Weaver blocks")
        self.model = model
        self.contract = split_forward_contract()

    @staticmethod
    def _batch_first(state: Any, mask: Any) -> tuple[Any, bool]:
        if not isinstance(state, torch.Tensor) or state.ndim != 3:
            raise RuntimeError("Weaver particle block output must be rank three")
        batch = int(mask.shape[0])
        if int(state.shape[0]) == batch:
            return state, True
        if int(state.shape[1]) == batch:
            return state.transpose(0, 1), False
        raise RuntimeError("cannot resolve Weaver particle-state layout")

    @staticmethod
    def _restore_layout(state: Any, batch_first: bool) -> Any:
        return state if batch_first else state.transpose(0, 1)

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        capture: Sequence[str] = tuple(TAP_BLOCKS),
        post_mid_transform: Callable[[Any, Any], Any] | None = None,
        later_block_transform: Callable[[int, Any, Any], Any] | None = None,
        later_pair_bias: Callable[[Any, Any], Any] | None = None,
    ) -> SplitForwardResult:
        requested = tuple(str(value) for value in capture)
        unknown = sorted(set(requested) - set(TAP_BLOCKS))
        if unknown:
            raise ValueError(f"unknown HOSD taps: {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("HOSD taps may not be duplicated")
        if (
            post_mid_transform is not None
            or later_block_transform is not None
            or later_pair_bias is not None
        ) and "TAP_MID" not in requested:
            raise ValueError("feedback transforms require TAP_MID capture")

        # The genuinely disabled path is the ordinary wrapper call: no hook is
        # installed and no additional tensor operation occurs.
        if (
            not requested
            and post_mid_transform is None
            and later_block_transform is None
            and later_pair_bias is None
        ):
            return SplitForwardResult(
                logits=self.model(points, features, lorentz_vectors, mask),
                states={},
                masks={},
            )

        valid = mask.bool()
        states: dict[str, Any] = {}
        masks: dict[str, Any] = {}
        padding_by_block: dict[int, Any] = {}
        feedback_pair_bias: list[Any | None] = [None]
        handles = []

        def pre_hook(index: int):
            def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
                padding = kwargs.get("padding_mask")
                if padding is not None:
                    padding_by_block[index] = padding.bool()
                if index < 4:
                    return None
                changed_args = args
                if later_block_transform is not None:
                    if not args:
                        raise RuntimeError("Weaver block received no particle state")
                    state = args[0]
                    batch_state, batch_first = self._batch_first(state, valid)
                    active = (
                        ~padding.bool()
                        if padding is not None
                        else valid[:, 0, : int(batch_state.shape[1])]
                    )
                    transformed = later_block_transform(
                        index + 1, batch_state, active
                    )
                    if (
                        not isinstance(transformed, torch.Tensor)
                        or transformed.shape != batch_state.shape
                        or transformed.dtype != batch_state.dtype
                        or transformed.device != batch_state.device
                    ):
                        raise ValueError(
                            "later-block transform must preserve state shape/dtype/device"
                        )
                    changed_args = (
                        self._restore_layout(transformed, batch_first),
                        *args[1:],
                    )
                bias = feedback_pair_bias[0]
                if bias is not None:
                    existing = kwargs.get("attn_mask")
                    if existing is None:
                        raise RuntimeError(
                            "pair feedback requires Weaver later-block attention masks"
                        )
                    batch, heads, length, other = bias.shape
                    if length != other:
                        raise ValueError("feedback pair bias must be square")
                    if existing.ndim == 3:
                        flattened = bias.reshape(batch * heads, length, length)
                        if flattened.shape != existing.shape:
                            raise ValueError(
                                "feedback and Weaver attention-mask shapes differ"
                            )
                        kwargs = dict(kwargs)
                        kwargs["attn_mask"] = existing + flattened.to(existing.dtype)
                    elif existing.ndim == 4 and existing.shape == bias.shape:
                        kwargs = dict(kwargs)
                        kwargs["attn_mask"] = existing + bias.to(existing.dtype)
                    else:
                        raise ValueError("unsupported Weaver attention-mask layout")
                if changed_args is args and bias is None:
                    return None
                return changed_args, kwargs

            return hook

        def post_hook(tap_id: str, index: int):
            def hook(_module: Any, _args: tuple[Any, ...], output: Any):
                state = output[0] if isinstance(output, tuple) else output
                batch_state, batch_first = self._batch_first(state, valid)
                padding = padding_by_block.get(index)
                if padding is None:
                    active_mask = valid[:, 0, : int(batch_state.shape[1])]
                else:
                    active_mask = ~padding
                if tuple(active_mask.shape) != tuple(batch_state.shape[:2]):
                    raise RuntimeError("tap state and exact Weaver mask disagree")
                captured = batch_state
                if tap_id == "TAP_MID" and post_mid_transform is not None:
                    captured = post_mid_transform(batch_state, active_mask)
                    if (
                        not isinstance(captured, torch.Tensor)
                        or tuple(captured.shape) != tuple(batch_state.shape)
                        or captured.dtype != batch_state.dtype
                        or captured.device != batch_state.device
                    ):
                        raise ValueError(
                            "post-mid transform must preserve state shape/dtype/device"
                        )
                if tap_id == "TAP_MID" and later_pair_bias is not None:
                    bias = later_pair_bias(captured, active_mask)
                    if (
                        not isinstance(bias, torch.Tensor)
                        or bias.ndim != 4
                        or int(bias.shape[0]) != int(captured.shape[0])
                        or tuple(bias.shape[2:])
                        != (int(captured.shape[1]), int(captured.shape[1]))
                        or bias.device != captured.device
                    ):
                        raise ValueError(
                            "later pair bias must be [batch,heads,particle,particle]"
                        )
                    feedback_pair_bias[0] = bias
                states[tap_id] = captured
                masks[tap_id] = active_mask
                if not (tap_id == "TAP_MID" and post_mid_transform is not None):
                    # Observation-only hooks do not replace Weaver's output
                    # object, even with an equivalent view.
                    return None
                replacement = self._restore_layout(captured, batch_first)
                if isinstance(output, tuple):
                    return (replacement, *output[1:])
                return replacement

            return hook

        try:
            for tap_id in requested:
                index = TAP_BLOCKS[tap_id] - 1
                block = self.model.mod.blocks[index]
                try:
                    handles.append(
                        block.register_forward_pre_hook(
                            pre_hook(index), with_kwargs=True
                        )
                    )
                except TypeError:  # pragma: no cover - old PyTorch fallback
                    pass
                handles.append(block.register_forward_hook(post_hook(tap_id, index)))
            logits = self.model(points, features, lorentz_vectors, valid)
        finally:
            for handle in reversed(handles):
                handle.remove()
        if set(states) != set(requested) or set(masks) != set(requested):
            raise RuntimeError("Weaver forward did not execute every requested tap")
        return SplitForwardResult(logits=logits, states=states, masks=masks)


class HBaseParticleTransformer(RelationalParticleTransformer):
    """Exact H_BASE with an opt-in, unregistered split-forward adapter."""

    def __init__(self, *, weaver_module: Any | None = None) -> None:
        super().__init__(weaver_module=weaver_module)
        object.__setattr__(self, "_hosd_split_adapter", WeaverSplitForwardAdapter(self))

    def forward_with_taps(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        capture: Sequence[str] = tuple(TAP_BLOCKS),
        post_mid_transform: Callable[[Any, Any], Any] | None = None,
        later_block_transform: Callable[[int, Any, Any], Any] | None = None,
        later_pair_bias: Callable[[Any, Any], Any] | None = None,
    ) -> SplitForwardResult:
        return self._hosd_split_adapter.forward(
            points,
            features,
            lorentz_vectors,
            mask,
            capture=capture,
            post_mid_transform=post_mid_transform,
            later_block_transform=later_block_transform,
            later_pair_bias=later_pair_bias,
        )


__all__ = [
    "HBaseParticleTransformer",
    "SplitForwardResult",
    "TAP_BLOCKS",
    "WeaverSplitForwardAdapter",
    "split_forward_contract",
]
