"""Non-selection Step-11 ablations proving use of reconstructed information."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any

from .tagger import HierarchyAwareDualStreamTagger, PseudoViewInputs


ABPH_TAGGER_DIAGNOSTIC_CONTRACT = "adaptive_binary_pseudooffline_tagger_diagnostics_v1"
ABPH_PSEUDO_INPUT_ABLATIONS: tuple[str, ...] = (
    "zero_pseudo_particles",
    "shuffle_pseudo_between_jets",
    "shuffle_hypotheses_within_jet",
    "remove_hierarchy_tokens",
    "remove_hierarchy_depth",
    "matched_pseudo_noise",
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - compute environment guard
        raise RuntimeError("PyTorch is required for ABPH tagger diagnostics") from exc
    return torch


def _is_view_tensor(name: str, value: Any, n_views: int) -> bool:
    return (
        name != "shared_root_ledger"
        and value.ndim >= 2
        and int(value.shape[1]) == int(n_views)
    )


def ablate_pseudo_inputs(
    pseudo: PseudoViewInputs,
    ablation: str,
    *,
    seed: int = 271828,
    hierarchy_depth: int | None = None,
) -> PseudoViewInputs:
    """Return an explicitly diagnostic copy; the selected artifact is untouched."""

    torch = _require_torch()
    pseudo.validate()
    mode = str(ablation)
    if mode not in ABPH_PSEUDO_INPUT_ABLATIONS:
        raise ValueError(f"unknown pseudo-input ablation {mode!r}")
    arrays = {name: value.clone() for name, value in pseudo.arrays.items()}
    generator = torch.Generator(device=arrays["shared_root_ledger"].device)
    generator.manual_seed(int(seed))

    if mode == "zero_pseudo_particles":
        for name, value in arrays.items():
            if not name.startswith("particle__"):
                continue
            if name.endswith("__mask"):
                value.zero_()
            elif value.is_floating_point():
                value.zero_()
            else:
                value.zero_()
    elif mode == "shuffle_pseudo_between_jets":
        batch = int(arrays["shared_root_ledger"].shape[0])
        if batch < 2:
            raise ValueError("between-jet shuffle requires at least two jets")
        order = torch.randperm(batch, generator=generator, device=arrays["shared_root_ledger"].device)
        arrays = {
            name: value.index_select(0, order) if value.ndim > 0 and value.shape[0] == batch else value
            for name, value in arrays.items()
        }
    elif mode == "shuffle_hypotheses_within_jet":
        batch = int(arrays["shared_root_ledger"].shape[0])
        views = len(pseudo.view_names)
        if views < 2:
            raise ValueError("hypothesis shuffle requires multiple views")
        orders = torch.stack(
            [torch.randperm(views, generator=generator, device=arrays["shared_root_ledger"].device) for _ in range(batch)]
        )
        for name, value in tuple(arrays.items()):
            if not _is_view_tensor(name, value, views):
                continue
            gather = orders.reshape((batch, views) + (1,) * (value.ndim - 2)).expand_as(value)
            arrays[name] = torch.gather(value, 1, gather)
    elif mode == "remove_hierarchy_tokens":
        for name, value in arrays.items():
            if name.startswith("frontier__") and name.endswith("__mask"):
                value.zero_()
    elif mode == "remove_hierarchy_depth":
        if hierarchy_depth is None or int(hierarchy_depth) <= 0:
            raise ValueError("remove_hierarchy_depth requires a positive non-root depth")
        marker = f"__depth_{int(hierarchy_depth):02d}__"
        matched = False
        for name, value in arrays.items():
            if name.startswith("frontier__") and marker in name and name.endswith("__mask"):
                value.zero_()
                matched = True
        if not matched:
            raise ValueError(f"hierarchy depth {hierarchy_depth} does not exist")
    elif mode == "matched_pseudo_noise":
        for hierarchy in pseudo.hierarchy_names:
            mask = arrays[f"particle__{hierarchy}__mask"].bool()
            for field in ("canonical_features", "side_channels"):
                key = f"particle__{hierarchy}__{field}"
                value = arrays[key]
                valid = value[mask]
                if valid.numel() == 0:
                    continue
                mean = valid.mean(dim=0)
                std = valid.std(dim=0, unbiased=False).clamp_min(1.0e-6)
                noise = torch.randn(
                    value.shape,
                    generator=generator,
                    device=value.device,
                    dtype=value.dtype,
                )
                arrays[key] = (mean + noise * std) * mask.unsqueeze(-1).to(value.dtype)

    diagnostics = dict(pseudo.diagnostics)
    diagnostics.update(
        {
            "diagnostic_contract": ABPH_TAGGER_DIAGNOSTIC_CONTRACT,
            "diagnostic_only": True,
            "checkpoint_selection_eligible": False,
            "ablation": mode,
            "seed": int(seed),
            "hierarchy_depth": hierarchy_depth,
        }
    )
    result = replace(pseudo, arrays=arrays, diagnostics=diagnostics)
    # Between-jet shuffling intentionally breaks HLT/pseudo pairing but must
    # retain the pseudo hierarchy's internal shared-root invariants.
    result.validate()
    return result


class TaggerDiagnosticOverride(AbstractContextManager["TaggerDiagnosticOverride"]):
    """Temporary fusion-location or trust-gate intervention for evaluation."""

    def __init__(
        self,
        model: HierarchyAwareDualStreamTagger,
        *,
        fusion_location_index: int | None = None,
        forced_trust: float | None = None,
    ) -> None:
        self.model = model
        self.fusion_location_index = fusion_location_index
        self.forced_trust = forced_trust
        self._saved: list[tuple[Any, ...]] = []
        if fusion_location_index is None and forced_trust is None:
            raise ValueError("a diagnostic override must change fusion or trust")
        if forced_trust is not None and float(forced_trust) not in (0.0, 1.0):
            raise ValueError("forced trust must be exactly zero or one")

    def __enter__(self) -> "TaggerDiagnosticOverride":
        torch = _require_torch()
        if self.fusion_location_index is not None:
            index = int(self.fusion_location_index)
            if not 0 <= index < len(self.model.fusion_stacks):
                raise ValueError("fusion location index is out of range")
            for block in self.model.fusion_stacks[index]:
                self._saved.append(("scale", block, block.hlt_rezero.detach().clone(), block.pseudo_rezero.detach().clone()))
                with torch.no_grad():
                    block.hlt_rezero.zero_()
                    block.pseudo_rezero.zero_()
        if self.forced_trust is not None:
            bias = -30.0 if float(self.forced_trust) == 0.0 else 30.0
            for stack in self.model.fusion_stacks:
                for block in stack:
                    gate = block.trust_gate
                    self._saved.append(("gate", gate, gate.weight.detach().clone(), gate.bias.detach().clone()))
                    with torch.no_grad():
                        gate.weight.zero_()
                        gate.bias.fill_(bias)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch = _require_torch()
        with torch.no_grad():
            for kind, target, first, second in reversed(self._saved):
                if kind == "scale":
                    target.hlt_rezero.copy_(first)
                    target.pseudo_rezero.copy_(second)
                else:
                    target.weight.copy_(first)
                    target.bias.copy_(second)
        self._saved.clear()
        return None


__all__ = [
    "ABPH_PSEUDO_INPUT_ABLATIONS",
    "ABPH_TAGGER_DIAGNOSTIC_CONTRACT",
    "TaggerDiagnosticOverride",
    "ablate_pseudo_inputs",
]
