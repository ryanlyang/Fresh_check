"""Validation-only relation perturbations and the trained unary endpoint control."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import canonical_sha256, require_sha256, with_content_hash
from .model import RelationalParticleTransformer
from .normalization import (
    DENSITY_NODE_FEATURE_NAMES,
    FeaturewiseNormalizer,
    GLOBAL_EPSILON,
    PT_RAW_FEATURE_NAMES,
    PT_ROBUST_FEATURE_NAMES,
    TRACK_NODE_CONTINUOUS_NAMES,
    validate_relation_normalization_artifact,
)
from .pair_base import STANDARD_FOUR_CHANNELS, require_torch
from .pair_builder import (
    SUPPORTED_FAMILY_DIMENSIONS,
    canonical_supported_families,
)
from .region_tree import EXCLUSIVE_RESOLUTIONS, canonical_leaf_key
from .relation_density import build_density_node_features
from .relation_pid_charge import pid_categories, quantize_charge
from .relation_pt import build_pt_raw_features
from .relation_region import (
    RegionNormalizer,
    build_batched_region_leaf_ranks,
    build_batched_region_raw_features,
)
from .relation_track import build_track_node_features, normalize_track_sentinel_policy

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SEMANTIC_PERTURBATION_CONTRACT = "relational_part_semantic_perturbations_v2"
UNARY_CONTROL_REGISTRY_CONTRACT = "relational_part_unary_control_registry_v1"
UNARY_MODEL_CONTRACT = "relational_part_unary_model_v1"
UNARY_FAMILY_WIDTHS = {
    "PT": 3,
    "TRACK": 7,
    "PID": 8,
    "CHARGE": 5,
    "DENSITY": 22,
    "REGION": 18,
}


def _identity_digest(salt: str, identity: str) -> bytes:
    return hashlib.sha256(
        salt.encode("utf-8") + b"\0" + identity.encode("utf-8")
    ).digest()


def within_jet_shuffled_relations(
    pair_features: Any,
    mask: Any,
    event_identities: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    """Apply a deterministic fixed-point-free cycle to active new relations."""

    module = require_torch()
    if pair_features.ndim != 4 or int(pair_features.shape[1]) <= 4:
        raise ValueError("semantic shuffle requires base4 plus new relations")
    if len(event_identities) != int(pair_features.shape[0]):
        raise ValueError("event identity count differs from relation batch")
    output = pair_features.clone()
    permutations = []
    excluded = permuted = fixed_points = 0
    for row, identity in enumerate(event_identities):
        valid = module.nonzero(
            mask[row, 0].bool(), as_tuple=False
        ).flatten().tolist()
        if len(valid) < 2:
            excluded += 1
            permutations.append({"identity": str(identity), "eligible": False})
            continue
        ordered = sorted(
            valid,
            key=lambda index: (
                _identity_digest(
                    "rpt_within_jet_shuffle_v1",
                    f"{identity}\0{index}",
                ),
                index,
            ),
        )
        mapped = ordered[1:] + ordered[:1]
        source = module.as_tensor(mapped, device=pair_features.device)
        destination = module.as_tensor(ordered, device=pair_features.device)
        relation = pair_features[row, STANDARD_FOUR_CHANNELS:]
        output[
            row,
            STANDARD_FOUR_CHANNELS:,
            destination[:, None],
            destination[None, :],
        ] = relation[:, source[:, None], source[None, :]]
        permuted += len(valid)
        fixed_points += sum(left == right for left, right in zip(ordered, mapped))
        permutations.append(
            {
                "identity": str(identity),
                "eligible": True,
                "destination_indices": ordered,
                "source_indices": mapped,
            }
        )
    if fixed_points:
        raise RuntimeError("within-jet shuffle unexpectedly contains fixed points")
    return output, {
        "kind": "within_jet_shuffled_relations",
        "eligible_event_count": len(event_identities) - excluded,
        "excluded_fewer_than_two_valid": excluded,
        "permuted_valid_particle_count": permuted,
        "fixed_point_count": fixed_points,
        "permutation_sha256": canonical_sha256(permutations),
        "base4_unchanged": bool(
            module.equal(
                output[:, :STANDARD_FOUR_CHANNELS],
                pair_features[:, :STANDARD_FOUR_CHANNELS],
            )
        ),
        "mask_agreement": True,
        "relation_norm_before": float(
            pair_features[:, STANDARD_FOUR_CHANNELS:].float().norm().cpu()
        ),
        "relation_norm_after": float(
            output[:, STANDARD_FOUR_CHANNELS:].float().norm().cpu()
        ),
    }


def directional_swap_relations(
    pair_features: Any,
    mask: Any,
) -> tuple[Any, dict[str, Any]]:
    module = require_torch()
    if pair_features.ndim != 4 or int(pair_features.shape[1]) <= 4:
        raise ValueError("directional swap requires base4 plus new relations")
    output = pair_features.clone()
    output[:, STANDARD_FOUR_CHANNELS:] = pair_features[
        :, STANDARD_FOUR_CHANNELS:
    ].transpose(-1, -2)
    pair_mask = mask.bool().unsqueeze(-1) & mask.bool().unsqueeze(-2)
    output[:, STANDARD_FOUR_CHANNELS:] = output[
        :, STANDARD_FOUR_CHANNELS:
    ].masked_fill(~pair_mask, 0)
    return output, {
        "kind": "directional_swap",
        "base4_unchanged": bool(
            module.equal(output[:, :4], pair_features[:, :4])
        ),
        "mask_agreement": True,
        "relation_norm_before": float(pair_features[:, 4:].float().norm().cpu()),
        "relation_norm_after": float(output[:, 4:].float().norm().cpu()),
    }


def zero_relation_family(
    pair_features: Any,
    mask: Any,
    *,
    families: Sequence[str],
    family: str,
) -> tuple[Any, dict[str, Any]]:
    """Zero exactly one encoded family while retaining base4 and all others."""

    module = require_torch()
    canonical = canonical_supported_families(families)
    if family not in canonical:
        raise ValueError(f"family dropout {family} is not active")
    offset = STANDARD_FOUR_CHANNELS
    selected_slice = None
    for name in canonical:
        width = SUPPORTED_FAMILY_DIMENSIONS[name]
        if name == family:
            selected_slice = slice(offset, offset + width)
        offset += width
    if selected_slice is None or offset != int(pair_features.shape[1]):
        raise ValueError("family dropout channel layout differs")
    output = pair_features.clone()
    output[:, selected_slice] = 0.0
    pair_mask = mask.bool().unsqueeze(-1) & mask.bool().unsqueeze(-2)
    output = output.masked_fill(~pair_mask, 0.0)
    retained = [
        name for name in canonical if name != family
    ]
    return output, {
        "kind": "inference_only_family_dropout",
        "zeroed_family": family,
        "retained_families": retained,
        "channel_start_inclusive": selected_slice.start,
        "channel_stop_exclusive": selected_slice.stop,
        "base4_unchanged": bool(
            module.equal(
                output[:, :STANDARD_FOUR_CHANNELS],
                pair_features[:, :STANDARD_FOUR_CHANNELS],
            )
        ),
        "selected_family_exactly_zero": bool(
            output[:, selected_slice].eq(0).all()
        ),
        "mask_agreement": True,
    }


def _physics_order(
    pt: np.ndarray,
    vectors: np.ndarray,
    raw_tokens: np.ndarray,
    valid: np.ndarray,
) -> list[int]:
    return sorted(
        np.flatnonzero(valid).tolist(),
        key=lambda index: (
            -float(pt[index]),
            canonical_leaf_key(vectors[index], raw_tokens[index]),
            index,
        ),
    )


def wrong_event_relations(
    pair_features: Any,
    mask: Any,
    event_identities: Sequence[str],
    lorentz_vectors: Any,
    raw_tokens: Any,
) -> tuple[Any, dict[str, Any]]:
    """Class-blind exact-multiplicity derangement with rank-aligned endpoints."""

    module = require_torch()
    batch = int(pair_features.shape[0])
    if len(event_identities) != batch:
        raise ValueError("wrong-event identity count differs from batch")
    valid_np = mask[:, 0].detach().cpu().numpy().astype(bool)
    vector_np = lorentz_vectors.detach().cpu().numpy().transpose(0, 2, 1)
    token_np = raw_tokens.detach().cpu().numpy()
    pt_np = np.hypot(vector_np[:, :, 0], vector_np[:, :, 1])
    strata: dict[int, list[int]] = {}
    for row in range(batch):
        strata.setdefault(int(valid_np[row].sum()), []).append(row)
    source_for: dict[int, int] = {}
    excluded = 0
    for count, rows in sorted(strata.items()):
        if len(rows) < 2:
            excluded += len(rows)
            continue
        ordered = sorted(
            rows,
            key=lambda row: (
                _identity_digest(
                    "rpt_wrong_event_derangement_v1",
                    str(event_identities[row]),
                ),
                str(event_identities[row]),
            ),
        )
        rotated = ordered[1:] + ordered[:1]
        source_for.update(zip(ordered, rotated))
    output = pair_features.clone()
    alignment = []
    for destination, source in sorted(source_for.items()):
        destination_order = _physics_order(
            pt_np[destination],
            vector_np[destination],
            token_np[destination],
            valid_np[destination],
        )
        source_order = _physics_order(
            pt_np[source],
            vector_np[source],
            token_np[source],
            valid_np[source],
        )
        if len(destination_order) != len(source_order):
            raise RuntimeError("wrong-event derangement crossed multiplicities")
        dst = module.as_tensor(destination_order, device=pair_features.device)
        src = module.as_tensor(source_order, device=pair_features.device)
        output[
            destination, 4:, dst[:, None], dst[None, :]
        ] = pair_features[source, 4:, src[:, None], src[None, :]]
        alignment.append(
            {
                "destination_identity": str(event_identities[destination]),
                "source_identity": str(event_identities[source]),
                "valid_count": len(dst),
                "destination_rank_order": destination_order,
                "source_rank_order": source_order,
            }
        )
    if any(destination == source for destination, source in source_for.items()):
        raise RuntimeError("wrong-event derangement contains a fixed event")
    return output, {
        "kind": "wrong_event_relations",
        "deranged_event_count": len(source_for),
        "excluded_underpopulated_stratum_event_count": excluded,
        "fixed_event_count": 0,
        "derangement_sha256": canonical_sha256(alignment),
        "exact_valid_multiplicity": True,
        "class_blind": True,
        "base4_unchanged": bool(module.equal(output[:, :4], pair_features[:, :4])),
        "mask_agreement": True,
        "relation_norm_before": float(pair_features[:, 4:].float().norm().cpu()),
        "relation_norm_after": float(output[:, 4:].float().norm().cpu()),
    }


def build_semantic_perturbation_artifact(
    *,
    nominal_winner_run_id: str,
    nominal_checkpoint_sha256: str,
    confirmation_summary_sha256: str,
    metrics: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "full_model",
        "within_jet_shuffled_relations",
        "wrong_event_relations",
        "directional_swap",
    }
    if (
        not required.issubset(metrics)
        or set(metrics) != set(diagnostics)
        or any(
            not name.startswith("family_dropout_")
            for name in set(metrics) - required
        )
    ):
        raise ValueError(
            "semantic artifact requires the full model, three controls, "
            "and only declared family dropouts"
        )
    return with_content_hash(
        {
            "contract": SEMANTIC_PERTURBATION_CONTRACT,
            "schema_version": 2,
            "nominal_winner_run_id": nominal_winner_run_id,
            "nominal_checkpoint_sha256": require_sha256(
                nominal_checkpoint_sha256, name="nominal_checkpoint_sha256"
            ),
            "confirmation_summary_sha256": require_sha256(
                confirmation_summary_sha256,
                name="confirmation_summary_sha256",
            ),
            "split": "val_select",
            "metrics": {name: dict(value) for name, value in metrics.items()},
            "diagnostics": {
                name: dict(value) for name, value in diagnostics.items()
            },
            "performance_gate": False,
            "final_test_accessed": False,
        }
    )


def _chunks_without_singletons(indices: Sequence[int], size: int = 64):
    values = list(indices)
    if len(values) == 1:
        raise ValueError("singleton groups must be excluded before batching")
    while len(values) > size:
        take = size - 1 if len(values) - size == 1 else size
        yield values[:take]
        values = values[take:]
    if values:
        yield values


def evaluate_semantic_perturbations(
    model: Any,
    loader: Any,
    *,
    device: str | Any = "cpu",
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Evaluate controls and predeclared family dropouts on val-select."""

    module = require_torch()
    from .data import collate_relational_batch
    from .evaluation import _move_batch, evaluate_logits, model_forward

    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "mask"):
        raise ValueError(
            "semantic evaluation requires the identity-bound relational dataset"
        )
    if str(getattr(dataset, "split", "")) not in ("stack_val", "val_select"):
        raise ValueError("semantic controls may only inspect val_select")
    counts = np.asarray(dataset.mask, dtype=bool).sum(axis=1).astype(np.int64)
    all_indices = list(range(len(dataset)))
    eligible_shuffle = [index for index in all_indices if counts[index] >= 2]
    strata: dict[int, list[int]] = {}
    for index in all_indices:
        strata.setdefault(int(counts[index]), []).append(index)
    wrong_groups = [
        group
        for _, group in sorted(strata.items())
        if len(group) >= 2
    ]
    wrong_indices = {index for group in wrong_groups for index in group}
    families = tuple(getattr(model, "families", ()))
    if families:
        families = canonical_supported_families(families)
    all_groups = [
        all_indices[index : index + 64]
        for index in range(0, len(all_indices), 64)
    ]
    specifications = {
        "full_model": all_groups,
        "within_jet_shuffled_relations": [
            eligible_shuffle[index : index + 64]
            for index in range(0, len(eligible_shuffle), 64)
        ],
        "wrong_event_relations": [
            list(chunk)
            for group in wrong_groups
            for chunk in _chunks_without_singletons(group)
        ],
        "directional_swap": all_groups,
    }
    if len(families) >= 2:
        for family in families:
            specifications[f"family_dropout_{family}"] = all_groups
    resolved_device = module.device(device)
    was_training = bool(model.training)
    model.eval()
    metrics: dict[str, Mapping[str, Any]] = {}
    diagnostics: dict[str, Mapping[str, Any]] = {}
    try:
        with module.no_grad():
            for name, groups in specifications.items():
                logits: list[np.ndarray] = []
                labels: list[np.ndarray] = []
                batch_diagnostics: list[Mapping[str, Any]] = []
                for indices in groups:
                    raw = collate_relational_batch(
                        [dataset[index] for index in indices]
                    )
                    identities = list(raw["event_identities"])

                    def transform(
                        pairs: Any,
                        *,
                        mask: Any,
                        lorentz_vectors: Any,
                        raw_tokens: Any,
                        **_: Any,
                    ) -> Any:
                        if name == "full_model":
                            output = pairs
                            detail = {
                                "kind": "unmodified_full_model",
                                "base4_unchanged": True,
                                "mask_agreement": True,
                            }
                        elif name == "within_jet_shuffled_relations":
                            output, detail = within_jet_shuffled_relations(
                                pairs, mask, identities
                            )
                        elif name == "wrong_event_relations":
                            output, detail = wrong_event_relations(
                                pairs,
                                mask,
                                identities,
                                lorentz_vectors,
                                raw_tokens,
                            )
                        elif name == "directional_swap":
                            output, detail = directional_swap_relations(
                                pairs, mask
                            )
                        else:
                            output, detail = zero_relation_family(
                                pairs,
                                mask,
                                families=families,
                                family=name.removeprefix(
                                    "family_dropout_"
                                ),
                            )
                        batch_diagnostics.append(detail)
                        return output

                    raw["pair_transform"] = transform
                    batch = _move_batch(raw, resolved_device)
                    output = model_forward(model, batch)
                    logits.append(output.detach().float().cpu().numpy())
                    labels.append(raw["labels"].detach().long().cpu().numpy())
                if not logits:
                    raise ValueError(f"{name} has no eligible val_select events")
                metrics[name] = evaluate_logits(
                    np.concatenate(logits),
                    np.concatenate(labels),
                    split="val_select",
                )
                diagnostics[name] = {
                    "batch_count": len(batch_diagnostics),
                    "evaluated_event_count": int(sum(len(group) for group in groups)),
                    "excluded_event_count": (
                        len(all_indices) - len(eligible_shuffle)
                        if name == "within_jet_shuffled_relations"
                        else (
                            len(all_indices) - len(wrong_indices)
                            if name == "wrong_event_relations"
                            else 0
                        )
                    ),
                    "fixed_point_count": int(
                        sum(
                            int(
                                detail.get(
                                    "fixed_point_count",
                                    detail.get("fixed_event_count", 0),
                                )
                            )
                            for detail in batch_diagnostics
                        )
                    ),
                    "mask_agreement": all(
                        detail.get("mask_agreement") is True
                        for detail in batch_diagnostics
                    ),
                    "base4_unchanged": all(
                        detail.get("base4_unchanged") is True
                        for detail in batch_diagnostics
                    ),
                    "batch_diagnostics_sha256": canonical_sha256(
                        batch_diagnostics
                    ),
                    "batch_diagnostics": batch_diagnostics,
                }
    finally:
        if was_training:
            model.train()
    return metrics, diagnostics


def unary_adapter_parameter_count(
    input_width: int,
    h1: int,
    h2: int,
) -> int:
    # Linear biases + RMSNorm affine weights are active.
    return (
        input_width * h1
        + h1
        + h1
        + h1 * h2
        + h2
        + h2
        + h2 * 128
        + 128
    )


def unary_adapter_flops(
    input_width: int,
    h1: int,
    h2: int,
    *,
    valid_particles: int = 128,
) -> int:
    return 2 * int(valid_particles) * (
        input_width * h1 + h1 * h2 + h2 * 128
    )


def select_unary_widths(
    *,
    families: Sequence[str],
    reference_incremental_parameters: int,
) -> dict[str, Any]:
    canonical = canonical_supported_families(families)
    input_width = sum(UNARY_FAMILY_WIDTHS[family] for family in canonical)
    embedding_parameters = (
        (48 if "PID" in canonical else 0)
        + (12 if "CHARGE" in canonical else 0)
    )
    target = int(reference_incremental_parameters)
    if target <= 0:
        raise ValueError("unary reference increment must be positive")
    selected = None
    for h1 in range(1, 513):
        for h2 in range(1, 513):
            candidate = embedding_parameters + unary_adapter_parameter_count(
                input_width, h1, h2
            )
            key = (
                abs(candidate - target),
                unary_adapter_flops(input_width, h1, h2),
                h1,
                h2,
            )
            if selected is None or key < selected[0]:
                selected = (key, h1, h2, candidate)
    _, h1, h2, candidate = selected
    relative = abs(candidate - target) / target
    if relative > 0.02:
        raise ValueError(
            f"unary parameter mismatch exceeds 2%: {relative:.6f}"
        )
    return {
        "families": list(canonical),
        "unary_input_width": input_width,
        "independent_categorical_embedding_parameters": embedding_parameters,
        "selected_widths": [h1, h2],
        "reference_incremental_parameters": target,
        "unary_incremental_parameters": candidate,
        "absolute_incremental_mismatch": abs(candidate - target),
        "relative_incremental_mismatch": relative,
        "adapter_flops_at_128_valid_particles": unary_adapter_flops(
            input_width, h1, h2
        ),
        "search_domain": {"h1": [1, 512], "h2": [1, 512]},
        "tie_breaks": [
            "absolute_incremental_parameter_mismatch",
            "adapter_FLOPs_at_128_valid_particles",
            "smaller_h1",
            "smaller_h2",
        ],
    }


def build_unary_control_registry(
    *,
    nominal_winner_run_id: str,
    unary_reference_run_id: str,
    families: Sequence[str],
    reference_incremental_parameters: int,
    reference_total_parameters: int,
    base_total_parameters: int,
    confirmation_summary_sha256: str,
    relation_normalization_sha256: str,
) -> dict[str, Any]:
    architecture_winner = nominal_winner_run_id in {
        "RPT_SELECTED_LAYERWISE",
        "RPT_SELECTED_EDGEVALUE",
    }
    if unary_reference_run_id in {
        "RPT_SELECTED_LAYERWISE",
        "RPT_SELECTED_EDGEVALUE",
        "RPT_BASE_LAYERWISE",
        "RPT_BASE_EDGEVALUE",
    }:
        raise ValueError("unary reference must be an ordinary shared-bias row")
    if not architecture_winner and unary_reference_run_id != nominal_winner_run_id:
        raise ValueError(
            "a shared-bias nominal winner is its own unary reference row"
        )
    if (
        int(reference_total_parameters) - int(base_total_parameters)
        != int(reference_incremental_parameters)
    ):
        raise ValueError("unary reference parameter equation is inconsistent")
    search = select_unary_widths(
        families=families,
        reference_incremental_parameters=reference_incremental_parameters,
    )
    return with_content_hash(
        {
            "contract": UNARY_CONTROL_REGISTRY_CONTRACT,
            "schema_version": 1,
            "run_id": "RPT_SELECTED_UNARY",
            "configuration_role": "semantic_control",
            "relational_selection_eligible": False,
            "nominal_winner_run_id": nominal_winner_run_id,
            "unary_reference_run_id": unary_reference_run_id,
            "unary_reference_architecture": "ordinary_shared_bias",
            "unary_source_relation_set": list(search["families"]),
            "confirmation_summary_sha256": require_sha256(
                confirmation_summary_sha256,
                name="confirmation_summary_sha256",
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "parameter_equations": {
                "reference_total_parameters": int(reference_total_parameters),
                "base_total_parameters": int(base_total_parameters),
                "reference_incremental_parameters": int(
                    reference_incremental_parameters
                ),
                "unary_incremental_parameters": search[
                    "unary_incremental_parameters"
                ],
            },
            "search": search,
            "seeds": [101, 202, 303],
            "standard_base4_pair_path_exact": True,
            "new_pairwise_channels": 0,
            "adapter_location": "before_first_particle_attention_block",
            "region_absolute_cluster_rank_transform": "fixed_unit_interval",
            "explicit_pair_only_quantities_forbidden": True,
        }
    )


class UnaryEndpointFeatureBuilder(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        families: Sequence[str],
        *,
        normalization_artifact: Mapping[str, Any],
        region_normalization_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        module = require_torch()
        super().__init__()
        self.families = canonical_supported_families(families)
        validate_relation_normalization_artifact(normalization_artifact)
        floor = normalization_artifact["track_uncertainty_floors"]
        self.d0_floor = float(floor["d0"]["floor"])
        self.dz_floor = float(floor["dz"]["floor"])
        self.sentinel_policy = normalize_track_sentinel_policy(
            normalization_artifact.get("track_sentinel_policy")
        )
        if "PT" in self.families:
            self.pt_normalizer = FeaturewiseNormalizer(
                family_id="PT",
                raw_feature_names=PT_RAW_FEATURE_NAMES,
                robust_feature_names=PT_ROBUST_FEATURE_NAMES,
                artifact=normalization_artifact,
            )
        if "TRACK" in self.families:
            self.track_normalizer = FeaturewiseNormalizer(
                family_id="TRACK",
                raw_feature_names=TRACK_NODE_CONTINUOUS_NAMES,
                robust_feature_names=TRACK_NODE_CONTINUOUS_NAMES,
                artifact=normalization_artifact,
            )
        if "PID" in self.families:
            self.pid_embedding = module.nn.Embedding(6, 8)
        if "CHARGE" in self.families:
            self.charge_embedding = module.nn.Embedding(3, 4)
        if "DENSITY" in self.families:
            self.density_normalizer = FeaturewiseNormalizer(
                family_id="DENSITY",
                raw_feature_names=DENSITY_NODE_FEATURE_NAMES,
                robust_feature_names=DENSITY_NODE_FEATURE_NAMES,
                artifact=normalization_artifact,
            )
        if "REGION" in self.families:
            if region_normalization_artifact is None:
                raise ValueError("unary REGION requires its normalizer")
            self.region_normalizer = RegionNormalizer(
                region_normalization_artifact
            )
        self.output_width = sum(
            UNARY_FAMILY_WIDTHS[family] for family in self.families
        )

    def _region(
        self,
        raw_tokens: Any,
        mask: Any,
        trees: Sequence[Mapping[str, Any]],
    ) -> Any:
        module = require_torch()
        raw = build_batched_region_raw_features(trees, raw_tokens, mask)
        normalized = self.region_normalizer(raw, mask)
        batch, _, length, _ = normalized.shape
        output = raw_tokens.new_zeros(batch, length, 18)
        diagonal_features = normalized.diagonal(dim1=-2, dim2=-1)
        ranks = build_batched_region_leaf_ranks(trees, mask)
        for k_index, _ in enumerate(EXCLUSIVE_RESOLUTIONS):
            descriptor = 8 + k_index * 6
            output[:, :, k_index * 6 : k_index * 6 + 3] = (
                diagonal_features[:, descriptor : descriptor + 3].transpose(
                    1, 2
                )
            )
            output[:, :, k_index * 6 + 3] = diagonal_features[
                :, 26 + k_index * 2
            ]
            output[:, :, k_index * 6 + 4] = diagonal_features[
                :, 32 + k_index * 2
            ]
            output[:, :, k_index * 6 + 5] = ranks[:, k_index]
        return output.masked_fill(~mask[:, 0].unsqueeze(-1), 0)

    def forward(
        self,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        module = require_torch()
        valid = mask.bool()
        pieces = []
        if "PT" in self.families:
            raw = build_pt_raw_features(lorentz_vectors, valid)
            normalized = self.pt_normalizer(raw, valid.unsqueeze(-1) & valid.unsqueeze(-2))
            diagonal_features = normalized.diagonal(dim1=-2, dim2=-1)
            pieces.append(
                diagonal_features[:, (0, 2, 7)].transpose(1, 2)
            )
        if any(
            family in self.families for family in ("TRACK", "DENSITY", "REGION")
        ) and raw_tokens is None:
            raise ValueError("selected unary families require raw HLT tokens")
        if "TRACK" in self.families:
            details = build_track_node_features(
                raw_tokens,
                valid,
                d0_uncertainty_floor=self.d0_floor,
                dz_uncertainty_floor=self.dz_floor,
                sentinel_policy=self.sentinel_policy,
            )
            normalized = self.track_normalizer(
                details["continuous"], details["track_valid"].unsqueeze(1)
            )
            pieces.append(
                module.cat(
                    (
                        normalized.transpose(1, 2),
                        details["track_valid"].unsqueeze(-1).to(normalized),
                    ),
                    dim=-1,
                )
            )
        if "PID" in self.families:
            category = pid_categories(features[:, 6:11], valid)
            pieces.append(self.pid_embedding(category))
        if "CHARGE" in self.families:
            quantized, state = quantize_charge(features[:, 5], valid)
            pieces.append(
                module.cat(
                    (quantized.unsqueeze(-1), self.charge_embedding(state)),
                    dim=-1,
                )
            )
        if "DENSITY" in self.families:
            details = build_density_node_features(
                raw_tokens,
                valid,
                d0_uncertainty_floor=self.d0_floor,
                dz_uncertainty_floor=self.dz_floor,
                sentinel_policy=self.sentinel_policy,
            )
            normalized = self.density_normalizer(
                details["descriptor"], valid
            )
            pieces.append(normalized.transpose(1, 2))
        if "REGION" in self.families:
            if region_trees is None:
                raise ValueError("unary REGION requires compact trees")
            pieces.append(self._region(raw_tokens, valid, region_trees))
        output = module.cat(pieces, dim=-1)
        if int(output.shape[-1]) != self.output_width:
            raise RuntimeError("unary endpoint width drifted")
        return output.masked_fill(~valid[:, 0].unsqueeze(-1), 0)


class UnaryEndpointParticleTransformer(
    torch.nn.Module if torch is not None else object
):
    """Standard base4 ParT plus a matched token-side unary adapter."""

    def __init__(
        self,
        *,
        unary_registry: Mapping[str, Any],
        normalization_artifact: Mapping[str, Any],
        region_normalization_artifact: Mapping[str, Any] | None = None,
        weaver_module: Any | None = None,
    ) -> None:
        module = require_torch()
        super().__init__()
        if unary_registry.get("contract") != UNARY_CONTROL_REGISTRY_CONTRACT:
            raise ValueError("unary registry contract mismatch")
        from .contracts import validate_content_hash

        validate_content_hash(
            unary_registry, expected_contract=UNARY_CONTROL_REGISTRY_CONTRACT
        )
        families = tuple(unary_registry["unary_source_relation_set"])
        self.base = RelationalParticleTransformer(weaver_module=weaver_module)
        self.unary_features = UnaryEndpointFeatureBuilder(
            families,
            normalization_artifact=normalization_artifact,
            region_normalization_artifact=region_normalization_artifact,
        )
        h1, h2 = map(int, unary_registry["search"]["selected_widths"])
        self.adapter = module.nn.Sequential(
            module.nn.Linear(self.unary_features.output_width, h1),
            module.nn.GELU(),
            module.nn.RMSNorm(h1, eps=GLOBAL_EPSILON),
            module.nn.Linear(h1, h2),
            module.nn.GELU(),
            module.nn.RMSNorm(h2, eps=GLOBAL_EPSILON),
            module.nn.Linear(h2, 128),
        )
        observed_increment = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(("unary_features.", "adapter."))
        )
        expected = int(
            unary_registry["search"]["unary_incremental_parameters"]
        )
        if observed_increment != expected:
            raise RuntimeError(
                f"unary active increment differs: {observed_increment} != {expected}"
            )
        self.run_id = "RPT_SELECTED_UNARY"
        self.families = families
        self.unary_registry = dict(unary_registry)

    def no_weight_decay(self) -> set[str]:
        return {"base.mod.cls_token"}

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        del points
        module = require_torch()
        valid = mask.bool()
        clean_features = features.masked_fill(~valid, 0)
        clean_vectors = lorentz_vectors.masked_fill(~valid, 0)
        unary = self.unary_features(
            clean_features,
            clean_vectors,
            valid,
            raw_tokens,
            region_trees,
        )
        adapter = self.adapter(unary).masked_fill(
            ~valid[:, 0].unsqueeze(-1), 0
        )
        base4 = self.base.explicit_standard_four(clean_vectors, valid)
        bias = self.base.mod.pair_embed(
            clean_vectors, uu=base4, mask=valid
        )
        x = self.base.mod.embed(clean_features) + adapter
        padding = ~valid[:, 0]
        for block in self.base.mod.blocks:
            x = block(x, padding_mask=padding, attn_mask=bias)
        cls = self.base.mod.cls_token.expand(x.shape[0], 1, -1)
        for block in self.base.mod.cls_blocks:
            cls = block(x, x_cls=cls, padding_mask=padding)
        return self.base.mod.fc(self.base.mod.norm(cls).squeeze(1))


def build_unary_model_contract(
    *,
    unary_registry_sha256: str,
    base_model_contract_sha256: str,
    relation_normalization_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": UNARY_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": "RPT_SELECTED_UNARY",
            "unary_registry_sha256": require_sha256(
                unary_registry_sha256, name="unary_registry_sha256"
            ),
            "base_model_contract_sha256": require_sha256(
                base_model_contract_sha256, name="base_model_contract_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "standard_base4_pair_path_exact": True,
            "new_pairwise_channels": 0,
            "edge_value": False,
            "adapter_location": "before_first_particle_attention_block",
            "initialization": "from_scratch",
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


__all__ = [
    "SEMANTIC_PERTURBATION_CONTRACT",
    "UNARY_CONTROL_REGISTRY_CONTRACT",
    "UNARY_FAMILY_WIDTHS",
    "UNARY_MODEL_CONTRACT",
    "UnaryEndpointFeatureBuilder",
    "UnaryEndpointParticleTransformer",
    "build_semantic_perturbation_artifact",
    "build_unary_control_registry",
    "build_unary_model_contract",
    "directional_swap_relations",
    "zero_relation_family",
    "evaluate_semantic_perturbations",
    "select_unary_widths",
    "unary_adapter_flops",
    "unary_adapter_parameter_count",
    "within_jet_shuffled_relations",
    "wrong_event_relations",
]
