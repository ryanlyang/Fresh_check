"""Canonical Step-2 declarations for every HOSD target coordinate.

This module defines target meaning and ordering only.  Numerical extraction is
implemented in Step 3.  Keeping the declarations independent makes it possible
to audit source admissibility before any target values are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from teacher_logit_reco.relation_expert_token_bridge.evaluation import CLASS_NAMES
from teacher_logit_reco.relational_part.normalization import (
    CHARGE_RAW_FEATURE_NAMES,
    DENSITY_NODE_FEATURE_NAMES,
    PT_RAW_FEATURE_NAMES,
)
from teacher_logit_reco.relational_part.pair_base import STANDARD_FOUR_FEATURE_NAMES
from teacher_logit_reco.relational_part.relation_pid_charge import PID_CATEGORY_NAMES
from teacher_logit_reco.relational_part.relation_region import (
    REGION_RAW_FEATURE_NAMES,
    REGION_SAME_CLUSTER_NAMES,
)
from teacher_logit_reco.relational_part.relation_track import (
    TRACK_COMPATIBILITY_FEATURE_NAMES,
    TRACK_NODE_CONTINUOUS_NAMES,
    TRACK_VALIDITY_STATE_NAMES,
)

AVAILABILITY_CLASSES = (
    "AUTHENTIC_TRUTH",
    "OFFLINE_RECO_DERIVED",
    "HLT_RECO_DERIVED",
    "DETERMINISTIC_PROXY",
    "TEACHER_DERIVED",
    "UNAVAILABLE",
)
CURRENT_REQUIRED = "current_required"
CURRENT_OPTIONAL = "current_optional"
FUTURE_ONLY = "future_only"
CONTINUOUS_REDUCTIONS = ("mean", "population_std", "q10", "q50", "q90")


@dataclass(frozen=True)
class TargetDeclaration:
    target_id: str
    availability_class: str
    campaign_status: str
    meaning: str
    components: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    source_view: str
    head_type: str
    symmetry: str
    allowed_parameterizations: tuple[str, ...]
    loss: str
    metrics: tuple[str, ...]
    feedback_permitted: bool
    availability_groups: tuple[str, ...] = ("target_available",)
    extractor_entrypoint: str = ""
    semantic_version: int = 1


def _relation_components(
    channels: Iterable[tuple[str, str, tuple[str, ...] | None]],
) -> tuple[str, ...]:
    output: list[str] = []
    for channel_name, channel_type, categories in channels:
        if channel_type.endswith("_continuous"):
            output.extend(
                f"{channel_name}__{reduction}"
                for reduction in CONTINUOUS_REDUCTIONS
            )
        elif channel_type.endswith("_binary"):
            output.append(f"{channel_name}__positive_fraction")
        elif channel_type == "categorical":
            if not categories:
                raise ValueError(f"categorical channel {channel_name!r} has no categories")
            output.extend(f"{channel_name}__frequency__{value}" for value in categories)
        else:
            raise ValueError(f"unknown relation channel type {channel_type!r}")
    if len(output) != len(set(output)):
        raise AssertionError("relation aggregate component names are not unique")
    return tuple(output)


def _continuous_channels(
    names: Iterable[str], channel_type: str
) -> tuple[tuple[str, str, None], ...]:
    return tuple((str(name), channel_type, None) for name in names)


BASE4_RELATION_CHANNELS = _continuous_channels(
    STANDARD_FOUR_FEATURE_NAMES, "unordered_pair_continuous"
)
PT_RELATION_CHANNELS = tuple(
    (
        name,
        (
            "unordered_pair_continuous"
            if name == "log_pair_scalar_pt_fraction"
            else "ordered_pair_continuous"
        ),
        None,
    )
    for name in PT_RAW_FEATURE_NAMES
)
_TRACK_DIRECTED_CHANNELS = {
    "context_minus_query_normalized_d0",
    "context_minus_query_normalized_dz",
    "sin_query_minus_context_delta_phi",
}
TRACK_RELATION_CHANNELS = (
    *_continuous_channels(TRACK_NODE_CONTINUOUS_NAMES, "node_continuous"),
    ("track_valid", "node_binary", None),
    *tuple(
        (
            name,
            (
                "ordered_pair_continuous"
                if name in _TRACK_DIRECTED_CHANNELS
                else "unordered_pair_continuous"
            ),
            None,
        )
        for name in TRACK_COMPATIBILITY_FEATURE_NAMES
    ),
    ("pair_track_validity_state", "categorical", TRACK_VALIDITY_STATE_NAMES),
)
PID_RELATION_CHANNELS = (
    ("query_pid", "categorical", PID_CATEGORY_NAMES),
    ("context_pid", "categorical", PID_CATEGORY_NAMES),
    (
        "directed_pid_pair",
        "categorical",
        tuple(
            f"{query}__to__{context}"
            for query in PID_CATEGORY_NAMES
            for context in PID_CATEGORY_NAMES
        ),
    ),
)
CHARGE_RELATION_CHANNELS = (
    (CHARGE_RAW_FEATURE_NAMES[0], "ordered_pair_continuous", None),
    (CHARGE_RAW_FEATURE_NAMES[1], "ordered_pair_continuous", None),
    *(
        (name, "unordered_pair_continuous", None)
        for name in CHARGE_RAW_FEATURE_NAMES[2:4]
    ),
    *((name, "pair_binary", None) for name in CHARGE_RAW_FEATURE_NAMES[4:]),
)
DENSITY_RELATION_CHANNELS = _continuous_channels(
    DENSITY_NODE_FEATURE_NAMES, "node_continuous"
)
_REGION_UNORDERED_CONTINUOUS = {
    "lca_normalized_depth",
    "lca_log_merge_delta_r",
    "lca_log_merge_kt",
    "lca_merge_z",
    "lca_log_merge_mass_fraction",
}
REGION_RELATION_CHANNELS = tuple(
    (
        name,
        (
            "pair_binary"
            if name in REGION_SAME_CLUSTER_NAMES
            else "unordered_pair_continuous"
            if name in _REGION_UNORDERED_CONTINUOUS
            else "ordered_pair_continuous"
        ),
        None,
    )
    for name in REGION_RAW_FEATURE_NAMES
)
RELATION_CHANNELS = {
    "BASE4": BASE4_RELATION_CHANNELS,
    "PT": PT_RELATION_CHANNELS,
    "TRACK": TRACK_RELATION_CHANNELS,
    "PID": PID_RELATION_CHANNELS,
    "CHARGE": CHARGE_RELATION_CHANNELS,
    "DENSITY": DENSITY_RELATION_CHANNELS,
    "REGION": REGION_RELATION_CHANNELS,
}

BASE4_RELATION_COMPONENTS = _relation_components(BASE4_RELATION_CHANNELS)
PT_RELATION_COMPONENTS = _relation_components(PT_RELATION_CHANNELS)
TRACK_RELATION_COMPONENTS = _relation_components(
    (
        *TRACK_RELATION_CHANNELS,
    )
)
PID_RELATION_COMPONENTS = _relation_components(PID_RELATION_CHANNELS)
CHARGE_RELATION_COMPONENTS = _relation_components(CHARGE_RELATION_CHANNELS)
DENSITY_RELATION_COMPONENTS = _relation_components(DENSITY_RELATION_CHANNELS)
REGION_RELATION_COMPONENTS = _relation_components(REGION_RELATION_CHANNELS)

RELATION_COMPONENTS = {
    "BASE4": BASE4_RELATION_COMPONENTS,
    "PT": PT_RELATION_COMPONENTS,
    "TRACK": TRACK_RELATION_COMPONENTS,
    "PID": PID_RELATION_COMPONENTS,
    "CHARGE": CHARGE_RELATION_COMPONENTS,
    "DENSITY": DENSITY_RELATION_COMPONENTS,
    "REGION": REGION_RELATION_COMPONENTS,
}

JET_COMPONENTS = (
    "log1p_jet_pt",
    "jet_eta",
    "sin_jet_phi",
    "cos_jet_phi",
    "log1p_jet_mass",
    "log1p_jet_energy",
    "log1p_valid_constituent_count",
    "leading_particle_pt_fraction",
    "subleading_particle_pt_fraction",
    "sqrt_sum_particle_pt2_over_sum_particle_pt",
)
COMPOSITION_COMPONENTS = (
    *(f"count_fraction__{name}" for name in PID_CATEGORY_NAMES),
    *(f"scalar_pt_fraction__{name}" for name in PID_CATEGORY_NAMES),
    "negative_charge_fraction",
    "zero_charge_fraction",
    "positive_charge_fraction",
    "net_charge_per_nonzero_charge_particle",
)
_QUANTILES = ("q10", "q25", "q50", "q75", "q90")
TRACK_COMPONENTS = (
    "valid_track_count_fraction",
    "valid_track_scalar_pt_fraction",
    "unavailable_charged_domain_count_fraction",
    "unavailable_charged_domain_scalar_pt_fraction",
    *(f"raw_d0_significance__{q}" for q in _QUANTILES),
    *(f"raw_dz_significance__{q}" for q in _QUANTILES),
    *(f"absolute_d0_significance__{q}" for q in _QUANTILES),
    *(f"absolute_dz_significance__{q}" for q in _QUANTILES),
    *(f"absolute_d0_significance_gt_{threshold}_fraction" for threshold in (1, 2, 3)),
    *(f"absolute_dz_significance_gt_{threshold}_fraction" for threshold in (1, 2, 3)),
    "scalar_pt_weighted_mean_absolute_d0_significance",
    "scalar_pt_weighted_mean_absolute_dz_significance",
)
DENSITY_COMPONENTS = tuple(
    f"scalar_pt_weighted_mean__{name}" for name in DENSITY_NODE_FEATURE_NAMES
)
CA_TREE_COMPONENTS = (
    "valid_particle_count_over_128",
    "node_count_over_255",
    "maximum_leaf_depth_over_127",
    "scalar_pt_weighted_mean_leaf_depth_over_127",
    "scalar_pt_weighted_population_std_leaf_depth_over_127",
    *(f"actual_cluster_count_K{k}_over_{k}" for k in (2, 4, 8)),
    *(
        f"cluster_rank_{rank}_scalar_pt_fraction_K{k}"
        for k in (2, 4, 8)
        for rank in (1, 2, 3)
    ),
    *(f"largest_cluster_mass_over_jet_mass_K{k}" for k in (2, 4, 8)),
    *(f"normalized_multiplicity_entropy_K{k}" for k in (2, 4, 8)),
    *(f"normalized_scalar_pt_entropy_K{k}" for k in (2, 4, 8)),
)
TRACK_COMPONENT_PROXY_COMPONENTS = (
    "component_count_capped_8_over_8",
    *(
        f"component_rank_{rank}__{quantity}"
        for rank in (1, 2, 3, 4)
        for quantity in (
            "track_count_over_40",
            "scalar_pt_fraction",
            "mean_absolute_d0_significance",
            "mean_absolute_dz_significance",
        )
    ),
)
HLT_REGION_PAIR_COMPONENTS = (
    "same_cluster_K2",
    "same_cluster_K4",
    "same_cluster_K8",
    "lca_normalized_depth",
    "lca_log1p_merge_delta_r",
    "lca_log1p_merge_kt",
    "lca_bounded_merge_z",
    "lca_log1p_merge_mass_over_jet_mass",
)


def target_component_availability_groups(
    target_id: str,
    components: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact loss-mask group for every registered coordinate."""

    base_id = target_id.replace("T_HLT_SELF_", "T_OFFLINE_", 1)
    groups: list[str] = []
    for index, name in enumerate(components):
        if base_id == "T_OFFLINE_JET_10":
            group = "jet_direction" if index in {1, 2, 3} else "nonempty_jet"
        elif base_id == "T_OFFLINE_TRACK_32":
            group = (
                "track_availability_observation"
                if index < 4
                else "has_valid_track"
            )
        elif base_id == "T_OFFLINE_TRACK_COMPONENT_PROXY_17":
            if index == 0:
                group = "component_count_observed"
            else:
                group = f"track_component_rank_{(index - 1) // 4 + 1}_present"
        elif target_id.startswith(_RELATION_TARGET_PREFIXES):
            raw_channel = name.split("__", 1)[0]
            group = f"relation_channel__{raw_channel}__applicable_set_nonempty"
        elif target_id == "T_HLT_TRACK_PAIR_13":
            group = "valid_directed_track_pair"
        elif target_id == "T_HLT_REGION_PAIR_8":
            group = "valid_unordered_particle_pair"
        elif base_id in {
            "T_OFFLINE_COMPOSITION_16",
            "T_OFFLINE_DENSITY_22",
            "T_OFFLINE_CA_TREE_26",
        }:
            group = "nonempty_jet"
        elif target_id.startswith("T_OFFLINE_LOGITS_"):
            group = "teacher_logits_available"
        elif target_id == "T_OFFLINE_POOLED_LATENT":
            group = "teacher_latent_available"
        else:
            group = "target_contract_available"
        groups.append(group)
    return tuple(groups)


_RELATION_TARGET_PREFIXES = (
    "T_OFFLINE_RELATION_",
    "T_HLT_SELF_RELATION_",
)


def _declaration(
    target_id: str,
    availability_class: str,
    campaign_status: str,
    meaning: str,
    components: tuple[str, ...],
    required_capabilities: tuple[str, ...],
    *,
    source_view: str,
    head_type: str = "global",
    symmetry: str = "not_applicable",
    allowed_parameterizations: tuple[str, ...] = ("ABS", "RES", "HET"),
    loss: str = "normalized_huber",
    metrics: tuple[str, ...] = ("masked_mae", "masked_rmse", "masked_r2"),
    feedback_permitted: bool = True,
    extractor_entrypoint: str | None = None,
) -> TargetDeclaration:
    return TargetDeclaration(
        target_id=target_id,
        availability_class=availability_class,
        campaign_status=campaign_status,
        meaning=meaning,
        components=components,
        required_capabilities=required_capabilities,
        source_view=source_view,
        head_type=head_type,
        symmetry=symmetry,
        allowed_parameterizations=allowed_parameterizations,
        loss=loss,
        metrics=metrics,
        feedback_permitted=feedback_permitted,
        extractor_entrypoint=extractor_entrypoint
        or (
            "teacher_logit_reco.hlt_offline_structure_distillation.extractors:"
            "extract_registered_target"
        ),
    )


def target_declarations() -> tuple[TargetDeclaration, ...]:
    kin = ("particle_four_vector",)
    track = ("particle_four_vector", "track_measurements")
    pid_charge = ("particle_four_vector", "charge", "pid_categories")
    density = (*track, "charge", "pid_categories")
    declarations: list[TargetDeclaration] = [
        _declaration("T_OFFLINE_JET_10", "OFFLINE_RECO_DERIVED", CURRENT_REQUIRED, "offline jet kinematics and multiplicity summary", JET_COMPONENTS, kin, source_view="offline"),
        _declaration("T_OFFLINE_COMPOSITION_16", "OFFLINE_RECO_DERIVED", CURRENT_REQUIRED, "offline PID and charge composition", COMPOSITION_COMPONENTS, pid_charge, source_view="offline"),
        _declaration("T_OFFLINE_TRACK_32", "OFFLINE_RECO_DERIVED", CURRENT_REQUIRED, "offline track availability and impact-parameter distributions", TRACK_COMPONENTS, track, source_view="offline"),
        _declaration("T_OFFLINE_DENSITY_22", "OFFLINE_RECO_DERIVED", CURRENT_REQUIRED, "offline local-density descriptors", DENSITY_COMPONENTS, density, source_view="offline"),
        _declaration("T_OFFLINE_CA_TREE_26", "OFFLINE_RECO_DERIVED", CURRENT_REQUIRED, "offline exclusive C/A tree summary", CA_TREE_COMPONENTS, kin, source_view="offline"),
        _declaration("T_OFFLINE_TRACK_COMPONENT_PROXY_17", "DETERMINISTIC_PROXY", CURRENT_REQUIRED, "deterministic track-compatibility connected-component proxy", TRACK_COMPONENT_PROXY_COMPONENTS, track, source_view="offline"),
    ]
    relation_requirements = {
        "BASE4": kin,
        "PT": kin,
        "TRACK": track,
        "PID": ("pid_categories",),
        "CHARGE": ("charge",),
        "DENSITY": density,
        "REGION": kin,
    }
    for family, components in RELATION_COMPONENTS.items():
        relation_parameterizations = (
            ("ABS", "RES", "HET")
            if any(
                kind.endswith("_continuous")
                for _, kind, _ in RELATION_CHANNELS[family]
            )
            else ("ABS", "RES")
        )
        declarations.append(
            _declaration(
                f"T_OFFLINE_RELATION_{family}",
                "OFFLINE_RECO_DERIVED",
                CURRENT_REQUIRED,
                f"offline raw {family} relation aggregates",
                components,
                relation_requirements[family],
                source_view="offline",
                allowed_parameterizations=relation_parameterizations,
            )
        )
    for teacher_id in ("O_BASE", "O_FULLREL"):
        declarations.append(
            _declaration(
                f"T_OFFLINE_LOGITS_{teacher_id}",
                "TEACHER_DERIVED",
                CURRENT_REQUIRED,
                f"locked {teacher_id} teacher logits",
                tuple(f"logit__{name}" for name in CLASS_NAMES),
                ("offline_teacher_checkpoint",),
                source_view="offline_teacher",
                allowed_parameterizations=("KD",),
                loss="temperature_2_kl",
                metrics=("kl", "teacher_argmax_agreement"),
                feedback_permitted=False,
            )
        )
    declarations.append(
        _declaration(
            "T_OFFLINE_POOLED_LATENT",
            "TEACHER_DERIVED",
            CURRENT_REQUIRED,
            "locked O_BASE normalized pre-classifier pooled representation",
            tuple(f"latent_{index:03d}" for index in range(128)),
            ("offline_teacher_checkpoint",),
            source_view="offline_teacher",
            allowed_parameterizations=("WHITENED_ABS",),
            loss="normalized_huber_plus_one_minus_cosine",
            metrics=("masked_mae", "cosine_similarity", "teacher_head_agreement"),
        )
    )
    hlt_self = (
        ("JET_10", JET_COMPONENTS, kin, "HLT_RECO_DERIVED"),
        ("COMPOSITION_16", COMPOSITION_COMPONENTS, pid_charge, "HLT_RECO_DERIVED"),
        ("TRACK_32", TRACK_COMPONENTS, track, "HLT_RECO_DERIVED"),
        ("DENSITY_22", DENSITY_COMPONENTS, density, "HLT_RECO_DERIVED"),
        ("CA_TREE_26", CA_TREE_COMPONENTS, kin, "HLT_RECO_DERIVED"),
        ("TRACK_COMPONENT_PROXY_17", TRACK_COMPONENT_PROXY_COMPONENTS, track, "DETERMINISTIC_PROXY"),
    )
    for suffix, components, requirements, availability in hlt_self:
        declarations.append(
            _declaration(
                f"T_HLT_SELF_{suffix}",
                availability,
                CURRENT_REQUIRED,
                f"HLT same-view {suffix.lower()} control",
                components,
                requirements,
                source_view="hlt",
            )
        )
    for family, components in RELATION_COMPONENTS.items():
        relation_parameterizations = (
            ("ABS", "RES", "HET")
            if any(
                kind.endswith("_continuous")
                for _, kind, _ in RELATION_CHANNELS[family]
            )
            else ("ABS", "RES")
        )
        declarations.append(
            _declaration(
                f"T_HLT_SELF_RELATION_{family}",
                "HLT_RECO_DERIVED",
                CURRENT_REQUIRED,
                f"HLT same-view raw {family} relation aggregate control",
                components,
                relation_requirements[family],
                source_view="hlt",
                allowed_parameterizations=relation_parameterizations,
            )
        )
    declarations.extend(
        (
            _declaration(
                "T_HLT_TRACK_PAIR_13",
                "HLT_RECO_DERIVED",
                CURRENT_REQUIRED,
                "exact inherited RPT TRACK compatibility over ordered HLT pairs",
                tuple(TRACK_COMPATIBILITY_FEATURE_NAMES),
                track,
                source_view="hlt",
                head_type="pair",
                symmetry="directed",
                allowed_parameterizations=("ABS",),
                loss="per_jet_masked_normalized_huber",
            ),
            _declaration(
                "T_HLT_REGION_PAIR_8",
                "HLT_RECO_DERIVED",
                CURRENT_REQUIRED,
                "exact inherited C/A REGION topology over unordered HLT pairs",
                HLT_REGION_PAIR_COMPONENTS,
                kin,
                source_view="hlt",
                head_type="pair",
                symmetry="symmetric",
                allowed_parameterizations=("ABS",),
                loss="three_bce_plus_five_normalized_huber",
            ),
            _declaration(
                "T_RETB_SUMMARY_TOKENS",
                "TEACHER_DERIVED",
                CURRENT_OPTIONAL,
                "optional authenticated RETB target-coordinate token comparator",
                ("coordinate_lock_defined_tokens",),
                ("authenticated_retb_token_parent",),
                source_view="retb_teacher",
                head_type="token_set",
                allowed_parameterizations=("ABS",),
                loss="coordinate_lock_defined",
                feedback_permitted=False,
            ),
            _declaration(
                "T_HLT_TRACK_ORIGIN_TRUTH",
                "UNAVAILABLE",
                FUTURE_ONLY,
                "authentic HLT-native track-origin truth categories",
                ("future_schema_defined_categories",),
                ("hlt_native_track_origin_truth",),
                source_view="hlt_truth",
                head_type="particle",
                allowed_parameterizations=(),
                loss="future_contract_required",
                feedback_permitted=False,
            ),
            _declaration(
                "T_HLT_COMMON_VERTEX_TRUTH",
                "UNAVAILABLE",
                FUTURE_ONLY,
                "authentic HLT-native common-production-vertex relation",
                ("future_schema_defined_common_vertex",),
                ("hlt_native_production_vertex_identity",),
                source_view="hlt_truth",
                head_type="pair",
                symmetry="symmetric",
                allowed_parameterizations=(),
                loss="future_contract_required",
                feedback_permitted=False,
            ),
            _declaration(
                "T_OFFLINE_SECONDARY_VERTEX_SET",
                "UNAVAILABLE",
                FUTURE_ONLY,
                "literal reconstructed offline secondary-vertex set",
                ("future_schema_defined_vertex_set",),
                ("offline_reconstructed_secondary_vertices",),
                source_view="offline_reco",
                head_type="set",
                allowed_parameterizations=(),
                loss="future_set_contract_required",
                feedback_permitted=False,
            ),
        )
    )
    ids = tuple(item.target_id for item in declarations)
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate HOSD target ID")
    expected_dimensions = {
        "T_OFFLINE_JET_10": 10,
        "T_OFFLINE_COMPOSITION_16": 16,
        "T_OFFLINE_TRACK_32": 32,
        "T_OFFLINE_DENSITY_22": 22,
        "T_OFFLINE_CA_TREE_26": 26,
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17": 17,
        "T_HLT_TRACK_PAIR_13": 13,
        "T_HLT_REGION_PAIR_8": 8,
    }
    for item in declarations:
        expected = expected_dimensions.get(item.target_id)
        if expected is not None and len(item.components) != expected:
            raise AssertionError(
                f"{item.target_id} declares {len(item.components)} != {expected} components"
            )
    return tuple(declarations)


__all__ = [
    "AVAILABILITY_CLASSES",
    "CA_TREE_COMPONENTS",
    "COMPOSITION_COMPONENTS",
    "CURRENT_OPTIONAL",
    "CURRENT_REQUIRED",
    "DENSITY_COMPONENTS",
    "FUTURE_ONLY",
    "HLT_REGION_PAIR_COMPONENTS",
    "JET_COMPONENTS",
    "RELATION_CHANNELS",
    "RELATION_COMPONENTS",
    "TRACK_COMPONENTS",
    "TRACK_COMPONENT_PROXY_COMPONENTS",
    "TargetDeclaration",
    "target_component_availability_groups",
    "target_declarations",
]
