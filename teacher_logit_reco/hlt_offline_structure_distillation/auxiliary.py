"""Exact Stage-D auxiliary graphs, losses, controls, and frozen selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    mean_log_selection_rejection,
)

from .baselines import component_seed
from .contracts import (
    AUXILIARY_OBJECTIVE_CONTRACT,
    AUXILIARY_PREDICTION_CONTRACT,
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_D_PLAN_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .heads import GlobalTargetHead, PairTargetHead
from .heads import deterministic_pair_indices
from .taps import HBaseParticleTransformer
from .target_schemas import (
    RELATION_CHANNELS,
    target_component_availability_groups,
    target_declarations,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


AUXILIARY_WEIGHTS = (0.10, 0.30, 1.00)
NULL_CONTROL_KINDS = (
    "DISABLED",
    "STOP_ENCODER",
    "TARGET_MEAN",
    "GLOBAL_SHUFFLE",
    "WITHIN_CLASS_SHUFFLE",
)
GLOBAL_PHYSICAL_TARGETS = (
    "T_OFFLINE_JET_10",
    "T_OFFLINE_COMPOSITION_16",
    "T_OFFLINE_TRACK_32",
    "T_OFFLINE_DENSITY_22",
    "T_OFFLINE_CA_TREE_26",
    "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
)
RELATION_TARGETS = tuple(
    f"T_OFFLINE_RELATION_{family}"
    for family in ("BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
)
CONTINUOUS_RELATION_TARGETS = tuple(
    target_id for target_id in RELATION_TARGETS if not target_id.endswith("_PID")
)
LOGIT_TARGETS = ("T_OFFLINE_LOGITS_O_BASE", "T_OFFLINE_LOGITS_O_FULLREL")
LATENT_TARGETS = ("T_OFFLINE_POOLED_LATENT",)
PAIR_TARGETS = ("T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8")
PRIMARY_TARGETS = (
    *GLOBAL_PHYSICAL_TARGETS,
    *RELATION_TARGETS,
    *LOGIT_TARGETS,
    *LATENT_TARGETS,
    *PAIR_TARGETS,
)
HLT_SELF_TARGETS = (
    "T_HLT_SELF_JET_10",
    "T_HLT_SELF_COMPOSITION_16",
    "T_HLT_SELF_TRACK_32",
    "T_HLT_SELF_DENSITY_22",
    "T_HLT_SELF_CA_TREE_26",
    "T_HLT_SELF_TRACK_COMPONENT_PROXY_17",
    *(
        f"T_HLT_SELF_RELATION_{family}"
        for family in ("BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
    ),
)


def _row_id(prefix: str, semantics: Mapping[str, Any]) -> str:
    return f"{prefix}_{canonical_sha256(semantics)[:16]}"


def _row(
    *,
    target_id: str,
    parameterization: str,
    weight: float | None,
    row_kind: str,
    phase: str,
    head_type: str,
    source_target_id: str | None = None,
    resolved: bool = True,
) -> dict[str, Any]:
    semantic = {
        "target_id": target_id,
        "parameterization": parameterization,
        "auxiliary_weight": weight,
        "row_kind": row_kind,
        "phase": phase,
        "source_target_id": source_target_id,
    }
    canonical_graph = (
        f"{source_target_id or target_id}||{parameterization}||"
        f"{0.30 if weight is None else weight}"
    )
    row_id = _row_id("AUX", semantic)
    return {
        "row_id": row_id,
        "target_id": target_id,
        "source_target_id": source_target_id,
        "parameterization": parameterization,
        "auxiliary_weight": weight,
        "row_kind": row_kind,
        "phase": phase,
        "head_type": head_type,
        "tap": "TAP_MID" if head_type == "pair" else "TAP_LATE",
        "pipeline_seed": 101,
        "encoder_component_seed": component_seed(101, "encoder", "H_BASE"),
        "head_component_seed": component_seed(101, "target_head", canonical_graph),
        "resolved": bool(resolved),
        "selection_eligible": row_kind == "SCIENTIFIC",
        "performance_can_omit_or_cancel": False,
        "fixed_epoch_budget": 40,
    }


def _target_rows(target_registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validate_content_hash(
        target_registry, expected_contract="hosd_structure_target_registry_v1"
    )
    rows = {
        str(row["target_id"]): row
        for row in target_registry.get("targets", ())
        if bool(row.get("executable_current_source"))
    }
    missing = sorted((set(PRIMARY_TARGETS) | set(HLT_SELF_TARGETS)) - set(rows))
    if missing:
        raise ValueError(f"Stage-D required target coverage differs: {missing}")
    return rows


def build_stage_d_plan(
    *,
    campaign_spec_sha256: str,
    target_registry: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the complete bounded matrix without reading target performance."""

    targets = _target_rows(target_registry)
    scientific: list[dict[str, Any]] = []
    for target_id in GLOBAL_PHYSICAL_TARGETS:
        for parameterization in ("ABS", "RES", "HET"):
            for weight in AUXILIARY_WEIGHTS:
                scientific.append(
                    _row(
                        target_id=target_id,
                        parameterization=parameterization,
                        weight=weight,
                        row_kind="SCIENTIFIC",
                        phase="PRIMARY",
                        head_type="global",
                    )
                )
    for target_id in RELATION_TARGETS:
        for parameterization in ("ABS", "RES"):
            for weight in AUXILIARY_WEIGHTS:
                scientific.append(
                    _row(
                        target_id=target_id,
                        parameterization=parameterization,
                        weight=weight,
                        row_kind="SCIENTIFIC",
                        phase="PRIMARY",
                        head_type="global",
                    )
                )
    for weight in AUXILIARY_WEIGHTS:
        scientific.append(
            _row(
                target_id="LOCKED_BEST_CONTINUOUS_RELATION",
                parameterization="HET",
                weight=weight,
                row_kind="SCIENTIFIC",
                phase="LOCKED_RELATION_HET",
                head_type="global",
                resolved=False,
            )
        )
    for target_id in PAIR_TARGETS:
        for weight in AUXILIARY_WEIGHTS:
            scientific.append(
                _row(
                    target_id=target_id,
                    parameterization="ABS",
                    weight=weight,
                    row_kind="SCIENTIFIC",
                    phase="PRIMARY",
                    head_type="pair",
                )
            )
    for weight in AUXILIARY_WEIGHTS:
        scientific.append(
            _row(
                target_id=LATENT_TARGETS[0],
                parameterization="WHITENED_ABS",
                weight=weight,
                row_kind="SCIENTIFIC",
                phase="PRIMARY",
                head_type="global",
            )
        )
    for target_id in LOGIT_TARGETS:
        scientific.append(
            _row(
                target_id=target_id,
                parameterization="KD",
                weight=1.0,
                row_kind="SCIENTIFIC",
                phase="PRIMARY",
                head_type="none",
            )
        )

    hlt_self = [
        _row(
            target_id=target_id,
            parameterization="MATCHED_LOCK",
            weight=None,
            row_kind="HLT_SELF",
            phase="MATCHED_HLT_SELF",
            head_type="global",
            source_target_id=target_id.replace("T_HLT_SELF_", "T_OFFLINE_", 1),
            resolved=False,
        )
        for target_id in HLT_SELF_TARGETS
    ]
    controls: list[dict[str, Any]] = []
    for target_id in PRIMARY_TARGETS:
        target = targets[target_id]
        head_type = str(target.get("head_type", "global"))
        canonical_parameterization = (
            "KD"
            if target_id in LOGIT_TARGETS
            else "WHITENED_ABS"
            if target_id in LATENT_TARGETS
            else "ABS"
        )
        canonical_weight = 1.0 if target_id in LOGIT_TARGETS else 0.30
        for kind in NULL_CONTROL_KINDS:
            control = _row(
                target_id=target_id,
                parameterization=canonical_parameterization,
                weight=0.0 if kind == "DISABLED" else canonical_weight,
                row_kind=kind,
                phase="NULL_CONTROLS",
                head_type=head_type,
            )
            control["head_component_seed"] = component_seed(
                101,
                "target_head",
                f"{target_id}||{canonical_parameterization}||{canonical_weight}",
            )
            controls.append(control)
    if (
        len(scientific) != 110
        or len(hlt_self) != 13
        or len(controls) != 90
    ):
        raise AssertionError("Stage-D matrix count drifted")
    rows = scientific + hlt_self + controls
    if len(rows) != 213:
        raise AssertionError("Stage-D executable row count drifted")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise AssertionError("Stage-D row IDs are not unique")
    return with_content_hash(
        {
            "contract": STAGE_D_PLAN_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "target_registry_sha256": target_registry["content_hash"],
            "scientific_rows": scientific,
            "hlt_self_rows": hlt_self,
            "control_rows": controls,
            "all_rows": rows,
            "row_count": len(rows),
            "hard_maximum": 213,
            "phase_order": [
                "PRIMARY_AND_NULL_CONTROLS",
                "SINGLE_FAMILY_PHASE_LOCK",
                "LOCKED_RELATION_HET_AND_MATCHED_HLT_SELF",
                "FINAL_SINGLE_FAMILY_SELECTION",
            ],
            "matrix_counts": {
                "global_abs_res_het_weight": 54,
                "relation_abs_res": 42,
                "locked_best_relation_het": 3,
                "pair": 6,
                "latent": 3,
                "kd": 2,
                "matched_hlt_self": 13,
                "null_controls": 90,
            },
            "all_targets_continue_regardless_of_stage_c": True,
            "performance_can_cancel_or_omit": False,
        }
    )


def validate_stage_d_plan(
    payload: Mapping[str, Any],
    *,
    target_registry: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(payload, expected_contract=STAGE_D_PLAN_CONTRACT)
    if int(payload.get("row_count", -1)) > int(payload.get("hard_maximum", -1)):
        raise ValueError("Stage-D plan exceeds its hard bound")
    if target_registry is not None:
        expected = build_stage_d_plan(
            campaign_spec_sha256=payload["campaign_spec_sha256"],
            target_registry=target_registry,
            source=payload["source"],
        )
        if dict(payload) != expected:
            raise ValueError("Stage-D plan differs from exact current contract")
    return digest


def build_auxiliary_model(
    row: Mapping[str, Any],
    *,
    weaver_module: Any | None = None,
    input_dimension: int = 128,
) -> tuple["AuxiliaryHBaseClassifier", tuple[str, ...]]:
    """Initialize matched encoders and independently seeded target heads."""

    if not bool(row.get("resolved")):
        raise ValueError("cannot build an unresolved Stage-D row")
    declarations = {item.target_id: item for item in target_declarations()}
    target_id = str(row["target_id"])
    if target_id not in declarations:
        raise ValueError("Stage-D target has no declaration")
    declaration = declarations[target_id]
    heteroscedastic = heteroscedastic_component_mask(
        target_id, declaration.components
    )
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(row["encoder_component_seed"]))
        classifier = HBaseParticleTransformer(weaver_module=weaver_module)
        torch.manual_seed(int(row["head_component_seed"]))
        groups = target_component_availability_groups(
            target_id, declaration.components
        )
        group_order = tuple(dict.fromkeys(groups))
        model = AuxiliaryHBaseClassifier(
            classifier,
            target_id=target_id,
            target_dimension=len(declaration.components),
            input_dimension=input_dimension,
            head_type=str(row["head_type"]),
            parameterization=str(row["parameterization"]),
            availability_group_count=max(1, len(group_order)),
            heteroscedastic_components=heteroscedastic,
            symmetric_pair=declaration.symmetry == "symmetric",
            stop_encoder=row["row_kind"] == "STOP_ENCODER",
        )
    return model, groups


def heteroscedastic_component_mask(
    target_id: str, components: Sequence[str]
) -> tuple[bool, ...]:
    prefix = next(
        (
            value
            for value in ("T_OFFLINE_RELATION_", "T_HLT_SELF_RELATION_")
            if str(target_id).startswith(value)
        ),
        None,
    )
    if prefix is None:
        return tuple(True for _ in components)
    family = str(target_id).removeprefix(prefix)
    kinds = {
        channel: kind for channel, kind, _ in RELATION_CHANNELS[family]
    }
    return tuple(
        kinds[str(component).split("__", 1)[0]].endswith("_continuous")
        for component in components
    )


class AuxiliaryHBaseClassifier(torch.nn.Module if torch is not None else object):
    """Classification-isolated H_BASE plus one training-only target head."""

    def __init__(
        self,
        classifier: Any,
        *,
        target_id: str,
        target_dimension: int,
        input_dimension: int = 128,
        head_type: str = "global",
        parameterization: str = "ABS",
        availability_group_count: int = 1,
        heteroscedastic_components: Sequence[bool] | None = None,
        symmetric_pair: bool = False,
        stop_encoder: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD auxiliary models")
        super().__init__()
        if head_type not in {"global", "pair", "none"}:
            raise ValueError("unknown auxiliary head type")
        self.classifier = classifier
        self.target_id = str(target_id)
        self.parameterization = str(parameterization)
        self.head_type = head_type
        self.tap = "TAP_MID" if head_type == "pair" else "TAP_LATE"
        self.stop_encoder = bool(stop_encoder)
        if head_type == "global":
            self.target_head = GlobalTargetHead(
                target_dimension,
                input_dimension=input_dimension,
                availability_groups=availability_group_count,
                heteroscedastic=parameterization == "HET",
                heteroscedastic_components=heteroscedastic_components,
            )
        elif head_type == "pair":
            self.target_head = PairTargetHead(
                input_dimension, target_dimension, symmetric=symmetric_pair
            )
        else:
            self.target_head = None

    def forward(self, points: Any, features: Any, vectors: Any, mask: Any) -> Any:
        """The deployable path is exactly the classifier and never calls the head."""

        return self.classifier(points, features, vectors, mask)

    def forward_with_aux(
        self,
        points: Any,
        features: Any,
        vectors: Any,
        mask: Any,
        *,
        sampled_pair_indices: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if self.head_type == "none":
            logits = self.classifier(points, features, vectors, mask)
            return logits, {"value": logits.detach() if self.stop_encoder else logits}
        split = self.classifier.forward_with_taps(
            points, features, vectors, mask, capture=(self.tap,)
        )
        state = split.states[self.tap]
        if self.stop_encoder:
            state = state.detach()
        if self.head_type == "global":
            prediction = self.target_head(state, split.masks[self.tap])
        else:
            if sampled_pair_indices is None:
                value, pair_mask = self.target_head(state, split.masks[self.tap])
                prediction = {"value": value, "pair_mask": pair_mask}
            else:
                value = self.target_head.forward_pairs(
                    state,
                    sampled_pair_indices["event_indices"],
                    sampled_pair_indices["left_indices"],
                    sampled_pair_indices["right_indices"],
                )
                prediction = {
                    "value": value,
                    "pair_mask": None,
                    "sampled_pair_indices": sampled_pair_indices,
                }
        return split.logits, prediction

    def shared_parameters(self) -> tuple[Any, ...]:
        return tuple(self.classifier.parameters())

    def head_parameters(self) -> tuple[Any, ...]:
        return (
            ()
            if self.target_head is None
            else tuple(self.target_head.parameters())
        )


def availability_targets(
    component_mask: Any, component_group_ids: Sequence[str]
) -> tuple[Any, tuple[str, ...]]:
    if component_mask.ndim != 2 or component_mask.shape[1] != len(
        component_group_ids
    ):
        raise ValueError("availability mask/group shape differs")
    order = tuple(dict.fromkeys(str(value) for value in component_group_ids))
    value = torch.stack(
        [
            component_mask[
                :, [name == group for name in component_group_ids]
            ].bool().any(dim=1)
            for group in order
        ],
        dim=1,
    )
    return value.to(dtype=torch.float32), order


def _masked_per_event(value: Any, mask: Any) -> tuple[Any, Any]:
    valid = mask.bool()
    flat_value = value.reshape(value.shape[0], -1)
    flat_mask = valid.reshape(valid.shape[0], -1)
    count = flat_mask.sum(dim=1)
    reduced = flat_value.masked_fill(~flat_mask, 0).sum(dim=1) / count.clamp_min(1)
    return reduced, count > 0


def _mean_applicable(per_event: Any, applicable: Any) -> Any:
    return (
        per_event[applicable].mean()
        if bool(applicable.any())
        else per_event.sum() * 0
    )


def _temperature_2_kl(student: Any, teacher: Any) -> Any:
    target_probability = torch.nn.functional.softmax(teacher.detach() / 2.0, dim=-1)
    per_event = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student / 2.0, dim=-1),
        target_probability,
        reduction="none",
    ).sum(dim=-1) * 4.0
    return per_event.mean()


def global_auxiliary_loss(
    prediction: Mapping[str, Any],
    target: Any,
    mask: Any,
    *,
    parameterization: str,
    component_group_ids: Sequence[str],
    target_id: str,
) -> tuple[Any, dict[str, Any]]:
    if target.shape != mask.shape or target.ndim != 2:
        raise ValueError("global auxiliary target must be [B,C]")
    if parameterization == "KD":
        value_loss = _temperature_2_kl(prediction["value"], target)
        return value_loss, {
            "value_loss": value_loss,
            "availability_loss": value_loss.detach() * 0,
            "family_loss": value_loss,
        }
    if parameterization == "HET":
        nll = 0.5 * (
            torch.exp(-prediction["log_variance"].clamp(-8.0, 5.0))
            * (target - prediction["mean"]).square()
            + prediction["log_variance"].clamp(-8.0, 5.0)
        )
        point_huber = torch.nn.functional.huber_loss(
            prediction["mean"], target, delta=1.0, reduction="none"
        )
        heteroscedastic_mask = prediction[
            "heteroscedastic_component_mask"
        ].bool().unsqueeze(0)
        point = torch.where(heteroscedastic_mask, nll, point_huber)
    else:
        point = torch.nn.functional.huber_loss(
            prediction["value"], target, delta=1.0, reduction="none"
        )
    per_event, applicable = _masked_per_event(point, mask)
    value_loss = _mean_applicable(per_event, applicable)
    if parameterization == "WHITENED_ABS":
        selected = mask.bool()
        predicted = prediction["value"].masked_fill(~selected, 0)
        truth = target.masked_fill(~selected, 0)
        cosine = 1.0 - torch.nn.functional.cosine_similarity(
            predicted, truth, dim=-1, eps=1.0e-8
        )
        cosine_loss = _mean_applicable(cosine, applicable)
        value_loss = 0.5 * (value_loss + cosine_loss)
    available, order = availability_targets(mask, component_group_ids)
    if prediction["availability_logits"].shape != available.shape:
        raise ValueError("availability prediction shape differs")
    availability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["availability_logits"], available
    )
    family = 0.5 * (value_loss + availability_loss)
    return family, {
        "value_loss": value_loss,
        "availability_loss": availability_loss,
        "family_loss": family,
        "availability_group_order": order,
        "target_id": target_id,
    }


def pair_auxiliary_loss(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    target_id: str,
) -> tuple[Any, dict[str, Any]]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("pair prediction, target, and mask shapes differ")
    if target.ndim != 4:
        raise ValueError("pair targets must be [B,N,N,C]")
    binary = 3 if target_id == "T_HLT_REGION_PAIR_8" else 0
    pieces = []
    if binary:
        pieces.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[..., :binary],
                target[..., :binary],
                reduction="none",
            )
        )
    if binary < target.shape[-1]:
        pieces.append(
            torch.nn.functional.huber_loss(
                prediction[..., binary:],
                target[..., binary:],
                delta=1.0,
                reduction="none",
            )
        )
    point = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
    per_event, applicable = _masked_per_event(point, mask)
    family = _mean_applicable(per_event, applicable)
    return family, {
        "value_loss": family,
        "availability_loss": family.detach() * 0,
        "family_loss": family,
        "binary_channel_count": binary,
        "pair_reduction": "mean_valid_components_and_pairs_per_jet_then_mean_jets",
    }


@dataclass(frozen=True)
class SampledPairBatch:
    event_indices: Any
    left_indices: Any
    right_indices: Any
    target: Any
    mask: Any
    positive: Any | None

    def indices(self) -> dict[str, Any]:
        return {
            "event_indices": self.event_indices,
            "left_indices": self.left_indices,
            "right_indices": self.right_indices,
        }


def build_sampled_pair_batch(
    target: Any,
    mask: Any,
    *,
    identities: Sequence[str],
    epoch: int,
    target_id: str,
) -> SampledPairBatch:
    """Select exact per-event pairs without invoking an RNG library."""

    if target.shape != mask.shape or target.ndim != 4:
        raise ValueError("pair sampling requires equal [B,N,N,C] arrays")
    if len(identities) != target.shape[0]:
        raise ValueError("pair sampling identity count differs")
    binary = 3 if target_id == "T_HLT_REGION_PAIR_8" else 0
    events, lefts, rights, values, masks, strata = [], [], [], [], [], []
    for event, identity in enumerate(identities):
        applicable = mask[event].bool().any(dim=-1)
        coordinate = applicable.nonzero(as_tuple=False)
        pair_ids = [
            f"{int(pair[0])}:{int(pair[1])}" for pair in coordinate
        ]
        positive = (
            None
            if binary == 0
            else [
                bool(
                    (
                        target[event, int(pair[0]), int(pair[1]), :binary]
                        >= 0.5
                    ).any()
                )
                for pair in coordinate
            ]
        )
        selected = deterministic_pair_indices(
            epoch=epoch,
            identity=str(identity),
            target_id=target_id,
            pair_ids=pair_ids,
            positive=positive,
        )
        for index in selected:
            left, right = (int(value) for value in coordinate[index].tolist())
            events.append(event)
            lefts.append(left)
            rights.append(right)
            values.append(target[event, left, right])
            masks.append(mask[event, left, right])
            if positive is not None:
                strata.append(positive[index])
    device = target.device
    empty_target = target.new_zeros((0, target.shape[-1]))
    return SampledPairBatch(
        event_indices=torch.as_tensor(events, dtype=torch.long, device=device),
        left_indices=torch.as_tensor(lefts, dtype=torch.long, device=device),
        right_indices=torch.as_tensor(rights, dtype=torch.long, device=device),
        target=torch.stack(values) if values else empty_target,
        mask=(
            torch.stack(masks)
            if masks
            else torch.zeros_like(empty_target, dtype=torch.bool)
        ),
        positive=(
            None
            if binary == 0
            else torch.as_tensor(strata, dtype=torch.bool, device=device)
        ),
    )


def sampled_pair_auxiliary_loss(
    prediction: Any,
    sampled: SampledPairBatch,
    *,
    target_id: str,
    event_count: int,
) -> tuple[Any, dict[str, Any]]:
    if prediction.shape != sampled.target.shape:
        raise ValueError("sampled pair prediction shape differs")
    binary = 3 if target_id == "T_HLT_REGION_PAIR_8" else 0
    channel_losses = []
    if binary:
        channel_losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[:, :binary],
                sampled.target[:, :binary],
                reduction="none",
            )
        )
    if binary < prediction.shape[-1]:
        channel_losses.append(
            torch.nn.functional.huber_loss(
                prediction[:, binary:],
                sampled.target[:, binary:],
                delta=1.0,
                reduction="none",
            )
        )
    point = (
        torch.cat(channel_losses, dim=-1)
        if len(channel_losses) > 1
        else channel_losses[0]
    )
    pair_loss, applicable = _masked_per_event(point, sampled.mask)
    event_values = []
    for event in range(int(event_count)):
        selected = (sampled.event_indices == event) & applicable
        if not bool(selected.any()):
            continue
        if sampled.positive is None:
            event_values.append(pair_loss[selected].mean())
            continue
        positive = selected & sampled.positive
        negative = selected & ~sampled.positive
        if bool(positive.any()) and bool(negative.any()):
            event_values.append(
                0.5 * pair_loss[positive].mean()
                + 0.5 * pair_loss[negative].mean()
            )
        elif bool(positive.any()):
            event_values.append(pair_loss[positive].mean())
        else:
            event_values.append(pair_loss[negative].mean())
    family = (
        torch.stack(event_values).mean()
        if event_values
        else prediction.sum() * 0
    )
    return family, {
        "value_loss": family,
        "availability_loss": family.detach() * 0,
        "family_loss": family,
        "binary_channel_count": binary,
        "sampled_pair_count": int(prediction.shape[0]),
        "pair_reduction": (
            "per_jet_equal_positive_negative_strata_when_both_exist"
        ),
    }


def auxiliary_objective(
    *,
    logits: Any,
    labels: Any,
    prediction: Mapping[str, Any],
    target: Any,
    target_mask: Any,
    target_id: str,
    parameterization: str,
    auxiliary_weight: float,
    component_group_ids: Sequence[str] | None = None,
    sampled_pair_batch: SampledPairBatch | None = None,
) -> tuple[Any, dict[str, Any]]:
    classification = torch.nn.functional.cross_entropy(logits, labels.long())
    if target_id in PAIR_TARGETS:
        if sampled_pair_batch is None:
            auxiliary, pieces = pair_auxiliary_loss(
                prediction["value"], target, target_mask, target_id=target_id
            )
        else:
            auxiliary, pieces = sampled_pair_auxiliary_loss(
                prediction["value"],
                sampled_pair_batch,
                target_id=target_id,
                event_count=int(logits.shape[0]),
            )
    else:
        groups = (
            tuple(component_group_ids)
            if component_group_ids is not None
            else target_component_availability_groups(
                target_id, tuple(f"component_{index}" for index in range(target.shape[-1]))
            )
        )
        auxiliary, pieces = global_auxiliary_loss(
            prediction,
            target,
            target_mask,
            parameterization=parameterization,
            component_group_ids=groups,
            target_id=target_id,
        )
    total = classification + float(auxiliary_weight) * auxiliary
    return total, {
        "classification_loss": classification,
        "auxiliary_loss": auxiliary,
        "weighted_auxiliary_loss": float(auxiliary_weight) * auxiliary,
        "total_loss": total,
        **pieces,
    }


def auxiliary_objective_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": AUXILIARY_OBJECTIVE_CONTRACT,
            "schema_version": 1,
            "classification": "ordinary_unweighted_per_event_ten_class_CE",
            "family_reduction": (
                "mean_valid_components_per_jet_then_mean_jets_with_any_valid"
            ),
            "continuous": "normalized_Huber_delta_1",
            "heteroscedastic": {
                "loss": "diagonal_Gaussian_NLL",
                "log_variance_clip": [-8.0, 5.0],
                "relation_scope": (
                    "continuous_aggregate_components_only;"
                    "bounded_frequency_components_retain_Huber"
                ),
            },
            "availability": (
                "BCE_mean_groups_and_jets_equal_weight_with_value_subtask"
            ),
            "latent": "equal_weight_normalized_Huber_and_one_minus_cosine",
            "teacher_logits": "T2_KL_temperature_2_lambda_1",
            "pair": {
                "reduction": "per_jet",
                "track": "13_normalized_Huber_channels",
                "region": "3_BCE_plus_5_normalized_Huber_channels",
                "validation": "all_applicable_pairs",
            },
            "weights": list(AUXILIARY_WEIGHTS),
            "classification_isolation": True,
        }
    )


def _selection_key(
    row: Mapping[str, Any], *, within_family: bool
) -> tuple[Any, ...]:
    metrics = row["design_select"]["classification_metrics"]
    return (
        -mean_log_selection_rejection(metrics),
        float(metrics["cross_entropy"]),
        float(row["deployed_analytical_flops"]),
        int(row["deployed_parameter_count"]),
        float(row["training_gpu_hours"]),
        (
            -float(row.get("standardized_improvement_over_prior", 0.0))
            if within_family
            else 0.0
        ),
        str(row["row_id"]),
    )


def select_utility_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if bool(row.get("selection_eligible", True))]
    if not eligible:
        raise ValueError("single-family selector has no eligible rows")
    maximum = max(
        float(row["design_select"]["classification_metrics"]["macro_per_class_accuracy"])
        for row in eligible
    )
    window = [
        row
        for row in eligible
        if maximum
        - float(
            row["design_select"]["classification_metrics"][
                "macro_per_class_accuracy"
            ]
        )
        <= 0.0001
    ]
    within_family = len({str(row["target_id"]) for row in eligible}) == 1
    return min(
        window,
        key=lambda row: _selection_key(row, within_family=within_family),
    )


def build_single_family_phase_lock(
    *,
    stage_d_plan: Mapping[str, Any],
    primary_results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_stage_d_plan(stage_d_plan)
    by_id = {str(row["row_id"]): row for row in primary_results}
    required = [
        row
        for row in stage_d_plan["scientific_rows"]
        if row["phase"] == "PRIMARY"
    ]
    if set(by_id) != {row["row_id"] for row in required}:
        raise ValueError("phase-lock primary result coverage differs")
    for row in by_id.values():
        validate_content_hash(
            row, expected_contract=AUXILIARY_PREDICTION_CONTRACT
        )
        if (
            row.get("source") != dict(source)
            or row.get("stage_d_plan_sha256") != stage_d_plan["content_hash"]
            or row.get("campaign_spec_sha256")
            != stage_d_plan["campaign_spec_sha256"]
        ):
            raise ValueError("phase-lock primary result lineage differs")
    expected_by_id = {row["row_id"]: row for row in required}
    for row_id, result in by_id.items():
        expected = expected_by_id[row_id]
        if any(
            result.get(key) != expected.get(key)
            for key in (
                "target_id",
                "parameterization",
                "auxiliary_weight",
                "row_kind",
            )
        ):
            raise ValueError("phase-lock primary result semantics differ")
    best_by_target = {}
    for target_id in (
        *GLOBAL_PHYSICAL_TARGETS,
        *RELATION_TARGETS,
    ):
        best_by_target[target_id] = dict(
            select_utility_row(
                [by_id[row["row_id"]] for row in required if row["target_id"] == target_id]
            )
        )
    relation_winner = select_utility_row(
        [
            best_by_target[target_id]
            for target_id in CONTINUOUS_RELATION_TARGETS
        ]
    )
    return with_content_hash(
        {
            "contract": SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "stage_d_plan_sha256": stage_d_plan["content_hash"],
            "primary_result_hashes": {
                row_id: require_sha256(
                    row["content_hash"], name=f"primary_result.{row_id}"
                )
                for row_id, row in sorted(by_id.items())
            },
            "best_offline_physical_by_target": {
                target_id: {
                    "row_id": row["row_id"],
                    "parameterization": row["parameterization"],
                    "auxiliary_weight": row["auxiliary_weight"],
                    "head_component_seed": next(
                        candidate["head_component_seed"]
                        for candidate in required
                        if candidate["row_id"] == row["row_id"]
                    ),
                }
                for target_id, row in sorted(best_by_target.items())
            },
            "locked_best_continuous_relation_target_id": relation_winner["target_id"],
            "selector": {
                "accuracy_window": 0.0001,
                "primary": "macro_per_class_accuracy",
                "tie_order": [
                    "higher_mean_log_Jeffreys_rejection",
                    "lower_cross_entropy",
                    "lower_deployed_analytical_flops",
                    "fewer_deployed_parameters",
                    "lower_training_gpu_hours",
                    (
                        "higher_standardized_improvement_over_P_PRIOR_"
                        "within_family_only"
                    ),
                    "lexicographically_smaller_row_id",
                ],
                "target_error_used_cross_family": False,
            },
        }
    )


def resolve_stage_d_phase_two(
    *,
    stage_d_plan: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_stage_d_plan(stage_d_plan)
    validate_content_hash(
        phase_lock, expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT
    )
    if phase_lock["stage_d_plan_sha256"] != stage_d_plan["content_hash"]:
        raise ValueError("Stage-D phase lock belongs to a different plan")
    output = []
    winner = phase_lock["locked_best_continuous_relation_target_id"]
    for unresolved in stage_d_plan["scientific_rows"]:
        if unresolved["phase"] != "LOCKED_RELATION_HET":
            continue
        output.append(
            {
                **unresolved,
                "target_id": winner,
                "source_target_id": winner,
                "head_component_seed": component_seed(
                    101,
                    "target_head",
                    f"{winner}||HET||{unresolved['auxiliary_weight']}",
                ),
                "resolved": True,
                "phase_lock_sha256": phase_lock["content_hash"],
            }
        )
    choices = phase_lock["best_offline_physical_by_target"]
    for unresolved in stage_d_plan["hlt_self_rows"]:
        source_target = unresolved["source_target_id"]
        choice = choices[source_target]
        output.append(
            {
                **unresolved,
                "parameterization": (
                    "HET" if choice["parameterization"] == "HET" else "ABS"
                ),
                "auxiliary_weight": choice["auxiliary_weight"],
                "head_component_seed": choice["head_component_seed"],
                "resolved": True,
                "matched_offline_row_id": choice["row_id"],
                "phase_lock_sha256": phase_lock["content_hash"],
            }
        )
    if len(output) != 16 or not all(row["resolved"] for row in output):
        raise AssertionError("Stage-D phase-two resolution differs")
    return output


def build_single_family_selection(
    *,
    stage_d_plan: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_stage_d_plan(stage_d_plan)
    validate_content_hash(
        phase_lock, expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT
    )
    phase_two = resolve_stage_d_phase_two(
        stage_d_plan=stage_d_plan, phase_lock=phase_lock
    )
    primary_rows = [
        row
        for row in stage_d_plan["scientific_rows"]
        if row["phase"] == "PRIMARY"
    ]
    eligible_rows = primary_rows + [
        row for row in phase_two if row["row_kind"] == "SCIENTIFIC"
    ]
    complete_rows = primary_rows + list(stage_d_plan["control_rows"]) + phase_two
    by_id = {str(row["row_id"]): row for row in results}
    if set(by_id) != {row["row_id"] for row in complete_rows}:
        raise ValueError("final Stage-D result/control coverage differs")
    for row in by_id.values():
        validate_content_hash(
            row, expected_contract=AUXILIARY_PREDICTION_CONTRACT
        )
        if (
            row.get("source") != dict(source)
            or row.get("stage_d_plan_sha256") != stage_d_plan["content_hash"]
            or row.get("campaign_spec_sha256")
            != stage_d_plan["campaign_spec_sha256"]
        ):
            raise ValueError("final Stage-D result lineage differs")
    expected_by_id = {row["row_id"]: row for row in complete_rows}
    for row_id, result in by_id.items():
        expected = expected_by_id[row_id]
        if any(
            result.get(key) != expected.get(key)
            for key in (
                "target_id",
                "parameterization",
                "auxiliary_weight",
                "row_kind",
            )
        ):
            raise ValueError("final Stage-D result semantics differ")
    selected_by_target = {}
    for target_id in PRIMARY_TARGETS:
        candidates = [
            by_id[row["row_id"]]
            for row in eligible_rows
            if row["target_id"] == target_id
        ]
        if candidates:
            selected_by_target[target_id] = select_utility_row(candidates)["row_id"]
    selected_results = [by_id[row_id] for row_id in selected_by_target.values()]
    winner = select_utility_row(selected_results)
    remaining = list(selected_results)
    cross_family_order = []
    while remaining:
        choice = select_utility_row(remaining)
        cross_family_order.append(
            {
                "ordinal": len(cross_family_order),
                "target_id": choice["target_id"],
                "row_id": choice["row_id"],
            }
        )
        remaining = [
            row for row in remaining if row["row_id"] != choice["row_id"]
        ]
    expected_by_row_id = {row["row_id"]: row for row in eligible_rows}
    return with_content_hash(
        {
            "contract": SINGLE_FAMILY_SELECTION_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "stage_d_plan_sha256": stage_d_plan["content_hash"],
            "phase_lock_sha256": phase_lock["content_hash"],
            "complete_result_hashes": {
                row_id: require_sha256(row["content_hash"], name=f"result.{row_id}")
                for row_id, row in sorted(by_id.items())
            },
            "selected_row_by_target": dict(sorted(selected_by_target.items())),
            "selected_definition_by_target": {
                target_id: {
                    key: expected_by_row_id[row_id][key]
                    for key in (
                        "target_id",
                        "parameterization",
                        "auxiliary_weight",
                        "head_type",
                    )
                }
                for target_id, row_id in sorted(selected_by_target.items())
            },
            "cross_family_order": cross_family_order,
            "global_winner_row_id": winner["row_id"],
            "negative_results_permitted": True,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "AUXILIARY_WEIGHTS",
    "AuxiliaryHBaseClassifier",
    "CONTINUOUS_RELATION_TARGETS",
    "GLOBAL_PHYSICAL_TARGETS",
    "HLT_SELF_TARGETS",
    "LATENT_TARGETS",
    "LOGIT_TARGETS",
    "NULL_CONTROL_KINDS",
    "PAIR_TARGETS",
    "PRIMARY_TARGETS",
    "RELATION_TARGETS",
    "SampledPairBatch",
    "auxiliary_objective",
    "auxiliary_objective_contract",
    "availability_targets",
    "build_single_family_phase_lock",
    "build_single_family_selection",
    "build_auxiliary_model",
    "build_sampled_pair_batch",
    "build_stage_d_plan",
    "global_auxiliary_loss",
    "heteroscedastic_component_mask",
    "pair_auxiliary_loss",
    "resolve_stage_d_phase_two",
    "sampled_pair_auxiliary_loss",
    "select_utility_row",
    "validate_stage_d_plan",
]
