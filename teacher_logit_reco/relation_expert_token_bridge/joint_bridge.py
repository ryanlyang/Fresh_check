"""Joint all-bank RETB predictor graphs for Stage J."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


JOINT_VARIANTS = (
    "J0_INDEPENDENT",
    "J1_SHARED_CONTEXT",
    "J2_COUPLED_DECODER",
    "J3_INDEPENDENT_PLUS_ADAPTER",
    "J4_BRIDGE_FINETUNE",
    "J5_END_TO_END",
)
JOINT_INPUT_POLICY = "R_MULTI"
JOINT_DIMENSION = 128


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB joint bridges")
    return torch


def validate_common_view_metadata(
    *,
    identities: Sequence[str],
    replica_ids: Any,
    degraded_view_hashes: Sequence[str],
) -> None:
    module = _require_torch()
    ids = tuple(str(value) for value in identities)
    hashes = tuple(str(value) for value in degraded_view_hashes)
    hexadecimal = set("0123456789abcdef")
    replicas = module.as_tensor(replica_ids, dtype=module.int64)
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(hashes) != len(ids)
        or replicas.shape != (len(ids),)
        or bool(((replicas < 0) | (replicas > 3)).any())
        or any(
            len(value) != 64
            or any(character not in hexadecimal for character in value)
            for value in hashes
        )
    ):
        raise ValueError("joint bridge common-view metadata differs")


class SharedHLTContext(torch.nn.Module if torch is not None else object):
    """Materialize one typed HLT context and reuse it for every decoder."""

    def __init__(self) -> None:
        module = _require_torch()
        super().__init__()
        self.bank_projections = module.nn.ModuleDict(
            {
                expert: module.nn.LazyLinear(JOINT_DIMENSION)
                for expert in EXPERT_ORDER
            }
        )
        self.particle_projections = module.nn.ModuleDict(
            {
                source: module.nn.LazyLinear(JOINT_DIMENSION)
                for source in ("BASE4", "PT", "TRACK", "REGION")
            }
        )
        self.expert_embedding = module.nn.Embedding(
            len(EXPERT_ORDER), JOINT_DIMENSION
        )
        self.slot_embedding = module.nn.Embedding(16, JOINT_DIMENSION)
        self.source_embedding = module.nn.Embedding(2, JOINT_DIMENSION)
        self.forward_call_count = 0

    def forward(
        self,
        *,
        hlt_token_banks: Mapping[str, Any],
        unbiased_particle_states: Any,
        particle_mask: Any,
        relation_particle_states: Mapping[str, Any] | None = None,
        relation_particle_masks: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        module = _require_torch()
        self.forward_call_count += 1
        if set(hlt_token_banks) != set(EXPERT_ORDER):
            raise ValueError("shared HLT context bank coverage differs")
        if (
            unbiased_particle_states.ndim != 3
            or tuple(particle_mask.shape)
            != tuple(unbiased_particle_states.shape[:2])
            or bool((particle_mask.bool().sum(dim=1) == 0).any())
        ):
            raise ValueError("shared HLT particle context differs")
        rows, masks = [], []
        batch = int(unbiased_particle_states.shape[0])
        for index, expert in enumerate(EXPERT_ORDER):
            bank = hlt_token_banks[expert]
            if (
                bank.ndim != 3
                or int(bank.shape[0]) != batch
                or not 1 <= int(bank.shape[1]) <= 16
            ):
                raise ValueError("shared HLT token bank differs")
            slot_ids = module.arange(bank.shape[1], device=bank.device)
            rows.append(
                self.bank_projections[expert](bank)
                + self.expert_embedding.weight[index][None, None]
                + self.slot_embedding(slot_ids)[None]
                + self.source_embedding.weight[0][None, None]
            )
            masks.append(
                module.zeros(
                    bank.shape[:2], dtype=module.bool, device=bank.device
                )
            )
        rows.append(
            self.particle_projections["BASE4"](unbiased_particle_states)
            + self.source_embedding.weight[1][None, None]
        )
        masks.append(~particle_mask.bool())
        if (relation_particle_states is None) != (
            relation_particle_masks is None
        ):
            raise ValueError("shared relation-particle context is partial")
        if relation_particle_states is not None:
            expected = {"PT", "TRACK", "REGION"}
            if (
                set(relation_particle_states) != expected
                or set(relation_particle_masks) != expected
            ):
                raise ValueError(
                    "shared relation-particle context coverage differs"
                )
            for source in ("PT", "TRACK", "REGION"):
                values = relation_particle_states[source]
                mask = relation_particle_masks[source]
                if (
                    values.ndim != 3
                    or int(values.shape[0]) != batch
                    or tuple(mask.shape) != tuple(values.shape[:2])
                ):
                    raise ValueError(
                        "shared relation-particle context shape differs"
                    )
                rows.append(
                    self.particle_projections[source](values)
                    + self.source_embedding.weight[1][None, None]
                )
                masks.append(~mask.bool())
        return module.cat(rows, dim=1), module.cat(masks, dim=1)


class CoupledExpertDecoder(torch.nn.Module if torch is not None else object):
    """One non-autoregressive decoder over every expert/slot target query."""

    def __init__(
        self,
        *,
        allocation: Mapping[str, Sequence[int]],
        offline_slot_queries: Mapping[str, Any],
        uncertainty_widths: Mapping[str, int],
        dropout: float,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            set(allocation) != set(EXPERT_ORDER)
            or set(offline_slot_queries) != set(EXPERT_ORDER)
            or set(uncertainty_widths) != set(EXPERT_ORDER)
            or float(dropout) not in {0.0, 0.1}
        ):
            raise ValueError("coupled decoder configuration differs")
        self.allocation = {
            expert: [int(value) for value in allocation[expert]]
            for expert in EXPERT_ORDER
        }
        self.output_projections = module.nn.ModuleDict()
        self.uncertainty_heads = module.nn.ModuleDict()
        query_rows = []
        self.slices: dict[str, tuple[int, int]] = {}
        start = 0
        for expert_index, expert in enumerate(EXPERT_ORDER):
            k, d = self.allocation[expert]
            queries = module.as_tensor(
                offline_slot_queries[expert], dtype=module.float32
            )
            if tuple(queries.shape) != (k, d):
                raise ValueError("coupled decoder offline queries differ")
            query_rows.append(
                module.nn.functional.pad(
                    queries, (0, JOINT_DIMENSION - d)
                )
            )
            self.output_projections[expert] = module.nn.Linear(
                JOINT_DIMENSION, d
            )
            self.uncertainty_heads[expert] = module.nn.Linear(
                JOINT_DIMENSION, int(uncertainty_widths[expert])
            )
            self.slices[expert] = (start, start + k)
            start += k
        self.queries = module.nn.Parameter(module.cat(query_rows, dim=0))
        self.expert_embedding = module.nn.Embedding(
            len(EXPERT_ORDER), JOINT_DIMENSION
        )
        self.slot_embedding = module.nn.Embedding(16, JOINT_DIMENSION)
        expert_ids, slot_ids = [], []
        for expert_index, expert in enumerate(EXPERT_ORDER):
            k = self.allocation[expert][0]
            expert_ids.extend([expert_index] * k)
            slot_ids.extend(range(k))
        self.register_buffer(
            "query_expert_ids",
            module.as_tensor(expert_ids, dtype=module.int64),
        )
        self.register_buffer(
            "query_slot_ids",
            module.as_tensor(slot_ids, dtype=module.int64),
        )
        layer = module.nn.TransformerDecoderLayer(
            d_model=JOINT_DIMENSION,
            nhead=8,
            dim_feedforward=4 * JOINT_DIMENSION,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = module.nn.TransformerDecoder(layer, num_layers=3)
        self.output_norm = module.nn.LayerNorm(JOINT_DIMENSION)

    def forward(self, memory: Any, memory_padding_mask: Any) -> dict[str, Any]:
        queries = (
            self.queries
            + self.expert_embedding(self.query_expert_ids)
            + self.slot_embedding(self.query_slot_ids)
        )[None].expand(memory.shape[0], -1, -1)
        decoded = self.output_norm(
            self.decoder(
                queries,
                memory,
                memory_key_padding_mask=memory_padding_mask,
            )
        )
        tokens, log_variances = {}, {}
        for expert in EXPERT_ORDER:
            start, stop = self.slices[expert]
            row = decoded[:, start:stop]
            tokens[expert] = self.output_projections[expert](row)
            log_variances[expert] = self.uncertainty_heads[expert](row).clamp(
                -8.0, 4.0
            )
        return {"predicted_tokens": tokens, "log_variance": log_variances}


class ResidualJointAdapter(torch.nn.Module if torch is not None else object):
    """J3-only deployable residual adapter with an exact identity start."""

    def __init__(self, *, allocation: Mapping[str, Sequence[int]]) -> None:
        module = _require_torch()
        super().__init__()
        self.projections = module.nn.ModuleDict(
            {
                expert: module.nn.Linear(
                    int(allocation[expert][1]), JOINT_DIMENSION
                )
                for expert in EXPERT_ORDER
            }
        )
        self.network = module.nn.Sequential(
            module.nn.LayerNorm(len(EXPERT_ORDER) * JOINT_DIMENSION),
            module.nn.Linear(
                len(EXPERT_ORDER) * JOINT_DIMENSION, 2 * JOINT_DIMENSION
            ),
            module.nn.GELU(),
            module.nn.Linear(2 * JOINT_DIMENSION, 10),
        )
        self.residual_scale = module.nn.Parameter(module.zeros(()))

    def forward(self, banks: Mapping[str, Any], base_logits: Any) -> Any:
        pooled = [
            self.projections[expert](banks[expert]).mean(dim=1)
            for expert in EXPERT_ORDER
        ]
        correction = self.network(_require_torch().cat(pooled, dim=-1))
        return base_logits + self.residual_scale * correction


def _inverse_normalize(values: Any, mean: Any, std: Any) -> Any:
    return values * std[None] + mean[None]


class JointBridgeGraph(torch.nn.Module if torch is not None else object):
    """J0-J5 graph with explicit frozen/deployable component boundaries."""

    def __init__(
        self,
        *,
        variant: str,
        predictors: Mapping[str, Any],
        frozen_offline_fusion: Any,
        frozen_expert_heads: Mapping[str, Any],
        token_means: Mapping[str, Any],
        token_standard_deviations: Mapping[str, Any],
        hlt_experts: Mapping[str, Any] | None = None,
        deployable_fusion: Any | None = None,
        coupled_decoder: CoupledExpertDecoder | None = None,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            variant not in JOINT_VARIANTS
            or set(predictors) != set(EXPERT_ORDER)
            or set(frozen_expert_heads) != set(EXPERT_ORDER)
            or set(token_means) != set(EXPERT_ORDER)
            or set(token_standard_deviations) != set(EXPERT_ORDER)
        ):
            raise ValueError("joint bridge graph coverage differs")
        if variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"} and (
            hlt_experts is None or set(hlt_experts) != set(EXPERT_ORDER)
        ):
            raise ValueError("live bridge variants require all HLT experts")
        if variant == "J2_COUPLED_DECODER" and coupled_decoder is None:
            raise ValueError("J2 requires the coupled decoder")
        if variant == "J5_END_TO_END" and (
            deployable_fusion is None
            or deployable_fusion is frozen_offline_fusion
        ):
            raise ValueError(
                "J5 requires a distinct deployable fusion initialized from J4"
            )
        self.variant = variant
        self.input_policy = JOINT_INPUT_POLICY
        self.predictors = module.nn.ModuleDict(
            {expert: predictors[expert] for expert in EXPERT_ORDER}
        )
        self.frozen_expert_heads = module.nn.ModuleDict(
            {expert: frozen_expert_heads[expert] for expert in EXPERT_ORDER}
        )
        self.frozen_offline_fusion = frozen_offline_fusion
        self.deployable_fusion = (
            frozen_offline_fusion
            if deployable_fusion is None
            else deployable_fusion
        )
        self.hlt_experts = (
            None
            if hlt_experts is None
            else module.nn.ModuleDict(
                {expert: hlt_experts[expert] for expert in EXPERT_ORDER}
            )
        )
        self.shared_context = (
            SharedHLTContext()
            if variant
            in {
                "J1_SHARED_CONTEXT",
                "J2_COUPLED_DECODER",
                "J4_BRIDGE_FINETUNE",
                "J5_END_TO_END",
            }
            else None
        )
        self.shared_memory_projections = (
            module.nn.ModuleDict(
                {
                    expert: module.nn.Linear(
                        JOINT_DIMENSION,
                        int(self.predictors[expert].token_dimension),
                    )
                    for expert in EXPERT_ORDER
                }
            )
            if self.shared_context is not None
            and variant != "J2_COUPLED_DECODER"
            else None
        )
        self.coupled_decoder = coupled_decoder
        self.allocation = {
            expert: [
                int(self.predictors[expert].token_count),
                int(self.predictors[expert].token_dimension),
            ]
            for expert in EXPERT_ORDER
        }
        for index, expert in enumerate(EXPERT_ORDER):
            mean = module.as_tensor(token_means[expert], dtype=module.float32)
            std = module.as_tensor(
                token_standard_deviations[expert], dtype=module.float32
            )
            if (
                list(mean.shape) != self.allocation[expert]
                or tuple(std.shape) != tuple(mean.shape)
                or bool((std < 0).any())
            ):
                raise ValueError("joint bridge token normalizer differs")
            self.register_buffer(f"token_mean_{index}", mean)
            self.register_buffer(f"token_std_{index}", std)
        self.adapter = (
            ResidualJointAdapter(allocation=self.allocation)
            if variant == "J3_INDEPENDENT_PLUS_ADAPTER"
            else None
        )
        self._semantic_predictor_control = "active"

    def set_semantic_predictor_control(self, mode: str) -> None:
        """Set a deterministic evaluation-only predictor evidence control."""
        if mode not in {
            "active",
            "zero_hlt_evidence",
            "shuffle_hlt_evidence_between_events",
            "remove_native_particle_context",
            "remove_noncorresponding_expert_banks",
        }:
            raise ValueError("predictor semantic control is unregistered")
        self._semantic_predictor_control = str(mode)

    def set_semantic_relation_transform(
        self, mode: str, *, expert_id: str | None = None
    ) -> None:
        """Apply one evaluation-only relation transform.

        ``expert_id`` scopes the perturbation to one biased expert.  This is
        required for the per-expert causal zero controls; all other experts
        are explicitly restored to ordinary inference.
        """
        if self.hlt_experts is None:
            raise ValueError("relation controls require live HLT experts")
        if expert_id is not None and (
            expert_id not in EXPERT_ORDER or expert_id == "BASE4"
        ):
            raise ValueError("relation-control expert is not biased")
        for expert in EXPERT_ORDER:
            encoder = self.hlt_experts[expert].particle_encoder
            if expert == "BASE4" or (
                expert_id is not None and expert != expert_id
            ):
                encoder.set_semantic_relation_transform("active")
            else:
                encoder.set_semantic_relation_transform(mode)

    def _controlled_evidence(
        self, evidence: Mapping[str, Any], *, target_expert: str | None = None
    ) -> dict[str, Any]:
        module = _require_torch()
        mode = self._semantic_predictor_control
        result = {
            name: (dict(value) if isinstance(value, Mapping) else value)
            for name, value in evidence.items()
        }
        if mode == "active":
            return result
        if mode == "shuffle_hlt_evidence_between_events":
            batch = int(result["unbiased_particle_states"].shape[0])
            if batch < 2:
                raise ValueError("evidence shuffle requires at least two events")
            permutation = module.arange(batch, device=result[
                "unbiased_particle_states"
            ].device).roll(1)
            for name in (
                "hlt_token_banks", "relation_particle_states",
                "relation_particle_masks",
            ):
                if name in result:
                    result[name] = {
                        key: value[permutation]
                        for key, value in result[name].items()
                    }
            result["unbiased_particle_states"] = result[
                "unbiased_particle_states"
            ][permutation]
            result["particle_mask"] = result["particle_mask"][permutation]
            return result
        if mode == "zero_hlt_evidence":
            for name in ("hlt_token_banks", "relation_particle_states"):
                if name in result:
                    result[name] = {
                        key: module.zeros_like(value)
                        for key, value in result[name].items()
                    }
            result["unbiased_particle_states"] = module.zeros_like(
                result["unbiased_particle_states"]
            )
            return result
        if mode == "remove_native_particle_context":
            result["unbiased_particle_states"] = module.zeros_like(
                result["unbiased_particle_states"]
            )
            if "relation_particle_states" in result:
                result["relation_particle_states"] = {
                    key: module.zeros_like(value)
                    for key, value in result["relation_particle_states"].items()
                }
            return result
        if mode == "remove_noncorresponding_expert_banks":
            if target_expert is None:
                raise ValueError("noncorresponding-bank control needs a target")
            result["hlt_token_banks"] = {
                key: (
                    value if key == target_expert else module.zeros_like(value)
                )
                for key, value in result["hlt_token_banks"].items()
            }
            return result
        raise RuntimeError("predictor semantic control differs")

    def token_normalizer(self, expert: str) -> tuple[Any, Any]:
        index = EXPERT_ORDER.index(expert)
        return (
            getattr(self, f"token_mean_{index}"),
            getattr(self, f"token_std_{index}"),
        )

    def _selected_predictor(
        self,
        expert: str,
        *,
        evidence: Mapping[str, Any],
        shared_memory: Any | None,
        shared_padding_mask: Any | None,
    ) -> dict[str, Any]:
        predictor = self.predictors[expert]
        if (
            shared_memory is not None
            and predictor.architecture
            in {"A3_SLOT_DECODER_DIRECT", "A4_SLOT_DECODER_GATED"}
        ):
            memory = self.shared_memory_projections[expert](shared_memory)
            queries = predictor.target_queries[None].expand(
                memory.shape[0], -1, -1
            )
            decoded = predictor.output_norm(
                predictor.decoder(
                    queries,
                    memory,
                    memory_key_padding_mask=shared_padding_mask,
                )
            )
            gate = None
            if predictor.architecture == "A3_SLOT_DECODER_DIRECT":
                predicted = decoded
            else:
                corresponding = evidence["hlt_token_banks"][expert]
                anchor = predictor.anchor_map(
                    predictor.anchor_norm(corresponding)
                )
                gate = _require_torch().sigmoid(predictor.gate_head(decoded))
                predicted = anchor + gate * decoded
            return {
                "predicted_tokens": predicted,
                "log_variance": predictor.log_variance_head(predicted).clamp(
                    -8.0, 4.0
                ),
                "gate": gate,
            }
        return predictor(
            corresponding_hlt_tokens=evidence["hlt_token_banks"][expert],
            hlt_token_banks=evidence["hlt_token_banks"],
            unbiased_particle_states=evidence["unbiased_particle_states"],
            particle_mask=evidence["particle_mask"],
            relation_particle_states=evidence.get(
                "relation_particle_states"
            ),
            relation_particle_masks=evidence.get("relation_particle_masks"),
        )

    def _live_evidence(self, shared_view: Mapping[str, Any]) -> dict[str, Any]:
        if self.hlt_experts is None:
            raise ValueError("joint graph has no live HLT experts")
        required = {
            "identities",
            "replica_ids",
            "degraded_view_hashes",
            "features",
            "vectors",
            "mask",
            "raw_tokens",
            "region_trees_by_expert",
        }
        if set(shared_view) != required:
            raise ValueError("live joint bridge shared-view fields differ")
        validate_common_view_metadata(
            identities=shared_view["identities"],
            replica_ids=shared_view["replica_ids"],
            degraded_view_hashes=shared_view["degraded_view_hashes"],
        )
        if set(shared_view["region_trees_by_expert"]) != set(EXPERT_ORDER):
            raise ValueError("live joint bridge REGION coverage differs")
        outputs = {}
        for expert in EXPERT_ORDER:
            outputs[expert] = self.hlt_experts[expert](
                features=shared_view["features"],
                vectors=shared_view["vectors"],
                mask=shared_view["mask"],
                raw_tokens=shared_view["raw_tokens"],
                region_trees=shared_view["region_trees_by_expert"][expert],
                return_details=True,
            )
        base = outputs["BASE4"]
        return {
            "hlt_token_banks": {
                expert: outputs[expert]["tokens"] for expert in EXPERT_ORDER
            },
            "unbiased_particle_states": base["particle_states"],
            "particle_mask": base["particle_mask"],
            "relation_particle_states": {
                expert: outputs[expert]["particle_states"]
                for expert in ("PT", "TRACK", "REGION")
            },
            "relation_particle_masks": {
                expert: outputs[expert]["particle_mask"]
                for expert in ("PT", "TRACK", "REGION")
            },
            "native_hlt_logits": {
                expert: outputs[expert]["logits"] for expert in EXPERT_ORDER
            },
        }

    def forward(
        self,
        *,
        evidence: Mapping[str, Any] | None = None,
        shared_view: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (evidence is None) == (shared_view is None):
            raise ValueError(
                "joint bridge requires exactly one evidence execution path"
            )
        if shared_view is not None:
            evidence = self._live_evidence(shared_view)
        assert evidence is not None
        if set(evidence["hlt_token_banks"]) != set(EXPERT_ORDER):
            raise ValueError("joint bridge evidence-bank coverage differs")
        if (
            self._semantic_predictor_control
            != "remove_noncorresponding_expert_banks"
        ):
            evidence = self._controlled_evidence(evidence)
        shared_memory = shared_padding = None
        if self.shared_context is not None:
            shared_memory, shared_padding = self.shared_context(
                hlt_token_banks=evidence["hlt_token_banks"],
                unbiased_particle_states=evidence["unbiased_particle_states"],
                particle_mask=evidence["particle_mask"],
                relation_particle_states=evidence.get(
                    "relation_particle_states"
                ),
                relation_particle_masks=evidence.get(
                    "relation_particle_masks"
                ),
            )
        if self.variant == "J2_COUPLED_DECODER":
            decoded = self.coupled_decoder(shared_memory, shared_padding)
            normalized = decoded["predicted_tokens"]
            log_variance = decoded["log_variance"]
            gates = {expert: None for expert in EXPERT_ORDER}
        else:
            outputs = {}
            for expert in EXPERT_ORDER:
                selected_evidence = evidence
                selected_memory = shared_memory
                selected_padding = shared_padding
                if (
                    self._semantic_predictor_control
                    == "remove_noncorresponding_expert_banks"
                ):
                    selected_evidence = self._controlled_evidence(
                        evidence, target_expert=expert
                    )
                    if self.shared_context is not None:
                        selected_memory, selected_padding = self.shared_context(
                            hlt_token_banks=selected_evidence["hlt_token_banks"],
                            unbiased_particle_states=selected_evidence[
                                "unbiased_particle_states"
                            ],
                            particle_mask=selected_evidence["particle_mask"],
                            relation_particle_states=selected_evidence.get(
                                "relation_particle_states"
                            ),
                            relation_particle_masks=selected_evidence.get(
                                "relation_particle_masks"
                            ),
                        )
                outputs[expert] = self._selected_predictor(
                    expert, evidence=selected_evidence,
                    shared_memory=selected_memory,
                    shared_padding_mask=selected_padding,
                )
            normalized = {
                expert: outputs[expert]["predicted_tokens"]
                for expert in EXPERT_ORDER
            }
            log_variance = {
                expert: outputs[expert]["log_variance"]
                for expert in EXPERT_ORDER
            }
            gates = {
                expert: outputs[expert].get("gate")
                for expert in EXPERT_ORDER
            }
        original, expert_logits = {}, {}
        for expert in EXPERT_ORDER:
            mean, std = self.token_normalizer(expert)
            original[expert] = _inverse_normalize(
                normalized[expert], mean, std
            )
            expert_logits[expert] = self.frozen_expert_heads[expert](
                original[expert]
            )
        base_logits = self.deployable_fusion(token_banks=original)
        logits = (
            self.adapter(original, base_logits)
            if self.adapter is not None
            else base_logits
        )
        return {
            "predicted_normalized_tokens": normalized,
            "predicted_tokens": original,
            "log_variance": log_variance,
            "predicted_expert_logits": expert_logits,
            "logits": logits,
            "native_hlt_logits": evidence.get("native_hlt_logits"),
            "gates": gates,
        }


def configure_joint_trainability(
    graph: JointBridgeGraph,
    *,
    final_particle_blocks: int | None = None,
) -> dict[str, Any]:
    """Apply the exact J0-J5 trainability boundary and learning rates."""

    if (
        graph.variant == "J4_BRIDGE_FINETUNE"
        and final_particle_blocks not in {2, 4}
    ) or (
        graph.variant != "J4_BRIDGE_FINETUNE"
        and final_particle_blocks is not None
    ):
        raise ValueError("HE_BRIDGE_TUNED final-block count differs")
    module = _require_torch()
    for parameter in graph.parameters():
        if isinstance(parameter, module.nn.parameter.UninitializedParameter):
            parameter.requires_grad = False
        else:
            parameter.requires_grad_(False)
    groups: dict[str, dict[str, Any]] = {}

    def enable(name: str, parameters: Any, learning_rate: float) -> None:
        rows = [
            parameter
            for parameter in parameters
            if not isinstance(
                parameter, module.nn.parameter.UninitializedParameter
            )
        ]
        for parameter in rows:
            parameter.requires_grad_(True)
        rows = [parameter for parameter in rows if parameter.requires_grad]
        if rows:
            groups[name] = {"params": rows, "lr": float(learning_rate)}

    if graph.variant in {
        "J1_SHARED_CONTEXT",
        "J4_BRIDGE_FINETUNE",
        "J5_END_TO_END",
    }:
        enable(
            "predictors",
            (
                parameter
                for name, parameter in graph.predictors.named_parameters()
                if ".evidence." not in name
            ),
            2.0e-4,
        )
        enable("shared_context", graph.shared_context.parameters(), 2.0e-4)
        enable(
            "shared_memory_projections",
            graph.shared_memory_projections.parameters(),
            2.0e-4,
        )
    elif graph.variant == "J2_COUPLED_DECODER":
        enable(
            "coupled_decoder", graph.coupled_decoder.parameters(), 2.0e-4
        )
        enable("shared_context", graph.shared_context.parameters(), 2.0e-4)
    elif graph.variant == "J3_INDEPENDENT_PLUS_ADAPTER":
        enable("adapter", graph.adapter.parameters(), 2.0e-4)
    if graph.variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        for expert in EXPERT_ORDER:
            model = graph.hlt_experts[expert]
            if graph.variant == "J5_END_TO_END":
                enable(
                    f"hlt_all.{expert}",
                    (
                        parameter
                        for name, parameter in model.named_parameters()
                        if not name.startswith("head.")
                    ),
                    5.0e-5,
                )
                continue
            enable(
                f"hlt_tokenizer.{expert}",
                model.tokenizer.parameters(),
                1.0e-4,
            )
            encoder = model.particle_encoder
            relation_parameters = [
                parameter
                for name, parameter in encoder.named_parameters()
                if any(
                    marker in name
                    for marker in (
                        "pair_bias_provider",
                        "concat_pair_embed",
                        "pair_builder",
                        "gate",
                    )
                )
            ]
            enable(
                f"hlt_relation.{expert}",
                relation_parameters,
                5.0e-5,
            )
            blocks = encoder.mod.blocks
            enable(
                f"hlt_final_blocks.{expert}",
                (
                    parameter
                    for block in blocks[-int(final_particle_blocks) :]
                    for parameter in block.parameters()
                ),
                5.0e-5,
            )
    if graph.variant == "J5_END_TO_END":
        enable(
            "deployable_fusion",
            graph.deployable_fusion.parameters(),
            2.0e-4,
        )
    return {
        "variant": graph.variant,
        "final_particle_blocks": (
            int(final_particle_blocks)
            if graph.variant == "J4_BRIDGE_FINETUNE"
            else None
        ),
        "parameter_groups": {
            name: {
                "learning_rate": row["lr"],
                "parameter_count": sum(
                    parameter.numel() for parameter in row["params"]
                ),
            }
            for name, row in groups.items()
        },
        "optimizer_groups": list(groups.values()),
        "offline_expert_heads_frozen": not any(
            parameter.requires_grad
            for parameter in graph.frozen_expert_heads.parameters()
        ),
        "offline_fusion_frozen": not any(
            parameter.requires_grad
            for parameter in graph.frozen_offline_fusion.parameters()
        ),
    }


def clone_predictors(
    predictors: Mapping[str, Any],
) -> dict[str, Any]:
    if set(predictors) != set(EXPERT_ORDER):
        raise ValueError("selected predictor clone coverage differs")
    return {
        expert: copy.deepcopy(predictors[expert]) for expert in EXPERT_ORDER
    }


__all__ = [
    "JOINT_DIMENSION",
    "JOINT_INPUT_POLICY",
    "JOINT_VARIANTS",
    "CoupledExpertDecoder",
    "JointBridgeGraph",
    "ResidualJointAdapter",
    "SharedHLTContext",
    "clone_predictors",
    "configure_joint_trainability",
    "validate_common_view_metadata",
]
