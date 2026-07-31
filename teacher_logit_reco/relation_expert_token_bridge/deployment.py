"""HLT-only export and complete-graph capacity attestation for RETB."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import io
from pathlib import Path
from typing import Any

from .contracts import (
    bind_source,
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    source_record,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .final_consumers import (
    FrozenPredictedOfflineFusion,
    HLTResidualAdapter,
    NativeConditionedTokenRefiner,
    UnrestrictedHLTFusion,
)
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEPLOYABLE_EXPORT_CONTRACT = "retb_deployable_graph_export_v2"
COMPLETE_GRAPH_CAPACITY_CONTRACT = "retb_complete_graph_capacity_v2"
DEPLOYABLE_PARENT_KEYS = frozenset(
    {
        "campaign_spec",
        "step12_bundle",
        "HLT_frontend_checkpoint",
        "joint_predictor_or_J_checkpoint",
        "final_consumer_checkpoint",
        "frozen_offline_fusion_checkpoint",
        "frozen_offline_expert_heads",
        "HLT_input_normalizer",
        "HLT_relation_normalizer",
        "HLT_region_normalizer",
        "degradation_profile",
        "uncertainty_calibration",
    }
)
CAPACITY_COMPONENTS = (
    "HLT_expert_encoders",
    "predictors",
    "dimension_projections",
    "uncertainty_reliability_heads",
    "token_refiner",
    "final_consumer",
)
FORBIDDEN_DEPLOYABLE_INPUT_TERMS = (
    "offline",
    "target",
    "oracle",
    "constituent_match",
    "label",
)
FORBIDDEN_DEPLOYABLE_PARENT_TERMS = (
    "offline_input",
    "offline_target",
    "target_cache",
    "oracle",
    "constituent_match",
    "label",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB deployment")
    return torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_forbidden_mapping(value: Any, *, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            path = f"{prefix}.{name}" if prefix else name
            if any(
                term in name
                for term in FORBIDDEN_DEPLOYABLE_INPUT_TERMS
            ):
                raise ValueError(
                    f"deployable inference forbids input field {path}"
                )
            _reject_forbidden_mapping(child, prefix=path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_mapping(child, prefix=f"{prefix}[{index}]")


class DeployableRetbGraph(
    torch.nn.Module if torch is not None else object
):
    """Exact HLT-only graph from selected frontend through final logits."""

    def __init__(
        self,
        *,
        frontend: Any,
        final_consumer: Any,
        consumer_kind: str,
        frozen_offline_fusion: Any,
        frozen_expert_heads: Mapping[str, Any],
        token_refiner: NativeConditionedTokenRefiner | None = None,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            consumer_kind
            not in {
                "PF_FROZEN",
                "OF_ROBUST",
                "TR_REFINE",
                "HF_ADAPTER",
                "HF_UNRESTRICTED",
            }
            or set(frozen_expert_heads) != set(EXPERT_ORDER)
        ):
            raise ValueError("deployable RETB graph configuration differs")
        if consumer_kind == "HF_ADAPTER" and not isinstance(
            final_consumer, HLTResidualAdapter
        ):
            raise ValueError("deployable adapter graph has wrong consumer")
        if consumer_kind == "HF_UNRESTRICTED" and not isinstance(
            final_consumer, UnrestrictedHLTFusion
        ):
            raise ValueError(
                "deployable unrestricted graph has wrong consumer"
            )
        self.frontend = frontend
        self.final_consumer = final_consumer
        self.consumer_kind = consumer_kind
        self.frozen_offline_fusion = frozen_offline_fusion
        self.frozen_expert_heads = module.nn.ModuleDict(
            {
                expert: frozen_expert_heads[expert]
                for expert in EXPERT_ORDER
            }
        )
        self.token_refiner = token_refiner
        for component in (
            self.frozen_offline_fusion,
            self.frozen_expert_heads,
        ):
            for parameter in component.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> Any:
        super().train(False)
        return self

    def forward(self, *, hlt_inputs: Mapping[str, Any]) -> dict[str, Any]:
        _reject_forbidden_mapping(hlt_inputs)
        evidence = self.frontend(hlt_inputs=hlt_inputs)
        if not isinstance(evidence, Mapping):
            raise ValueError("deployable frontend output differs")
        _reject_forbidden_mapping(evidence)
        required = {
            "predicted_banks",
            "calibrated_log_variance",
            "native_banks",
            "native_expert_logits",
        }
        if set(evidence) != required:
            raise ValueError("deployable frontend evidence fields differ")
        predicted = evidence["predicted_banks"]
        before = predicted
        refiner_details = None
        if self.token_refiner is not None:
            refiner_details = self.token_refiner(
                predicted_banks=predicted,
                calibrated_log_variance=evidence[
                    "calibrated_log_variance"
                ],
                native_banks=evidence["native_banks"],
            )
            predicted = refiner_details["refined_banks"]
        predicted_expert_logits = {
            expert: self.frozen_expert_heads[expert](predicted[expert])
            for expert in EXPERT_ORDER
        }
        frozen_logits = self.frozen_offline_fusion(
            token_banks=predicted
        )
        if self.consumer_kind in {"PF_FROZEN", "TR_REFINE"}:
            if isinstance(
                self.final_consumer, FrozenPredictedOfflineFusion
            ):
                logits = self.final_consumer(
                    predicted_banks=predicted
                )
            else:
                logits = frozen_logits
            details = {"frozen_path_logits": logits}
        elif self.consumer_kind == "OF_ROBUST":
            logits = self.final_consumer(token_banks=predicted)
            details = {"robust_fusion_logits": logits}
        elif self.consumer_kind == "HF_ADAPTER":
            details = self.final_consumer(
                frozen_offline_logits=frozen_logits,
                predicted_banks=predicted,
                calibrated_log_variance=evidence[
                    "calibrated_log_variance"
                ],
                native_banks=evidence["native_banks"],
            )
            logits = details["combined_logits"]
        else:
            details = self.final_consumer(
                token_banks=predicted,
                calibrated_log_variance=evidence[
                    "calibrated_log_variance"
                ],
                native_banks=evidence["native_banks"],
                native_expert_logits=evidence[
                    "native_expert_logits"
                ],
                predicted_expert_logits=predicted_expert_logits,
            )
            logits = details["logits"]
        if logits.ndim != 2 or int(logits.shape[1]) != 10 or not bool(
            _require_torch().isfinite(logits).all()
        ):
            raise FloatingPointError("deployable RETB logits differ")
        return {
            "logits": logits,
            "consumer_details": details,
            "token_refiner_details": refiner_details,
            "predicted_banks_before_refinement": before,
            "predicted_banks_after_refinement": predicted,
        }


class JointBridgeDeployableFrontend(
    torch.nn.Module if torch is not None else object
):
    """Selected live J5 bridge exposed as typed Step-12 HLT evidence.

    The wrapper owns the exact selected joint graph and immutable additive
    uncertainty offsets.  Its public input is the authenticated HLT shared
    view only; target caches and offline/oracle arrays are neither accepted
    nor retained.
    """

    def __init__(
        self,
        *,
        joint_graph: Any,
        calibration_offsets: Mapping[str, Any],
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            getattr(joint_graph, "variant", None) != "J5_END_TO_END"
            or set(calibration_offsets) != set(EXPERT_ORDER)
        ):
            raise ValueError("deployable joint frontend configuration differs")
        self.joint_graph = joint_graph
        for index, expert in enumerate(EXPERT_ORDER):
            offsets = module.as_tensor(
                calibration_offsets[expert], dtype=module.float32
            ).reshape(-1)
            if len(offsets) <= 0 or not bool(module.isfinite(offsets).all()):
                raise ValueError(
                    "deployable joint frontend calibration differs"
                )
            self.register_buffer(f"calibration_offset_{index}", offsets)

    def train(self, mode: bool = True) -> Any:
        super().train(False)
        self.joint_graph.eval()
        return self

    def forward(self, *, hlt_inputs: Mapping[str, Any]) -> dict[str, Any]:
        _reject_forbidden_mapping(hlt_inputs)
        evidence = self.joint_graph._live_evidence(hlt_inputs)
        prediction = self.joint_graph(evidence=evidence)
        calibrated = {}
        for index, expert in enumerate(EXPERT_ORDER):
            values = prediction["log_variance"][expert]
            offsets = getattr(self, f"calibration_offset_{index}").to(
                dtype=values.dtype, device=values.device
            )
            if values.ndim != 3 or int(values.shape[-1]) != len(offsets):
                raise ValueError(
                    "deployable joint frontend uncertainty shape differs"
                )
            calibrated[expert] = (
                values + offsets[None, None]
            ).clamp(-8.0, 4.0)
        return {
            "predicted_banks": prediction["predicted_tokens"],
            "calibrated_log_variance": calibrated,
            "native_banks": evidence["hlt_token_banks"],
            "native_expert_logits": evidence["native_hlt_logits"],
        }


def _tensor_digest(value: Any) -> str:
    stream = io.BytesIO()
    _require_torch().save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def export_deployable_retb_graph(
    *,
    output_dir: str | Path,
    graph: DeployableRetbGraph,
    hlt_smoke_inputs: Mapping[str, Any],
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    expected_parents = set(DEPLOYABLE_PARENT_KEYS)
    if graph.token_refiner is not None:
        expected_parents.add("token_refiner_checkpoint")
    if set(parent_hashes) != expected_parents or any(
        any(
            term in name.lower()
            for term in FORBIDDEN_DEPLOYABLE_PARENT_TERMS
        )
        for name in parent_hashes
    ):
        raise ValueError("deployable RETB parent coverage differs")
    _reject_forbidden_mapping(hlt_smoke_inputs)
    graph.eval()
    with _require_torch().no_grad():
        expected = graph(hlt_inputs=hlt_smoke_inputs)["logits"].float().cpu()
    root = Path(output_dir)
    path = root / "deployable_retb_graph.pt"
    stream = io.BytesIO()
    _require_torch().save(
        {
            "contract": DEPLOYABLE_EXPORT_CONTRACT,
            "schema_version": 1,
            "graph": graph,
        },
        stream,
    )
    publication = write_immutable_bytes(path, stream.getvalue())
    loaded_payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    loaded = loaded_payload.get("graph")
    if (
        loaded_payload.get("contract") != DEPLOYABLE_EXPORT_CONTRACT
        or not isinstance(loaded, DeployableRetbGraph)
    ):
        raise ValueError("deployable RETB reload payload differs")
    loaded.eval()
    with _require_torch().no_grad():
        actual = loaded(hlt_inputs=hlt_smoke_inputs)["logits"].float().cpu()
    if not bool(
        _require_torch().allclose(
            expected, actual, atol=1.0e-6, rtol=1.0e-6
        )
    ):
        raise RuntimeError("deployable RETB reload parity failed")
    manifest = bind_source(
        with_content_hash(
            {
                "contract": DEPLOYABLE_EXPORT_CONTRACT,
                "schema_version": 1,
                "consumer_kind": graph.consumer_kind,
                "graph_filename": path.name,
                "graph_sha256": publication["file_sha256"],
                "graph_state_sha256": _tensor_digest(graph.state_dict()),
                "parent_hashes": {
                    name: require_sha256(
                        value, name=f"parent_hashes.{name}"
                    )
                    for name, value in sorted(parent_hashes.items())
                },
                "smoke": {
                    "logits_sha256": _tensor_digest(expected),
                    "shape": list(expected.shape),
                    "absolute_tolerance": 1.0e-6,
                    "relative_tolerance": 1.0e-6,
                    "reload_parity": True,
                },
                "inference_inputs": "HLT_arrays_and_selected_checkpoints_only",
                "offline_inputs_accepted": False,
                "target_caches_loadable": False,
                "oracle_targets_requestable": False,
                "labels_accepted": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "deployable_retb_graph.json", manifest)
    return manifest


def load_deployable_retb_graph(
    manifest_path: str | Path,
    *,
    expected_source: Mapping[str, Any],
) -> DeployableRetbGraph:
    manifest = load_hashed_json(
        manifest_path, expected_contract=DEPLOYABLE_EXPORT_CONTRACT
    )
    path = Path(manifest_path).parent / manifest["graph_filename"]
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        manifest.get("source") != expected_source_record
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != manifest["graph_sha256"]
        or manifest.get("offline_inputs_accepted")
        or manifest.get("target_caches_loadable")
        or manifest.get("oracle_targets_requestable")
        or manifest.get("labels_accepted")
    ):
        raise ValueError("deployable RETB manifest lineage differs")
    payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    graph = payload.get("graph")
    if (
        payload.get("contract") != DEPLOYABLE_EXPORT_CONTRACT
        or not isinstance(graph, DeployableRetbGraph)
        or _tensor_digest(graph.state_dict())
        != manifest["graph_state_sha256"]
    ):
        raise ValueError("deployable RETB graph semantics differ")
    graph.eval()
    return graph


def _unique_parameter_count(modules: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    seen: set[int] = set()
    rows = {}
    for name in CAPACITY_COMPONENTS:
        count = 0
        for parameter in modules[name].parameters():
            identifier = id(parameter)
            if identifier in seen:
                continue
            seen.add(identifier)
            if isinstance(
                parameter,
                _require_torch().nn.parameter.UninitializedParameter,
            ):
                raise ValueError(
                    "complete-graph capacity has uninitialized parameters"
                )
            count += parameter.numel()
        rows[name] = count
    return sum(rows.values()), rows


def _parameter_multiset(module_or_modules: Any) -> Counter[tuple[Any, ...]]:
    modules = (
        module_or_modules.values()
        if isinstance(module_or_modules, Mapping)
        else (module_or_modules,)
    )
    seen: set[int] = set()
    values: Counter[tuple[Any, ...]] = Counter()
    for component in modules:
        for parameter in component.parameters():
            identifier = id(parameter)
            if identifier in seen:
                continue
            seen.add(identifier)
            if isinstance(
                parameter,
                _require_torch().nn.parameter.UninitializedParameter,
            ):
                raise ValueError(
                    "complete-graph capacity has uninitialized parameters"
                )
            tensor = parameter.detach().cpu().contiguous()
            digest = hashlib.sha256(
                tensor.reshape(-1)
                .view(_require_torch().uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
            values[
                (
                    str(tensor.dtype),
                    tuple(int(value) for value in tensor.shape),
                    int(tensor.numel()),
                    digest,
                )
            ] += 1
    return values


class _CapacityParameterBucket(
    torch.nn.Module if torch is not None else object
):
    """Non-executable view over one disjoint exported-parameter partition."""

    def __init__(self, parameters: list[Any]) -> None:
        super().__init__()
        self.parameters_view = _require_torch().nn.ParameterList(parameters)


def _capacity_component_for_name(name: str) -> str:
    """Assign every deployable parameter to one scientific component."""

    lowered = str(name).lower()
    if "token_refiner" in lowered:
        return "token_refiner"
    if (
        "log_variance" in lowered
        or "reliability" in lowered
        or "calibration_offset" in lowered
    ):
        return "uncertainty_reliability_heads"
    if "frontend.joint_graph.hlt_experts" in lowered:
        return "HLT_expert_encoders"
    if "shared_memory_projections" in lowered:
        return "dimension_projections"
    if any(
        marker in lowered
        for marker in (
            "frontend.joint_graph.predictors",
            "frontend.joint_graph.shared_context",
            "frontend.joint_graph.coupled_decoder",
            "frontend.joint_graph.adapter",
        )
    ):
        return "predictors"
    return "final_consumer"


def build_capacity_parameter_components(exported_graph: Any) -> dict[str, Any]:
    """Return an exact, disjoint parameter view of an exported RETB graph."""

    buckets: dict[str, list[Any]] = {
        name: [] for name in CAPACITY_COMPONENTS
    }
    seen: set[int] = set()
    for name, parameter in exported_graph.named_parameters():
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        buckets[_capacity_component_for_name(name)].append(parameter)
    if not seen:
        raise ValueError("deployable graph has no capacity parameters")
    components = {
        name: _CapacityParameterBucket(buckets[name])
        for name in CAPACITY_COMPONENTS
    }
    if _parameter_multiset(components) != _parameter_multiset(exported_graph):
        raise RuntimeError("capacity parameter partition is not exact")
    return components


def _module_flops(module: Any, inputs: tuple[Any, ...], output: Any) -> int:
    """Count fixed-shape multiply/add and declared elementwise operations."""

    torch_module = _require_torch()

    def tensors(value: Any) -> list[Any]:
        if torch_module.is_tensor(value):
            return [value]
        if isinstance(value, Mapping):
            return [
                tensor
                for child in value.values()
                for tensor in tensors(child)
            ]
        if isinstance(value, (tuple, list)):
            return [
                tensor for child in value for tensor in tensors(child)
            ]
        return []

    outputs = tensors(output)
    if isinstance(module, torch_module.nn.Linear):
        positions = sum(
            int(tensor.numel() // module.out_features)
            for tensor in outputs
            if tensor.ndim and int(tensor.shape[-1]) == module.out_features
        )
        return positions * (
            2 * int(module.in_features) * int(module.out_features)
            + (int(module.out_features) if module.bias is not None else 0)
        )
    if isinstance(module, torch_module.nn.Conv1d):
        if not outputs:
            return 0
        count = sum(int(tensor.numel()) for tensor in outputs)
        kernel = int(module.kernel_size[0])
        return count * (
            2 * int(module.in_channels // module.groups) * kernel
            + (1 if module.bias is not None else 0)
        )
    if isinstance(module, torch_module.nn.MultiheadAttention):
        query = inputs[0] if inputs else None
        if not torch_module.is_tensor(query) or query.ndim != 3:
            return 0
        if module.batch_first:
            batch, target, width = map(int, query.shape)
        else:
            target, batch, width = map(int, query.shape)
        key = inputs[1] if len(inputs) > 1 else query
        source = int(key.shape[1 if module.batch_first else 0])
        # Q/K/V projections, attention scores and value reduction, output
        # projection. A multiply and add count as two FLOPs.
        return batch * (
            6 * target * width * width
            + 4 * target * source * width
            + 2 * target * width * width
        )
    if isinstance(module, torch_module.nn.LayerNorm):
        return 5 * sum(int(tensor.numel()) for tensor in outputs)
    if isinstance(
        module,
        (
            torch_module.nn.GELU,
            torch_module.nn.ReLU,
            torch_module.nn.SiLU,
            torch_module.nn.Sigmoid,
            torch_module.nn.Softmax,
        ),
    ):
        return 4 * sum(int(tensor.numel()) for tensor in outputs)
    return 0


def analytical_export_flops(
    *,
    exported_graph: Any,
    hlt_smoke_inputs: Mapping[str, Any],
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Measure tensor shapes once and analytically count the fixed graph."""

    module = _require_torch()
    totals = {name: 0 for name in CAPACITY_COMPONENTS}
    hooks = []

    def component_for_module(name: str) -> str:
        # Buffers and parameter-free operations inherit their enclosing
        # scientific component from the same deterministic name rule.
        return _capacity_component_for_name(name)

    supported = (
        module.nn.Linear,
        module.nn.Conv1d,
        module.nn.MultiheadAttention,
        module.nn.LayerNorm,
        module.nn.GELU,
        module.nn.ReLU,
        module.nn.SiLU,
        module.nn.Sigmoid,
        module.nn.Softmax,
    )
    for name, child in exported_graph.named_modules():
        if name and isinstance(child, supported):
            component = component_for_module(name)

            def capture(
                child_module: Any,
                child_inputs: tuple[Any, ...],
                child_output: Any,
                *,
                component_name: str = component,
            ) -> None:
                totals[component_name] += _module_flops(
                    child_module, child_inputs, child_output
                )

            hooks.append(child.register_forward_hook(capture))
    exported_graph.eval()
    try:
        with module.no_grad():
            result = exported_graph(hlt_inputs=hlt_smoke_inputs)
    finally:
        for hook in hooks:
            hook.remove()
    logits = result["logits"]
    batch = int(logits.shape[0])
    if batch <= 0:
        raise ValueError("capacity smoke batch is empty")
    per_event = {
        name: int((value + batch - 1) // batch)
        for name, value in totals.items()
    }
    # A structurally absent refiner is the only component permitted to have
    # zero work. All other categories are part of every J5 export.
    if any(
        per_event[name] <= 0
        for name in CAPACITY_COMPONENTS
        if name != "token_refiner"
    ):
        raise ValueError(
            "analytical export FLOP component coverage is incomplete"
        )
    ledger = {
        name: {"1": value, "128": 128 * value}
        for name, value in per_event.items()
    }
    diagnostics = {
        "shape_capture_event_count": batch,
        "logits_shape": list(logits.shape),
        "estimator": "module_shape_analytical_v1",
        "multiply_add_flops": 2,
        "measured_wall_time_used": False,
        "selection_uses_analytical_ledger_only": True,
    }
    return ledger, diagnostics


def build_complete_graph_capacity(
    *,
    graph_id: str,
    deployment_export_sha256: str,
    component_modules: Mapping[str, Any],
    exported_graph: Any,
    analytical_component_flops: Mapping[str, Mapping[int | str, int]],
    measured_diagnostics: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(component_modules) != set(CAPACITY_COMPONENTS)
        or set(analytical_component_flops) != set(CAPACITY_COMPONENTS)
    ):
        raise ValueError("complete-graph capacity component coverage differs")
    total_parameters, parameters = _unique_parameter_count(
        component_modules
    )
    if _parameter_multiset(component_modules) != _parameter_multiset(
        exported_graph
    ):
        raise ValueError(
            "capacity components do not exactly cover exported graph"
        )
    flops_by_batch = {}
    components = {}
    for component in CAPACITY_COMPONENTS:
        values = analytical_component_flops[component]
        flops = {
            batch: int(values.get(batch, values.get(str(batch), 0)))
            for batch in (1, 128)
        }
        if min(flops.values()) < 0 or (
            component != "token_refiner" and min(flops.values()) <= 0
        ) or flops[128] != 128 * flops[1]:
            raise ValueError(
                "complete-graph analytical FLOP ledger differs"
            )
        components[component] = {
            "parameter_count": parameters[component],
            "analytical_inference_flops": {
                str(batch): flops[batch] for batch in (1, 128)
            },
        }
        for batch in (1, 128):
            flops_by_batch[batch] = (
                flops_by_batch.get(batch, 0) + flops[batch]
            )
    if total_parameters <= 0 or min(flops_by_batch.values()) <= 0:
        raise ValueError("complete-graph capacity totals differ")
    return bind_source(
        with_content_hash(
            {
                "contract": COMPLETE_GRAPH_CAPACITY_CONTRACT,
                "schema_version": 2,
                "graph_id": str(graph_id),
                "deployment_export_sha256": require_sha256(
                    deployment_export_sha256,
                    name="deployment_export_sha256",
                ),
                "components": components,
                "totals": {
                    "parameter_count": total_parameters,
                    "analytical_inference_flops_batch1": flops_by_batch[1],
                    "analytical_inference_flops_batch128": flops_by_batch[
                        128
                    ],
                },
                "measured_diagnostics": dict(measured_diagnostics),
                "parameter_identity_deduplication": True,
                "exact_export_parameter_multiset_match": True,
                "training_only_offline_teachers_excluded": True,
                "deployable_frozen_offline_heads_and_fusion_included": True,
                "deployable_frozen_components_accounted_under": (
                    "final_consumer"
                ),
                "target_caches_excluded": True,
                "analytical_batches": [1, 128],
                "analytical_flop_contract": (
                    "retb_fixed_shape_multiply_add_flops_v1"
                ),
                "batch128_equals_128_times_batch1": True,
                "measured_latency_used_for_selection": False,
            }
        ),
        source_snapshot=source_snapshot,
    )


__all__ = [
    "CAPACITY_COMPONENTS",
    "COMPLETE_GRAPH_CAPACITY_CONTRACT",
    "DEPLOYABLE_EXPORT_CONTRACT",
    "DEPLOYABLE_PARENT_KEYS",
    "DeployableRetbGraph",
    "JointBridgeDeployableFrontend",
    "analytical_export_flops",
    "build_capacity_parameter_components",
    "build_complete_graph_capacity",
    "export_deployable_retb_graph",
    "load_deployable_retb_graph",
]
