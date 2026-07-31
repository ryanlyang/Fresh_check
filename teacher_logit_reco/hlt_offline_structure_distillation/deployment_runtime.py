"""Self-contained HLT-only exports and label-free inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (
    collate_native_hlt_expert_batch,
)

from .auxiliary_data import HLTArrayDataset
from .baselines import build_baseline_model
from .combination_runtime import build_combination_model
from .contracts import (
    canonical_sha256,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .feedback import build_feedback_model
from .taps import HBaseParticleTransformer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEPLOYABLE_GRAPH_EXPORT_CONTRACT = "hosd_deployable_graph_export_v1"
DEPLOYABLE_INFERENCE_CONTRACT = "hosd_deployable_inference_v1"


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_model(
    descriptor: Mapping[str, Any], *, weaver_module: Any
) -> Any:
    kind = descriptor["graph_kind"]
    if kind == "BASELINE":
        return build_baseline_model(
            descriptor["baseline_id"], weaver_module=weaver_module
        )
    if kind in {"AUXILIARY", "COMBINATION"}:
        return HBaseParticleTransformer(weaver_module=weaver_module)
    if kind == "FEEDBACK":
        return build_feedback_model(
            descriptor["row"], weaver_module=weaver_module
        )
    if kind == "CAPACITY":
        from .capacity import build_capacity_model

        return build_capacity_model(
            descriptor["capacity_kind"],
            descriptor["configuration"],
            weaver_module=weaver_module,
        )
    raise ValueError("unknown deployable graph kind")


def _forward(model: Any, batch: Mapping[str, Any]) -> Any:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    points = batch.get("points")
    if points is None:
        points = batch["features"][:, 15:17]
    return model(points, batch["features"], vectors, batch["mask"])


def export_deployable_graph(
    *,
    descriptor: Mapping[str, Any],
    research_model: Any,
    representative_batch: Mapping[str, Any],
    output_path: str | Path,
    checkpoint_sha256: str,
    lineage_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    weaver_module: Any,
    precision: str = "FP32",
    analytical_inference_flops_batch1_n128: int,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for HOSD export")
    if descriptor["graph_kind"] in {"AUXILIARY", "COMBINATION"}:
        runtime_state = {
            key.removeprefix("classifier."): value.detach().cpu()
            for key, value in research_model.state_dict().items()
            if key.startswith("classifier.")
        }
    else:
        runtime_state = {
            key: value.detach().cpu()
            for key, value in research_model.state_dict().items()
        }
    runtime = _runtime_model(descriptor, weaver_module=weaver_module)
    runtime.load_state_dict(runtime_state, strict=True)
    research_model.eval()
    runtime.eval()
    with torch.no_grad():
        expected = (
            _forward(research_model.classifier, representative_batch)
            if descriptor["graph_kind"] in {"AUXILIARY", "COMBINATION"}
            else _forward(research_model, representative_batch)
        )
        actual = _forward(runtime, representative_batch)
    tolerance = (
        {"absolute": 1e-6, "relative": 1e-5}
        if precision == "FP32"
        else {"absolute": 2e-3, "relative": 2e-3}
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=tolerance["absolute"],
        rtol=tolerance["relative"],
    )
    forbidden = (
        "offline",
        "oracle",
        "target_cache",
        "target_mask",
        "teacher_state",
        "label",
    )
    if any(any(value in key.lower() for value in forbidden) for key in runtime_state):
        raise ValueError("deployable runtime state reaches forbidden evidence")
    complete_trainable_parameters = sum(
        int(value.numel())
        for value in research_model.parameters()
        if value.requires_grad
    )
    deployed_trainable_parameters = sum(
        int(value.numel()) for value in runtime.parameters() if value.requires_grad
    )
    if (
        int(analytical_inference_flops_batch1_n128) <= 0
        or complete_trainable_parameters < deployed_trainable_parameters
    ):
        raise ValueError("deployable capacity evidence differs")
    output = Path(output_path)
    payload = {
        "contract": DEPLOYABLE_GRAPH_EXPORT_CONTRACT,
        "schema_version": 1,
        "descriptor": dict(descriptor),
        "runtime_state_dict": runtime_state,
        "checkpoint_sha256": require_sha256(
            checkpoint_sha256, name="checkpoint_sha256"
        ),
        "lineage_hashes": {
            key: require_sha256(value, name=f"lineage.{key}")
            for key, value in sorted(lineage_hashes.items())
        },
        "source": dict(source),
        "precision": precision,
        "parity_tolerance": tolerance,
        "runtime_inputs": ["features", "vectors", "mask"],
        "forbidden_runtime_dependencies": [],
        "target_heads_removed": descriptor["graph_kind"]
        in {"AUXILIARY", "COMBINATION"},
        "complete_trainable_parameters": complete_trainable_parameters,
        "deployed_trainable_parameters": deployed_trainable_parameters,
        "target_head_removal_parameter_savings": (
            complete_trainable_parameters - deployed_trainable_parameters
        ),
        "analytical_inference_flops_batch1_n128": int(
            analytical_inference_flops_batch1_n128
        ),
        "hlt_only": True,
    }
    _atomic_torch_save(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = with_content_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "runtime_state_dict"
        }
        | {
            "export_file": output.name,
            "export_sha256": digest,
            "state_key_count": len(runtime_state),
            "research_export_logits_parity": True,
        }
    )
    write_immutable_json(output.with_suffix(output.suffix + ".json"), manifest)
    return manifest


def load_deployable_graph(
    path: str | Path, *, weaver_module: Any, source: Mapping[str, Any]
) -> tuple[Any, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("contract") != DEPLOYABLE_GRAPH_EXPORT_CONTRACT
        or payload.get("source") != dict(source)
        or payload.get("forbidden_runtime_dependencies")
        or not payload.get("hlt_only")
    ):
        raise ValueError("deployable graph export contract differs")
    model = _runtime_model(payload["descriptor"], weaver_module=weaver_module)
    model.load_state_dict(payload["runtime_state_dict"], strict=True)
    model._hosd_deployable_descriptor = dict(payload["descriptor"])
    return model, payload


def build_label_free_hlt_loader(
    *,
    cache_paths: Mapping[int, str | Path],
    identities: Sequence[str],
    logical_role: str,
    realization_policy: str,
    batch_size: int,
) -> Any:
    ids = tuple(str(value) for value in identities)
    arrays_by_replica = {}
    for replica, path in sorted(cache_paths.items()):
        arrays, _ = load_hlt_v3_cache(Path(path))
        source_ids = tuple(str(value) for value in arrays["identities"].tolist())
        lookup = {value: index for index, value in enumerate(source_ids)}
        if not set(ids).issubset(lookup):
            raise ValueError("deployable cache lacks requested identities")
        order = np.asarray([lookup[value] for value in ids], dtype=np.int64)
        arrays_by_replica[int(replica)] = {
            name: np.asarray(arrays[name])[order]
            for name in ("tokens", "mask", "measurement_states")
        }
    dataset = HLTArrayDataset(
        replica_arrays=arrays_by_replica,
        labels=np.zeros(len(ids), dtype=np.int64),
        identities=ids,
        logical_role=logical_role,
        realization_policy=realization_policy,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=torch.utils.data.SequentialSampler(dataset),
        num_workers=0,
        drop_last=False,
        collate_fn=collate_native_hlt_expert_batch,
    )


def infer_deployable_graph(
    model: Any,
    loader: Any,
    *,
    device: str | Any,
) -> tuple[tuple[str, ...], np.ndarray]:
    descriptor = getattr(model, "_hosd_deployable_descriptor", {})
    if (
        descriptor.get("inference_control")
        == "IDENTITY_BOUND_WRONG_EVENT_PREDICTION"
    ):
        return _infer_identity_bound_wrong_event(
            model,
            loader,
            device=device,
            descriptor=descriptor,
        )
    resolved = torch.device(device)
    model.to(resolved).eval()
    identities, logits = [], []
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(resolved) if hasattr(value, "to") else value
                for key, value in raw.items()
            }
            values = _forward(model, batch)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("deployable inference logits are nonfinite")
            identities.extend(str(value) for value in raw["event_identities"])
            logits.append(values.float().cpu().numpy())
    ids = tuple(identities)
    values = np.concatenate(logits)
    if len(ids) != len(set(ids)) or values.shape != (len(ids), 10):
        raise ValueError("deployable inference population differs")
    return ids, values


def _wrong_event_rank(
    identity: str, *, target_id: str, split: str
) -> tuple[str, str]:
    digest = hashlib.sha256()
    for item in (
        "hosd_final_feedback_shuffle_v1",
        str(target_id),
        str(split),
        str(identity),
    ):
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest(), str(identity)


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _feedback_inputs(batch: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    points = batch.get("points")
    if points is None:
        points = batch["features"][:, 15:17]
    return points, batch["features"], vectors, batch["mask"]


def _infer_identity_bound_wrong_event(
    model: Any,
    loader: Any,
    *,
    device: str | Any,
    descriptor: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate wrong-event feedback by joining donors before batching.

    Only one donor batch and one recipient batch are resident at once, so the
    pair-feedback control remains practical for the full final-test set.
    """

    if not hasattr(model, "forward_with_feedback"):
        raise ValueError("wrong-event export is not a feedback graph")
    dataset = loader.dataset
    identities = tuple(str(value) for value in dataset.identities)
    if len(identities) < 2 or len(identities) != len(set(identities)):
        raise ValueError("wrong-event inference requires distinct identities")
    split = str(getattr(dataset, "logical_role", "final_test"))
    ordered = sorted(
        range(len(identities)),
        key=lambda index: _wrong_event_rank(
            identities[index],
            target_id=descriptor["row"]["target_id"],
            split=split,
        ),
    )
    donor = [0] * len(identities)
    for ordinal, recipient in enumerate(ordered):
        donor[recipient] = ordered[(ordinal + 1) % len(ordered)]
    if sorted(donor) != list(range(len(identities))) or any(
        index == value for index, value in enumerate(donor)
    ):
        raise AssertionError("wrong-event donor map is not a derangement")
    batch_size = int(loader.batch_size)
    collate = loader.collate_fn
    resolved = torch.device(device)
    model.to(resolved).eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(identities), batch_size):
            indices = list(
                range(start, min(start + batch_size, len(identities)))
            )
            recipient_raw = collate([dataset[index] for index in indices])
            donor_raw = collate([dataset[donor[index]] for index in indices])
            recipient_batch = _move_batch(recipient_raw, resolved)
            donor_batch = _move_batch(donor_raw, resolved)
            _, prediction = model.forward_with_feedback(
                *_feedback_inputs(donor_batch)
            )
            override = {
                key: value
                for key, value in prediction.items()
                if key
                in {
                    "value",
                    "mean",
                    "log_variance",
                    "availability_logits",
                }
            }
            values, _ = model.forward_with_feedback(
                *_feedback_inputs(recipient_batch),
                predicted_feedback_override=override,
            )
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError(
                    "wrong-event deployable logits are nonfinite"
                )
            observed = tuple(
                str(value) for value in recipient_raw["event_identities"]
            )
            if observed != tuple(identities[index] for index in indices):
                raise ValueError("wrong-event recipient identity order changed")
            logits.append(values.float().cpu().numpy())
    values = np.concatenate(logits)
    if values.shape != (len(identities), 10):
        raise ValueError("wrong-event deployable population differs")
    return identities, values


__all__ = [
    "DEPLOYABLE_GRAPH_EXPORT_CONTRACT",
    "DEPLOYABLE_INFERENCE_CONTRACT",
    "build_label_free_hlt_loader",
    "export_deployable_graph",
    "infer_deployable_graph",
    "load_deployable_graph",
]
