"""Selection-gated deployment runtime benchmark for the P7b fusion campaign."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from jetclass_fresh.hlt_baseline import amp_autocast_context, resolve_device

from .data import move_local_particle_residual_field_batch_to_device
from .curriculum import LocalResidualFieldCurriculumJointModel
from .fusion import LocalResidualFieldPredictionConfig, _make_prediction_loader, _prediction_dataset, load_local_residual_field_tagger_from_checkpoint
from .fusion_campaign import FUSION_FINAL_SPLIT, default_fusion_candidate_specs, stable_fusion_json_hash
from .fusion_features import PreClassifierEmbeddingCapture, _campaign_forward, _model_logits
from .fusion_final import _development_feature_config, _read_json, _validate_selection_dependencies
from .fusion_seed_control import sha256_file
from .fusion_selection import _atomic_json, _validate_candidate_report, load_selected_fusion_set
from .fusion_train import load_representation_fusion_head_from_checkpoint


LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_CONTRACT = "local_residual_field_fusion_runtime_v1"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_summary(latencies_ms: list[float], batch_size: int) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    median = float(np.median(values))
    return {
        "measured_batches": int(len(values)), "batch_size": int(batch_size),
        "median_batch_latency_ms": median, "p95_batch_latency_ms": float(np.quantile(values, 0.95)),
        "jets_per_second": float(batch_size * 1000.0 / median),
    }


def _time_callable(call: Callable[[], Any], *, device: torch.device, warmup: int, batches: int, batch_size: int) -> dict[str, Any]:
    with torch.no_grad():
        for _ in range(warmup):
            call()
        _synchronize(device)
        rows: list[float] = []
        for _ in range(batches):
            start = time.perf_counter()
            call()
            _synchronize(device)
            rows.append((time.perf_counter() - start) * 1000.0)
    return _timing_summary(rows, batch_size)


def _benchmark_member(member: str, config: Mapping[str, Any], *, batch_size: int, warmup: int, batches: int) -> dict[str, Any]:
    device = resolve_device(str(config.get("device", "auto")))
    model, _payload = load_local_residual_field_tagger_from_checkpoint(config["checkpoint"], device=device)
    model.to(device)
    if isinstance(model, LocalResidualFieldCurriculumJointModel) and model.oracle_consumer is not None:
        raise ValueError(f"runtime benchmark refuses oracle-attached deployable member {member}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prediction = LocalResidualFieldPredictionConfig(
        checkpoint=str(config["checkpoint"]), prediction_dir="unused", model_name=member,
        hlt_cache_dir=str(config["hlt_cache_dir"]), target_cache_dir=None,
        manifest_path=str(config["manifest_path"]), splits=(FUSION_FINAL_SPLIT,),
        batch_size=batch_size, num_workers=int(config.get("num_workers", 0)),
        device=str(config.get("device", "auto")), amp=bool(config.get("amp", True)),
        confirm_final_test=True, allow_oracle_final_test=False,
    )
    loader = _make_prediction_loader(
        _prediction_dataset(prediction, FUSION_FINAL_SPLIT, model=model), batch_size=batch_size,
        num_workers=prediction.num_workers, seed=0, hlt_only=True,
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"runtime dataset is empty for {member}") from exc
    batch = move_local_particle_residual_field_batch_to_device(batch, device)
    actual_batch_size = int(batch["labels"].shape[0])
    amp_enabled = bool(prediction.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def forward() -> torch.Tensor:
        with amp_autocast_context(amp_enabled):
            return _campaign_forward(model, batch)

    timing = _time_callable(forward, device=device, warmup=warmup, batches=batches, batch_size=actual_batch_size)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    checkpoint = Path(str(config["checkpoint"]))
    return {
        "run_id": f"member/{member}", "member_id": member, **timing,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": 0, "peak_gpu_memory_bytes": peak,
        "artifact_size_bytes": int(checkpoint.stat().st_size),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device), "timing_scope": "forward_only_excludes_data_loading_and_host_to_device",
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "deployable": True,
    }


def _torch_late_fusion(
    candidate_id: str, parameters: Mapping[str, Any], logits_a: torch.Tensor, logits_b: torch.Tensor,
) -> torch.Tensor:
    if candidate_id == "L0_mean_logits":
        return 0.5 * (logits_a + logits_b)
    if candidate_id == "L1_mean_probs":
        return torch.log(torch.clamp(0.5 * (torch.softmax(logits_a, dim=1) + torch.softmax(logits_b, dim=1)), min=1.0e-12))
    if candidate_id == "L2_temp_mean_logits":
        temperatures = parameters["temperatures"]
        return 0.5 * (logits_a / float(temperatures[0]) + logits_b / float(temperatures[1]))
    if candidate_id == "L3_scalar_simplex_logits":
        weight = float(parameters["weight"])
        return weight * logits_a + (1.0 - weight) * logits_b
    if candidate_id == "L4_classwise_simplex_logits":
        weights = torch.as_tensor(parameters["weights"], dtype=logits_a.dtype, device=logits_a.device)[None, :]
        return weights * logits_a + (1.0 - weights) * logits_b
    if candidate_id == "L5_linear_stacker":
        mode = str(parameters["feature_mode"])
        probabilities = torch.cat([torch.softmax(logits_a, dim=1), torch.softmax(logits_b, dim=1)], dim=1)
        if mode == "logits":
            features = torch.cat([logits_a, logits_b], dim=1)
        elif mode == "probabilities":
            features = probabilities
        elif mode == "logits+probabilities":
            features = torch.cat([logits_a, logits_b, probabilities], dim=1)
        else:
            raise ValueError(f"unsupported stacker feature mode {mode!r}")
        mean = torch.as_tensor(parameters["feature_mean"], dtype=features.dtype, device=features.device)
        weight = torch.as_tensor(parameters["weight"], dtype=features.dtype, device=features.device)
        bias = torch.as_tensor(parameters["bias"], dtype=features.dtype, device=features.device)
        return (features - mean) @ weight.T + bias
    raise ValueError(f"unsupported late fusion candidate {candidate_id!r}")


def _load_runtime_model(member: str, config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model, _payload = load_local_residual_field_tagger_from_checkpoint(config["checkpoint"], device=device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if isinstance(model, LocalResidualFieldCurriculumJointModel) and model.oracle_consumer is not None:
        raise ValueError(f"runtime benchmark refuses oracle-attached deployable member {member}")
    return model


def _benchmark_fusion_end_to_end(
    *, candidate_id: str, member_a: str, member_b: str, parameters: Mapping[str, Any],
    candidate_report: Mapping[str, Any], feature_configs: Mapping[str, Mapping[str, Any]],
    batch_size: int, warmup: int, batches: int,
) -> dict[str, Any]:
    device = resolve_device("auto")
    config_a, config_b = feature_configs[member_a], feature_configs[member_b]
    for key in ("hlt_cache_dir", "manifest_path"):
        if Path(str(config_a[key])).resolve() != Path(str(config_b[key])).resolve():
            raise ValueError(f"fused runtime members do not share the same {key}")
    model_a = _load_runtime_model(member_a, config_a, device)
    model_b = _load_runtime_model(member_b, config_b, device)
    prediction = LocalResidualFieldPredictionConfig(
        checkpoint=str(config_a["checkpoint"]), prediction_dir="unused", model_name=member_a,
        hlt_cache_dir=str(config_a["hlt_cache_dir"]), target_cache_dir=None,
        manifest_path=str(config_a["manifest_path"]), splits=(FUSION_FINAL_SPLIT,),
        batch_size=int(batch_size), num_workers=0, device=str(device),
        amp=bool(config_a.get("amp", True) and config_b.get("amp", True)),
        confirm_final_test=True, allow_oracle_final_test=False,
    )
    loader = _make_prediction_loader(
        _prediction_dataset(prediction, FUSION_FINAL_SPLIT, model=model_a), batch_size=int(batch_size),
        num_workers=0, seed=0, hlt_only=True,
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("fused runtime dataset is empty") from exc
    batch = move_local_particle_residual_field_batch_to_device(batch, device)
    actual_batch_size = int(batch["labels"].shape[0])
    amp_enabled = bool(prediction.amp and device.type == "cuda")
    family = str(candidate_report["family"])
    head_models: list[torch.nn.Module] = []
    capture_a = capture_b = None
    head_artifact_size = 0
    head_parameters = int(candidate_report["trainable_parameter_count"])
    deployed_head_artifacts: list[dict[str, Any]] = []
    if family == "representation":
        capture_a = PreClassifierEmbeddingCapture(model_a)
        capture_b = PreClassifierEmbeddingCapture(model_b)
        deployed_heads, deployed_head_artifacts = _verified_deployed_heads(candidate_id, candidate_report)
        head_parameters = 0
        for head, binding in zip(deployed_heads, deployed_head_artifacts, strict=True):
            checkpoint_path = Path(binding["checkpoint_path"])
            model, _payload = load_representation_fusion_head_from_checkpoint(checkpoint_path, device=device)
            model.eval()
            head_models.append(model)
            head_parameters += int(sum(parameter.numel() for parameter in model.parameters()))
            head_artifact_size += int(checkpoint_path.stat().st_size)
    else:
        head_artifact_size = sum(int(Path(row["path"]).stat().st_size) for row in candidate_report["fit_artifacts"])

    def forward() -> torch.Tensor:
        with amp_autocast_context(amp_enabled):
            if family == "representation":
                assert capture_a is not None and capture_b is not None
                logits_a, embedding_a = capture_a.capture(lambda: _campaign_forward(model_a, batch))
                logits_b, embedding_b = capture_b.capture(lambda: _campaign_forward(model_b, batch))
                return torch.stack([
                    head(embedding_a, embedding_b, logits_a, logits_b).logits for head in head_models
                ]).mean(dim=0)
            logits_a = _model_logits(_campaign_forward(model_a, batch))
            logits_b = _model_logits(_campaign_forward(model_b, batch))
            return _torch_late_fusion(candidate_id, parameters, logits_a, logits_b)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing = _time_callable(forward, device=device, warmup=warmup, batches=batches, batch_size=actual_batch_size)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    member_artifact_size = sum(int(Path(str(config["checkpoint"])).stat().st_size) for config in (config_a, config_b))
    member_parameters = sum(sum(parameter.numel() for parameter in model.parameters()) for model in (model_a, model_b))
    return {
        **timing, "device": str(device),
        "timing_scope": "measured_two_member_forward_plus_fusion_excludes_data_loading_and_host_to_device",
        "end_to_end_measured": True, "peak_gpu_memory_bytes": peak,
        "member_parameter_count": int(member_parameters), "fusion_parameter_count": int(head_parameters),
        "total_parameter_count": int(member_parameters + head_parameters),
        "member_artifact_size_bytes": member_artifact_size, "fusion_artifact_size_bytes": head_artifact_size,
        "total_artifact_size_bytes": member_artifact_size + head_artifact_size,
        "deployed_head_artifacts": deployed_head_artifacts,
    }


def _verified_deployed_heads(
    candidate_id: str, candidate_report: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Resolve and hash-bind exactly the heads used by deployment runtime."""

    heads = list(candidate_report["head_artifacts"])
    deployed = heads[:1] if candidate_id == "R0_linear_embeddings" else heads
    bindings: list[dict[str, Any]] = []
    for head in deployed:
        checkpoint_path = Path(head["checkpoint_path"]).resolve()
        observed_hash = sha256_file(checkpoint_path)
        if observed_hash != head.get("checkpoint_hash"):
            raise ValueError(f"selected fusion-head checkpoint changed before runtime benchmark: {checkpoint_path}")
        bindings.append({
            "seed": int(head["seed"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": observed_hash,
        })
    return deployed, bindings


def benchmark_selected_fusion_runtime(
    selected_fusion_json: str | Path, *, batch_size: int = 128, warmup_batches: int = 10, measured_batches: int = 50,
) -> dict[str, Any]:
    """Measure frozen member forwards and selected fusion overhead; never changes selection."""

    if min(int(batch_size), int(warmup_batches), int(measured_batches)) <= 0:
        raise ValueError("runtime batch size, warmup, and measured batch counts must be positive")
    selection_path = Path(selected_fusion_json).resolve()
    selection = load_selected_fusion_set(selection_path)
    _validate_selection_dependencies(selection_path, selection)
    final_root = selection_path.parent.parent / "final_evaluation" / selection["artifact_hash"][:16]
    final_path = final_root / "final_evaluation.json"
    final = _read_json(final_path)
    unsigned_final = dict(final)
    if unsigned_final.pop("artifact_hash", None) != stable_fusion_json_hash(unsigned_final):
        raise ValueError("final evaluation logical hash mismatch")
    output_path = final_root / "runtime_metrics.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable runtime artifact: {output_path}")

    feature_root = Path(selection["feature_root"])
    member_ids = sorted(final["member_metrics"])
    feature_configs = {
        member: _development_feature_config(feature_root, member)
        for member in member_ids
    }
    member_rows = [
        _benchmark_member(
            member, feature_configs[member], batch_size=int(batch_size),
            warmup=int(warmup_batches), batches=int(measured_batches),
        )
        for member in member_ids
    ]
    member_lookup = {row["member_id"]: row for row in member_rows}
    specs = {spec.candidate_id: spec for spec in default_fusion_candidate_specs()}
    bindings = {(row["group_id"], row["candidate_id"]): row for row in selection["selection_bindings"]}
    selected_records = {(row["group_id"], row["candidate_id"]): row for row in selection["selections"]}
    final_rows = {(row["group_id"], row["candidate_id"]): row for row in final["selected_results"]}
    fusion_rows: list[dict[str, Any]] = []
    for key, result in final_rows.items():
        group_id, candidate_id = key
        selected = selected_records[key]
        member_a, member_b = selected["member_ids"]
        spec = specs[candidate_id]
        candidate_report = _validate_candidate_report(bindings[key]["candidate_report_path"])
        timing = _benchmark_fusion_end_to_end(
            candidate_id=candidate_id, member_a=member_a, member_b=member_b,
            parameters=selected["hyperparameters"], candidate_report=candidate_report,
            feature_configs=feature_configs, batch_size=int(batch_size), warmup=int(warmup_batches),
            batches=int(measured_batches),
        )
        component_sum = member_lookup[member_a]["median_batch_latency_ms"] + member_lookup[member_b]["median_batch_latency_ms"]
        added = timing["median_batch_latency_ms"] - member_lookup["A0"]["median_batch_latency_ms"]
        accuracy_delta = float(result["multiclass"]["accuracy"]) - float(final["member_metrics"]["A0"]["multiclass"]["accuracy"])
        fusion_rows.append({
            "run_id": result["run_id"], "group_id": group_id, "candidate_id": candidate_id,
            "family": spec.family, "member_ids": [member_a, member_b], **timing,
            "component_member_median_sum_ms": component_sum,
            "added_median_batch_latency_ms_vs_A0": added,
            "accuracy_delta_vs_A0": accuracy_delta,
            "accuracy_points_per_added_ms": None if added <= 0 else 100.0 * accuracy_delta / added,
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        })
    report = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_CONTRACT,
        "campaign_id": selection["campaign_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_fusion_path": str(selection_path), "selected_fusion_sha256": sha256_file(selection_path),
        "selected_fusion_artifact_hash": selection["artifact_hash"],
        "final_evaluation_path": str(final_path.resolve()), "final_evaluation_sha256": sha256_file(final_path),
        "benchmark": {
            "warmup_batches": int(warmup_batches), "measured_batches": int(measured_batches),
            "requested_batch_size": int(batch_size),
            "canonical_fusion_latency": "measured_end_to_end_two_member_forward_plus_fusion",
            "excludes": ["data_loading", "host_to_device_transfer"],
        },
        "member_rows": member_rows, "fusion_rows": fusion_rows,
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "deployable": True,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(output_path, report)
    return report


__all__ = ["LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_CONTRACT", "benchmark_selected_fusion_runtime"]
