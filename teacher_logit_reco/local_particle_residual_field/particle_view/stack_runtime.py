"""Production runtime for sealed stack validation and HLT-only fusion.

The stack jobs are siblings in the production graph.  Consequently, no stack
job may consume an output from another stack job.  Each task authenticates the
PV06 selection and all PV07 controls, rebuilds the same sealed authorization,
and evaluates its required HLT-only checkpoints directly on ``stack_val``.
Fusion tasks materialize their two source-logit arrays in RAM, fit only on the
fixed stack-fit half, and report only on the disjoint evaluation half.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jetclass_fresh.hlt_baseline import collate_particle_transformer_batch
from jetclass_fresh.hlt_cache import load_cached_hlt_view

from .campaign import (
    _FAIRNESS_CONTROL_IDS,
    _STACK_STATIC_IDS,
    _WINNER_FAMILIES,
)
from .confirmation_runtime import (
    _candidate_records,
    resolve_confirmation_role,
)
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .direct_control import _direct_model
from .distillation import _consumer_forward
from .distillation_runtime import (
    _consumer_for_row,
    _pview_registration_and_model,
    _target_for_alias,
)
from .fairness_runtime import (
    PARTICLE_VIEW_STAGE_G_RESULT_CONTRACT,
    validate_fairness_factory_config,
)
from .fusion import (
    A0_A0_PAIRS,
    build_fusion_recipe,
    build_stack_fusion_partition,
    evaluate_fusion_recipe,
    fit_linear_logit_fusion,
)
from .metrics import (
    PAIRED_BOOTSTRAP_REPLICATES,
    build_paired_statistics_report,
    classification_metrics,
)
from .post_target_runtime import _artifact, _task_artifacts, _teacher_from_task
from .registry import validate_particle_view_registry
from .runtime_data import (
    _load_bound_manifests,
    validate_runtime_data_config,
)
from .splits import logical_split_identities
from .selection import (
    assert_split_access,
    build_sealed_split_authorization,
)


PARTICLE_VIEW_STACK_FACTORY_CONFIG_CONTRACT = (
    "particle_view_stack_factory_config_v1"
)
PARTICLE_VIEW_STACK_EVALUATION_CONTRACT = (
    "particle_view_stack_evaluation_v1"
)
PARTICLE_VIEW_STACK_FUSION_RESULT_CONTRACT = (
    "particle_view_stack_fusion_result_v1"
)
PARTICLE_VIEW_OPTIONAL_FUSION_STATUS_CONTRACT = (
    "particle_view_optional_fusion_status_v1"
)

_FAMILY_SELECTION_KEYS = {
    "PRIVILEGED_SCIENTIFIC": "selected_privileged_scientific_model",
    "PRE_STAGE_G_DEPLOYABLE": (
        "selected_pre_stage_g_hlt_deployable_model"
    ),
}
_PAIR_BY_STATIC_ID = {
    f"A0_A0_PAIR_{left}_{right}": (left, right)
    for left, right in A0_A0_PAIRS
}
_FUSION_STATIC_IDS = set(_STACK_STATIC_IDS) - {
    "MATCHED_CE_ONLY_COMPARATOR"
}


def build_stack_factory_config(
    *,
    fairness_factory_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    batch_size: int = 128,
    max_stack_batches: int | None = None,
    bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
    linear_fusion_steps: int = 300,
    optional_p7b_resource: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind PV08 to the exact PV07 factory and immutable runtime sources."""

    validate_fairness_factory_config(fairness_factory_config)
    data = fairness_factory_config["runtime_data_config"]
    validate_runtime_data_config(data, verify_cache_files=True)
    if (
        not isinstance(device, str)
        or not device
        or int(num_workers) < 0
        or int(batch_size) <= 0
        or int(bootstrap_replicates) <= 0
        or int(linear_fusion_steps) <= 0
    ):
        raise ValueError("stack runtime settings are invalid")
    if max_stack_batches is not None and int(max_stack_batches) <= 0:
        raise ValueError("max_stack_batches must be positive when set")
    p7b = _normalize_optional_p7b(optional_p7b_resource)
    unified = load_hashed_json(data["unified_manifest"]["path"])
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_FACTORY_CONFIG_CONTRACT,
            "fairness_factory_config": dict(fairness_factory_config),
            "fairness_factory_config_sha256": fairness_factory_config[
                "content_hash"
            ],
            "runtime_data_config_sha256": data["content_hash"],
            "runtime": {
                "device": device,
                "num_workers": int(num_workers),
                "batch_size": int(batch_size),
                "max_stack_batches": max_stack_batches,
                "bootstrap_replicates": int(bootstrap_replicates),
                "linear_fusion_steps": int(linear_fusion_steps),
            },
            "class_names": list(
                unified["split_config"]["class_names"]
            ),
            "winner_families": list(_WINNER_FAMILIES),
            "fairness_control_ids": list(_FAIRNESS_CONTROL_IDS),
            "stack_static_ids": list(_STACK_STATIC_IDS),
            "optional_p7b_resource": p7b,
            "stack_partition_policy": {
                "fit_fraction": 0.5,
                "fit_purpose": "linear_fusion_only",
                "evaluation_purpose": "sealed_fusion_reporting_only",
            },
            "stack_val_may_change_winner": False,
            "final_test_loaded": False,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_stack_factory_config(artifact)
    return artifact


def _normalize_optional_p7b(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    expected = {
        "logits_npz_path",
        "logits_npz_sha256",
        "logits_key",
        "labels_key",
        "event_identities_key",
        "bundle_sha256",
        "class_order",
        "stack_split_sha256",
        "stack_identity_sha256",
        "deployment_sha256",
        "requires_oracle",
        "final_test_hlt_only",
    }
    if set(value) != expected:
        raise ValueError("optional P7b resource field inventory mismatch")
    path = Path(value["logits_npz_path"]).resolve()
    digest = require_sha256("logits_npz_sha256", value["logits_npz_sha256"])
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError("optional P7b logit resource is absent or stale")
    result = dict(value)
    result["logits_npz_path"] = str(path)
    for name in (
        "bundle_sha256",
        "stack_split_sha256",
        "stack_identity_sha256",
        "deployment_sha256",
    ):
        result[name] = require_sha256(name, result[name])
    if (
        not result["class_order"]
        or len(set(result["class_order"])) != len(result["class_order"])
        or result["requires_oracle"] is not False
        or result["final_test_hlt_only"] is not True
    ):
        raise ValueError("optional P7b resource is not HLT-only/class-aligned")
    return result


def validate_stack_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_STACK_FACTORY_CONFIG_CONTRACT
    )
    expected = {
        "contract",
        "fairness_factory_config",
        "fairness_factory_config_sha256",
        "runtime_data_config_sha256",
        "runtime",
        "class_names",
        "winner_families",
        "fairness_control_ids",
        "stack_static_ids",
        "optional_p7b_resource",
        "stack_partition_policy",
        "stack_val_may_change_winner",
        "final_test_loaded",
        "performance_gates",
        "quality_warnings_stop_execution",
        "content_hash",
    }
    fairness = payload["fairness_factory_config"]
    validate_fairness_factory_config(fairness)
    validate_runtime_data_config(
        fairness["runtime_data_config"], verify_cache_files=False
    )
    runtime = payload["runtime"]
    if (
        set(payload) != expected
        or payload["fairness_factory_config_sha256"]
        != fairness["content_hash"]
        or payload["runtime_data_config_sha256"]
        != fairness["runtime_data_config"]["content_hash"]
        or set(runtime)
        != {
            "device",
            "num_workers",
            "batch_size",
            "max_stack_batches",
            "bootstrap_replicates",
            "linear_fusion_steps",
        }
        or int(runtime["num_workers"]) < 0
        or int(runtime["batch_size"]) <= 0
        or int(runtime["bootstrap_replicates"]) <= 0
        or int(runtime["linear_fusion_steps"]) <= 0
        or (
            runtime["max_stack_batches"] is not None
            and int(runtime["max_stack_batches"]) <= 0
        )
        or payload["winner_families"] != list(_WINNER_FAMILIES)
        or payload["fairness_control_ids"] != list(_FAIRNESS_CONTROL_IDS)
        or payload["stack_static_ids"] != list(_STACK_STATIC_IDS)
        or payload["stack_partition_policy"]
        != {
            "fit_fraction": 0.5,
            "fit_purpose": "linear_fusion_only",
            "evaluation_purpose": "sealed_fusion_reporting_only",
        }
        or payload["stack_val_may_change_winner"] is not False
        or payload["final_test_loaded"] is not False
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("stack factory policy changed")
    _normalize_optional_p7b(payload["optional_p7b_resource"])
    return {"ok": True, "content_hash": payload["content_hash"]}


@dataclass(frozen=True)
class _StackHLTLogicalView:
    hlt: Any
    parent_row_indices: np.ndarray
    logical_split_sha256: str
    ordered_identity_sha256: str

    def __len__(self) -> int:
        return int(self.parent_row_indices.size)


def _load_stack_hlt_view(data: Mapping[str, Any]) -> _StackHLTLogicalView:
    """Load only fixed-HLT stack inputs; never open the offline stack cache."""

    validate_runtime_data_config(data, verify_cache_files=False)
    parent, unified, config = _load_bound_manifests(data)
    split = unified["logical_splits"]["stack_val"]
    parent_split = str(split["parent_split"])
    record = next(
        row
        for row in data["parent_cache_records"]
        if row["parent_split"] == parent_split
    )
    for kind in ("hlt_array", "hlt_metadata"):
        path = Path(record[kind]["path"])
        if sha256_file(path) != record[kind]["sha256"]:
            raise ValueError(f"stack HLT {kind} changed")
    hlt = load_cached_hlt_view(
        data["hlt_cache_dir"], parent_split, verify_hash=True
    )
    expected_parent = parent.splits[parent_split]
    if [row.key() for row in hlt.jet_ids] != [
        row.key() for row in expected_parent
    ]:
        raise ValueError("stack HLT identities differ from the parent split")
    expected_labels = np.asarray(
        [row.label for row in expected_parent], dtype=np.int64
    )
    if not np.array_equal(hlt.labels, expected_labels):
        raise ValueError("stack HLT labels differ from the parent split")
    expected = logical_split_identities(
        unified,
        parent=parent,
        split_name="stack_val",
        config=config,
    )
    if split["membership_kind"] == "complete_parent_alias":
        indices = np.arange(len(expected), dtype=np.int64)
    else:
        indices = np.asarray(split["parent_row_indices"], dtype=np.int64)
    if [
        hlt.jet_ids[int(index)].key() for index in indices
    ] != [row.key() for row in expected]:
        raise ValueError("stack HLT logical slicing changed identity order")
    return _StackHLTLogicalView(
        hlt=hlt,
        parent_row_indices=indices,
        logical_split_sha256=str(split["content_hash"]),
        ordered_identity_sha256=str(split["ordered_identity_sha256"]),
    )


class _StackHLTDataset:
    def __init__(self, aligned: _StackHLTLogicalView) -> None:
        self.aligned = aligned

    def __len__(self) -> int:
        return len(self.aligned)

    def __getitem__(self, index: int):
        parent = int(self.aligned.parent_row_indices[index])
        return (
            self.aligned.hlt.tokens[parent],
            self.aligned.hlt.mask[parent],
            np.int64(self.aligned.hlt.labels[parent]),
            np.int64(parent),
        )


def _collate_stack_hlt(samples):
    batch = collate_particle_transformer_batch(
        [(row[0], row[1], row[2]) for row in samples],
        source_view="fixed_hlt",
    )
    batch["parent_indices"] = torch.as_tensor(
        [int(row[3]) for row in samples], dtype=torch.long
    )
    return batch


def _stack_loader(
    aligned: _StackHLTLogicalView,
    *,
    batch_size: int,
    num_workers: int,
):
    kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(
        _StackHLTDataset(aligned),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_stack_hlt,
        **kwargs,
    )


def _resolved_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _source_logits(source: Mapping[str, Any], batch: Mapping[str, Any]):
    if source["kind"] == "particle_view":
        predictor = source["predictor"]
        view = predictor(
            batch["features"], batch["lorentz_vectors"], batch["mask"]
        ).mean
        output = _consumer_forward(source["consumer"], batch, view)
    else:
        output = source["model"](
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
        )
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("stack source output has no logits tensor")
    return logits


def _move_source(source: Mapping[str, Any], device: torch.device) -> None:
    names = (
        ("predictor", "consumer")
        if source["kind"] == "particle_view"
        else ("model",)
    )
    for name in names:
        source[name].to(device).eval()


def _release_source(source: Mapping[str, Any]) -> None:
    names = (
        ("predictor", "consumer")
        if source["kind"] == "particle_view"
        else ("model",)
    )
    for name in names:
        source[name].to(torch.device("cpu"))


def _collect_source_logits(
    *,
    sources: Sequence[Mapping[str, Any]],
    loader: Any,
    device: str,
    max_batches: int | None,
) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    resolved = _resolved_device(device)
    for source in sources:
        _move_source(source, resolved)
    logits = [[] for _ in sources]
    labels = []
    identities: list[int] = []
    try:
        with torch.no_grad():
            for batch_index, raw in enumerate(loader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                batch = {
                    key: (
                        value.to(resolved, non_blocking=True)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in raw.items()
                }
                labels.append(batch["labels"].detach().cpu().numpy())
                identities.extend(
                    int(value)
                    for value in batch["parent_indices"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                for index, source in enumerate(sources):
                    logits[index].append(
                        _source_logits(source, batch)
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )
    finally:
        for source in sources:
            _release_source(source)
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    if not labels:
        raise ValueError("stack-validation loader is empty")
    return (
        [np.concatenate(rows, axis=0) for rows in logits],
        np.concatenate(labels, axis=0),
        identities,
    )


def _load_selection_and_authorization(
    *,
    root: Path,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection_artifacts = _task_artifacts(
        root, registry, "SELECT_WINNER_FAMILIES", 101
    )
    selection = load_hashed_json(
        _artifact(selection_artifacts, "winner_selection.json")
    )
    ledger_artifacts = _task_artifacts(
        root, registry, "SELECTED_PATH_FAIRNESS_LEDGER", 101
    )
    fairness = load_hashed_json(
        _artifact(
            ledger_artifacts, "selected_path_fairness_ledger.json"
        )
    )
    ce_rows = []
    for seed in (101, 202, 303):
        artifacts = _task_artifacts(
            root, registry, "CONFIRM_CE_ONLY_UPPER_BOUND", seed
        )
        replica = load_hashed_json(
            _artifact(artifacts, "confirmation_replica.json")
        )
        ce_rows.append(
            {
                "bundle_sha256": replica["bundle_sha256"],
                "seed": seed,
                "role": "matched_ce_only_comparator",
            }
        )
    control_rows = []
    controls: dict[tuple[str, str, int], dict[str, Any]] = {}
    for seed in (101, 202, 303):
        a0 = _load_a0_source(root, registry, seed)
        control_rows.append(
            {
                "bundle_sha256": a0["bundle_sha256"],
                "seed": seed,
                "role": "matched_a0_baseline",
            }
        )
    for family in _WINNER_FAMILIES:
        for control in _FAIRNESS_CONTROL_IDS:
            run_id = f"FAIR_{family}_{control}"
            for seed in (101, 202, 303):
                artifacts = _task_artifacts(root, registry, run_id, seed)
                result = load_hashed_json(
                    _artifact(artifacts, "stage_g_result.json")
                )
                validate_content_hash(
                    result,
                    expected_contract=PARTICLE_VIEW_STAGE_G_RESULT_CONTRACT,
                )
                checkpoint = _artifact(
                    artifacts, "best_model_val_stop_within_budget.pt"
                )
                if result["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise ValueError("Stage-G checkpoint lineage changed")
                controls[(family, control, seed)] = {
                    "result": result,
                    "artifacts": artifacts,
                    "checkpoint": checkpoint,
                }
                control_rows.append(
                    {
                        "bundle_sha256": result["checkpoint_sha256"],
                        "seed": seed,
                        "role": "stage_g_control",
                    }
                )
    if config["optional_p7b_resource"] is not None:
        control_rows.append(
            {
                "bundle_sha256": config["optional_p7b_resource"][
                    "bundle_sha256"
                ],
                "seed": 101,
                "role": "optional_p7b_hlt_only",
            }
        )
    data = config["fairness_factory_config"]["runtime_data_config"]
    unified = load_hashed_json(data["unified_manifest"]["path"])
    authorization = build_sealed_split_authorization(
        selection=selection,
        fairness_ledger=fairness,
        stack_split_sha256=unified["logical_splits"]["stack_val"][
            "content_hash"
        ],
        final_test_split_sha256=unified["logical_splits"]["final_test"][
            "content_hash"
        ],
        ce_only_comparator_bundles=ce_rows,
        stage_g_control_bundles=control_rows,
    )
    return selection, authorization, controls


def _load_a0_source(
    root: Path, registry: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    _, checkpoint, model = _teacher_from_task(
        root, registry, "A0_VIEW", int(seed)
    )
    return {
        "kind": "direct",
        "model": model,
        "bundle_sha256": sha256_file(checkpoint),
        "configuration_id": "A0_VIEW",
        "seed": int(seed),
    }


def _winner_replica(
    selection: Mapping[str, Any], family: str, seed: int
) -> Mapping[str, Any]:
    winner = selection[_FAMILY_SELECTION_KEYS[family]]
    rows = [row for row in winner["replicas"] if int(row["seed"]) == int(seed)]
    if len(rows) != 1:
        raise ValueError("winner family does not resolve to one seed replica")
    return rows[0]


def _load_confirmation_source(
    *,
    root: Path,
    registry: Mapping[str, Any],
    replica: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(replica["seed"])
    checkpoint = Path(replica["bundle_path"]).resolve()
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != replica["bundle_sha256"]
    ):
        raise ValueError("confirmation winner bundle is absent or stale")
    source = resolve_confirmation_role(
        str(replica["role_id"]), _candidate_records(root, registry)
    )
    row = source["campaign_row"]
    target, _ = _target_for_alias(
        root=root,
        registry=registry,
        seed=seed,
        alias=row["target_id"],
    )
    view_dim = int(
        target.registration["generator_config"]["bottleneck_width"]
    )
    consumer, _, _ = _consumer_for_row(
        root=root,
        registry=registry,
        seed=seed,
        alias=row["target_id"],
        consumer_id=row["consumer_id"],
        view_dim=view_dim,
    )
    predictor, _, _ = _pview_registration_and_model(
        root=root,
        registry=registry,
        seed=seed,
        architecture_id=row["architecture_id"],
        view_dim=view_dim,
        consumer=consumer,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    if "consumer_state_dict" in payload:
        consumer.load_state_dict(payload["consumer_state_dict"], strict=True)
    return {
        "kind": "particle_view",
        "predictor": predictor,
        "consumer": consumer,
        "bundle_sha256": replica["bundle_sha256"],
        "configuration_id": replica["configuration_id"],
        "seed": seed,
    }


def _load_stage_g_source(
    *,
    root: Path,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
    family: str,
    control: str,
    seed: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if control in {"A0_VIEW_LONG_DEPLOY", "A0_VIEW_TOTAL_LABEL_BUDGET"}:
        _, _, model = _teacher_from_task(root, registry, "A0_VIEW", seed)
    else:
        result = binding["result"]
        candidate = next(
            row
            for row in config["fairness_factory_config"][
                "direct_candidates"
            ]
            if row["config_sha256"]
            == result["selected_direct_config_sha256"]
        )
        model = _direct_model(candidate)
        summary = load_hashed_json(
            _artifact(binding["artifacts"], "direct_trial_summary.json")
        )
        selected_id = summary["selected_trial_id"]
        selected = next(
            row for row in summary["trials"] if row["trial_id"] == selected_id
        )
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = float(selected["dropout"])
    payload = torch.load(
        binding["checkpoint"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {
        "kind": "direct",
        "model": model,
        "bundle_sha256": binding["result"]["checkpoint_sha256"],
        "configuration_id": (
            f"{family}/{control}/{binding['result']['configuration_id']}"
        ),
        "seed": int(seed),
    }


def run_stack_evaluation(
    *,
    evaluations: Sequence[Mapping[str, Any]],
    loader: Any,
    authorization: Mapping[str, Any],
    output_dir: str,
    class_names: Sequence[str],
    stack_split_sha256: str,
    full_stack_identity_sha256: str,
    expected_stack_count: int,
    device: str,
    max_stack_batches: int | None,
    bootstrap_replicates: int,
) -> None:
    """Evaluate preauthorized candidates against matched A0 checkpoints."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    paired = []
    for item in evaluations:
        candidate = item["candidate"]
        baseline = item["baseline"]
        assert_split_access(
            authorization,
            split="stack_val",
            artifact_sha256=candidate["bundle_sha256"],
        )
        assert_split_access(
            authorization,
            split="stack_val",
            artifact_sha256=baseline["bundle_sha256"],
        )
        arrays, labels, identities = _collect_source_logits(
            sources=(baseline, candidate),
            loader=loader,
            device=device,
            max_batches=max_stack_batches,
        )
        observed_identity = canonical_sha256(
            [str(value) for value in identities]
        )
        complete = len(labels) == int(expected_stack_count)
        identity_sha = (
            full_stack_identity_sha256 if complete else observed_identity
        )
        baseline_metrics = classification_metrics(
            arrays[0], labels, split="stack_val", class_names=class_names
        )
        candidate_metrics = classification_metrics(
            arrays[1], labels, split="stack_val", class_names=class_names
        )
        statistics = build_paired_statistics_report(
            baseline_logits=arrays[0],
            candidate_logits=arrays[1],
            labels=labels,
            split="stack_val",
            baseline_artifact_sha256=baseline["bundle_sha256"],
            candidate_artifact_sha256=candidate["bundle_sha256"],
            split_sha256=stack_split_sha256,
            event_identity_sha256=identity_sha,
            replicates=int(bootstrap_replicates),
        )
        paired.append(statistics)
        rows.append(
            {
                "bundle_sha256": candidate["bundle_sha256"],
                "configuration_id": candidate["configuration_id"],
                "seed": int(candidate["seed"]),
                "role": item["role"],
                "winner_family": item.get("winner_family"),
                "split": "stack_val",
                "accuracy": candidate_metrics["accuracy"],
                "cross_entropy": candidate_metrics["cross_entropy"],
                "ece_top_label_15_equal_width": candidate_metrics[
                    "ece_top_label_15_equal_width"
                ],
                "multiclass_brier": candidate_metrics[
                    "multiclass_brier"
                ],
                "metrics": candidate_metrics,
                "matched_a0_bundle_sha256": baseline["bundle_sha256"],
                "matched_a0_metrics": baseline_metrics,
                "paired_statistics_sha256": statistics["content_hash"],
                "event_count": len(labels),
                "complete_stack_split": complete,
            }
        )
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_EVALUATION_CONTRACT,
            "authorization_sha256": authorization["content_hash"],
            "stack_split_sha256": stack_split_sha256,
            "full_stack_identity_sha256": full_stack_identity_sha256,
            "rows": rows,
            "paired_statistics": paired,
            "selection_changed": False,
            "warnings": (
                []
                if all(row["complete_stack_split"] for row in rows)
                else ["WARN_PARTIAL_STACK_REHEARSAL"]
            ),
            "warnings_are_non_gating": True,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(output / "sealed_split_authorization.json", authorization)
    write_immutable_json(output / "stack_evaluation.json", artifact)


def run_stack_fusion(
    *,
    sources: Sequence[Mapping[str, Any]],
    loader: Any,
    authorization: Mapping[str, Any],
    output_dir: str,
    fusion_id: str,
    method: str,
    class_names: Sequence[str],
    stack_split_sha256: str,
    expected_stack_count: int,
    device: str,
    max_stack_batches: int | None,
    linear_fusion_steps: int,
    optional_p7b_resource: Mapping[str, Any] | None = None,
) -> None:
    """Fit/evaluate one frozen fusion recipe without selecting a new model."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if fusion_id == "OPTIONAL_P7B_FUSION" and optional_p7b_resource is None:
        status = with_content_hash(
            {
                "contract": PARTICLE_VIEW_OPTIONAL_FUSION_STATUS_CONTRACT,
                "fusion_id": fusion_id,
                "status": "not_run",
                "reason": "no_authenticated_hlt_only_p7b_resource",
                "selection_changed": False,
                "warning_is_non_gating": True,
            }
        )
        write_immutable_json(
            output / "sealed_split_authorization.json", authorization
        )
        write_immutable_json(output / "fusion_status.json", status)
        return
    for source in sources:
        assert_split_access(
            authorization,
            split="stack_val",
            artifact_sha256=source["bundle_sha256"],
        )
    arrays, labels, identities = _collect_source_logits(
        sources=sources,
        loader=loader,
        device=device,
        max_batches=max_stack_batches,
    )
    source_hashes = [source["bundle_sha256"] for source in sources]
    p7b_provenance = None
    if optional_p7b_resource is not None:
        resource = optional_p7b_resource
        with np.load(resource["logits_npz_path"], allow_pickle=False) as npz:
            external = np.asarray(npz[resource["logits_key"]], dtype=np.float32)
            external_labels = np.asarray(
                npz[resource["labels_key"]], dtype=np.int64
            )
            external_ids = [
                int(value)
                for value in np.asarray(
                    npz[resource["event_identities_key"]]
                ).tolist()
            ]
        external_identity_sha = canonical_sha256(
            [str(value) for value in external_ids]
        )
        if (
            resource["stack_split_sha256"] != stack_split_sha256
            or resource["stack_identity_sha256"] != external_identity_sha
            or list(resource["class_order"]) != list(class_names)
            or not np.array_equal(external_labels, labels)
            or external_ids != identities
            or external.shape != arrays[0].shape
        ):
            raise ValueError("optional P7b stack logits are not aligned")
        assert_split_access(
            authorization,
            split="stack_val",
            artifact_sha256=resource["bundle_sha256"],
        )
        arrays.append(external)
        source_hashes.append(resource["bundle_sha256"])
        p7b_provenance = {
            "requires_oracle": False,
            "final_test_hlt_only": True,
            "deployment_sha256": resource["deployment_sha256"],
        }
    if len(labels) != int(expected_stack_count):
        raise ValueError(
            "fusion is forbidden on a partial stack-validation rehearsal"
        )
    partition = build_stack_fusion_partition(
        event_identities=identities,
        stack_split_sha256=stack_split_sha256,
    )
    parameters = (
        fit_linear_logit_fusion(
            source_logits=arrays,
            labels=labels,
            fit_indices=partition["fit_indices"],
            steps=int(linear_fusion_steps),
        )
        if method == "linear_logit"
        else None
    )
    recipe = build_fusion_recipe(
        fusion_id=fusion_id,
        source_bundle_sha256=source_hashes,
        class_order=class_names,
        stack_partition=partition,
        method=method,
        linear_parameters=parameters,
        optional_p7b=p7b_provenance is not None,
        p7b_hlt_only_provenance=p7b_provenance,
    )
    report = evaluate_fusion_recipe(
        recipe=recipe,
        stack_partition=partition,
        source_logits=arrays,
        labels=labels,
        source_bundle_sha256=source_hashes,
    )
    source_metrics = [
        classification_metrics(
            values[np.asarray(partition["evaluation_indices"])],
            labels[np.asarray(partition["evaluation_indices"])],
            split="stack_val_evaluation",
            class_names=class_names,
        )
        for values in arrays
    ]
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_FUSION_RESULT_CONTRACT,
            "authorization_sha256": authorization["content_hash"],
            "stack_partition_sha256": partition["content_hash"],
            "fusion_recipe_sha256": recipe["content_hash"],
            "fusion_report_sha256": report["content_hash"],
            "source_bundle_sha256": source_hashes,
            "source_metrics": source_metrics,
            "selection_changed": False,
            "evaluation_only": True,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(output / "sealed_split_authorization.json", authorization)
    write_immutable_json(output / "stack_partition.json", partition)
    write_immutable_json(output / "fusion_recipe.json", recipe)
    write_immutable_json(output / "fusion_report.json", report)
    write_immutable_json(output / "stack_fusion_result.json", result)


def _stack_context(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    output: Path,
):
    root = output.parent.parent
    selection, authorization, controls = _load_selection_and_authorization(
        root=root, registry=registry, config=config
    )
    data = config["fairness_factory_config"]["runtime_data_config"]
    aligned = _load_stack_hlt_view(data)
    loader = _stack_loader(
        aligned,
        batch_size=config["runtime"]["batch_size"],
        num_workers=config["runtime"]["num_workers"],
    )
    return root, selection, authorization, controls, aligned, loader


def build_stack_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_stack_factory_config(config)
    validate_particle_view_registry(registry)
    if task_id != f"{run_id}__seed_{int(seed)}":
        raise ValueError("stack task identity changed")
    output = Path(output_dir).resolve()
    root, selection, authorization, controls, aligned, loader = _stack_context(
        config=config, registry=registry, output=output
    )
    common = {
        "loader": loader,
        "authorization": authorization,
        "output_dir": str(output),
        "class_names": config["class_names"],
        "stack_split_sha256": aligned.logical_split_sha256,
        "expected_stack_count": len(aligned),
        "device": config["runtime"]["device"],
        "max_stack_batches": config["runtime"]["max_stack_batches"],
    }
    if operation == "stack_evaluation":
        evaluations = []
        if run_id.startswith("STACK_WINNER_"):
            family = run_id[len("STACK_WINNER_") :]
            replica = _winner_replica(selection, family, seed)
            candidate = _load_confirmation_source(
                root=root, registry=registry, replica=replica
            )
            evaluations.append(
                {
                    "candidate": candidate,
                    "baseline": _load_a0_source(root, registry, seed),
                    "role": "preselected_winner_replica",
                    "winner_family": family,
                }
            )
        elif run_id.startswith("STACK_FAIR_"):
            remainder = run_id[len("STACK_FAIR_") :]
            family = next(
                value
                for value in _WINNER_FAMILIES
                if remainder.startswith(value + "_")
            )
            control = remainder[len(family) + 1 :]
            binding = controls[(family, control, int(seed))]
            candidate = _load_stage_g_source(
                root=root,
                registry=registry,
                config=config,
                family=family,
                control=control,
                seed=int(seed),
                binding=binding,
            )
            evaluations.append(
                {
                    "candidate": candidate,
                    "baseline": _load_a0_source(root, registry, seed),
                    "role": "stage_g_control",
                    "winner_family": family,
                }
            )
        elif run_id == "STACK_MATCHED_CE_ONLY_COMPARATOR":
            for comparator_seed in (101, 202, 303):
                artifacts = _task_artifacts(
                    root,
                    registry,
                    "CONFIRM_CE_ONLY_UPPER_BOUND",
                    comparator_seed,
                )
                replica = load_hashed_json(
                    _artifact(artifacts, "confirmation_replica.json")
                )
                evaluations.append(
                    {
                        "candidate": _load_confirmation_source(
                            root=root, registry=registry, replica=replica
                        ),
                        "baseline": _load_a0_source(
                            root, registry, comparator_seed
                        ),
                        "role": "matched_ce_only_comparator",
                        "winner_family": None,
                    }
                )
        else:
            raise ValueError("unknown stack-evaluation run")
        return {
            "kwargs": {
                **common,
                "evaluations": evaluations,
                "full_stack_identity_sha256": (
                    aligned.ordered_identity_sha256
                ),
                "bootstrap_replicates": config["runtime"][
                    "bootstrap_replicates"
                ],
            },
            "artifact_paths": [
                str(output / "sealed_split_authorization.json"),
                str(output / "stack_evaluation.json"),
            ],
            "action": None,
        }
    if operation != "fusion" or not run_id.startswith("STACK_"):
        raise ValueError("stack operation/run identity changed")
    static_id = run_id[len("STACK_") :]
    if static_id not in _FUSION_STATIC_IDS:
        raise ValueError("unknown stack-fusion run")
    if static_id in _PAIR_BY_STATIC_ID:
        seeds = _PAIR_BY_STATIC_ID[static_id]
        sources = [_load_a0_source(root, registry, value) for value in seeds]
        method = "logit_average"
    else:
        family = (
            "PRIVILEGED_SCIENTIFIC"
            if static_id.startswith("PRIVILEGED_")
            or static_id == "OPTIONAL_P7B_FUSION"
            else "PRE_STAGE_G_DEPLOYABLE"
        )
        winner = selection[_FAMILY_SELECTION_KEYS[family]]
        representative_seed = int(winner["representative_seed"])
        replica = _winner_replica(
            selection, family, representative_seed
        )
        sources = [
            _load_a0_source(root, registry, representative_seed),
            _load_confirmation_source(
                root=root, registry=registry, replica=replica
            ),
        ]
        method = (
            "linear_logit"
            if static_id.endswith("LINEAR_FUSION")
            else "logit_average"
        )
    optional = (
        config["optional_p7b_resource"]
        if static_id == "OPTIONAL_P7B_FUSION"
        else None
    )
    artifact_paths = [
        str(output / "sealed_split_authorization.json"),
    ]
    if static_id == "OPTIONAL_P7B_FUSION" and optional is None:
        artifact_paths.append(str(output / "fusion_status.json"))
    else:
        artifact_paths.extend(
            [
                str(output / "stack_partition.json"),
                str(output / "fusion_recipe.json"),
                str(output / "fusion_report.json"),
                str(output / "stack_fusion_result.json"),
            ]
        )
    return {
        "kwargs": {
            **common,
            "sources": sources,
            "fusion_id": static_id,
            "method": method,
            "linear_fusion_steps": config["runtime"][
                "linear_fusion_steps"
            ],
            "optional_p7b_resource": optional,
        },
        "artifact_paths": artifact_paths,
        "action": None,
    }


def build_stack_task_specs(
    *, factory_config_path: str | Path
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_stack_factory_config(load_hashed_json(path))
    common = {
        "factory": (
            "teacher_logit_reco.local_particle_residual_field."
            "particle_view.stack_runtime:build_stack_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    specs: dict[str, dict[str, str]] = {}
    for family in _WINNER_FAMILIES:
        specs[f"STACK_WINNER_{family}"] = {
            **common,
            "operation": "stack_evaluation",
        }
        for control in _FAIRNESS_CONTROL_IDS:
            specs[f"STACK_FAIR_{family}_{control}"] = {
                **common,
                "operation": "stack_evaluation",
            }
    for static_id in _STACK_STATIC_IDS:
        specs[f"STACK_{static_id}"] = {
            **common,
            "operation": (
                "stack_evaluation"
                if static_id == "MATCHED_CE_ONLY_COMPARATOR"
                else "fusion"
            ),
        }
    return specs


__all__ = [
    "PARTICLE_VIEW_OPTIONAL_FUSION_STATUS_CONTRACT",
    "PARTICLE_VIEW_STACK_EVALUATION_CONTRACT",
    "PARTICLE_VIEW_STACK_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_STACK_FUSION_RESULT_CONTRACT",
    "build_stack_factory",
    "build_stack_factory_config",
    "build_stack_task_specs",
    "run_stack_evaluation",
    "run_stack_fusion",
    "validate_stack_factory_config",
]
