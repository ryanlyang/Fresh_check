"""PV10 one-time, HLT-only final evaluation and immutable publication.

The first final-test task evaluates every distinct authorized source exactly
once.  The sibling family task reuses the same authenticated publication and
does not reopen the final-test cache.  Offline cache bindings remain part of
the campaign provenance, but this module deliberately never reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jetclass_fresh.hlt_baseline import collate_particle_transformer_batch
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .deployment import (
    PARTICLE_VIEW_BUNDLE_INPUT_NAMES,
    load_exported_particle_view_bundle,
    validate_particle_view_bundle_export,
)
from .fusion import (
    PARTICLE_VIEW_FUSION_RECIPE_CONTRACT,
    apply_linear_logit_fusion,
)
from .metrics import (
    PAIRED_BOOTSTRAP_REPLICATES,
    build_paired_statistics_report,
    classification_metrics,
)
from .post_target_runtime import _artifact, _task_artifacts, _teacher_from_task
from .production import build_quality_warning, write_quality_warning_jsonl
from .report_runtime import (
    _FINAL_FUSION_IDS,
    _PERMIT_FAMILIES,
    _canonical_permit_exports,
    _collect_pv08,
    _export_and_reload_artifacts,
    validate_report_factory_config,
)
from .reporting import (
    PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT,
    PARTICLE_VIEW_REPORT_SECTIONS,
    _validate_final_test_permit,
    build_separated_campaign_report,
)
from .runtime_data import _load_bound_manifests, validate_runtime_data_config
from .splits import logical_split_identities
from .stack_runtime import _resolved_device


PARTICLE_VIEW_FINAL_FACTORY_CONFIG_CONTRACT = (
    "particle_view_final_factory_config_v1"
)
PARTICLE_VIEW_FINAL_HLT_SOURCE_AUDIT_CONTRACT = (
    "particle_view_final_hlt_source_audit_v1"
)
PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT = (
    "particle_view_final_evaluation_plan_v1"
)
PARTICLE_VIEW_FINAL_BASELINE_RESULT_CONTRACT = (
    "particle_view_final_a0_result_v1"
)
PARTICLE_VIEW_FINAL_FUSION_RESULT_CONTRACT = (
    "particle_view_final_fusion_result_v1"
)
PARTICLE_VIEW_FINAL_RESULT_INDEX_CONTRACT = (
    "particle_view_final_result_index_v1"
)
PARTICLE_VIEW_FINAL_PUBLICATION_CONTRACT = (
    "particle_view_final_publication_v1"
)
PARTICLE_VIEW_FINAL_FAMILY_BINDING_CONTRACT = (
    "particle_view_final_family_binding_v1"
)
PARTICLE_VIEW_FINAL_ACCESS_CLAIM_CONTRACT = (
    "particle_view_final_access_claim_v1"
)
PARTICLE_VIEW_FINAL_RECOVERY_AUTHORIZATION_CONTRACT = (
    "particle_view_final_recovery_authorization_v1"
)
PARTICLE_VIEW_FINAL_RECOVERY_CONSUMPTION_CONTRACT = (
    "particle_view_final_recovery_consumption_v1"
)
PARTICLE_VIEW_FINAL_ACCESS_RECEIPT_CONTRACT = (
    "particle_view_final_access_receipt_v1"
)

_FINAL_RUN_FAMILIES = {
    "FINAL_PRIVILEGED_SCIENTIFIC": "PRIVILEGED_SCIENTIFIC",
    "FINAL_PRE_STAGE_G_DEPLOYABLE": "PRE_STAGE_G_DEPLOYABLE",
}
_FORBIDDEN_BATCH_FRAGMENTS = (
    "offline",
    "oracle",
    "teacher",
    "true_view",
    "selected_view",
    "target_logit",
    "attention_map",
    "gview",
)


@dataclass(frozen=True)
class FinalHLTLogicalView:
    hlt: Any
    parent_row_indices: np.ndarray
    logical_split_sha256: str
    ordered_identity_sha256: str
    parent_split: str

    def __len__(self) -> int:
        return int(self.parent_row_indices.size)


class _FinalHLTDataset:
    def __init__(self, view: FinalHLTLogicalView) -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, index: int):
        parent = int(self.view.parent_row_indices[index])
        return (
            self.view.hlt.tokens[parent],
            self.view.hlt.mask[parent],
            np.int64(self.view.hlt.labels[parent]),
            np.int64(parent),
        )


def _collate_final_hlt(samples):
    batch = collate_particle_transformer_batch(
        [(row[0], row[1], row[2]) for row in samples],
        source_view="fixed_hlt",
    )
    batch["parent_indices"] = torch.as_tensor(
        [int(row[3]) for row in samples], dtype=torch.long
    )
    return batch


def _final_loader(
    view: FinalHLTLogicalView, *, batch_size: int, num_workers: int
):
    kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(
        _FinalHLTDataset(view),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_final_hlt,
        **kwargs,
    )


def load_final_hlt_view(
    runtime_data_config: Mapping[str, Any],
) -> tuple[FinalHLTLogicalView, dict[str, Any]]:
    """Authenticate and open only the HLT parent backing ``final_test``."""

    validate_runtime_data_config(
        runtime_data_config, verify_cache_files=False
    )
    parent, unified, config = _load_bound_manifests(runtime_data_config)
    split = unified["logical_splits"]["final_test"]
    parent_split = str(split["parent_split"])
    record = next(
        row
        for row in runtime_data_config["parent_cache_records"]
        if row["parent_split"] == parent_split
    )
    for kind in ("hlt_array", "hlt_metadata"):
        path = Path(record[kind]["path"])
        if sha256_file(path) != record[kind]["sha256"]:
            raise ValueError(f"final-test {kind} changed")
    hlt = load_cached_hlt_view(
        runtime_data_config["hlt_cache_dir"],
        parent_split,
        verify_hash=True,
    )
    expected_parent = parent.splits[parent_split]
    if [row.key() for row in hlt.jet_ids] != [
        row.key() for row in expected_parent
    ]:
        raise ValueError("final HLT identity order differs from parent")
    expected_labels = np.asarray(
        [row.label for row in expected_parent], dtype=np.int64
    )
    if not np.array_equal(hlt.labels, expected_labels):
        raise ValueError("final HLT labels differ from parent")
    expected = logical_split_identities(
        unified,
        parent=parent,
        split_name="final_test",
        config=config,
    )
    indices = (
        np.arange(len(expected), dtype=np.int64)
        if split["membership_kind"] == "complete_parent_alias"
        else np.asarray(split["parent_row_indices"], dtype=np.int64)
    )
    observed = [hlt.jet_ids[int(index)].key() for index in indices]
    if observed != [row.key() for row in expected]:
        raise ValueError("final HLT logical slicing changed identity order")
    view = FinalHLTLogicalView(
        hlt=hlt,
        parent_row_indices=indices,
        logical_split_sha256=str(split["content_hash"]),
        ordered_identity_sha256=str(split["ordered_identity_sha256"]),
        parent_split=parent_split,
    )
    audit = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_HLT_SOURCE_AUDIT_CONTRACT,
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "final_test_split_sha256": view.logical_split_sha256,
            "final_test_identity_sha256": view.ordered_identity_sha256,
            "parent_split": parent_split,
            "event_count": len(view),
            "verified_cache_bindings": {
                kind: record[kind]["sha256"]
                for kind in ("hlt_array", "hlt_metadata")
            },
            "opened_cache_kinds": ["hlt_array", "hlt_metadata"],
            "offline_cache_opened": False,
            "oracle_model_loaded": False,
            "selected_view_cache_loaded": False,
            "hlt_only": True,
        }
    )
    return view, audit


def build_final_factory_config(
    *,
    report_factory_config: Mapping[str, Any],
    device: str = "auto",
    batch_size: int = 128,
    num_workers: int = 0,
    bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    validate_report_factory_config(report_factory_config)
    if (
        not isinstance(device, str)
        or not device
        or int(batch_size) <= 0
        or int(num_workers) < 0
        or int(bootstrap_replicates) <= 0
    ):
        raise ValueError("invalid PV10 runtime configuration")
    payload = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_FACTORY_CONFIG_CONTRACT,
            "report_factory_config": dict(report_factory_config),
            "report_factory_config_sha256": report_factory_config[
                "content_hash"
            ],
            "runtime_data_config_sha256": report_factory_config[
                "runtime_data_config_sha256"
            ],
            "runtime": {
                "device": device,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "bootstrap_replicates": int(bootstrap_replicates),
            },
            "final_test_split": "final_test",
            "evaluate_each_distinct_source_once": True,
            "offline_cache_forbidden": True,
            "oracle_models_forbidden": True,
            "selection_changed": False,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_final_factory_config(payload)
    return payload


def validate_final_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_FINAL_FACTORY_CONFIG_CONTRACT
    )
    report = payload["report_factory_config"]
    validate_report_factory_config(report)
    runtime = payload["runtime"]
    expected = {
        "contract",
        "report_factory_config",
        "report_factory_config_sha256",
        "runtime_data_config_sha256",
        "runtime",
        "final_test_split",
        "evaluate_each_distinct_source_once",
        "offline_cache_forbidden",
        "oracle_models_forbidden",
        "selection_changed",
        "performance_gates",
        "quality_warnings_stop_execution",
        "content_hash",
    }
    if (
        set(payload) != expected
        or payload["report_factory_config_sha256"] != report["content_hash"]
        or payload["runtime_data_config_sha256"]
        != report["runtime_data_config_sha256"]
        or set(runtime)
        != {"device", "batch_size", "num_workers", "bootstrap_replicates"}
        or not isinstance(runtime["device"], str)
        or not runtime["device"]
        or int(runtime["batch_size"]) <= 0
        or int(runtime["num_workers"]) < 0
        or int(runtime["bootstrap_replicates"]) <= 0
        or payload["final_test_split"] != "final_test"
        or payload["evaluate_each_distinct_source_once"] is not True
        or payload["offline_cache_forbidden"] is not True
        or payload["oracle_models_forbidden"] is not True
        or payload["selection_changed"] is not False
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("PV10 final-test policy changed")
    return {"ok": True, "content_hash": payload["content_hash"]}


def _collect_logits(
    *,
    loader,
    bundle_modules: Mapping[str, torch.nn.Module],
    baseline_models: Mapping[str, torch.nn.Module],
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    resolved = _resolved_device(device)
    modules = {**bundle_modules, **baseline_models}
    for module in modules.values():
        module.to(resolved).eval()
    rows: dict[str, list[np.ndarray]] = {key: [] for key in modules}
    labels: list[np.ndarray] = []
    identities: list[int] = []
    try:
        with torch.no_grad():
            for raw in loader:
                if set(raw) != {
                    *PARTICLE_VIEW_BUNDLE_INPUT_NAMES,
                    "labels",
                    "parent_indices",
                }:
                    raise PermissionError(
                        "final batch inventory is not the locked HLT-only schema"
                    )
                if any(
                    fragment in str(name).lower()
                    for name in raw
                    for fragment in _FORBIDDEN_BATCH_FRAGMENTS
                ):
                    raise PermissionError("final batch exposes privileged data")
                inputs = tuple(
                    raw[name].to(resolved, non_blocking=True)
                    for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
                )
                labels.append(raw["labels"].cpu().numpy())
                identities.extend(
                    int(value) for value in raw["parent_indices"].tolist()
                )
                for key, module in bundle_modules.items():
                    output = module(*inputs)
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("bundle logits are nonfinite")
                    rows[key].append(output.detach().cpu().float().numpy())
                for key, module in baseline_models.items():
                    output = module(*inputs)
                    logits = (
                        output
                        if isinstance(output, torch.Tensor)
                        else getattr(output, "logits", None)
                    )
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError("A0 final output has no logits")
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("A0 logits are nonfinite")
                    rows[key].append(logits.detach().cpu().float().numpy())
    finally:
        for module in modules.values():
            module.to(torch.device("cpu"))
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    if not labels:
        raise ValueError("final-test loader is empty")
    return (
        {key: np.concatenate(value) for key, value in rows.items()},
        np.concatenate(labels),
        identities,
    )


def build_final_result_payloads(
    *,
    permit: Mapping[str, Any],
    evaluation_plan: Mapping[str, Any],
    logits_by_source: Mapping[str, np.ndarray],
    labels: np.ndarray,
    fusion_recipes: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] = LABEL_NAMES,
    bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic result artifacts from one in-memory final pass."""

    _validate_final_test_permit(permit)
    validate_content_hash(
        evaluation_plan,
        expected_contract=PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT,
    )
    split_sha = evaluation_plan["final_test_split_sha256"]
    identity_sha = evaluation_plan["final_test_identity_sha256"]
    if int(labels.size) != int(evaluation_plan["event_count"]):
        raise ValueError("final result builder received a partial split")
    baselines = []
    for row in permit["authorized_hlt_baselines"]:
        source = row["bundle_sha256"]
        metrics = classification_metrics(
            logits_by_source[source],
            labels,
            split="final_test",
            class_names=class_names,
        )
        baselines.append(
            with_content_hash(
                {
                    "contract": PARTICLE_VIEW_FINAL_BASELINE_RESULT_CONTRACT,
                    "permit_sha256": permit["content_hash"],
                    "source_bundle_sha256": source,
                    "seed": int(row["seed"]),
                    "winner_families": row["winner_families"],
                    "final_test_split_sha256": split_sha,
                    "final_test_identity_sha256": identity_sha,
                    "metrics": metrics,
                    "model_input_names": list(
                        PARTICLE_VIEW_BUNDLE_INPUT_NAMES
                    ),
                    "offline_inputs_loaded": False,
                    "oracle_model_loaded": False,
                    "selection_changed": False,
                    "one_time_evaluation": True,
                }
            )
        )
    baseline_by_family = {}
    for row in permit["authorized_hlt_baselines"]:
        for family in row["winner_families"]:
            if family in baseline_by_family:
                raise ValueError("winner family has multiple A0 baselines")
            baseline_by_family[family] = row["bundle_sha256"]
    standalone = []
    for entry in permit["authorized_exports"]:
        source = entry["source_bundle_sha256"]
        baseline_sources = {
            baseline_by_family[family] for family in entry["winner_families"]
        }
        if len(baseline_sources) != 1:
            raise ValueError("shared winner resolves to different A0 baselines")
        baseline = baseline_sources.pop()
        metrics = classification_metrics(
            logits_by_source[source],
            labels,
            split="final_test",
            class_names=class_names,
        )
        baseline_metrics = next(
            row["metrics"]
            for row in baselines
            if row["source_bundle_sha256"] == baseline
        )
        paired = build_paired_statistics_report(
            baseline_logits=logits_by_source[baseline],
            candidate_logits=logits_by_source[source],
            labels=labels,
            split="final_test",
            baseline_artifact_sha256=baseline,
            candidate_artifact_sha256=source,
            split_sha256=split_sha,
            event_identity_sha256=identity_sha,
            replicates=int(bootstrap_replicates),
        )
        standalone.append(
            with_content_hash(
                {
                    "contract": PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT,
                    "permit_sha256": permit["content_hash"],
                    "bundle_export_sha256": entry[
                        "bundle_export_sha256"
                    ],
                    "source_bundle_sha256": source,
                    "matched_a0_bundle_sha256": baseline,
                    "final_test_split_sha256": split_sha,
                    "final_test_identity_sha256": identity_sha,
                    "winner_families": entry["winner_families"],
                    "bundle_kind": evaluation_plan[
                        "bundle_kind_by_source"
                    ][source],
                    "seed": int(entry["seed"]),
                    "metrics": metrics,
                    "matched_a0_metrics": baseline_metrics,
                    "accuracy_gain_over_matched_a0": (
                        metrics["accuracy"] - baseline_metrics["accuracy"]
                    ),
                    "paired_statistics": paired,
                    "model_input_names": list(
                        PARTICLE_VIEW_BUNDLE_INPUT_NAMES
                    ),
                    "labels_used_for_evaluation_only": True,
                    "offline_inputs_loaded": False,
                    "selected_view_cache_loaded": False,
                    "oracle_model_loaded": False,
                    "selection_changed": False,
                    "one_time_evaluation": True,
                    "complete_final_split": True,
                }
            )
        )
    fusions = []
    for recipe in fusion_recipes:
        validate_content_hash(
            recipe, expected_contract=PARTICLE_VIEW_FUSION_RECIPE_CONTRACT
        )
        sources = list(recipe["source_bundle_sha256"])
        arrays = [logits_by_source[source] for source in sources]
        fused = (
            np.mean(arrays, axis=0)
            if recipe["method"] == "logit_average"
            else apply_linear_logit_fusion(
                arrays, recipe["linear_parameters"]
            )
        )
        reference = sources[0]
        metrics = classification_metrics(
            fused, labels, split="final_test", class_names=class_names
        )
        reference_metrics = classification_metrics(
            logits_by_source[reference],
            labels,
            split="final_test",
            class_names=class_names,
        )
        paired = build_paired_statistics_report(
            baseline_logits=logits_by_source[reference],
            candidate_logits=fused,
            labels=labels,
            split="final_test",
            baseline_artifact_sha256=reference,
            candidate_artifact_sha256=recipe["content_hash"],
            split_sha256=split_sha,
            event_identity_sha256=identity_sha,
            replicates=int(bootstrap_replicates),
        )
        fusions.append(
            with_content_hash(
                {
                    "contract": PARTICLE_VIEW_FINAL_FUSION_RESULT_CONTRACT,
                    "permit_sha256": permit["content_hash"],
                    "fusion_recipe_sha256": recipe["content_hash"],
                    "fusion_id": recipe["fusion_id"],
                    "source_bundle_sha256": sources,
                    "reference_bundle_sha256": reference,
                    "final_test_split_sha256": split_sha,
                    "final_test_identity_sha256": identity_sha,
                    "metrics": metrics,
                    "reference_metrics": reference_metrics,
                    "accuracy_gain_over_reference": (
                        metrics["accuracy"] - reference_metrics["accuracy"]
                    ),
                    "paired_statistics": paired,
                    "model_input_names": list(
                        PARTICLE_VIEW_BUNDLE_INPUT_NAMES
                    ),
                    "offline_inputs_loaded": False,
                    "oracle_model_loaded": False,
                    "selection_changed": False,
                    "one_time_evaluation": True,
                    "complete_final_split": True,
                }
            )
        )
    return {
        "baselines": baselines,
        "standalone": standalone,
        "fusions": fusions,
    }


def _result_paths(
    central: Path,
    permit: Mapping[str, Any],
    recipes: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    paths = {
        row["source_bundle_sha256"]: central / row["result_file"]
        for row in permit["authorized_exports"]
    }
    paths.update(
        {
            row["bundle_sha256"]: (
                central
                / f"final_test_a0_seed_{row['seed']}_{row['bundle_sha256'][:16]}.json"
            )
            for row in permit["authorized_hlt_baselines"]
        }
    )
    paths.update(
        {
            row["content_hash"]: (
                central / f"final_test_fusion_{row['content_hash'][:16]}.json"
            )
            for row in recipes
        }
    )
    return paths


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if load_hashed_json(path) != dict(payload):
            raise FileExistsError(f"existing final artifact differs: {path}")
    else:
        write_immutable_json(path, payload)


def build_final_recovery_authorization(
    *,
    access_claim: Mapping[str, Any],
    reason: str,
    authorized_by: str,
    previous_recovery_consumption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        access_claim,
        expected_contract=PARTICLE_VIEW_FINAL_ACCESS_CLAIM_CONTRACT,
    )
    if not reason or not authorized_by:
        raise ValueError("final recovery reason/authorizer must be nonempty")
    previous_sha256 = None
    if previous_recovery_consumption is not None:
        validate_content_hash(
            previous_recovery_consumption,
            expected_contract=PARTICLE_VIEW_FINAL_RECOVERY_CONSUMPTION_CONTRACT,
        )
        if (
            previous_recovery_consumption.get("access_claim_sha256")
            != access_claim["content_hash"]
        ):
            raise ValueError("previous recovery consumption belongs to another claim")
        previous_sha256 = previous_recovery_consumption["content_hash"]
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_RECOVERY_AUTHORIZATION_CONTRACT,
            "access_claim_sha256": access_claim["content_hash"],
            "reason": reason,
            "authorized_by": authorized_by,
            "previous_recovery_consumption_sha256": previous_sha256,
            "allow_one_recovery_access": True,
        }
    )


def _recovery_consumption_tail(
    central: Path,
    *,
    access_claim_sha256: str,
) -> tuple[str | None, list[str]]:
    root = central / "recovery_authorization_consumptions"
    if not root.exists():
        return None, []
    rows = []
    for path in sorted(root.glob("*.json")):
        row = load_hashed_json(path)
        validate_content_hash(
            row,
            expected_contract=PARTICLE_VIEW_FINAL_RECOVERY_CONSUMPTION_CONTRACT,
        )
        if row.get("access_claim_sha256") != access_claim_sha256:
            raise ValueError("recovery consumption belongs to another access claim")
        rows.append(row)
    by_hash = {row["content_hash"]: row for row in rows}
    if len(by_hash) != len(rows):
        raise ValueError("duplicate final recovery consumption")
    children: dict[str | None, list[str]] = {}
    for row in rows:
        previous = row.get("previous_recovery_consumption_sha256")
        if previous is not None and previous not in by_hash:
            raise ValueError("final recovery consumption chain is broken")
        children.setdefault(previous, []).append(row["content_hash"])
    cursor = None
    ordered: list[str] = []
    while children.get(cursor):
        candidates = children[cursor]
        if len(candidates) != 1:
            raise ValueError("final recovery consumption chain forks")
        cursor = candidates[0]
        if cursor in ordered:
            raise ValueError("final recovery consumption chain cycles")
        ordered.append(cursor)
    if len(ordered) != len(rows):
        raise ValueError("final recovery consumption chain is disconnected")
    return cursor, ordered


def _consume_final_recovery_authorization(
    *,
    central: Path,
    claim: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        claim,
        expected_contract=PARTICLE_VIEW_FINAL_ACCESS_CLAIM_CONTRACT,
    )
    validate_content_hash(
        authorization,
        expected_contract=PARTICLE_VIEW_FINAL_RECOVERY_AUTHORIZATION_CONTRACT,
    )
    if (
        authorization.get("access_claim_sha256") != claim["content_hash"]
        or authorization.get("allow_one_recovery_access") is not True
    ):
        raise PermissionError("final recovery authorization belongs to another claim")
    declared_predecessor = authorization.get(
        "previous_recovery_consumption_sha256"
    )
    root = central / "recovery_authorization_consumptions"
    predecessor_slot = declared_predecessor or "initial"
    destination = root / f"consumption_after_{predecessor_slot}.json"
    if destination.exists():
        raise PermissionError(
            "a final recovery authorization for this predecessor was already "
            "consumed"
        )
    tail_sha256, _ = _recovery_consumption_tail(
        central,
        access_claim_sha256=claim["content_hash"],
    )
    if declared_predecessor != tail_sha256:
        raise PermissionError(
            "final recovery authorization is stale or omits the previous "
            "recovery consumption"
        )
    consumption = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_RECOVERY_CONSUMPTION_CONTRACT,
            "access_claim_sha256": claim["content_hash"],
            "recovery_authorization_sha256": authorization["content_hash"],
            "previous_recovery_consumption_sha256": tail_sha256,
            "task_id": os.environ.get(
                "PARTICLE_VIEW_TASK_ID", "manual_final_recovery"
            ),
            "run_id": os.environ.get(
                "PARTICLE_VIEW_RUN_ID", "manual_final_recovery"
            ),
            "authorization_consumed_before_cache_open": True,
        }
    )
    write_immutable_json(destination, consumption)
    return consumption


def _finalize_publication(
    *,
    central: Path,
    permit: Mapping[str, Any],
    plan: Mapping[str, Any],
    pre_final_report: Mapping[str, Any],
    pre_final_strong_support: Mapping[str, Any],
    deployment_export_sha256: Sequence[str],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    source_audit: Mapping[str, Any],
) -> None:
    result_rows = [
        row for group in results.values() for row in group
    ]
    result_hashes = sorted(row["content_hash"] for row in result_rows)
    sections = {
        name: [dict(row) for row in pre_final_report["sections"][name]]
        for name in PARTICLE_VIEW_REPORT_SECTIONS
    }
    for row in results["baselines"]:
        sections["pre_stage_g_hlt_performance_controls"].append(
            {
                "row_id": (
                    f"final_a0__seed_{row['seed']}__"
                    f"{row['source_bundle_sha256'][:12]}"
                ),
                "artifact_sha256": row["content_hash"],
                "split": "final_test",
                "metrics": row["metrics"],
                "requires_oracle": False,
                "deployable": True,
            }
        )
    for row in results["standalone"]:
        section = (
            "frozen_consumer_hlt_deployable"
            if row["bundle_kind"] == "frozen_consumer"
            else "joint_hlt_deployable"
        )
        sections[section].append(
            {
                "row_id": (
                    "final_selected__"
                    f"{row['source_bundle_sha256'][:16]}"
                ),
                "artifact_sha256": row["content_hash"],
                "split": "final_test",
                "metrics": {
                    **row["metrics"],
                    "accuracy_gain_over_matched_a0": row[
                        "accuracy_gain_over_matched_a0"
                    ],
                },
                "requires_oracle": False,
                "deployable": True,
            }
        )
    for row in results["fusions"]:
        sections["fusion_ensemble_results"].append(
            {
                "row_id": f"final_fusion__{row['fusion_id']}",
                "artifact_sha256": row["content_hash"],
                "split": "final_test",
                "metrics": {
                    **row["metrics"],
                    "accuracy_gain_over_reference": row[
                        "accuracy_gain_over_reference"
                    ],
                },
                "requires_oracle": False,
                "deployable": True,
            }
        )
    warning_specs = []
    for row in results["standalone"]:
        if row["accuracy_gain_over_matched_a0"] <= 0:
            warning_specs.append(
                (
                    row,
                    row["source_bundle_sha256"],
                    int(row["seed"]),
                    row["accuracy_gain_over_matched_a0"],
                )
            )
    for row in results["fusions"]:
        if row["accuracy_gain_over_reference"] <= 0:
            warning_specs.append(
                (
                    row,
                    row["fusion_id"],
                    101,
                    row["accuracy_gain_over_reference"],
                )
            )
    result_path_by_hash = {}
    for path in central.glob("final_test_*.json"):
        payload = load_hashed_json(path)
        result_path_by_hash[payload["content_hash"]] = path
    warnings = [
        build_quality_warning(
            warning_code="WARN_NEGATIVE_FINAL_TEST_GAIN",
            severity="warning",
            graph_node="pv10_hlt_only_final_test",
            configuration_id=configuration_id,
            seed=seed,
            split="final_test",
            observed_value=float(gain),
            reference_value=0.0,
            warning_threshold=0.0,
            interpretation=(
                "The preselected deployable result did not retain a positive "
                "accuracy gain on final_test."
            ),
            suggested_diagnostic=(
                "Inspect the paired final statistics; do not change the "
                "already frozen winner selection."
            ),
            supporting_artifacts=[
                {
                    "path": str(
                        result_path_by_hash[row["content_hash"]].resolve()
                    ),
                    "sha256": sha256_file(
                        result_path_by_hash[row["content_hash"]]
                    ),
                }
            ],
            source_commit=plan["source_commit"],
            timestamp_utc="1970-01-01T00:00:00Z",
        )
        for row, configuration_id, seed, gain in warning_specs
    ]
    warning_stream = central / "final_scientific_warnings.jsonl"
    write_quality_warning_jsonl(warning_stream, warnings)
    warning_index = with_content_hash(
        {
            "contract": "particle_view_pv10_warning_index_v1",
            "warning_sha256": sorted(
                row["content_hash"] for row in warnings
            ),
            "warning_count": len(warnings),
            "warning_stream_path": str(warning_stream.resolve()),
            "warning_stream_sha256": sha256_file(warning_stream),
            "warnings_are_non_gating": True,
        }
    )
    _write_or_validate(central / "final_quality_warning_index.json", warning_index)
    privileged_final = [
        row
        for row in results["standalone"]
        if "selected_privileged_scientific_model" in row["winner_families"]
    ]
    final_gain = (
        float(privileged_final[0]["accuracy_gain_over_matched_a0"])
        if len(privileged_final) == 1
        else None
    )
    criteria = [dict(row) for row in pre_final_strong_support["criteria"]]
    criterion9 = next(row for row in criteria if row["criterion"] == 9)
    criterion9.update(
        {
            "status": "pass" if final_gain is not None and final_gain > 0 else "fail",
            "passed": bool(final_gain is not None and final_gain > 0),
            "evidence": {"final_test_accuracy_gain_over_matched_a0": final_gain},
            "warning_code": (
                None
                if final_gain is not None and final_gain > 0
                else "WARN_STRONG_SUPPORT_CRITERION_9"
            ),
        }
    )
    final_assessment = with_content_hash(
        {
            "contract": "particle_view_strong_support_assessment_v1",
            "selection_sha256": pre_final_strong_support["selection_sha256"],
            "split": "final",
            "criteria": criteria,
            "passed_count": sum(row["passed"] is True for row in criteria),
            "failed_count": sum(row["passed"] is False for row in criteria),
            "pending_count": 0,
            "strong_support_status": (
                "strongly_supported"
                if all(row["passed"] is True for row in criteria)
                else "not_strongly_supported"
            ),
            "warning_codes": sorted(
                row["warning_code"] for row in criteria if row["warning_code"]
            ),
            "warnings_are_non_gating": True,
        }
    )
    _write_or_validate(
        central / "final_strong_support_assessment.json",
        final_assessment,
    )
    report = build_separated_campaign_report(
        sections=sections,
        selection_sha256=pre_final_report["selection_sha256"],
        stack_report_sha256=pre_final_report["stack_report_sha256"],
        fairness_ledger_sha256=pre_final_report["fairness_ledger_sha256"],
        label_exposure_ledger_sha256=pre_final_report[
            "label_exposure_ledger_sha256"
        ],
        storage_reservation_sha256=pre_final_report[
            "storage_reservation_sha256"
        ],
        lineage_graph_sha256=pre_final_report["lineage_graph_sha256"],
        deployment_export_sha256=deployment_export_sha256,
        aggregate_warning_summary_sha256=warning_index["content_hash"],
        final_test_permit_sha256=permit["content_hash"],
        final_test_result_sha256=result_hashes,
    )
    _write_or_validate(central / "final_campaign_report.json", report)
    index = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_RESULT_INDEX_CONTRACT,
            "evaluation_plan_sha256": plan["content_hash"],
            "permit_sha256": permit["content_hash"],
            "source_audit_sha256": source_audit["content_hash"],
            "result_sha256": result_hashes,
            "baseline_result_count": len(results["baselines"]),
            "standalone_result_count": len(results["standalone"]),
            "fusion_result_count": len(results["fusions"]),
            "unique_model_execution_count": (
                len(results["baselines"]) + len(results["standalone"])
            ),
            "final_campaign_report_sha256": report["content_hash"],
            "complete_final_split": True,
            "offline_inputs_loaded": False,
            "oracle_model_loaded": False,
            "selection_changed": False,
        }
    )
    _write_or_validate(central / "final_result_index.json", index)
    publication = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_PUBLICATION_CONTRACT,
            "permit_sha256": permit["content_hash"],
            "evaluation_plan_sha256": plan["content_hash"],
            "result_index_sha256": index["content_hash"],
            "final_campaign_report_sha256": report["content_hash"],
            "warning_index_sha256": warning_index["content_hash"],
            "one_time_hlt_only_final_test": True,
            "all_warnings_non_gating": True,
        }
    )
    _write_or_validate(central / "final_publication.json", publication)


def run_final_test_campaign(
    *,
    output_dir: str,
    permit_family: str,
    central_output_dir: str,
    permit: Mapping[str, Any],
    evaluation_plan: Mapping[str, Any],
    runtime_data_config: Mapping[str, Any],
    bundle_manifest_paths: Sequence[str],
    baseline_models: Mapping[str, torch.nn.Module],
    fusion_recipes: Sequence[Mapping[str, Any]],
    pre_final_report: Mapping[str, Any],
    pre_final_strong_support: Mapping[str, Any],
    deployment_export_sha256: Sequence[str],
    class_names: Sequence[str],
    device: str,
    batch_size: int,
    num_workers: int,
    bootstrap_replicates: int,
) -> None:
    """Execute or reuse the single central PV10 publication."""

    _validate_final_test_permit(permit)
    validate_content_hash(
        evaluation_plan,
        expected_contract=PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT,
    )
    if (
        evaluation_plan["permit_sha256"] != permit["content_hash"]
        or evaluation_plan["runtime_data_config_sha256"]
        != runtime_data_config["content_hash"]
        or evaluation_plan["final_test_split_sha256"]
        != permit["final_test_split_sha256"]
        or set(evaluation_plan["bundle_kind_by_source"])
        != {
            row["source_bundle_sha256"]
            for row in permit["authorized_exports"]
        }
        or evaluation_plan["distinct_sources_evaluated_once"] is not True
        or evaluation_plan["offline_cache_forbidden"] is not True
        or evaluation_plan["oracle_models_forbidden"] is not True
        or evaluation_plan["selection_changed"] is not False
    ):
        raise ValueError("PV10 evaluation plan lineage/policy changed")
    central = Path(central_output_dir)
    central.mkdir(parents=True, exist_ok=True)
    _write_or_validate(central / "final_evaluation_plan.json", evaluation_plan)
    claim_path = central / "final_access_claim.json"
    claim_existed = claim_path.exists()
    claim = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_ACCESS_CLAIM_CONTRACT,
            "permit_sha256": permit["content_hash"],
            "evaluation_plan_sha256": evaluation_plan["content_hash"],
            "final_test_split_sha256": evaluation_plan[
                "final_test_split_sha256"
            ],
            "final_test_identity_sha256": evaluation_plan[
                "final_test_identity_sha256"
            ],
            "source_commit": evaluation_plan["source_commit"],
            "central_output_dir": str(central.resolve()),
            "one_time_access_claimed": True,
        }
    )
    _write_or_validate(claim_path, claim)
    index_path = central / "final_result_index.json"
    recovery_consumption = None
    if not index_path.exists():
        paths = _result_paths(central, permit, fusion_recipes)
        audit_path = central / "final_hlt_source_audit.json"
        if all(path.is_file() for path in paths.values()) and audit_path.is_file():
            results = {"baselines": [], "standalone": [], "fusions": []}
            for expected_key, path in paths.items():
                row = load_hashed_json(path)
                group = {
                    PARTICLE_VIEW_FINAL_BASELINE_RESULT_CONTRACT: "baselines",
                    PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT: "standalone",
                    PARTICLE_VIEW_FINAL_FUSION_RESULT_CONTRACT: "fusions",
                }.get(row["contract"])
                if group is None:
                    raise ValueError("unknown existing PV10 result contract")
                observed_key = (
                    row["fusion_recipe_sha256"]
                    if group == "fusions"
                    else row["source_bundle_sha256"]
                )
                if (
                    observed_key != expected_key
                    or row["permit_sha256"] != permit["content_hash"]
                    or row["final_test_split_sha256"]
                    != evaluation_plan["final_test_split_sha256"]
                    or row["final_test_identity_sha256"]
                    != evaluation_plan["final_test_identity_sha256"]
                ):
                    raise ValueError("existing PV10 result lineage changed")
                results[group].append(row)
            audit = load_hashed_json(audit_path)
        else:
            if claim_existed:
                authorization_path = os.environ.get(
                    "PARTICLE_VIEW_FINAL_RECOVERY_AUTHORIZATION"
                )
                if not authorization_path:
                    raise PermissionError(
                        "an incomplete claimed final-test access requires an "
                        "explicit recovery authorization"
                    )
                authorization = load_hashed_json(authorization_path)
                validate_content_hash(
                    authorization,
                    expected_contract=(
                        PARTICLE_VIEW_FINAL_RECOVERY_AUTHORIZATION_CONTRACT
                    ),
                )
                if (
                    authorization.get("access_claim_sha256")
                    != claim["content_hash"]
                    or authorization.get("allow_one_recovery_access") is not True
                ):
                    raise PermissionError(
                        "final recovery authorization belongs to another claim"
                    )
                recovery_consumption = _consume_final_recovery_authorization(
                    central=central,
                    claim=claim,
                    authorization=authorization,
                )
            view, audit = load_final_hlt_view(runtime_data_config)
            if (
                view.logical_split_sha256
                != evaluation_plan["final_test_split_sha256"]
                or view.ordered_identity_sha256
                != evaluation_plan["final_test_identity_sha256"]
                or len(view) != evaluation_plan["event_count"]
            ):
                raise ValueError(
                    "loaded final HLT view differs from evaluation plan"
                )
            _write_or_validate(audit_path, audit)
            modules = {}
            for path in bundle_manifest_paths:
                validation = validate_particle_view_bundle_export(path)
                modules[validation["source_bundle_sha256"]] = (
                    load_exported_particle_view_bundle(
                        path,
                        expected_source_bundle_sha256=validation[
                            "source_bundle_sha256"
                        ],
                        device="cpu",
                    )
                )
            loader = _final_loader(
                view, batch_size=batch_size, num_workers=num_workers
            )
            logits, labels, identities = _collect_logits(
                loader=loader,
                bundle_modules=modules,
                baseline_models=baseline_models,
                device=device,
            )
            if (
                len(identities) != len(view)
                or canonical_sha256([str(value) for value in identities])
                != evaluation_plan["parent_row_identity_sha256"]
            ):
                raise ValueError("final loader identity order changed")
            results = build_final_result_payloads(
                permit=permit,
                evaluation_plan=evaluation_plan,
                logits_by_source=logits,
                labels=labels,
                fusion_recipes=fusion_recipes,
                class_names=class_names,
                bootstrap_replicates=bootstrap_replicates,
            )
            for group in results.values():
                for row in group:
                    key = (
                        row["fusion_recipe_sha256"]
                        if row["contract"]
                        == PARTICLE_VIEW_FINAL_FUSION_RESULT_CONTRACT
                        else row["source_bundle_sha256"]
                    )
                    _write_or_validate(paths[key], row)
        _finalize_publication(
            central=central,
            permit=permit,
            plan=evaluation_plan,
            pre_final_report=pre_final_report,
            pre_final_strong_support=pre_final_strong_support,
            deployment_export_sha256=deployment_export_sha256,
            results=results,
            source_audit=audit,
        )
    index = load_hashed_json(index_path)
    validate_content_hash(
        index, expected_contract=PARTICLE_VIEW_FINAL_RESULT_INDEX_CONTRACT
    )
    report = load_hashed_json(central / "final_campaign_report.json")
    publication = load_hashed_json(central / "final_publication.json")
    source_audit = load_hashed_json(central / "final_hlt_source_audit.json")
    if (
        index["evaluation_plan_sha256"] != evaluation_plan["content_hash"]
        or index["permit_sha256"] != permit["content_hash"]
        or index["source_audit_sha256"] != source_audit["content_hash"]
        or index["final_campaign_report_sha256"] != report["content_hash"]
        or index["complete_final_split"] is not True
        or index["offline_inputs_loaded"] is not False
        or index["oracle_model_loaded"] is not False
        or publication["result_index_sha256"] != index["content_hash"]
        or publication["final_campaign_report_sha256"]
        != report["content_hash"]
        or publication["one_time_hlt_only_final_test"] is not True
    ):
        raise ValueError("PV10 central publication lineage changed")
    _, recovery_consumption_sha256 = _recovery_consumption_tail(
        central,
        access_claim_sha256=claim["content_hash"],
    )
    receipt = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_ACCESS_RECEIPT_CONTRACT,
            "access_claim_sha256": claim["content_hash"],
            "permit_sha256": permit["content_hash"],
            "evaluation_plan_sha256": evaluation_plan["content_hash"],
            "result_index_sha256": index["content_hash"],
            "final_publication_sha256": publication["content_hash"],
            "recovery_consumption_sha256": recovery_consumption_sha256,
            "recovery_authorization_consumed_before_cache_open": (
                bool(recovery_consumption_sha256)
            ),
            "final_cache_access_completed": True,
            "one_time_hlt_only_final_test": True,
        }
    )
    _write_or_validate(central / "final_access_receipt.json", receipt)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_json(
        output / "final_family_binding.json",
        with_content_hash(
            {
                "contract": PARTICLE_VIEW_FINAL_FAMILY_BINDING_CONTRACT,
                "permit_family": permit_family,
                "permit_sha256": permit["content_hash"],
                "result_index_sha256": index["content_hash"],
                "final_campaign_report_sha256": report["content_hash"],
                "final_publication_sha256": publication["content_hash"],
                "central_output_dir": str(central.resolve()),
                "final_cache_reopened_when_publication_existed": False,
                "selection_changed": False,
            }
        ),
    )


def _load_common_permit(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    permits = []
    for run_id in _PERMIT_FAMILIES:
        artifacts = _task_artifacts(root, registry, run_id, 101)
        permits.append(
            load_hashed_json(_artifact(artifacts, "final_test_permit.json"))
        )
    if permits[0] != permits[1]:
        raise ValueError("PV09 final-test family permits differ")
    _validate_final_test_permit(permits[0])
    return permits[0]


def build_final_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_final_factory_config(config)
    if (
        operation != "final_test"
        or run_id not in _FINAL_RUN_FAMILIES
        or int(seed) != 101
        or task_id != f"{run_id}__seed_101"
    ):
        raise ValueError("PV10 final task identity changed")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    central = root / "final_test_publication"
    permit = _load_common_permit(root, registry)
    report_config = config["report_factory_config"]
    pv08 = _collect_pv08(
        root=root, registry=registry, config=report_config
    )
    manifests, _, audits = _export_and_reload_artifacts(root, registry)
    manifests, _ = _canonical_permit_exports(manifests, audits)
    exports = [load_hashed_json(path) for path in manifests]
    if {
        row["content_hash"] for row in exports
    } != {
        row["bundle_export_sha256"]
        for row in permit["authorized_exports"]
    }:
        raise ValueError("PV10 exports differ from the permit")
    recipes = [
        row["recipe"]
        for row in pv08["fusions"]
        if row["recipe"]["content_hash"]
        in set(permit["authorized_fusion_recipe_sha256"])
    ]
    if {row["fusion_id"] for row in recipes} != set(_FINAL_FUSION_IDS):
        raise ValueError("PV10 final fusion inventory changed")
    runtime_data = report_config["stack_factory_config"][
        "fairness_factory_config"
    ]["runtime_data_config"]
    _, unified, _ = _load_bound_manifests(runtime_data)
    split = unified["logical_splits"]["final_test"]
    indices = (
        list(range(int(split["count"])))
        if split["membership_kind"] == "complete_parent_alias"
        else [int(value) for value in split["parent_row_indices"]]
    )
    plan = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT,
            "permit_sha256": permit["content_hash"],
            "runtime_data_config_sha256": runtime_data["content_hash"],
            "source_commit": report_config["source_commit"],
            "final_test_split_sha256": split["content_hash"],
            "final_test_identity_sha256": split[
                "ordered_identity_sha256"
            ],
            "parent_row_identity_sha256": canonical_sha256(
                [str(value) for value in indices]
            ),
            "event_count": int(split["count"]),
            "authorized_source_bundle_sha256": sorted(
                {
                    row["source_bundle_sha256"]
                    for row in permit["authorized_exports"]
                }
                | {
                    row["bundle_sha256"]
                    for row in permit["authorized_hlt_baselines"]
                }
            ),
            "bundle_kind_by_source": {
                row["source_bundle_sha256"]: row["deployment_manifest"][
                    "bundle_kind"
                ]
                for row in exports
            },
            "authorized_fusion_recipe_sha256": sorted(
                row["content_hash"] for row in recipes
            ),
            "distinct_sources_evaluated_once": True,
            "labels_used_for_evaluation_only": True,
            "offline_cache_forbidden": True,
            "oracle_models_forbidden": True,
            "selection_changed": False,
        }
    )
    baseline_models = {}
    for row in permit["authorized_hlt_baselines"]:
        _, checkpoint, model = _teacher_from_task(
            root, registry, "A0_VIEW", int(row["seed"])
        )
        if sha256_file(checkpoint) != row["bundle_sha256"]:
            raise ValueError("PV10 matched A0 checkpoint changed")
        baseline_models[row["bundle_sha256"]] = model
    aggregate = _task_artifacts(
        root, registry, "REPORT_AGGREGATE_REPORT", 101
    )
    pre_report = load_hashed_json(
        _artifact(aggregate, "pre_final_campaign_report.json")
    )
    pre_strong_support = load_hashed_json(
        _artifact(aggregate, "strong_support_assessment.json")
    )
    paths = _result_paths(central, permit, recipes)
    artifact_paths = [
        central / "final_evaluation_plan.json",
        central / "final_access_claim.json",
        central / "final_access_receipt.json",
        central / "final_hlt_source_audit.json",
        *paths.values(),
        central / "final_quality_warning_index.json",
        central / "final_strong_support_assessment.json",
        central / "final_campaign_report.json",
        central / "final_result_index.json",
        central / "final_publication.json",
        central / "final_scientific_warnings.jsonl",
        output / "final_family_binding.json",
    ]
    runtime = config["runtime"]
    return {
        "kwargs": {
            "output_dir": str(output),
            "permit_family": _FINAL_RUN_FAMILIES[run_id],
            "central_output_dir": str(central),
            "permit": permit,
            "evaluation_plan": plan,
            "runtime_data_config": runtime_data,
            "bundle_manifest_paths": manifests,
            "baseline_models": baseline_models,
            "fusion_recipes": recipes,
            "pre_final_report": pre_report,
            "pre_final_strong_support": pre_strong_support,
            "deployment_export_sha256": [
                row["content_hash"] for row in exports
            ],
            "class_names": report_config["stack_factory_config"][
                "class_names"
            ],
            "device": runtime["device"],
            "batch_size": int(runtime["batch_size"]),
            "num_workers": int(runtime["num_workers"]),
            "bootstrap_replicates": int(
                runtime["bootstrap_replicates"]
            ),
        },
        "artifact_paths": [str(path) for path in artifact_paths],
        "action": None,
    }


def build_final_task_specs(
    *, factory_config_path: str | Path
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_final_factory_config(load_hashed_json(path))
    common = {
        "operation": "final_test",
        "factory": (
            "teacher_logit_reco.local_particle_residual_field."
            "particle_view.final_runtime:build_final_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    return {run_id: dict(common) for run_id in _FINAL_RUN_FAMILIES}


__all__ = [
    "PARTICLE_VIEW_FINAL_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_FINAL_HLT_SOURCE_AUDIT_CONTRACT",
    "PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT",
    "PARTICLE_VIEW_FINAL_BASELINE_RESULT_CONTRACT",
    "PARTICLE_VIEW_FINAL_FUSION_RESULT_CONTRACT",
    "PARTICLE_VIEW_FINAL_RESULT_INDEX_CONTRACT",
    "PARTICLE_VIEW_FINAL_PUBLICATION_CONTRACT",
    "PARTICLE_VIEW_FINAL_FAMILY_BINDING_CONTRACT",
    "FinalHLTLogicalView",
    "build_final_factory",
    "build_final_factory_config",
    "build_final_result_payloads",
    "build_final_task_specs",
    "load_final_hlt_view",
    "run_final_test_campaign",
    "validate_final_factory_config",
]
