"""Audited from-scratch A0_seed1 recipe for the P7b fusion campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .fusion_campaign import (
    FUSION_MEMBER_A0,
    FUSION_MEMBER_A0_SEED1,
    INDEPENDENT_SEED_CANONICAL_CONFIG_ID,
    INDEPENDENT_SEED_DISPLAY_ALIAS,
    stable_fusion_json_hash,
)
from .tagger import RESIDUAL_FIELD_SOURCE_HLT_ONLY
from .tagger_train import LocalResidualFieldTaggerTrainConfig


LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_CONTRACT = "local_residual_field_a0_seed_recipe_v1"
LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_AUDIT_CONTRACT = "local_residual_field_a0_seed_recipe_audit_v1"
A0_TRAINING_SEED = 20421
A0_SEED1_TRAINING_SEED = 20522
A0_SEED1_ALLOWED_CONFIG_DIFFERENCES = frozenset({"output_dir", "seed"})

_FORBIDDEN_RUNTIME_OR_TRAINING_INPUT_FIELDS = (
    "baseline_checkpoint",
    "reconstructor_checkpoint",
    "teacher_logits_dir",
    "teacher_logits_train_path",
    "teacher_logits_val_path",
    "teacher_logits_stack_val_path",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_train_config(value: LocalResidualFieldTaggerTrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, LocalResidualFieldTaggerTrainConfig):
        config = value
    elif isinstance(value, Mapping):
        config = LocalResidualFieldTaggerTrainConfig(**dict(value))
    else:
        raise TypeError("tagger train config must be a LocalResidualFieldTaggerTrainConfig or mapping")
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def extract_a0_train_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and normalize the stored A0 config from source_metadata or a report."""

    candidates = (
        payload.get("config"),
        (payload.get("source_metadata") or {}).get("config")
        if isinstance(payload.get("source_metadata"), Mapping)
        else None,
        (payload.get("training") or {}).get("config")
        if isinstance(payload.get("training"), Mapping)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return _normalize_train_config(candidate)
    raise ValueError("A0 metadata does not contain a tagger training config mapping")


def build_a0_seed1_train_config(
    a0_config: LocalResidualFieldTaggerTrainConfig | Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Clone A0's normalized config while changing only seed and output directory."""

    source = _normalize_train_config(a0_config)
    candidate = dict(source)
    candidate["output_dir"] = str(output_dir)
    candidate["seed"] = A0_SEED1_TRAINING_SEED
    return _normalize_train_config(candidate)


def _config_differences(a0: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for key in sorted(set(a0) | set(candidate)):
        left = a0.get(key, {"missing": True})
        right = candidate.get(key, {"missing": True})
        if left != right:
            differences.append(
                {
                    "path": str(key),
                    "a0_value": left,
                    "candidate_value": right,
                    "allowed": str(key) in A0_SEED1_ALLOWED_CONFIG_DIFFERENCES,
                }
            )
    return differences


def audit_a0_seed1_recipe(
    a0_config: LocalResidualFieldTaggerTrainConfig | Mapping[str, Any],
    candidate_config: LocalResidualFieldTaggerTrainConfig | Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed audit of the independent HLT seed recipe."""

    a0 = _normalize_train_config(a0_config)
    candidate = _normalize_train_config(candidate_config)
    differences = _config_differences(a0, candidate)
    problems: list[str] = []

    if int(a0.get("seed", -1)) != A0_TRAINING_SEED:
        problems.append(f"stored A0 seed must be {A0_TRAINING_SEED}")
    if int(candidate.get("seed", -1)) != A0_SEED1_TRAINING_SEED:
        problems.append(f"A0_seed1 seed must be {A0_SEED1_TRAINING_SEED}")
    if a0.get("output_dir") == candidate.get("output_dir"):
        problems.append("A0_seed1 output_dir must differ from A0 output_dir")
    if str(a0.get("field_source")) != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        problems.append("stored A0 recipe must use field_source='hlt_only'")
    if str(candidate.get("field_source")) != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        problems.append("A0_seed1 recipe must use field_source='hlt_only'")
    if bool(a0.get("require_baseline_warm_start")) or bool(candidate.get("require_baseline_warm_start")):
        problems.append("A0 and A0_seed1 must not require baseline warm start")
    for field_name in _FORBIDDEN_RUNTIME_OR_TRAINING_INPUT_FIELDS:
        if a0.get(field_name) not in (None, ""):
            problems.append(f"stored A0 recipe unexpectedly sets {field_name}")
        if candidate.get(field_name) not in (None, ""):
            problems.append(f"A0_seed1 recipe must not set {field_name}")
    for field_name in ("kd_loss_weight", "reconstructor_loss_weight"):
        if float(a0.get(field_name, 0.0)) != 0.0 or float(candidate.get(field_name, 0.0)) != 0.0:
            problems.append(f"A0 and A0_seed1 must use {field_name}=0")
    forbidden_differences = [row["path"] for row in differences if not bool(row["allowed"])]
    if forbidden_differences:
        problems.append("non-allowlisted config differences: " + ", ".join(forbidden_differences))
    observed_allowed = {str(row["path"]) for row in differences if bool(row["allowed"])}
    missing_required = sorted(A0_SEED1_ALLOWED_CONFIG_DIFFERENCES - observed_allowed)
    if missing_required:
        problems.append("required config differences are missing: " + ", ".join(missing_required))

    provenance_payload = dict(provenance or {})
    if "training_source_match" in provenance_payload and provenance_payload.get("training_source_match") is not True:
        problems.append("current training source does not match the source used for A0")

    report: dict[str, Any] = {
        "contract": LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_AUDIT_CONTRACT,
        "ok": not problems,
        "problems": problems,
        "a0_run_id": FUSION_MEMBER_A0,
        "candidate_run_id": FUSION_MEMBER_A0_SEED1,
        "candidate_display_alias": INDEPENDENT_SEED_DISPLAY_ALIAS,
        "candidate_canonical_config_id": INDEPENDENT_SEED_CANONICAL_CONFIG_ID,
        "allowed_config_difference_paths": sorted(A0_SEED1_ALLOWED_CONFIG_DIFFERENCES),
        "differences": differences,
        "a0_config_hash": stable_fusion_json_hash(a0),
        "candidate_config_hash": stable_fusion_json_hash(candidate),
        "a0_seed": int(a0.get("seed", -1)),
        "candidate_seed": int(candidate.get("seed", -1)),
        "from_scratch": True,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "provenance": provenance_payload,
    }
    report["audit_hash"] = stable_fusion_json_hash(report)
    return report


def require_a0_seed1_recipe_audit(audit: Mapping[str, Any]) -> None:
    if str(audit.get("contract")) != LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_AUDIT_CONTRACT:
        raise ValueError("unexpected A0_seed1 recipe-audit contract")
    expected_hash = audit.get("audit_hash")
    without_hash = {key: value for key, value in dict(audit).items() if key != "audit_hash"}
    if expected_hash != stable_fusion_json_hash(without_hash):
        raise ValueError("A0_seed1 recipe audit hash mismatch")
    if audit.get("ok") is not True:
        raise ValueError(f"A0_seed1 recipe audit failed: {audit.get('problems')}")


@dataclass(frozen=True)
class A0Seed1Recipe:
    """Immutable identity envelope around the audited training config."""

    config: Mapping[str, Any]
    a0_config_hash: str
    audit_hash: str
    run_id: str = FUSION_MEMBER_A0_SEED1
    display_alias: str = INDEPENDENT_SEED_DISPLAY_ALIAS
    canonical_config_id: str = INDEPENDENT_SEED_CANONICAL_CONFIG_ID
    contract: str = LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_CONTRACT:
            raise ValueError("unexpected A0_seed1 recipe contract")
        if self.run_id != FUSION_MEMBER_A0_SEED1:
            raise ValueError("seed recipe run_id must be A0_seed1")
        if self.display_alias != INDEPENDENT_SEED_DISPLAY_ALIAS:
            raise ValueError("seed recipe display alias mismatch")
        if self.canonical_config_id != INDEPENDENT_SEED_CANONICAL_CONFIG_ID:
            raise ValueError("seed recipe canonical config ID mismatch")
        normalized = _normalize_train_config(self.config)
        if int(normalized["seed"]) != A0_SEED1_TRAINING_SEED:
            raise ValueError("seed recipe config does not use the locked independent seed")
        object.__setattr__(self, "config", normalized)
        for name in ("a0_config_hash", "audit_hash"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)

    @property
    def config_hash(self) -> str:
        return stable_fusion_json_hash(self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "run_id": self.run_id,
            "display_alias": self.display_alias,
            "canonical_config_id": self.canonical_config_id,
            "a0_config_hash": self.a0_config_hash,
            "audit_hash": self.audit_hash,
            "config_hash": self.config_hash,
            "config": dict(self.config),
        }


def build_a0_seed1_recipe(
    a0_config: LocalResidualFieldTaggerTrainConfig | Mapping[str, Any],
    *,
    output_dir: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[A0Seed1Recipe, dict[str, Any]]:
    a0 = _normalize_train_config(a0_config)
    candidate = build_a0_seed1_train_config(a0, output_dir=output_dir)
    audit = audit_a0_seed1_recipe(a0, candidate, provenance=provenance)
    require_a0_seed1_recipe_audit(audit)
    recipe = A0Seed1Recipe(
        config=candidate,
        a0_config_hash=str(audit["a0_config_hash"]),
        audit_hash=str(audit["audit_hash"]),
    )
    return recipe, audit


def load_a0_seed1_recipe(
    recipe_payload: Mapping[str, Any],
    audit_payload: Mapping[str, Any],
) -> tuple[A0Seed1Recipe, LocalResidualFieldTaggerTrainConfig]:
    require_a0_seed1_recipe_audit(audit_payload)
    recipe = A0Seed1Recipe(
        config=dict(recipe_payload.get("config") or {}),
        a0_config_hash=str(recipe_payload.get("a0_config_hash") or ""),
        audit_hash=str(recipe_payload.get("audit_hash") or ""),
        run_id=str(recipe_payload.get("run_id") or ""),
        display_alias=str(recipe_payload.get("display_alias") or ""),
        canonical_config_id=str(recipe_payload.get("canonical_config_id") or ""),
        contract=str(recipe_payload.get("contract") or ""),
    )
    if recipe_payload.get("config_hash") != recipe.config_hash:
        raise ValueError("A0_seed1 recipe config hash mismatch")
    if recipe.audit_hash != audit_payload.get("audit_hash"):
        raise ValueError("A0_seed1 recipe does not bind the supplied audit")
    if recipe.a0_config_hash != audit_payload.get("a0_config_hash"):
        raise ValueError("A0_seed1 recipe does not bind the audited A0 config")
    if recipe.config_hash != audit_payload.get("candidate_config_hash"):
        raise ValueError("A0_seed1 recipe config differs from the audited candidate config")
    return recipe, LocalResidualFieldTaggerTrainConfig(**dict(recipe.config))


__all__ = [
    "LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_A0_SEED_RECIPE_AUDIT_CONTRACT",
    "A0_TRAINING_SEED",
    "A0_SEED1_TRAINING_SEED",
    "A0_SEED1_ALLOWED_CONFIG_DIFFERENCES",
    "A0Seed1Recipe",
    "sha256_file",
    "extract_a0_train_config",
    "build_a0_seed1_train_config",
    "audit_a0_seed1_recipe",
    "require_a0_seed1_recipe_audit",
    "build_a0_seed1_recipe",
    "load_a0_seed1_recipe",
]
