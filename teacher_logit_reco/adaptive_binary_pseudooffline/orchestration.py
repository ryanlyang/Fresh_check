"""Step-12 Slurm graph construction and immutable campaign approvals."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    ABPH_HIGHDATA_SPLIT_SIZES,
    ABPH_PILOT_SPLIT_SIZES,
    canonical_hash,
)
from .accounting_preflight import ABPH_STEP4_PREFLIGHT_CONTRACT
from .report import ABPH_CAMPAIGN_REPORT_CONTRACT
from .variants import ABPH_EXPECTED_VARIANT_NAMES, resolve_variant_config, variant_spec


ABPH_SLURM_ORCHESTRATION_CONTRACT = "adaptive_binary_pseudooffline_slurm_orchestration_v1"
ABPH_FINAL_CLAIM_CONTRACT = "adaptive_binary_pseudooffline_final_claim_contract_v1"
ABPH_STAGE_MODES: tuple[str, ...] = (
    "full",
    "predictions",
    "fusion",
    "diagnostics",
    "report",
    "final_claims",
)
ABPH_CLUSTER_PROFILES: tuple[str, ...] = ("tigris", "tier3")
ABPH_RECONSTRUCTOR_VARIANTS: tuple[str, ...] = tuple(
    name for name in ABPH_EXPECTED_VARIANT_NAMES if variant_spec(name).tier in {"B", "C"}
)
ABPH_RENDERER_VARIANTS: tuple[str, ...] = tuple(
    name for name in ABPH_EXPECTED_VARIANT_NAMES if variant_spec(name).tier == "D"
)
ABPH_NEURAL_TAGGER_VARIANTS: tuple[str, ...] = tuple(
    name
    for name in ABPH_EXPECTED_VARIANT_NAMES
    if variant_spec(name).tier in {"E", "F"} or variant_spec(name).run_id in {"G0", "G1"}
)
ABPH_BASELINE_VARIANTS: tuple[str, ...] = tuple(
    name for name in ABPH_EXPECTED_VARIANT_NAMES if variant_spec(name).tier == "A"
)
ABPH_POSTHOC_VARIANTS: tuple[str, ...] = tuple(
    name for name in ABPH_EXPECTED_VARIANT_NAMES if variant_spec(name).run_id in {"G2", "G3", "G4", "G5"}
)
ABPH_DEPLOYABLE_PSEUDO_SOURCES: tuple[str, ...] = (
    "D1_kt32_mh4_particles",
    "D2_ca32_mh4_particles",
)
ABPH_DIAGNOSTIC_VARIANTS: tuple[str, ...] = (
    "E5_kt32_mh4_dualcross",
    "E7_dual_hierarchy_dualcross",
)
ABPH_FUSION_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "G2_kt_ca_logit_fusion": (
        "E5_kt32_mh4_dualcross",
        "E6_ca32_mh4_dualcross",
    ),
    "G3_particle_and_logit_fusion": (
        "E7_dual_hierarchy_dualcross",
        "A0_hlt_part",
        "E5_kt32_mh4_dualcross",
        "E6_ca32_mh4_dualcross",
        "F0_ce_reco_primary",
        "F0_ce_reco_primary__seed2",
        "F0_ce_reco_primary__seed3",
    ),
    "G4_seed_ensemble_primary": (
        "F0_ce_reco_primary",
        "F0_ce_reco_primary__seed2",
        "F0_ce_reco_primary__seed3",
    ),
    "G5_best_complementary_ensemble": (
        "E5_kt32_mh4_dualcross",
        "E6_ca32_mh4_dualcross",
        "E7_dual_hierarchy_dualcross",
        "F0_ce_reco_primary",
        "F0_ce_reco_primary__seed2",
        "F0_ce_reco_primary__seed3",
        "F3_ce_reco_branch_aux",
        "F4_ce_logit_kd",
        "F5_ce_reco_logit_kd",
    ),
}
ABPH_LOGIT_PREDICTION_MEMBERS: tuple[str, ...] = tuple(
    dict.fromkeys(member for members in ABPH_FUSION_CANDIDATES.values() for member in members)
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must contain a JSON mapping")
    return payload


def _require_successful_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = _read_json(path)
    if payload.get("ok") is not True:
        raise ValueError(f"{label} has ok!=true: {path}")
    return payload


def require_successful_selection_report(path: str | Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_CAMPAIGN_REPORT_CONTRACT:
        raise ValueError("selection report contract mismatch")
    if payload.get("ok") is not True:
        raise ValueError("selection report is not successful")
    if payload.get("final_test_policy", {}).get("confirmed") is not False:
        raise ValueError("selection report must precede final-test evaluation")
    saved_hash = payload.get("report_content_hash")
    if not saved_hash:
        raise ValueError("selection report lacks its content hash")
    hash_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"outputs", "report_content_hash"}
    }
    if saved_hash != canonical_hash(hash_payload):
        raise ValueError("selection report content hash mismatch")
    return payload


def require_actual_target_preflight(path: str | Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_STEP4_PREFLIGHT_CONTRACT:
        raise ValueError("actual-target feasibility preflight contract mismatch")
    if payload.get("ok") is not True or payload.get("problems"):
        raise ValueError("actual-target feasibility preflight is not successful")
    reports = payload.get("reports")
    if not isinstance(reports, Mapping) or not reports:
        raise ValueError("actual-target feasibility preflight contains no real-target reports")
    for name, row in reports.items():
        if not isinstance(row, Mapping) or row.get("ok") is not True:
            raise ValueError(f"preflight subgroup {name} failed")
        if int(row.get("compiler_failure_count", -1)) != 0:
            raise ValueError(f"preflight subgroup {name} has compiler failures")
        counts = row.get("class_counts")
        if not isinstance(counts, Mapping) or not counts or any(int(value) <= 0 for value in counts.values()):
            raise ValueError(f"preflight subgroup {name} is not class-stratified")
    synthetic = payload.get("synthetic_edge_cases")
    if not isinstance(synthetic, Mapping) or synthetic.get("ok") is not True:
        raise ValueError("preflight synthetic edge-case matrix failed")
    return payload


def freeze_final_claim_contract(
    selection_report_path: str | Path,
    output_path: str | Path,
    *,
    claim_variants: Sequence[str],
    fusion_artifact_hashes: Mapping[str, str],
    fusion_memberships: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    report_path = Path(selection_report_path)
    report = require_successful_selection_report(report_path)
    variants = tuple(str(name) for name in claim_variants)
    unknown = sorted(set(variants) - set(ABPH_EXPECTED_VARIANT_NAMES))
    if unknown or not variants or len(set(variants)) != len(variants):
        raise ValueError(f"invalid final claim membership: unknown={unknown}")
    for name in variants:
        if not bool(resolve_variant_config(name)["evaluation"].get("final_test_eligible")):
            raise ValueError(f"{name} is not final-test eligible")
    fusion_hashes = {str(name): str(value) for name, value in fusion_artifact_hashes.items()}
    missing_fusion = [name for name in variants if name in ABPH_POSTHOC_VARIANTS and not fusion_hashes.get(name)]
    if missing_fusion:
        raise ValueError(f"final claim lacks frozen fusion artifacts {missing_fusion}")
    memberships = {
        str(name): tuple(str(member) for member in members)
        for name, members in dict(fusion_memberships or {}).items()
    }
    for name in variants:
        if name not in ABPH_POSTHOC_VARIANTS:
            continue
        members = memberships.get(name, ())
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(f"final claim fusion {name} lacks an immutable multi-member declaration")
    payload = {
        "contract": ABPH_FINAL_CLAIM_CONTRACT,
        "selection_report_path": str(report_path.resolve()),
        "selection_report_sha256": _sha256_file(report_path),
        "selection_report_content_hash": report["report_content_hash"],
        "claim_variants": list(variants),
        "fusion_artifact_hashes": fusion_hashes,
        "fusion_memberships": {name: list(members) for name, members in memberships.items()},
        "teacher_logits_allowed": False,
        "offline_inputs_allowed": False,
        "membership_overrides_allowed": False,
    }
    payload["claim_contract_hash"] = canonical_hash(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return payload


def load_final_claim_contract(
    path: str | Path,
    *,
    selection_report_path: str | Path,
) -> Mapping[str, Any]:
    payload = dict(_read_json(path))
    if payload.get("contract") != ABPH_FINAL_CLAIM_CONTRACT:
        raise ValueError("final claim contract mismatch")
    saved_hash = payload.pop("claim_contract_hash", None)
    if saved_hash != canonical_hash(payload):
        raise ValueError("final claim contract content hash mismatch")
    report_path = Path(selection_report_path)
    report = require_successful_selection_report(report_path)
    if payload.get("selection_report_sha256") != _sha256_file(report_path):
        raise ValueError("selection report bytes changed after claim freeze")
    if payload.get("selection_report_content_hash") != report.get("report_content_hash"):
        raise ValueError("selection report identity changed after claim freeze")
    if payload.get("teacher_logits_allowed") is not False or payload.get("offline_inputs_allowed") is not False:
        raise ValueError("final claim contract permits privileged inputs")
    if payload.get("membership_overrides_allowed") is not False:
        raise ValueError("final claim contract permits membership overrides")
    claims = tuple(str(name) for name in payload.get("claim_variants", ()))
    memberships = payload.get("fusion_memberships")
    if not isinstance(memberships, Mapping):
        raise ValueError("final claim contract lacks frozen fusion memberships")
    for name in claims:
        if name not in ABPH_POSTHOC_VARIANTS:
            continue
        members = tuple(str(member) for member in memberships.get(name, ()))
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(f"final claim fusion {name} has invalid frozen membership")
    payload["claim_contract_hash"] = saved_hash
    return payload


@dataclass(frozen=True)
class SlurmResourceProfile:
    account: str
    partition: str
    gpu_gres: str
    gpu_cpus: int
    gpu_memory: str
    cpu_cpus: int
    cpu_memory: str
    gpu_time: str = "3-00:00:00"
    cpu_time: str = "2-00:00:00"

    @classmethod
    def for_cluster(cls, cluster: str, *, account: str | None = None) -> "SlurmResourceProfile":
        name = str(cluster).lower()
        if name == "tigris":
            return cls(
                account=account or "reu-aisocial",
                partition="tigris",
                gpu_gres="gpu:gh200:1",
                gpu_cpus=16,
                gpu_memory="220G",
                cpu_cpus=16,
                cpu_memory="220G",
            )
        if name == "tier3":
            return cls(
                account=account or "",
                partition="tier3",
                gpu_gres="gpu:1",
                gpu_cpus=16,
                gpu_memory="300G",
                cpu_cpus=16,
                cpu_memory="300G",
            )
        raise ValueError(f"unknown ABPH cluster profile {cluster!r}")


@dataclass(frozen=True)
class AdaptiveBinaryCampaignPaths:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def manifest(self) -> Path:
        return self.inputs / "split_manifest" / "split_manifest.json.gz"

    @property
    def hlt_cache(self) -> Path:
        return self.inputs / "hlt_cache"

    @property
    def offline_cache(self) -> Path:
        return self.inputs / "offline_cache"

    @property
    def target_cache(self) -> Path:
        return self.root / "targets"

    @property
    def preflight_report(self) -> Path:
        return self.root / "audits" / "actual_target_feasibility.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def pseudo_predictions(self) -> Path:
        return self.root / "pseudo_predictions"

    @property
    def logits(self) -> Path:
        return self.root / "logit_predictions"

    @property
    def fusion(self) -> Path:
        return self.root / "fusion"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"

    @property
    def report(self) -> Path:
        return self.root / "report"


@dataclass(frozen=True)
class SlurmJobSpec:
    key: str
    stage: str
    script: str
    arguments: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    gpu: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.stage or not self.script:
            raise ValueError("Slurm jobs require key, stage, and script")
        if self.key in self.dependencies:
            raise ValueError(f"job {self.key} depends on itself")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "script": self.script,
            "arguments": list(self.arguments),
            "dependencies": list(self.dependencies),
            "gpu": self.gpu,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class AdaptiveBinarySubmissionConfig:
    campaign_root: str | Path
    data_dir: str | Path
    campaign_mode: str = "pilot"
    stage_mode: str = "full"
    cluster: str = "tigris"
    account: str | None = None
    approve_highdata: bool = False
    pilot_report_path: str | Path | None = None
    approve_final_test: bool = False
    selection_report_path: str | Path | None = None
    final_claim_contract_path: str | Path | None = None
    confirm_final_test: bool = False
    rebuild_inputs: bool = True
    rebuild_targets: bool = True
    rebuild_models: bool = True
    rebuild_predictions: bool = True

    def __post_init__(self) -> None:
        if self.campaign_mode not in {"pilot", "highdata"}:
            raise ValueError("campaign_mode must be pilot or highdata")
        if self.stage_mode not in ABPH_STAGE_MODES:
            raise ValueError(f"unknown stage mode {self.stage_mode!r}")
        if self.cluster not in ABPH_CLUSTER_PROFILES:
            raise ValueError(f"unknown cluster {self.cluster!r}")
        if self.campaign_mode == "highdata" and self.stage_mode != "final_claims":
            if not self.approve_highdata:
                raise PermissionError("high-data submission requires canonical approval")
            if self.pilot_report_path is None:
                raise PermissionError("high-data submission requires a successful pilot report")
            require_successful_selection_report(self.pilot_report_path)
        if self.stage_mode == "final_claims":
            if self.campaign_mode != "highdata":
                raise ValueError("final claims are permitted only for high-data campaigns")
            if not self.approve_final_test or not self.confirm_final_test:
                raise PermissionError("final claims require canonical final-test approval")
            if self.selection_report_path is None or self.final_claim_contract_path is None:
                raise PermissionError("final claims require selection report and frozen claim contract")
            selection = require_successful_selection_report(self.selection_report_path)
            selected_root = Path(str(selection.get("campaign_root", ""))).resolve()
            if selected_root != Path(self.campaign_root).resolve():
                raise ValueError("final-claim selection report belongs to a different campaign root")
            load_final_claim_contract(
                self.final_claim_contract_path,
                selection_report_path=self.selection_report_path,
            )
        elif self.confirm_final_test:
            raise PermissionError("only the final_claims stage may confirm final_test")
        if self.stage_mode in {"predictions", "fusion", "diagnostics", "report"}:
            if self.rebuild_inputs or self.rebuild_targets or self.rebuild_models:
                raise ValueError(
                    f"partial stage {self.stage_mode} cannot rebuild upstream inputs/targets/models"
                )

    @property
    def paths(self) -> AdaptiveBinaryCampaignPaths:
        return AdaptiveBinaryCampaignPaths(Path(self.campaign_root))

    @property
    def split_sizes(self) -> Mapping[str, int]:
        return ABPH_PILOT_SPLIT_SIZES if self.campaign_mode == "pilot" else ABPH_HIGHDATA_SPLIT_SIZES


def _run_dir(paths: AdaptiveBinaryCampaignPaths, member: str) -> Path:
    return paths.runs / member


def _fusion_member_names(name: str) -> tuple[str, ...]:
    return ABPH_FUSION_CANDIDATES[name]


def require_partial_stage_inputs(config: AdaptiveBinarySubmissionConfig) -> dict[str, Any]:
    """Fail closed before submitting a partial graph against reused artifacts."""

    if config.stage_mode == "full":
        return {"stage_mode": "full", "checked": []}
    paths = config.paths
    checked: list[str] = []

    def require_run(member: str) -> None:
        run_dir = _run_dir(paths, member)
        _require_successful_json(run_dir / "run_report.json", label=f"{member} run report")
        checkpoint = run_dir / "best_model_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{member} checkpoint is missing: {checkpoint}")
        checked.extend((str(run_dir / "run_report.json"), str(checkpoint)))

    if config.stage_mode in {"predictions", "diagnostics", "final_claims"}:
        _require_successful_json(
            paths.root / "audits" / "step1_input_audit.json",
            label="Step-1 input audit",
        )
        checked.append(str(paths.root / "audits" / "step1_input_audit.json"))
    if config.stage_mode == "predictions":
        require_actual_target_preflight(paths.preflight_report)
        checked.append(str(paths.preflight_report))
        for member in ABPH_DEPLOYABLE_PSEUDO_SOURCES:
            require_run(member)
    elif config.stage_mode == "fusion":
        for name in ABPH_POSTHOC_VARIANTS:
            for member in _fusion_member_names(name):
                for split in ("stack_train", "stack_val"):
                    prediction = paths.logits / member / f"{split}.npz"
                    metadata = paths.logits / member / f"{split}_metadata.json"
                    if not prediction.is_file() or not metadata.is_file():
                        raise FileNotFoundError(
                            f"fusion input {member}/{split} is incomplete under {paths.logits}"
                        )
                    checked.extend((str(prediction), str(metadata)))
    elif config.stage_mode == "diagnostics":
        for member in ABPH_DIAGNOSTIC_VARIANTS:
            require_run(member)
    elif config.stage_mode == "report":
        for member in (
            *ABPH_BASELINE_VARIANTS,
            *ABPH_RECONSTRUCTOR_VARIANTS,
            *ABPH_RENDERER_VARIANTS,
            *ABPH_NEURAL_TAGGER_VARIANTS,
        ):
            _require_successful_json(
                _run_dir(paths, member) / "run_report.json",
                label=f"{member} run report",
            )
            checked.append(str(_run_dir(paths, member) / "run_report.json"))
        for name in ABPH_POSTHOC_VARIANTS:
            artifact = paths.fusion / name / "frozen_fusion.json"
            report = _run_dir(paths, name) / "run_report.json"
            if not artifact.is_file():
                raise FileNotFoundError(f"{name} frozen fusion is missing: {artifact}")
            _require_successful_json(report, label=f"{name} run report")
            checked.extend((str(artifact), str(report)))
    elif config.stage_mode == "final_claims":
        contract = load_final_claim_contract(
            config.final_claim_contract_path,
            selection_report_path=config.selection_report_path,
        )
        memberships = dict(contract["fusion_memberships"])
        members = tuple(
            dict.fromkeys(
                (
                    *(name for name in contract["claim_variants"] if name not in ABPH_POSTHOC_VARIANTS),
                    *(member for name in contract["claim_variants"] for member in memberships.get(name, ())),
                )
            )
        )
        for member in members:
            require_run(str(member))
        for name in contract["claim_variants"]:
            if name in ABPH_POSTHOC_VARIANTS:
                artifact = paths.fusion / name / "frozen_fusion.json"
                if not artifact.is_file():
                    raise FileNotFoundError(f"{name} frozen fusion is missing: {artifact}")
                if _sha256_file(artifact) != contract["fusion_artifact_hashes"].get(name):
                    raise ValueError(f"{name} frozen fusion hash differs from claim contract")
                checked.append(str(artifact))
    return {"stage_mode": config.stage_mode, "checked": checked}


def _variant_job_key(name: str) -> str:
    return f"variant:{name}"


def _seed_member_name(name: str, seed_index: int) -> str:
    return f"{name}__seed{int(seed_index)}"


def _dependency_job_key(name: str) -> str:
    return f"baseline:{name}" if variant_spec(name).tier == "A" else _variant_job_key(name)


def _variant_dependencies(name: str) -> tuple[str, ...]:
    dependencies = tuple(_dependency_job_key(dep) for dep in variant_spec(name).dependencies)
    tier = variant_spec(name).tier
    if tier in {"B", "C", "D"}:
        dependencies = ("preflight:actual_targets", *dependencies)
    if tier in {"E", "F"} or variant_spec(name).run_id in {"G0", "G1"}:
        sources: list[str] = ["baseline:A0_hlt_part"]
        if name in {"E6_ca32_mh4_dualcross"}:
            sources.append("prediction:D2_ca32_mh4_particles")
        elif name in {"E7_dual_hierarchy_dualcross", "E11_independent_root_dual_hierarchy_diagnostic", "G1_kt_ca_early_fusion"}:
            sources.extend(("prediction:D1_kt32_mh4_particles", "prediction:D2_ca32_mh4_particles"))
            if name != "E11_independent_root_dual_hierarchy_diagnostic":
                sources.append("prediction:E7_shared_root_dual")
        else:
            sources.append("prediction:D1_kt32_mh4_particles")
        dependencies = tuple(dict.fromkeys((*dependencies, *sources)))
    if bool(resolve_variant_config(name)["data"].get("requires_teacher_logits")):
        dependencies = tuple(dict.fromkeys((*dependencies, "teacher_logits:A4_offline_part_ceiling")))
    return dependencies


def _topological_validate(jobs: Sequence[SlurmJobSpec]) -> None:
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("ABPH graph has duplicate job keys")
    known: set[str] = set()
    all_keys = set(keys)
    for job in jobs:
        missing = [dep for dep in job.dependencies if dep not in all_keys]
        if missing:
            raise ValueError(f"job {job.key} has missing dependencies {missing}")
        unresolved = [dep for dep in job.dependencies if dep not in known]
        if unresolved:
            raise ValueError(f"job {job.key} is not topologically ordered: {unresolved}")
        known.add(job.key)


def build_submission_graph(config: AdaptiveBinarySubmissionConfig) -> tuple[SlurmJobSpec, ...]:
    """Build the canonical, topologically ordered campaign graph."""

    paths = config.paths
    common_env = {
        "ABPH_ROOT": str(paths.root),
        "ABPH_CAMPAIGN_MODE": config.campaign_mode,
        "ABPH_DATA_DIR": str(config.data_dir),
        "ABPH_CONFIRM_FINAL_TEST": "1" if config.confirm_final_test else "0",
    }
    if config.selection_report_path is not None:
        common_env["ABPH_SELECTION_REPORT_PATH"] = str(config.selection_report_path)
    jobs: list[SlurmJobSpec] = []

    if config.stage_mode == "full":
        jobs.append(SlurmJobSpec("input:splits", "splits", "run_adaptive_binary_inputs.sh", ("splits",), environment=common_env))
        jobs.append(SlurmJobSpec("input:hlt_cache", "hlt_cache", "run_adaptive_binary_inputs.sh", ("hlt_cache",), ("input:splits",), environment=common_env))
        jobs.append(SlurmJobSpec("input:offline_cache", "offline_cache", "run_adaptive_binary_inputs.sh", ("offline_cache",), ("input:splits",), environment=common_env))
        jobs.append(SlurmJobSpec("input:audit", "input_audit", "run_adaptive_binary_inputs.sh", ("audit",), ("input:hlt_cache", "input:offline_cache"), environment=common_env))
        for name in ABPH_BASELINE_VARIANTS:
            baseline_dependencies = tuple(
                _dependency_job_key(dep) for dep in variant_spec(name).dependencies
            ) or ("input:audit",)
            jobs.append(
                SlurmJobSpec(
                    f"baseline:{name}",
                    "baseline",
                    "run_adaptive_binary_variant.sh",
                    (name,),
                    baseline_dependencies,
                    gpu=True,
                    environment=common_env,
                )
            )
        jobs.append(
            SlurmJobSpec(
                "teacher_logits:A4_offline_part_ceiling",
                "teacher_logits",
                "run_adaptive_binary_prediction.sh",
                ("A4_offline_part_ceiling", "model_train", "model_val", "--teacher-logits"),
                ("baseline:A4_offline_part_ceiling", "input:offline_cache"),
                gpu=True,
                environment=common_env,
            )
        )
        jobs.append(SlurmJobSpec("target:cache", "targets", "run_adaptive_binary_targets.sh", ("cache",), ("input:audit",), environment=common_env))
        jobs.append(SlurmJobSpec("preflight:actual_targets", "preflight", "run_adaptive_binary_targets.sh", ("preflight",), ("target:cache",), environment=common_env))
        for name in (*ABPH_RECONSTRUCTOR_VARIANTS, *ABPH_RENDERER_VARIANTS):
            jobs.append(
                SlurmJobSpec(
                    _variant_job_key(name),
                    "renderer" if variant_spec(name).tier == "D" else "reconstructor",
                    "run_adaptive_binary_variant.sh",
                    (name,),
                    _variant_dependencies(name),
                    gpu=True,
                    environment=common_env,
                )
            )
        for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES:
            jobs.append(
                SlurmJobSpec(
                    f"prediction:{name}",
                    "pseudo_prediction",
                    "run_adaptive_binary_prediction.sh",
                    (name, "model_train", "model_val", "stack_train", "stack_val"),
                    (_variant_job_key(name), "input:hlt_cache"),
                    gpu=True,
                    environment=common_env,
                )
            )
        jobs.append(
            SlurmJobSpec(
                "prediction:E7_shared_root_dual",
                "pseudo_prediction",
                "run_adaptive_binary_prediction.sh",
                (
                    "E7_shared_root_dual",
                    "model_train",
                    "model_val",
                    "stack_train",
                    "stack_val",
                ),
                tuple(f"prediction:{name}" for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES),
                gpu=True,
                environment=common_env,
            )
        )
        for name in ABPH_NEURAL_TAGGER_VARIANTS:
            jobs.append(
                SlurmJobSpec(
                    _variant_job_key(name),
                    "tagger",
                    "run_adaptive_binary_variant.sh",
                    (name,),
                    _variant_dependencies(name),
                    gpu=True,
                    environment=common_env,
                )
            )
        primary_seed = "F0_ce_reco_primary"
        for seed_index in (2, 3):
            jobs.append(
                SlurmJobSpec(
                    f"variant:{_seed_member_name(primary_seed, seed_index)}",
                    "tagger_seed",
                    "run_adaptive_binary_variant.sh",
                    (primary_seed, str(seed_index)),
                    _variant_dependencies(primary_seed),
                    gpu=True,
                    environment=common_env,
                )
            )
        for member in ABPH_LOGIT_PREDICTION_MEMBERS:
            model_key = (
                f"variant:{member}"
                if "__seed" in member
                else _dependency_job_key(member)
            )
            jobs.append(
                SlurmJobSpec(
                    f"logit_prediction:{member}",
                    "logit_prediction",
                    "run_adaptive_binary_prediction.sh",
                    (member, "model_val", "stack_train", "stack_val"),
                    (model_key, "input:hlt_cache"),
                    gpu=True,
                    environment=common_env,
                )
            )
        for name in ABPH_POSTHOC_VARIANTS:
            member_names = _fusion_member_names(name)
            members = tuple(f"logit_prediction:{member}" for member in member_names)
            jobs.append(
                SlurmJobSpec(
                    _variant_job_key(name),
                    "fusion",
                    "run_adaptive_binary_fusion.sh",
                    (name, *member_names),
                    members,
                    environment=common_env,
                )
            )
        jobs.append(
            SlurmJobSpec(
                "diagnostic:tagger_use",
                "diagnostics",
                "run_adaptive_binary_diagnostics.sh",
                ABPH_DIAGNOSTIC_VARIANTS,
                tuple(_variant_job_key(name) for name in ABPH_DIAGNOSTIC_VARIANTS),
                gpu=True,
                environment=common_env,
            )
        )
        terminal = tuple(
            dict.fromkeys(
                (
                    *(f"baseline:{name}" for name in ABPH_BASELINE_VARIANTS),
                    *(_variant_job_key(name) for name in ABPH_RECONSTRUCTOR_VARIANTS),
                    *(_variant_job_key(name) for name in ABPH_RENDERER_VARIANTS),
                    *(f"prediction:{name}" for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES),
                    "prediction:E7_shared_root_dual",
                    *(_variant_job_key(name) for name in ABPH_NEURAL_TAGGER_VARIANTS),
                    f"variant:{_seed_member_name(primary_seed, 2)}",
                    f"variant:{_seed_member_name(primary_seed, 3)}",
                    *(f"logit_prediction:{member}" for member in ABPH_LOGIT_PREDICTION_MEMBERS),
                    *(_variant_job_key(name) for name in ABPH_POSTHOC_VARIANTS),
                    "diagnostic:tagger_use",
                )
            )
        )
        jobs.append(SlurmJobSpec("report:model_selection", "report", "run_adaptive_binary_report.sh", ("selection",), terminal, environment=common_env))
    elif config.stage_mode == "predictions":
        for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES:
            jobs.append(SlurmJobSpec(f"prediction:{name}", "pseudo_prediction", "run_adaptive_binary_prediction.sh", (name, "model_train", "model_val", "stack_train", "stack_val"), gpu=True, environment=common_env))
        jobs.append(SlurmJobSpec("prediction:E7_shared_root_dual", "pseudo_prediction", "run_adaptive_binary_prediction.sh", ("E7_shared_root_dual", "model_train", "model_val", "stack_train", "stack_val"), gpu=True, environment=common_env))
    elif config.stage_mode == "fusion":
        for name in ABPH_POSTHOC_VARIANTS:
            members = _fusion_member_names(name)
            jobs.append(SlurmJobSpec(_variant_job_key(name), "fusion", "run_adaptive_binary_fusion.sh", (name, *members), environment=common_env))
    elif config.stage_mode == "diagnostics":
        jobs.append(SlurmJobSpec("diagnostic:tagger_use", "diagnostics", "run_adaptive_binary_diagnostics.sh", ABPH_DIAGNOSTIC_VARIANTS, gpu=True, environment=common_env))
    elif config.stage_mode == "report":
        jobs.append(SlurmJobSpec("report:model_selection", "report", "run_adaptive_binary_report.sh", ("selection",), environment=common_env))
    elif config.stage_mode == "final_claims":
        contract = load_final_claim_contract(
            config.final_claim_contract_path,
            selection_report_path=config.selection_report_path,
        )
        claims = tuple(str(name) for name in contract["claim_variants"])
        memberships = {
            str(name): tuple(str(member) for member in members)
            for name, members in dict(contract["fusion_memberships"]).items()
        }
        direct_claims = tuple(name for name in claims if name not in ABPH_POSTHOC_VARIANTS)
        prediction_members = tuple(
            dict.fromkeys(
                (
                    *direct_claims,
                    *(member for name in claims for member in memberships.get(name, ())),
                )
            )
        )
        for name in prediction_members:
            jobs.append(
                SlurmJobSpec(
                    f"final_prediction:{name}",
                    "final_prediction",
                    "run_adaptive_binary_prediction.sh",
                    (name, "final_test"),
                    gpu=True,
                    environment={**common_env, "ABPH_FINAL_CLAIM_CONTRACT": str(config.final_claim_contract_path)},
                )
            )
        fusion_claims = tuple(name for name in claims if name in ABPH_POSTHOC_VARIANTS)
        for name in fusion_claims:
            members = memberships[name]
            jobs.append(
                SlurmJobSpec(
                    f"final_fusion:{name}",
                    "final_fusion",
                    "run_adaptive_binary_fusion.sh",
                    (name, *members, "--apply-split", "final_test"),
                    tuple(f"final_prediction:{member}" for member in members),
                    environment={**common_env, "ABPH_FINAL_CLAIM_CONTRACT": str(config.final_claim_contract_path)},
                )
            )
        report_dependencies = tuple(
            f"final_fusion:{name}" if name in fusion_claims else f"final_prediction:{name}"
            for name in claims
        )
        jobs.append(
            SlurmJobSpec(
                "report:final_claim",
                "final_report",
                "run_adaptive_binary_report.sh",
                ("final_claim",),
                report_dependencies,
                environment={**common_env, "ABPH_FINAL_CLAIM_CONTRACT": str(config.final_claim_contract_path)},
            )
        )
    _topological_validate(jobs)
    if config.stage_mode == "full":
        preflight_index = next(index for index, job in enumerate(jobs) if job.key == "preflight:actual_targets")
        for index, job in enumerate(jobs):
            if job.stage in {"reconstructor", "renderer"}:
                if index <= preflight_index or "preflight:actual_targets" not in job.dependencies:
                    raise RuntimeError("reconstructor/renderer escaped the actual-target hard gate")
    return tuple(jobs)


def submission_manifest(
    config: AdaptiveBinarySubmissionConfig,
    jobs: Sequence[SlurmJobSpec],
) -> dict[str, Any]:
    payload = {
        "contract": ABPH_SLURM_ORCHESTRATION_CONTRACT,
        "campaign_root": str(config.paths.root),
        "campaign_mode": config.campaign_mode,
        "stage_mode": config.stage_mode,
        "cluster": config.cluster,
        "split_sizes": dict(config.split_sizes),
        "confirm_final_test": config.confirm_final_test,
        "jobs": [job.to_dict() for job in jobs],
    }
    payload["graph_hash"] = canonical_hash(payload)
    return payload


__all__ = [
    "ABPH_BASELINE_VARIANTS",
    "ABPH_CLUSTER_PROFILES",
    "ABPH_DEPLOYABLE_PSEUDO_SOURCES",
    "ABPH_DIAGNOSTIC_VARIANTS",
    "ABPH_FINAL_CLAIM_CONTRACT",
    "ABPH_FUSION_CANDIDATES",
    "ABPH_LOGIT_PREDICTION_MEMBERS",
    "ABPH_NEURAL_TAGGER_VARIANTS",
    "ABPH_POSTHOC_VARIANTS",
    "ABPH_RECONSTRUCTOR_VARIANTS",
    "ABPH_RENDERER_VARIANTS",
    "ABPH_SLURM_ORCHESTRATION_CONTRACT",
    "ABPH_STAGE_MODES",
    "AdaptiveBinaryCampaignPaths",
    "AdaptiveBinarySubmissionConfig",
    "SlurmJobSpec",
    "SlurmResourceProfile",
    "build_submission_graph",
    "freeze_final_claim_contract",
    "load_final_claim_contract",
    "require_successful_selection_report",
    "require_actual_target_preflight",
    "require_partial_stage_inputs",
    "submission_manifest",
]
