"""Step-12 Slurm graph construction and immutable campaign approvals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .cache import load_adaptive_binary_target_cache_metadata
from .convergence_schedule import ABPH_ACCELERATED_SCHEDULE_CONTRACT
from .report import ABPH_CAMPAIGN_REPORT_CONTRACT
from .runtime_acceptance import require_runtime_acceptance
from .tagger_distributed import require_tagger_ddp_acceptance
from .bundled_scoring import group_scoring_members, scoring_source_family
from .storage_quota import (
    ABPH_CACHE_HEAVY_STORAGE_PROFILE,
    ABPH_STORAGE_PROFILE_NAMES,
    require_storage_projection,
    resolve_storage_profile,
)
from .storage_lifecycle import (
    cleanup_receipt_path,
    require_artifact_manifest,
    require_cleanup_receipt,
    verify_bundled_logits,
)
from .target_mode import (
    ABPH_RANK_LOCAL_TARGET_MODE,
    load_target_mode_selection,
    rank_local_target_metadata,
)
from .variants import ABPH_EXPECTED_VARIANT_NAMES, resolve_variant_config, variant_spec


ABPH_SLURM_ORCHESTRATION_CONTRACT = "adaptive_binary_pseudooffline_slurm_orchestration_v1"
ABPH_RECONSTRUCTOR_PARALLELISM_CONTRACT = (
    "adaptive_binary_reconstructor_parallelism_v1"
)
ABPH_RECONSTRUCTOR_PARALLELISM_MODES: tuple[str, ...] = ("single", "ddp4")
ABPH_FINAL_CLAIM_CONTRACT = "adaptive_binary_pseudooffline_final_claim_contract_v1"
ABPH_STAGE_MODES: tuple[str, ...] = (
    "full",
    "models",
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
ABPH_BUNDLED_SCORING_FAMILIES: Mapping[str, tuple[str, ...]] = (
    group_scoring_members(ABPH_LOGIT_PREDICTION_MEMBERS)
)
ABPH_JOINT_TARGET_TAGGER_VARIANTS: tuple[str, ...] = tuple(
    name
    for name in ABPH_NEURAL_TAGGER_VARIANTS
    if variant_spec(name).tier == "F" and variant_spec(name).run_id not in {"F1", "F4"}
)
ABPH_JOINT_TARGET_TAGGER_MEMBERS: tuple[str, ...] = (
    *ABPH_JOINT_TARGET_TAGGER_VARIANTS,
    "F0_ce_reco_primary__seed2",
    "F0_ce_reco_primary__seed3",
)
def _neural_prerequisite_closure(names: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"neural prerequisite cycle contains {name}")
        visiting.add(name)
        for dependency in variant_spec(name).dependencies:
            if dependency in ABPH_NEURAL_TAGGER_VARIANTS:
                visit(dependency)
                if dependency not in ordered and dependency not in names:
                    ordered.append(dependency)
        visiting.remove(name)

    for name in names:
        visit(name)
    return tuple(ordered)


ABPH_PRIVILEGED_TAGGER_PREREQUISITES: tuple[str, ...] = (
    _neural_prerequisite_closure(ABPH_JOINT_TARGET_TAGGER_VARIANTS)
)
ABPH_PRIVILEGED_CONSUMERS: tuple[str, ...] = (
    *ABPH_RECONSTRUCTOR_VARIANTS,
    *ABPH_RENDERER_VARIANTS,
    *ABPH_JOINT_TARGET_TAGGER_MEMBERS,
)
ABPH_RUNTIME_BATCH_PROBE_SPECS: tuple[tuple[str, int], ...] = (
    ("root_hierarchy", 256),
    ("root_hierarchy", 128),
    ("root_hierarchy", 64),
    ("renderer_distribution", 128),
    ("renderer_distribution", 64),
    ("renderer_distribution", 32),
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
    screening = payload.get("schedule_screening")
    if not isinstance(screening, Mapping):
        raise ValueError("selection report lacks accelerated schedule screening")
    if screening.get("automatic_highdata_promotion_allowed") is not True:
        raise ValueError(
            "selection report blocks automatic high-data promotion because the "
            "accelerated schedule is incomplete or truncated"
        )
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
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    distributed_world_size: int = 1
    launcher: str = "direct"

    def __post_init__(self) -> None:
        for name in (
            "nodes",
            "ntasks",
            "ntasks_per_node",
            "gpus_per_node",
            "distributed_world_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"Slurm resource {name} must be positive")
        if self.ntasks != self.nodes * self.ntasks_per_node:
            raise ValueError("Slurm resource task topology is inconsistent")
        if self.distributed_world_size != self.ntasks:
            raise ValueError("Slurm resource world size must equal ntasks")
        if self.launcher not in {"direct", "srun"}:
            raise ValueError("Slurm resource launcher must be direct or srun")

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
    def target_mode_report(self) -> Path:
        return self.root / "audits" / "target_mode_selection.json"

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

    @property
    def storage(self) -> Path:
        return self.root / "storage"


@dataclass(frozen=True)
class SlurmJobSpec:
    key: str
    stage: str
    script: str
    arguments: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    gpu: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int | None = None
    distributed_world_size: int = 1
    launcher: str = "direct"

    def __post_init__(self) -> None:
        if not self.key or not self.stage or not self.script:
            raise ValueError("Slurm jobs require key, stage, and script")
        if self.key in self.dependencies:
            raise ValueError(f"job {self.key} depends on itself")
        for name in (
            "nodes",
            "ntasks",
            "ntasks_per_node",
            "distributed_world_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"job {self.key} has invalid {name}")
        if self.gpus_per_node is not None and int(self.gpus_per_node) < 0:
            raise ValueError(f"job {self.key} has invalid gpus_per_node")
        if self.ntasks != self.nodes * self.ntasks_per_node:
            raise ValueError(f"job {self.key} has inconsistent task topology")
        if self.distributed_world_size != self.ntasks:
            raise ValueError(f"job {self.key} world size differs from ntasks")
        if self.launcher not in {"direct", "srun"}:
            raise ValueError(f"job {self.key} has an invalid launcher")
        if self.launcher == "direct" and self.distributed_world_size != 1:
            raise ValueError(f"job {self.key} cannot launch multiple direct ranks")
        if not self.gpu and self.distributed_world_size != 1:
            raise ValueError(f"CPU job {self.key} cannot use reconstructor DDP")

    @property
    def resolved_gpus_per_node(self) -> int:
        if self.gpus_per_node is not None:
            return int(self.gpus_per_node)
        return 1 if self.gpu else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "script": self.script,
            "arguments": list(self.arguments),
            "dependencies": list(self.dependencies),
            "gpu": self.gpu,
            "environment": dict(self.environment),
            "nodes": int(self.nodes),
            "ntasks": int(self.ntasks),
            "ntasks_per_node": int(self.ntasks_per_node),
            "gpus_per_node": self.resolved_gpus_per_node,
            "distributed_world_size": int(self.distributed_world_size),
            "launcher": self.launcher,
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
    reconstructor_parallelism: str = "ddp4"
    runtime_acceptance_path: str | Path | None = None
    tagger_ddp_acceptance_path: str | Path | None = None
    allow_debug_single_reconstructor: bool = False
    storage_profile: str = ABPH_CACHE_HEAVY_STORAGE_PROFILE
    storage_projection_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.campaign_mode not in {"pilot", "highdata"}:
            raise ValueError("campaign_mode must be pilot or highdata")
        if self.stage_mode not in ABPH_STAGE_MODES:
            raise ValueError(f"unknown stage mode {self.stage_mode!r}")
        if self.cluster not in ABPH_CLUSTER_PROFILES:
            raise ValueError(f"unknown cluster {self.cluster!r}")
        if self.storage_profile not in ABPH_STORAGE_PROFILE_NAMES:
            raise ValueError(f"unknown storage profile {self.storage_profile!r}")
        if self.tagger_ddp_acceptance_path is not None:
            if self.storage_profile != "streaming_30gb_v1":
                raise ValueError(
                    "tagger DDP promotion is valid only for streaming_30gb_v1"
                )
            if self.cluster != "tigris":
                raise ValueError("tagger DDP4 is currently certified only for Tigris")
        storage_profile = resolve_storage_profile(self.storage_profile)
        if storage_profile.enforce_quota:
            if self.storage_projection_path is None:
                raise PermissionError(
                    "quota-enforced storage submission requires a measured storage projection"
                )
            require_storage_projection(
                self.storage_projection_path,
                campaign_root=self.campaign_root,
                campaign_mode=self.campaign_mode,
                profile=storage_profile,
            )
        elif self.storage_projection_path is not None:
            require_storage_projection(
                self.storage_projection_path,
                campaign_root=self.campaign_root,
                campaign_mode=self.campaign_mode,
                profile=storage_profile,
            )
        if self.reconstructor_parallelism not in ABPH_RECONSTRUCTOR_PARALLELISM_MODES:
            raise ValueError("reconstructor_parallelism must be single or ddp4")
        if self.reconstructor_parallelism == "ddp4" and self.cluster != "tigris":
            raise ValueError("ABPH ddp4 is currently certified only for Tigris")
        if (
            self.reconstructor_parallelism == "single"
            and self.stage_mode in {"full", "models"}
            and not self.allow_debug_single_reconstructor
        ):
            raise PermissionError(
                "single-GPU full/models submission is debug-only; pass the explicit "
                "debug override or use the gated ddp4 production profile"
            )
        if self.reconstructor_parallelism == "ddp4" and self.stage_mode in {
            "full",
            "models",
        }:
            if self.runtime_acceptance_path is None:
                raise PermissionError(
                    "ABPH ddp4 full campaigns require a Step-10 runtime acceptance artifact"
                )
            scope = "highdata" if self.campaign_mode == "highdata" else "optimized_pilot"
            require_runtime_acceptance(self.runtime_acceptance_path, scope=scope)
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
        if self.stage_mode == "models":
            if self.rebuild_inputs or self.rebuild_targets or not self.rebuild_models:
                raise ValueError(
                    "models stage must reuse inputs/targets and rebuild model outputs"
                )

    @property
    def paths(self) -> AdaptiveBinaryCampaignPaths:
        return AdaptiveBinaryCampaignPaths(Path(self.campaign_root))

    @property
    def split_sizes(self) -> Mapping[str, int]:
        return ABPH_PILOT_SPLIT_SIZES if self.campaign_mode == "pilot" else ABPH_HIGHDATA_SPLIT_SIZES

    @property
    def reconstructor_topology(self) -> Mapping[str, Any]:
        if self.reconstructor_parallelism == "ddp4":
            return {
                "contract": ABPH_RECONSTRUCTOR_PARALLELISM_CONTRACT,
                "mode": "ddp4",
                "nodes": 4,
                "ntasks": 4,
                "ntasks_per_node": 1,
                "gpus_per_node": 1,
                "distributed_world_size": 4,
                "launcher": "srun",
                "promotion_status": "requires_step10_runtime_acceptance",
            }
        return {
            "contract": ABPH_RECONSTRUCTOR_PARALLELISM_CONTRACT,
            "mode": "single",
            "nodes": 1,
            "ntasks": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "distributed_world_size": 1,
            "launcher": "direct",
            "promotion_status": "debug_compatibility",
        }

    @property
    def tagger_topology(self) -> Mapping[str, Any]:
        if self.storage_profile != "streaming_30gb_v1":
            return {
                "mode": "single",
                "nodes": 1,
                "ntasks": 1,
                "ntasks_per_node": 1,
                "gpus_per_node": 1,
                "distributed_world_size": 1,
                "launcher": "direct",
                "promotion_status": "persistent_cache_profile_not_promoted",
            }
        if self.tagger_ddp_acceptance_path is None:
            return {
                "mode": "single",
                "nodes": 1,
                "ntasks": 1,
                "ntasks_per_node": 1,
                "gpus_per_node": 1,
                "distributed_world_size": 1,
                "launcher": "direct",
                "promotion_status": "single_rank_ram_fallback_gate_absent",
            }
        require_tagger_ddp_acceptance(self.tagger_ddp_acceptance_path)
        return {
            "mode": "ddp4",
            "nodes": 4,
            "ntasks": 4,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "distributed_world_size": 4,
            "launcher": "srun",
            "promotion_status": "e7_f0_parity_and_speed_gate_passed",
        }


def _run_dir(paths: AdaptiveBinaryCampaignPaths, member: str) -> Path:
    return paths.runs / member


def _fusion_member_names(name: str) -> tuple[str, ...]:
    return ABPH_FUSION_CANDIDATES[name]


def _bundled_scoring_job_key_for_member(member: str) -> str:
    return f"logit_bundle:{scoring_source_family(member).key}"


def require_partial_stage_inputs(config: AdaptiveBinarySubmissionConfig) -> dict[str, Any]:
    """Fail closed before submitting a partial graph against reused artifacts."""

    if config.stage_mode == "full":
        return {"stage_mode": "full", "checked": []}
    paths = config.paths
    checked: list[str] = []
    streaming = config.storage_profile == "streaming_30gb_v1"
    if streaming:
        lifecycle_manifest = require_artifact_manifest(paths.root)
        checked.append(str(paths.storage / "artifact_manifest.json"))
        if config.stage_mode == "models" and cleanup_receipt_path(
            paths.root, "privileged"
        ).exists():
            raise PermissionError(
                "models reuse is impossible after privileged inputs were cleaned; "
                "start a fresh full campaign or explicitly rebuild its inputs"
            )
        if config.stage_mode == "diagnostics" and cleanup_receipt_path(
            paths.root, "deployable"
        ).exists():
            raise PermissionError(
                "tagger diagnostics cannot be rerun after deployable HLT cleanup"
            )

    def require_exact(actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")

    def require_run(
        member: str, *, expected_provenance: Mapping[str, Any] | None = None
    ) -> None:
        run_dir = _run_dir(paths, member)
        report = _require_successful_json(
            run_dir / "run_report.json", label=f"{member} run report"
        )
        checkpoint = run_dir / "best_model_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{member} checkpoint is missing: {checkpoint}")
        checkpoint_hash = _sha256_file(checkpoint)
        require_exact(
            report.get("selected_checkpoint_hash")
            or report.get("best_model_val_checkpoint_sha256"),
            checkpoint_hash,
            f"{member} selected checkpoint hash",
        )
        if expected_provenance is not None:
            provenance_rows = report.get("provenance")
            if not isinstance(provenance_rows, Mapping) or not isinstance(
                provenance_rows.get("model_val"), Mapping
            ):
                raise ValueError(f"{member} lacks model_val provenance")
            observed = provenance_rows["model_val"]
            for key, value in expected_provenance.items():
                require_exact(
                    observed.get(key), value, f"{member} model_val {key}"
                )
        checked.extend((str(run_dir / "run_report.json"), str(checkpoint)))

    if config.stage_mode in {"models", "predictions", "diagnostics", "final_claims"}:
        input_audit = _require_successful_json(
            paths.root / "audits" / "step1_input_audit.json",
            label="Step-1 input audit",
        )
        checked.append(str(paths.root / "audits" / "step1_input_audit.json"))
    if config.stage_mode == "models":
        from .production import reconstructor_runtime_provenance
        from .runtime_batch import load_runtime_batch_contract

        manifest_report = input_audit.get("manifest")
        hlt_reports = input_audit.get("hlt_splits")
        offline_reports = input_audit.get("offline_splits")
        if not all(
            isinstance(value, Mapping)
            for value in (manifest_report, hlt_reports, offline_reports)
        ):
            raise ValueError("Step-1 input audit lacks split provenance maps")
        manifest_sha = manifest_report.get("manifest_hash")
        current_hlt_metadata_by_split: dict[str, Mapping[str, Any]] = {}
        target_metadata_by_grouping: dict[str, Mapping[str, Any]] = {}
        require_actual_target_preflight(paths.preflight_report)
        checked.append(str(paths.preflight_report))
        target_mode_selection = None
        if config.storage_profile == "streaming_30gb_v1":
            target_mode_selection = load_target_mode_selection(
                paths.target_mode_report, campaign_root=paths.root
            )
            checked.append(str(paths.target_mode_report))
        for split in ("model_train", "model_val"):
            current_hlt_metadata = _read_json(
                paths.hlt_cache / f"{split}_fixed_hlt_metadata.json"
            )
            current_offline_metadata = _read_json(
                paths.offline_cache / f"{split}_offline_metadata.json"
            )
            current_hlt_metadata_by_split[split] = current_hlt_metadata
            hlt_expected = hlt_reports.get(split)
            offline_expected = offline_reports.get(split)
            if not isinstance(hlt_expected, Mapping) or not isinstance(
                offline_expected, Mapping
            ):
                raise ValueError(f"input audit lacks active {split} cache provenance")
            for key in (
                "source_manifest_hash",
                "hlt_content_hash",
                "jet_identity_hash",
            ):
                require_exact(
                    current_hlt_metadata.get(key),
                    hlt_expected.get(key),
                    f"current HLT {split} {key}",
                )
            for key in (
                "source_manifest_hash",
                "offline_content_hash",
                "jet_identity_hash",
            ):
                require_exact(
                    current_offline_metadata.get(key),
                    offline_expected.get(key),
                    f"current offline {split} {key}",
                )
            for grouping in ("exclusive_kt", "cambridge_aachen"):
                if (
                    target_mode_selection is not None
                    and target_mode_selection.get("selected_mode")
                    == ABPH_RANK_LOCAL_TARGET_MODE
                ):
                    metadata = rank_local_target_metadata(
                        selection=target_mode_selection,
                        split=split,
                        grouping=grouping,
                        n_jets=int(current_hlt_metadata["n_jets"]),
                        jet_identity_hash_value=str(hlt_expected.get("jet_identity_hash")),
                    )
                else:
                    metadata = load_adaptive_binary_target_cache_metadata(
                        paths.target_cache, split, grouping
                    )
                if split == "model_train":
                    target_metadata_by_grouping[grouping] = metadata
                if metadata.get("target_content_hash") in {None, ""}:
                    raise ValueError(f"reused target metadata lacks a hash: {split}/{grouping}")
                for key, expected in {
                    "source_manifest_hash": manifest_sha,
                    "hlt_content_hash": hlt_expected.get("hlt_content_hash"),
                    "offline_content_hash": offline_expected.get(
                        "offline_content_hash"
                    ),
                    "jet_identity_hash": hlt_expected.get("jet_identity_hash"),
                }.items():
                    require_exact(
                        metadata.get(key),
                        expected,
                        f"target {split}/{grouping} {key}",
                    )
                if metadata.get("target_mode") != ABPH_RANK_LOCAL_TARGET_MODE:
                    checked.append(
                        str(
                            paths.target_cache
                            / f"{split}_{grouping}_adaptive_binary_targets_metadata.json"
                        )
                    )
        hlt_model_val = hlt_reports.get("model_val")
        offline_model_val = offline_reports.get("model_val")
        if not isinstance(hlt_model_val, Mapping) or not isinstance(
            offline_model_val, Mapping
        ):
            raise ValueError("input audit lacks model_val baseline provenance")
        for member in ABPH_BASELINE_VARIANTS:
            if member == "A4_offline_part_ceiling":
                expected = {
                    "source_manifest_hash": manifest_sha,
                    "jet_identity_hash": offline_model_val.get("jet_identity_hash"),
                    "label_hash": offline_model_val.get("label_hash"),
                    "offline_content_hash": offline_model_val.get(
                        "offline_content_hash"
                    ),
                }
            else:
                expected = {
                    "source_manifest_hash": manifest_sha,
                    "jet_identity_hash": hlt_model_val.get("jet_identity_hash"),
                    "label_hash": hlt_model_val.get("label_hash"),
                    "hlt_content_hash": hlt_model_val.get("hlt_content_hash"),
                }
            require_run(member, expected_provenance=expected)
        teacher_root = paths.root / "teacher_logits" / "A4_offline_part_ceiling"
        teacher_checkpoint_hash = _sha256_file(
            _run_dir(paths, "A4_offline_part_ceiling") / "best_model_val.pt"
        )
        for split in ("model_train", "model_val"):
            prediction = teacher_root / f"{split}.npz"
            metadata = teacher_root / f"{split}_metadata.json"
            if not prediction.is_file():
                raise FileNotFoundError(f"reused teacher logits are missing: {prediction}")
            teacher_metadata = _require_successful_json(
                metadata, label=f"{split} teacher-logit metadata"
            )
            offline_expected = offline_reports.get(split)
            if not isinstance(offline_expected, Mapping):
                raise ValueError(f"input audit lacks offline {split} provenance")
            teacher_provenance = teacher_metadata.get("provenance")
            if not isinstance(teacher_provenance, Mapping):
                raise ValueError(f"{split} teacher logits lack provenance")
            for key, expected in {
                "source_manifest_hash": manifest_sha,
                "jet_identity_hash": offline_expected.get("jet_identity_hash"),
                "label_hash": offline_expected.get("label_hash"),
                "offline_content_hash": offline_expected.get(
                    "offline_content_hash"
                ),
            }.items():
                require_exact(
                    teacher_provenance.get(key),
                    expected,
                    f"{split} teacher-logit {key}",
                )
            require_exact(
                teacher_metadata.get("checkpoint_sha256"),
                teacher_checkpoint_hash,
                f"{split} teacher-logit checkpoint",
            )
            require_exact(
                teacher_metadata.get("prediction_sha256"),
                _sha256_file(prediction),
                f"{split} teacher-logit prediction",
            )
            checked.extend((str(prediction), str(metadata)))
        expected_world_size = 4 if config.reconstructor_parallelism == "ddp4" else 1
        for member in (*ABPH_RECONSTRUCTOR_VARIANTS, *ABPH_RENDERER_VARIANTS):
            resolved = resolve_variant_config(member)
            grouping = str(
                resolved["model"]["hierarchy"].get(
                    "grouping", "exclusive_kt"
                )
            )
            target_metadata = target_metadata_by_grouping.get(grouping)
            if target_metadata is None:
                raise ValueError(
                    f"active model_train target metadata is missing for {grouping}"
                )
            runtime_provenance = reconstructor_runtime_provenance(
                variant_name=member,
                target_metadata=target_metadata,
                hlt_metadata=current_hlt_metadata_by_split["model_train"],
            )
            contract_path = (
                paths.root
                / "runtime_batch_contracts"
                / member
                / "runtime_batch_contract.json"
            )
            load_runtime_batch_contract(
                contract_path,
                expected_variant_name=member,
                expected_resolved_variant_config_hash=resolved[
                    "resolved_config_hash"
                ],
                expected_runtime_provenance_hash=canonical_hash(
                    runtime_provenance
                ),
                expected_world_size=expected_world_size,
            )
            checked.append(str(contract_path))
    elif config.stage_mode == "predictions":
        require_actual_target_preflight(paths.preflight_report)
        checked.append(str(paths.preflight_report))
        for member in ABPH_DEPLOYABLE_PSEUDO_SOURCES:
            require_run(member)
    elif config.stage_mode == "fusion":
        if streaming:
            verified = verify_bundled_logits(
                paths.root, ABPH_LOGIT_PREDICTION_MEMBERS
            )
            checked.extend(
                str(paths.logits / row["member"] / f"{row['split']}.npz")
                for row in verified["verified"]
            )
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
        if streaming:
            for barrier in ("privileged", "deployable"):
                require_cleanup_receipt(paths.root, barrier)
                checked.append(str(cleanup_receipt_path(paths.root, barrier)))
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
        if streaming:
            final_hlt = paths.hlt_cache / "final_test_fixed_hlt.npz"
            retained = {
                row["relative_path"]: row
                for row in lifecycle_manifest["artifacts"]
                if row.get("role") == "hlt_final_test_payload"
            }
            relative = final_hlt.resolve().relative_to(paths.root.resolve()).as_posix()
            row = retained.get(relative)
            if row is None or not final_hlt.is_file():
                raise FileNotFoundError(
                    "streaming final claims require the manifest-retained final-test HLT"
                )
            if _sha256_file(final_hlt) != row.get("sha256"):
                raise ValueError("retained final-test HLT hash differs from lifecycle manifest")
            checked.append(str(final_hlt))
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


def _runtime_batch_probe_job_key(
    name: str, stage_family: str, local_batch_size: int
) -> str:
    return f"runtime_batch_probe:{name}:{stage_family}:b{int(local_batch_size)}"


def _runtime_batch_contract_job_key(name: str) -> str:
    return f"runtime_batch_contract:{name}"


def _runtime_batch_contract_jobs(
    *,
    names: Sequence[str],
    dependencies: Sequence[str],
    common_env: Mapping[str, str],
    topology: Mapping[str, Any],
) -> list[SlurmJobSpec]:
    """Create measured topology-matched probes and one compiler per variant."""

    jobs: list[SlurmJobSpec] = []
    base_dependencies = tuple(dict.fromkeys(str(value) for value in dependencies))
    for name in names:
        probe_keys: list[str] = []
        for stage_family, local_batch_size in ABPH_RUNTIME_BATCH_PROBE_SPECS:
            key = _runtime_batch_probe_job_key(
                name, stage_family, local_batch_size
            )
            probe_keys.append(key)
            jobs.append(
                SlurmJobSpec(
                    key,
                    "runtime_batch_probe",
                    "run_adaptive_binary_runtime_batch_probe.sh",
                    (name, stage_family, str(local_batch_size)),
                    base_dependencies,
                    gpu=True,
                    environment=common_env,
                    nodes=int(topology["nodes"]),
                    ntasks=int(topology["ntasks"]),
                    ntasks_per_node=int(topology["ntasks_per_node"]),
                    gpus_per_node=int(topology["gpus_per_node"]),
                    distributed_world_size=int(topology["distributed_world_size"]),
                    launcher=str(topology["launcher"]),
                )
            )
        jobs.append(
            SlurmJobSpec(
                _runtime_batch_contract_job_key(name),
                "runtime_batch_contract",
                "run_compile_adaptive_binary_runtime_batch_contract.sh",
                (name, str(topology["distributed_world_size"])),
                tuple(probe_keys),
                environment=common_env,
            )
        )
    return jobs


def _logit_scoring_jobs(
    *,
    common_env: Mapping[str, str],
    streaming: bool,
    reused_preparation: bool,
) -> list[SlurmJobSpec]:
    jobs: list[SlurmJobSpec] = []
    if not streaming:
        for member in ABPH_LOGIT_PREDICTION_MEMBERS:
            model_key = (
                f"variant:{member}"
                if "__seed" in member
                else _dependency_job_key(member)
            )
            dependencies = (
                ()
                if reused_preparation and model_key.startswith("baseline:")
                else (model_key,)
            )
            if not reused_preparation:
                dependencies = (*dependencies, "input:hlt_cache")
            jobs.append(
                SlurmJobSpec(
                    f"logit_prediction:{member}",
                    "logit_prediction",
                    "run_adaptive_binary_prediction.sh",
                    (member, "model_val", "stack_train", "stack_val"),
                    tuple(dict.fromkeys(dependencies)),
                    gpu=True,
                    environment=common_env,
                )
            )
        return jobs
    for family_key, members in ABPH_BUNDLED_SCORING_FAMILIES.items():
        dependencies: list[str] = []
        for member in members:
            model_key = (
                f"variant:{member}"
                if "__seed" in member
                else _dependency_job_key(member)
            )
            if not (reused_preparation and model_key.startswith("baseline:")):
                dependencies.append(model_key)
        if not reused_preparation:
            dependencies.append("input:hlt_cache")
        jobs.append(
            SlurmJobSpec(
                f"logit_bundle:{family_key}",
                "logit_bundle",
                "run_adaptive_binary_bundled_scoring.sh",
                tuple(members),
                tuple(dict.fromkeys(dependencies)),
                gpu=True,
                environment=common_env,
            )
        )
    return jobs


def _fusion_scoring_dependencies(
    members: Sequence[str], *, streaming: bool
) -> tuple[str, ...]:
    if streaming:
        return tuple(
            dict.fromkeys(_bundled_scoring_job_key_for_member(member) for member in members)
        )
    return tuple(f"logit_prediction:{member}" for member in members)


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


def _streaming_variant_dependencies(
    name: str, *, reused_models: bool = False
) -> tuple[str, ...]:
    base = (
        _reused_model_dependencies(name)
        if reused_models
        else _variant_dependencies(name)
    )
    dependencies = [value for value in base if not value.startswith("prediction:")]
    tier = variant_spec(name).tier
    if tier in {"E", "F"} or variant_spec(name).run_id in {"G0", "G1"}:
        run_id = variant_spec(name).run_id
        grouping = str(
            resolve_variant_config(name)["model"]["hierarchy"].get(
                "grouping", "exclusive_kt"
            )
        )
        source_models = (
            ("D1_kt32_mh4_particles", "D2_ca32_mh4_particles")
            if bool(resolve_variant_config(name)["model"]["fusion"].get("dual_hierarchy"))
            or run_id == "E11"
            else (
                ("D2_ca32_mh4_particles",)
                if grouping == "cambridge_aachen"
                else ("D1_kt32_mh4_particles",)
            )
        )
        dependencies.extend(_variant_job_key(value) for value in source_models)
    return tuple(dict.fromkeys(dependencies))


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


def _reused_model_dependencies(name: str) -> tuple[str, ...]:
    """Keep only dependencies rebuilt by the models-onward reuse stage."""

    external = {
        "preflight:actual_targets",
        "input:hlt_cache",
        "input:offline_cache",
        "teacher_logits:A4_offline_part_ceiling",
        *(f"baseline:{member}" for member in ABPH_BASELINE_VARIANTS),
    }
    return tuple(
        dependency
        for dependency in _variant_dependencies(name)
        if dependency not in external
    )


def _build_reused_model_graph(
    *,
    paths: AdaptiveBinaryCampaignPaths,
    common_env: Mapping[str, str],
    reconstructor_env: Mapping[str, str],
    topology: Mapping[str, Any],
    tagger_env: Mapping[str, str],
    tagger_topology: Mapping[str, Any],
    streaming: bool,
) -> list[SlurmJobSpec]:
    """Rebuild B-through-report while treating validated preparation as immutable."""

    jobs: list[SlurmJobSpec] = []
    reconstructor_names = (
        *ABPH_RECONSTRUCTOR_VARIANTS,
        *ABPH_RENDERER_VARIANTS,
    )
    for name in reconstructor_names:
        jobs.append(
            SlurmJobSpec(
                _variant_job_key(name),
                "renderer" if variant_spec(name).tier == "D" else "reconstructor",
                "run_adaptive_binary_variant.sh",
                (name,),
                _reused_model_dependencies(name),
                gpu=True,
                environment=reconstructor_env,
                nodes=int(topology["nodes"]),
                ntasks=int(topology["ntasks"]),
                ntasks_per_node=int(topology["ntasks_per_node"]),
                gpus_per_node=int(topology["gpus_per_node"]),
                distributed_world_size=int(topology["distributed_world_size"]),
                launcher=str(topology["launcher"]),
            )
        )
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    f"receipt:{name}",
                    "consumer_receipt",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    (
                        "consumer_receipt",
                        name,
                        str(_run_dir(paths, name) / "run_report.json"),
                        "target_consumer",
                    ),
                    (_variant_job_key(name),),
                    environment=common_env,
                )
            )
    if not streaming:
        for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES:
            jobs.append(
                SlurmJobSpec(
                    f"prediction:{name}",
                    "pseudo_prediction",
                    "run_adaptive_binary_prediction.sh",
                    (name, "model_train", "model_val", "stack_train", "stack_val"),
                    (_variant_job_key(name),),
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
    primary_seed = "F0_ce_reco_primary"

    def append_reused_tagger(
        name: str,
        *,
        member_name: str | None = None,
        seed_index: int | None = None,
        cross_privileged_barrier: bool = False,
    ) -> None:
        member = member_name or name
        dependencies = (
            _streaming_variant_dependencies(name, reused_models=True)
            if streaming
            else _reused_model_dependencies(name)
        )
        if cross_privileged_barrier:
            dependencies = tuple(dict.fromkeys((*dependencies, "wave:3")))
        jobs.append(
            SlurmJobSpec(
                f"variant:{member}",
                "tagger_seed" if seed_index is not None else "tagger",
                "run_adaptive_binary_variant.sh",
                (name,) if seed_index is None else (name, str(seed_index)),
                dependencies,
                gpu=True,
                environment=tagger_env,
                nodes=int(tagger_topology["nodes"]),
                ntasks=int(tagger_topology["ntasks"]),
                ntasks_per_node=int(tagger_topology["ntasks_per_node"]),
                gpus_per_node=int(tagger_topology["gpus_per_node"]),
                distributed_world_size=int(tagger_topology["distributed_world_size"]),
                launcher=str(tagger_topology["launcher"]),
            )
        )

    if streaming:
        for name in ABPH_PRIVILEGED_TAGGER_PREREQUISITES:
            append_reused_tagger(name)
        for name in ABPH_JOINT_TARGET_TAGGER_VARIANTS:
            append_reused_tagger(name)
            jobs.append(
                SlurmJobSpec(
                    f"receipt:{name}",
                    "consumer_receipt",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    (
                        "consumer_receipt",
                        name,
                        str(_run_dir(paths, name) / "run_report.json"),
                        "target_consumer",
                    ),
                    (_variant_job_key(name),),
                    environment=common_env,
                )
            )
        for seed_index in (2, 3):
            member = _seed_member_name(primary_seed, seed_index)
            append_reused_tagger(
                primary_seed, member_name=member, seed_index=seed_index
            )
            jobs.append(
                SlurmJobSpec(
                    f"receipt:{member}",
                    "consumer_receipt",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    (
                        "consumer_receipt",
                        member,
                        str(_run_dir(paths, member) / "run_report.json"),
                        "target_consumer",
                    ),
                    (f"variant:{member}",),
                    environment=common_env,
                )
            )
        jobs.append(
            SlurmJobSpec(
                "barrier:privileged_cleanup",
                "storage_cleanup",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("cleanup_privileged", *ABPH_PRIVILEGED_CONSUMERS),
                tuple(f"receipt:{name}" for name in ABPH_PRIVILEGED_CONSUMERS),
                environment=common_env,
            )
        )
        jobs.append(
            SlurmJobSpec(
                "wave:3",
                "storage_wave",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("wave_receipt", "3"),
                ("barrier:privileged_cleanup",),
                environment=common_env,
            )
        )
        for name in ABPH_NEURAL_TAGGER_VARIANTS:
            if name not in (
                *ABPH_JOINT_TARGET_TAGGER_VARIANTS,
                *ABPH_PRIVILEGED_TAGGER_PREREQUISITES,
            ):
                append_reused_tagger(name, cross_privileged_barrier=True)
    else:
        for name in ABPH_NEURAL_TAGGER_VARIANTS:
            append_reused_tagger(name)
        for seed_index in (2, 3):
            append_reused_tagger(
                primary_seed,
                member_name=_seed_member_name(primary_seed, seed_index),
                seed_index=seed_index,
            )
    tagger_members = (
        *ABPH_NEURAL_TAGGER_VARIANTS,
        _seed_member_name(primary_seed, 2),
        _seed_member_name(primary_seed, 3),
    )
    if streaming:
        jobs.append(
            SlurmJobSpec(
                "wave:4",
                "storage_wave",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("wave_receipt", "4"),
                tuple(f"variant:{name}" for name in tagger_members),
                environment=common_env,
            )
        )
    scoring_jobs = _logit_scoring_jobs(
            common_env=common_env,
            streaming=streaming,
            reused_preparation=True,
        )
    if streaming:
        scoring_jobs = [
            replace(
                job,
                dependencies=tuple(dict.fromkeys((*job.dependencies, "wave:4"))),
            )
            for job in scoring_jobs
        ]
    jobs.extend(scoring_jobs)
    for name in ABPH_POSTHOC_VARIANTS:
        member_names = _fusion_member_names(name)
        jobs.append(
            SlurmJobSpec(
                _variant_job_key(name),
                "fusion",
                "run_adaptive_binary_fusion.sh",
                (name, *member_names),
                _fusion_scoring_dependencies(member_names, streaming=streaming),
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
    if streaming:
        bundle_jobs = tuple(
            f"logit_bundle:{name}" for name in ABPH_BUNDLED_SCORING_FAMILIES
        )
        jobs.append(
            SlurmJobSpec(
                "barrier:deployable_cleanup",
                "storage_cleanup",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("cleanup_deployable", *ABPH_LOGIT_PREDICTION_MEMBERS),
                (*bundle_jobs, "diagnostic:tagger_use"),
                environment=common_env,
            )
        )
        jobs.append(
            SlurmJobSpec(
                "wave:5",
                "storage_wave",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("wave_receipt", "5"),
                (
                    "barrier:deployable_cleanup",
                    *(_variant_job_key(name) for name in ABPH_POSTHOC_VARIANTS),
                ),
                environment=common_env,
            )
        )
        report_dependencies = ("wave:5",)
    else:
        report_dependencies = tuple(
            dict.fromkeys(
                (
                    *(_variant_job_key(name) for name in ABPH_RECONSTRUCTOR_VARIANTS),
                    *(_variant_job_key(name) for name in ABPH_RENDERER_VARIANTS),
                    *(f"prediction:{name}" for name in ABPH_DEPLOYABLE_PSEUDO_SOURCES),
                    "prediction:E7_shared_root_dual",
                    *(_variant_job_key(name) for name in ABPH_NEURAL_TAGGER_VARIANTS),
                    f"variant:{_seed_member_name(primary_seed, 2)}",
                    f"variant:{_seed_member_name(primary_seed, 3)}",
                    *(
                        f"logit_prediction:{member}"
                        for member in ABPH_LOGIT_PREDICTION_MEMBERS
                    ),
                    *(_variant_job_key(name) for name in ABPH_POSTHOC_VARIANTS),
                    "diagnostic:tagger_use",
                )
            )
        )
    jobs.append(
        SlurmJobSpec(
            "report:model_selection",
            "report",
            "run_adaptive_binary_report.sh",
            ("selection",),
            report_dependencies,
            environment=common_env,
        )
    )
    if streaming:
        jobs.append(
            SlurmJobSpec(
                "wave:6",
                "storage_wave",
                "run_adaptive_binary_storage_lifecycle.sh",
                ("wave_receipt", "6"),
                ("report:model_selection",),
                environment=common_env,
            )
        )
    return jobs


def build_submission_graph(config: AdaptiveBinarySubmissionConfig) -> tuple[SlurmJobSpec, ...]:
    """Build the canonical, topologically ordered campaign graph."""

    paths = config.paths
    common_env = {
        "ABPH_ROOT": str(paths.root),
        "ABPH_CAMPAIGN_MODE": config.campaign_mode,
        "ABPH_RECONSTRUCTOR_SCHEDULE_CONTRACT": ABPH_ACCELERATED_SCHEDULE_CONTRACT,
        "ABPH_RECONSTRUCTOR_SCHEDULE_POLICY": "accelerated_screening_v1",
        "ABPH_DATA_DIR": str(config.data_dir),
        "ABPH_CONFIRM_FINAL_TEST": "1" if config.confirm_final_test else "0",
        "ABPH_STORAGE_PROFILE": config.storage_profile,
        "ABPH_STORAGE_CONTRACT_PATH": str(paths.storage / "storage_contract.json"),
        "ABPH_STORAGE_LEDGER_PATH": str(paths.storage / "quota_ledger.json"),
        "ABPH_TARGET_MODE_REPORT": str(paths.target_mode_report),
    }
    if config.storage_projection_path is not None:
        common_env["ABPH_STORAGE_PROJECTION_PATH"] = str(
            Path(config.storage_projection_path).resolve()
        )
    if config.selection_report_path is not None:
        common_env["ABPH_SELECTION_REPORT_PATH"] = str(config.selection_report_path)
    topology = dict(config.reconstructor_topology)
    tagger_topology = dict(config.tagger_topology)
    reconstructor_env = {
        **common_env,
        "ABPH_RECONSTRUCTOR_PARALLELISM": str(topology["mode"]),
        "ABPH_JOB_LAUNCHER": str(topology["launcher"]),
        "ABPH_DISTRIBUTED_NODES": str(topology["nodes"]),
        "ABPH_DISTRIBUTED_NTASKS": str(topology["ntasks"]),
        "ABPH_DISTRIBUTED_NTASKS_PER_NODE": str(topology["ntasks_per_node"]),
        "ABPH_DISTRIBUTED_GPUS_PER_NODE": str(topology["gpus_per_node"]),
        "ABPH_DISTRIBUTED_WORLD_SIZE": str(topology["distributed_world_size"]),
        "ABPH_DDP_TIMEOUT_SECONDS": "300",
    }
    tagger_env = {
        **common_env,
        "ABPH_TAGGER_PARALLELISM": str(tagger_topology["mode"]),
        "ABPH_TAGGER_DISTRIBUTED_WORLD_SIZE": str(
            tagger_topology["distributed_world_size"]
        ),
        "ABPH_DDP_TIMEOUT_SECONDS": "300",
    }
    jobs: list[SlurmJobSpec] = []
    streaming = config.storage_profile == "streaming_30gb_v1"

    if config.stage_mode == "full":
        jobs.append(SlurmJobSpec("input:splits", "splits", "run_adaptive_binary_inputs.sh", ("splits",), environment=common_env))
        input_parent = "input:splits"
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    "wave:0",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "0"),
                    ("input:splits",),
                    environment=common_env,
                )
            )
            input_parent = "wave:0"
        jobs.append(SlurmJobSpec("input:hlt_cache", "hlt_cache", "run_adaptive_binary_inputs.sh", ("hlt_cache",), (input_parent,), environment=common_env))
        jobs.append(SlurmJobSpec("input:offline_cache", "offline_cache", "run_adaptive_binary_inputs.sh", ("offline_cache",), (input_parent,), environment=common_env))
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
        wave_one_parent = "input:audit"
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    "wave:1",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "1"),
                    (
                        "input:audit",
                        *(f"baseline:{name}" for name in ABPH_BASELINE_VARIANTS),
                        "teacher_logits:A4_offline_part_ceiling",
                    ),
                    environment=common_env,
                )
            )
            wave_one_parent = "wave:1"
        target_dependencies: tuple[str, ...] = ("input:audit",)
        if config.storage_profile == "streaming_30gb_v1":
            jobs.append(
                SlurmJobSpec(
                    "target:mode_preflight",
                    "preflight",
                    "run_adaptive_binary_targets.sh",
                    ("mode_preflight",),
                    (wave_one_parent,),
                    environment=common_env,
                )
            )
            target_dependencies = ("target:mode_preflight",)
        jobs.append(SlurmJobSpec("target:cache", "targets", "run_adaptive_binary_targets.sh", ("cache",), target_dependencies, environment=common_env))
        jobs.append(SlurmJobSpec("preflight:actual_targets", "preflight", "run_adaptive_binary_targets.sh", ("preflight",), ("target:cache",), environment=common_env))
        reconstructor_preflight = "preflight:actual_targets"
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    "lifecycle:artifact_manifest",
                    "storage_manifest",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("manifest",),
                    ("preflight:actual_targets",),
                    environment=common_env,
                )
            )
            jobs.append(
                SlurmJobSpec(
                    "wave:2",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "2"),
                    ("lifecycle:artifact_manifest",),
                    environment=common_env,
                )
            )
            reconstructor_preflight = "wave:2"
        reconstructor_names = (
            *ABPH_RECONSTRUCTOR_VARIANTS,
            *ABPH_RENDERER_VARIANTS,
        )
        jobs.extend(
            _runtime_batch_contract_jobs(
                names=reconstructor_names,
                dependencies=(reconstructor_preflight,),
                common_env=common_env,
                topology=topology,
            )
        )
        for name in reconstructor_names:
            jobs.append(
                SlurmJobSpec(
                    _variant_job_key(name),
                    "renderer" if variant_spec(name).tier == "D" else "reconstructor",
                    "run_adaptive_binary_variant.sh",
                    (name,),
                    (
                        _runtime_batch_contract_job_key(name),
                        *_variant_dependencies(name),
                    ),
                    gpu=True,
                    environment=reconstructor_env,
                    nodes=int(topology["nodes"]),
                    ntasks=int(topology["ntasks"]),
                    ntasks_per_node=int(topology["ntasks_per_node"]),
                    gpus_per_node=int(topology["gpus_per_node"]),
                    distributed_world_size=int(topology["distributed_world_size"]),
                    launcher=str(topology["launcher"]),
                )
            )
            if streaming:
                jobs.append(
                    SlurmJobSpec(
                        f"receipt:{name}",
                        "consumer_receipt",
                        "run_adaptive_binary_storage_lifecycle.sh",
                        (
                            "consumer_receipt",
                            name,
                            str(_run_dir(paths, name) / "run_report.json"),
                            "target_consumer",
                        ),
                        (_variant_job_key(name), "lifecycle:artifact_manifest"),
                        environment=common_env,
                    )
                )
        if not streaming:
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
        primary_seed = "F0_ce_reco_primary"

        def append_tagger(
            name: str,
            *,
            member_name: str | None = None,
            seed_index: int | None = None,
            cross_privileged_barrier: bool = False,
        ) -> None:
            member = member_name or name
            dependencies = (
                _streaming_variant_dependencies(name)
                if streaming
                else _variant_dependencies(name)
            )
            if cross_privileged_barrier:
                dependencies = tuple(dict.fromkeys((*dependencies, "wave:3")))
            jobs.append(
                SlurmJobSpec(
                    f"variant:{member}",
                    "tagger_seed" if seed_index is not None else "tagger",
                    "run_adaptive_binary_variant.sh",
                    (name,) if seed_index is None else (name, str(seed_index)),
                    dependencies,
                    gpu=True,
                    environment=tagger_env,
                    nodes=int(tagger_topology["nodes"]),
                    ntasks=int(tagger_topology["ntasks"]),
                    ntasks_per_node=int(tagger_topology["ntasks_per_node"]),
                    gpus_per_node=int(tagger_topology["gpus_per_node"]),
                    distributed_world_size=int(tagger_topology["distributed_world_size"]),
                    launcher=str(tagger_topology["launcher"]),
                )
            )

        if streaming:
            for name in ABPH_PRIVILEGED_TAGGER_PREREQUISITES:
                append_tagger(name)
            for name in ABPH_JOINT_TARGET_TAGGER_VARIANTS:
                append_tagger(name)
                jobs.append(
                    SlurmJobSpec(
                        f"receipt:{name}",
                        "consumer_receipt",
                        "run_adaptive_binary_storage_lifecycle.sh",
                        (
                            "consumer_receipt",
                            name,
                            str(_run_dir(paths, name) / "run_report.json"),
                            "target_consumer",
                        ),
                        (_variant_job_key(name), "lifecycle:artifact_manifest"),
                        environment=common_env,
                    )
                )
            for seed_index in (2, 3):
                member = _seed_member_name(primary_seed, seed_index)
                append_tagger(
                    primary_seed,
                    member_name=member,
                    seed_index=seed_index,
                )
                jobs.append(
                    SlurmJobSpec(
                        f"receipt:{member}",
                        "consumer_receipt",
                        "run_adaptive_binary_storage_lifecycle.sh",
                        (
                            "consumer_receipt",
                            member,
                            str(_run_dir(paths, member) / "run_report.json"),
                            "target_consumer",
                        ),
                        (f"variant:{member}", "lifecycle:artifact_manifest"),
                        environment=common_env,
                    )
                )
            jobs.append(
                SlurmJobSpec(
                    "barrier:privileged_cleanup",
                    "storage_cleanup",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("cleanup_privileged", *ABPH_PRIVILEGED_CONSUMERS),
                    (
                        *(f"receipt:{name}" for name in ABPH_PRIVILEGED_CONSUMERS),
                        "teacher_logits:A4_offline_part_ceiling",
                    ),
                    environment=common_env,
                )
            )
            jobs.append(
                SlurmJobSpec(
                    "wave:3",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "3"),
                    ("barrier:privileged_cleanup",),
                    environment=common_env,
                )
            )
            for name in ABPH_NEURAL_TAGGER_VARIANTS:
                if name not in (
                    *ABPH_JOINT_TARGET_TAGGER_VARIANTS,
                    *ABPH_PRIVILEGED_TAGGER_PREREQUISITES,
                ):
                    append_tagger(name, cross_privileged_barrier=True)
        else:
            for name in ABPH_NEURAL_TAGGER_VARIANTS:
                append_tagger(name)
            for seed_index in (2, 3):
                append_tagger(
                    primary_seed,
                    member_name=_seed_member_name(primary_seed, seed_index),
                    seed_index=seed_index,
                )

        tagger_members = (
            *ABPH_NEURAL_TAGGER_VARIANTS,
            _seed_member_name(primary_seed, 2),
            _seed_member_name(primary_seed, 3),
        )
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    "wave:4",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "4"),
                    tuple(f"variant:{name}" for name in tagger_members),
                    environment=common_env,
                )
            )
        scoring_jobs = _logit_scoring_jobs(
                common_env=common_env,
                streaming=streaming,
                reused_preparation=False,
            )
        if streaming:
            scoring_jobs = [
                replace(
                    job,
                    dependencies=tuple(
                        dict.fromkeys((*job.dependencies, "wave:4"))
                    ),
                )
                for job in scoring_jobs
            ]
        jobs.extend(scoring_jobs)
        for name in ABPH_POSTHOC_VARIANTS:
            member_names = _fusion_member_names(name)
            members = _fusion_scoring_dependencies(
                member_names, streaming=streaming
            )
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
        if streaming:
            bundle_jobs = tuple(
                f"logit_bundle:{name}" for name in ABPH_BUNDLED_SCORING_FAMILIES
            )
            jobs.append(
                SlurmJobSpec(
                    "barrier:deployable_cleanup",
                    "storage_cleanup",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("cleanup_deployable", *ABPH_LOGIT_PREDICTION_MEMBERS),
                    (*bundle_jobs, "diagnostic:tagger_use"),
                    environment=common_env,
                )
            )
            jobs.append(
                SlurmJobSpec(
                    "wave:5",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "5"),
                    (
                        "barrier:deployable_cleanup",
                        *(_variant_job_key(name) for name in ABPH_POSTHOC_VARIANTS),
                    ),
                    environment=common_env,
                )
            )
            report_dependencies = ("wave:5",)
        else:
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
                        *(
                            f"logit_prediction:{member}"
                            for member in ABPH_LOGIT_PREDICTION_MEMBERS
                        ),
                        *(_variant_job_key(name) for name in ABPH_POSTHOC_VARIANTS),
                        "diagnostic:tagger_use",
                    )
                )
            )
            report_dependencies = terminal
        jobs.append(
            SlurmJobSpec(
                "report:model_selection",
                "report",
                "run_adaptive_binary_report.sh",
                ("selection",),
                report_dependencies,
                environment=common_env,
            )
        )
        if streaming:
            jobs.append(
                SlurmJobSpec(
                    "wave:6",
                    "storage_wave",
                    "run_adaptive_binary_storage_lifecycle.sh",
                    ("wave_receipt", "6"),
                    ("report:model_selection",),
                    environment=common_env,
                )
            )
    elif config.stage_mode == "models":
        jobs.extend(
            _build_reused_model_graph(
                paths=paths,
                common_env=common_env,
                reconstructor_env=reconstructor_env,
                topology=topology,
                tagger_env=tagger_env,
                tagger_topology=tagger_topology,
                streaming=streaming,
            )
        )
    elif config.stage_mode == "predictions":
        if streaming:
            raise ValueError(
                "the streaming_30gb_v1 profile forbids persistent pseudo-prediction "
                "stages; score taggers through RAM-backed logit prediction jobs"
            )
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
    topology = dict(config.reconstructor_topology)
    topology["reconstructor_job_keys"] = [
        job.key for job in jobs if job.stage in {"reconstructor", "renderer"}
    ]
    topology["parallelism_hash"] = canonical_hash(topology)
    tagger_topology = dict(config.tagger_topology)
    tagger_topology["tagger_job_keys"] = [
        job.key for job in jobs if job.stage in {"tagger", "tagger_seed"}
    ]
    tagger_topology["parallelism_hash"] = canonical_hash(tagger_topology)
    runtime_acceptance = None
    if config.runtime_acceptance_path is not None:
        acceptance_path = Path(config.runtime_acceptance_path).resolve()
        acceptance = require_runtime_acceptance(
            acceptance_path,
            scope=(
                "highdata"
                if config.campaign_mode == "highdata"
                else (
                    "optimized_pilot"
                    if config.stage_mode in {"full", "models"}
                    else "ddp4_runtime"
                )
            ),
        )
        runtime_acceptance = {
            "path": str(acceptance_path),
            "sha256": _sha256_file(acceptance_path),
            "acceptance_content_hash": acceptance["acceptance_content_hash"],
        }
    tagger_ddp_acceptance = None
    if config.tagger_ddp_acceptance_path is not None:
        tagger_acceptance_path = Path(
            config.tagger_ddp_acceptance_path
        ).resolve()
        tagger_acceptance = require_tagger_ddp_acceptance(
            tagger_acceptance_path
        )
        tagger_ddp_acceptance = {
            "path": str(tagger_acceptance_path),
            "sha256": _sha256_file(tagger_acceptance_path),
            "content_hash": tagger_acceptance["content_hash"],
        }
    storage_projection = None
    if config.storage_projection_path is not None:
        projection_path = Path(config.storage_projection_path).resolve()
        projection = require_storage_projection(
            projection_path,
            campaign_root=config.campaign_root,
            campaign_mode=config.campaign_mode,
            profile=config.storage_profile,
        )
        storage_projection = {
            "path": str(projection_path),
            "sha256": _sha256_file(projection_path),
            "content_hash": projection["content_hash"],
            "projected_peak_persistent_bytes": projection[
                "projected_peak_persistent_bytes"
            ],
            "projected_final_retained_bytes": projection[
                "projected_final_retained_bytes"
            ],
        }
    payload = {
        "contract": ABPH_SLURM_ORCHESTRATION_CONTRACT,
        "campaign_root": str(config.paths.root),
        "campaign_mode": config.campaign_mode,
        "stage_mode": config.stage_mode,
        "cluster": config.cluster,
        "split_sizes": dict(config.split_sizes),
        "reconstructor_schedule": {
            "contract": ABPH_ACCELERATED_SCHEDULE_CONTRACT,
            "policy_label": "accelerated_screening_v1",
            "campaign_profile": config.campaign_mode,
            "profile_selected_from_split_sizes": True,
        },
        "reconstructor_parallelism": topology,
        "runtime_acceptance": runtime_acceptance,
        "tagger_parallelism": tagger_topology,
        "tagger_ddp_acceptance": tagger_ddp_acceptance,
        "storage_profile": config.storage_profile,
        "storage_projection": storage_projection,
        "confirm_final_test": config.confirm_final_test,
        "jobs": [job.to_dict() for job in jobs],
    }
    payload["graph_hash"] = canonical_hash(payload)
    return payload


__all__ = [
    "ABPH_BASELINE_VARIANTS",
    "ABPH_BUNDLED_SCORING_FAMILIES",
    "ABPH_CLUSTER_PROFILES",
    "ABPH_DEPLOYABLE_PSEUDO_SOURCES",
    "ABPH_DIAGNOSTIC_VARIANTS",
    "ABPH_FINAL_CLAIM_CONTRACT",
    "ABPH_FUSION_CANDIDATES",
    "ABPH_LOGIT_PREDICTION_MEMBERS",
    "ABPH_JOINT_TARGET_TAGGER_MEMBERS",
    "ABPH_JOINT_TARGET_TAGGER_VARIANTS",
    "ABPH_NEURAL_TAGGER_VARIANTS",
    "ABPH_POSTHOC_VARIANTS",
    "ABPH_PRIVILEGED_CONSUMERS",
    "ABPH_PRIVILEGED_TAGGER_PREREQUISITES",
    "ABPH_RECONSTRUCTOR_VARIANTS",
    "ABPH_RECONSTRUCTOR_PARALLELISM_CONTRACT",
    "ABPH_RECONSTRUCTOR_PARALLELISM_MODES",
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
