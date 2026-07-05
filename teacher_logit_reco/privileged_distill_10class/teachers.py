"""PD10 Step 3 ParT teacher training, registration, and final-test evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import (
    HLTBaselineTrainConfig,
    JetViewTorchDataset,
    ParticleTransformerHLTClassifier,
    ParticleViewTorchDataset,
    build_particle_transformer_classifier,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    train_hlt_baseline,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view, load_hlt_metadata
from jetclass_fresh.heterogeneous_hlt import balanced_limit_jet_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_offline_view, load_split_manifest, manifest_hash
from jetclass_fresh.offline_teacher import OfflineTeacherTrainConfig, train_offline_teacher

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_HLT_DEGRADATION_STRENGTH,
    PD10_HLT_PROFILE,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_ALLOWED_INPUTS,
    PD10_TEACHER_HLT,
    PD10_TEACHER_MODEL_NAMES,
    PD10_TEACHER_OFFLINE,
    default_pd10_experiment_layout,
    normalize_pd10_teacher_target,
)
from .inputs import pd10_hlt_params_dict


PD10_STEP3_EXPERIMENT_STEP = "pd10_step3_part_teachers"
PD10_STEP3_TRAIN_EXPERIMENT_STEP = f"{PD10_STEP3_EXPERIMENT_STEP}:train"
PD10_STEP3_REGISTER_EXPERIMENT_STEP = f"{PD10_STEP3_EXPERIMENT_STEP}:register"
PD10_STEP3_EVALUATE_EXPERIMENT_STEP = f"{PD10_STEP3_EXPERIMENT_STEP}:final_test"

PD10_PART_TEACHER_TARGETS: tuple[str, ...] = (PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE)
PD10_HLT_TEACHER_SEED = 101
PD10_OFFLINE_TEACHER_SEED = 707


def normalize_pd10_part_teacher_target(value: str) -> str:
    target = normalize_pd10_teacher_target(value)
    if target not in PD10_PART_TEACHER_TARGETS:
        raise ValueError(f"Step 3 trains only ParT HLT/offline teachers, got {target!r}")
    return target


def pd10_part_teacher_model_name(teacher_target: str) -> str:
    return PD10_TEACHER_MODEL_NAMES[normalize_pd10_part_teacher_target(teacher_target)]


def pd10_part_teacher_dir(
    teacher_target: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.teacher_dir(normalize_pd10_part_teacher_target(teacher_target))


def pd10_part_teacher_checkpoint(
    teacher_target: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    return pd10_part_teacher_dir(teacher_target, output_root=output_root) / "best_model_val.pt"


def default_pd10_teacher_seed(teacher_target: str) -> int:
    target = normalize_pd10_part_teacher_target(teacher_target)
    if target == PD10_TEACHER_HLT:
        return PD10_HLT_TEACHER_SEED
    return PD10_OFFLINE_TEACHER_SEED


@dataclass
class PD10PartTeacherTrainConfig:
    """Strict Step 3 config for one 10-class ParT teacher."""

    teacher_target: str
    output_dir: str
    manifest_path: str
    cache_dir: str
    data_dir: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int | None = None
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    model_size: str = "base"
    compile_model: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    confirm_final_test: bool = False
    evaluate_final_test: bool = True

    def __post_init__(self) -> None:
        target = normalize_pd10_part_teacher_target(self.teacher_target)
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"PD10 Step 3 split order must be {PD10_SPLIT_ORDER}")
        if self.evaluate_final_test and not self.confirm_final_test:
            raise ValueError("PD10 Step 3 final-test evaluation requires confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if self.model_size not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        seed = default_pd10_teacher_seed(target) if self.seed is None else int(self.seed)
        object.__setattr__(self, "teacher_target", target)
        object.__setattr__(self, "seed", seed)

    @property
    def model_name(self) -> str:
        return pd10_part_teacher_model_name(self.teacher_target)

    @property
    def source_view(self) -> str:
        return "fixed_hlt" if self.teacher_target == PD10_TEACHER_HLT else "offline"

    @property
    def allowed_inputs(self) -> str:
        return PD10_TEACHER_ALLOWED_INPUTS[self.teacher_target]

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    try:
        torch = require_torch()
        payload = torch.load(Path(path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - depends on file/env
        return {"checkpoint_metadata_loaded": False, "checkpoint_metadata_error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"checkpoint_metadata_loaded": False, "checkpoint_metadata_error": "checkpoint payload is not a mapping"}
    metrics = dict(payload.get("metrics") or {})
    model_val = dict(metrics.get("model_val") or {})
    return {
        "checkpoint_metadata_loaded": True,
        "epoch": payload.get("epoch"),
        "experiment_step": payload.get("experiment_step"),
        "model_config": dict(payload.get("model_config") or {}),
        "config": dict(payload.get("config") or {}),
        "label_names": list(payload.get("label_names") or []),
        "model_val_accuracy": model_val.get("accuracy"),
        "model_val_loss": model_val.get("loss"),
    }


def _manifest_report(config: PD10PartTeacherTrainConfig) -> dict[str, Any]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "manifest_exists": False, "manifest_hash": None}
    manifest = load_split_manifest(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": True,
        "manifest_hash": manifest_hash(manifest),
        "data_dir": manifest.data_dir,
        "max_constits": int(manifest.max_constits),
    }


def _hlt_contract_splits(
    config: PD10PartTeacherTrainConfig,
    *,
    include_final_test: bool | None = None,
) -> tuple[str, ...]:
    splits = [config.train_split, config.val_split]
    should_include_final = config.evaluate_final_test if include_final_test is None else bool(include_final_test)
    if should_include_final:
        splits.append(config.final_test_split)
    return tuple(dict.fromkeys(splits))


def _require_hlt_cache_contract(
    config: PD10PartTeacherTrainConfig,
    *,
    include_final_test: bool | None = None,
) -> dict[str, Any]:
    """Hard-fail if an HLT teacher would train/evaluate on the wrong cache profile."""

    if config.teacher_target != PD10_TEACHER_HLT:
        return {"ok": True, "skipped": True, "reason": "teacher is not HLT"}
    manifest_report = _manifest_report(config)
    manifest_sha = manifest_report.get("manifest_hash")
    expected_params = pd10_hlt_params_dict()
    split_reports: dict[str, Any] = {}
    problems: list[str] = []
    splits = _hlt_contract_splits(config, include_final_test=include_final_test)
    for split in splits:
        try:
            metadata = load_hlt_metadata(config.cache_dir, split)
        except Exception as exc:
            split_reports[split] = {"ok": False, "error": str(exc)}
            problems.append(f"{split}: could not load HLT metadata: {exc}")
            continue
        split_problems: list[str] = []
        if metadata.get("hlt_profile") != PD10_HLT_PROFILE:
            split_problems.append(
                f"hlt_profile is {metadata.get('hlt_profile')!r}, expected {PD10_HLT_PROFILE!r}"
            )
        actual_strength = metadata.get("hlt_degradation_strength")
        if actual_strength is None or abs(float(actual_strength) - float(PD10_HLT_DEGRADATION_STRENGTH)) > 1.0e-12:
            split_problems.append(
                "hlt_degradation_strength is "
                f"{actual_strength!r}, expected {PD10_HLT_DEGRADATION_STRENGTH:g}"
            )
        if metadata.get("hlt_params") != expected_params:
            split_problems.append("hlt_params do not match the configured PD10 HLT profile")
        if manifest_sha and metadata.get("source_manifest_hash") != manifest_sha:
            split_problems.append("source_manifest_hash does not match active manifest")
        split_reports[split] = {
            "ok": not split_problems,
            "metadata_path": str(Path(config.cache_dir) / f"{split}_fixed_hlt_metadata.json"),
            "hlt_profile": metadata.get("hlt_profile"),
            "expected_hlt_profile": PD10_HLT_PROFILE,
            "hlt_degradation_strength": actual_strength,
            "expected_hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
            "source_manifest_hash": metadata.get("source_manifest_hash"),
            "expected_source_manifest_hash": manifest_sha,
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "problems": split_problems,
        }
        problems.extend(f"{split}: {problem}" for problem in split_problems)
    report = {
        "ok": not problems,
        "cache_dir": config.cache_dir,
        "manifest_hash": manifest_sha,
        "expected_hlt_profile": PD10_HLT_PROFILE,
        "expected_hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
        "splits": list(splits),
        "split_reports": split_reports,
        "problems": problems,
    }
    if problems:
        raise ValueError("PD10 HLT cache contract check failed: " + "; ".join(problems))
    return report


def _hlt_metadata_summary(config: PD10PartTeacherTrainConfig) -> dict[str, Any]:
    if config.teacher_target != PD10_TEACHER_HLT:
        return {}
    summaries: dict[str, Any] = {}
    for split in _hlt_contract_splits(config):
        try:
            metadata = load_hlt_metadata(config.cache_dir, split)
        except Exception as exc:  # pragma: no cover - exercised by compute-side failures
            summaries[split] = {"ok": False, "error": str(exc)}
            continue
        summaries[split] = {
            "ok": True,
            "n_jets": int(metadata.get("n_jets", 0)),
            "seed": metadata.get("seed"),
            "source_manifest_hash": metadata.get("source_manifest_hash"),
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
            "hlt_profile": metadata.get("hlt_profile"),
            "expected_hlt_profile": PD10_HLT_PROFILE,
            "hlt_profile_version": metadata.get("hlt_profile_version"),
            "hlt_params": metadata.get("hlt_params"),
            "expected_hlt_params": pd10_hlt_params_dict(),
            "hlt_profile_match_pd10": metadata.get("hlt_profile") == PD10_HLT_PROFILE,
            "hlt_params_match_pd10": metadata.get("hlt_params") == pd10_hlt_params_dict(),
        }
    return summaries


def _source_metadata(
    config: PD10PartTeacherTrainConfig,
    *,
    source_type: str,
    checkpoint_sha256: str | None,
    registration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_report = _manifest_report(config)
    metadata = {
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP3_TRAIN_EXPERIMENT_STEP if source_type.startswith("trained") else PD10_STEP3_REGISTER_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "architecture": "part",
        "source_type": source_type,
        "source_view": config.source_view,
        "allowed_inputs": config.allowed_inputs,
        "manifest_path": config.manifest_path,
        "manifest_hash": manifest_report.get("manifest_hash"),
        "cache_dir": config.cache_dir,
        "data_dir": config.data_dir,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "final_test_split": config.final_test_split,
        "split_sizes": {split: int(PD10_SPLIT_SIZES[split]) for split in PD10_SPLIT_ORDER},
        "max_train_jets": config.max_train_jets,
        "max_val_jets": config.max_val_jets,
        "max_final_test_jets": config.max_final_test_jets,
        "selection_split": "model_val",
        "final_test_after_model_val_selection": bool(config.evaluate_final_test),
        "confirm_final_test": bool(config.confirm_final_test),
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "no_stack_partitions_loaded": True,
        "student_deployment_inputs": "HLT_only",
        "teacher_is_train_time_only_for_distillation": True,
        "hlt_profile": PD10_HLT_PROFILE,
        "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
        "expected_hlt_params": pd10_hlt_params_dict(),
        "manifest": manifest_report,
        "hlt_cache_metadata": _hlt_metadata_summary(config),
    }
    if registration is not None:
        metadata["registration"] = dict(registration)
    return metadata


def _enriched_model_val_report(
    config: PD10PartTeacherTrainConfig,
    base_report: Mapping[str, Any],
    *,
    source_type: str,
    checkpoint_sha256: str | None,
    registration: bool,
) -> dict[str, Any]:
    return {
        **dict(base_report),
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP3_REGISTER_EXPERIMENT_STEP if registration else PD10_STEP3_TRAIN_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "architecture": "part",
        "source_type": source_type,
        "source_view": config.source_view,
        "allowed_inputs": config.allowed_inputs,
        "train_split": config.train_split,
        "selection_split": config.val_split,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_metadata_path": str(Path(config.output_dir) / "source_metadata.json"),
        "final_test_report_path": str(Path(config.output_dir) / "final_test_report.json"),
        "no_final_test_in_checkpoint_selection": True,
        "student_deployment_inputs": "HLT_only",
    }


def _write_run_artifacts(
    config: PD10PartTeacherTrainConfig,
    *,
    source_metadata: Mapping[str, Any],
    model_val_report: Mapping[str, Any],
    final_test_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    save_json(output_dir / "source_metadata.json", source_metadata)
    save_json(
        output_dir / "config.json",
        {
            "config": asdict(config),
            "source_metadata": dict(source_metadata),
        },
    )
    save_json(output_dir / "model_val_report.json", model_val_report)
    run_report = {
        "ok": True,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": source_metadata["experiment_step"],
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "checkpoint": str(config.checkpoint_path),
        "source_metadata_path": str(output_dir / "source_metadata.json"),
        "model_val_report": dict(model_val_report),
        "final_test_report": None if final_test_report is None else dict(final_test_report),
        "required_artifacts": [
            "best_model_val.pt",
            "run_report.json",
            "model_val_report.json",
            "final_test_report.json",
            "source_metadata.json",
        ],
    }
    save_json(output_dir / "run_report.json", run_report)
    return run_report


def _load_pd10_part_teacher_model(checkpoint: str | Path, *, device, fallback_model_size: str = "base"):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    if model_config:
        model_config.pop("architecture", None)
        model = ParticleTransformerHLTClassifier(**model_config)
    else:
        cfg = dict(payload.get("config") or {})
        model = build_particle_transformer_classifier(
            num_classes=len(LABEL_NAMES),
            model_size=cfg.get("model_size", fallback_model_size),
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def load_pd10_part_teacher_model_from_checkpoint(
    checkpoint: str | Path,
    *,
    device,
    fallback_model_size: str = "base",
):
    """Load a frozen PD10 ParT teacher checkpoint for evaluation/logit caching."""

    return _load_pd10_part_teacher_model(
        checkpoint,
        device=device,
        fallback_model_size=fallback_model_size,
    )


def evaluate_pd10_part_teacher_final_test(
    config: PD10PartTeacherTrainConfig,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if not config.confirm_final_test:
        raise ValueError("PD10 final-test evaluation requires confirm_final_test=True")
    if config.teacher_target == PD10_TEACHER_HLT:
        _require_hlt_cache_contract(config, include_final_test=True)

    torch = require_torch()
    device = resolve_device(config.device)
    checkpoint_path = Path(checkpoint or config.checkpoint_path)
    model, payload = _load_pd10_part_teacher_model(
        checkpoint_path,
        device=device,
        fallback_model_size=config.model_size,
    )
    if config.teacher_target == PD10_TEACHER_HLT:
        view = load_cached_hlt_view(config.cache_dir, config.final_test_split)
        view, selection_report = balanced_limit_jet_view(
            view,
            config.max_final_test_jets,
            seed=int(config.seed) + 20_003,
        )
        dataset = JetViewTorchDataset(view)
        source_view = "fixed_hlt"
    else:
        manifest = load_split_manifest(config.manifest_path)
        view = load_offline_view(
            manifest,
            config.final_test_split,
            data_dir=config.data_dir,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
        view, selection_report = balanced_limit_jet_view(
            view,
            config.max_final_test_jets,
            seed=int(config.seed) + 20_003,
        )
        dataset = ParticleViewTorchDataset(view, expected_view="offline")
        source_view = "offline"

    loader = make_data_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 20_004,
        source_view=source_view,
    )
    criterion = torch.nn.CrossEntropyLoss()
    metrics = run_epoch(
        model,
        loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_final_test_batches,
    )
    report = {
        "ok": True,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP3_EVALUATE_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "architecture": "part",
        "split": config.final_test_split,
        "source_view": source_view,
        "metrics": metrics,
        "accuracy": metrics.get("accuracy"),
        "loss": metrics.get("loss"),
        "n_jets": metrics.get("n_jets"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_experiment_step": payload.get("experiment_step"),
        "model_val_selection_only_before_final_test": True,
        "confirm_final_test": True,
        "subset_selection": selection_report,
        "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
    }
    save_json(Path(config.output_dir) / "final_test_report.json", report)
    return report


def train_pd10_part_teacher(
    config: PD10PartTeacherTrainConfig,
    *,
    model=None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.teacher_target == PD10_TEACHER_HLT:
        _require_hlt_cache_contract(config)
        base_config = HLTBaselineTrainConfig(
            output_dir=config.output_dir,
            cache_dir=config.cache_dir,
            train_split=config.train_split,
            val_split=config.val_split,
            seed=int(config.seed),
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            num_workers=config.num_workers,
            device=config.device,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            early_stop_patience=config.early_stop_patience,
            max_train_batches=config.max_train_batches,
            max_val_batches=config.max_val_batches,
            model_size=config.model_size,
            compile_model=config.compile_model,
        )
        base_report = train_hlt_baseline(
            base_config,
            model=model,
            max_train_jets=config.max_train_jets,
            max_val_jets=config.max_val_jets,
        )
        source_type = "trained_hlt_view"
    else:
        base_config = OfflineTeacherTrainConfig(
            output_dir=config.output_dir,
            manifest_path=config.manifest_path,
            data_dir=config.data_dir,
            train_split=config.train_split,
            val_split=config.val_split,
            seed=int(config.seed),
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            num_workers=config.num_workers,
            device=config.device,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            early_stop_patience=config.early_stop_patience,
            max_train_batches=config.max_train_batches,
            max_val_batches=config.max_val_batches,
            model_size=config.model_size,
            compile_model=config.compile_model,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
        base_report = train_offline_teacher(
            base_config,
            model=model,
            max_train_jets=config.max_train_jets,
            max_val_jets=config.max_val_jets,
        )
        source_type = "trained_offline_view"

    checkpoint_sha = sha256_file(config.checkpoint_path)
    source_metadata = _source_metadata(
        config,
        source_type=source_type,
        checkpoint_sha256=checkpoint_sha,
    )
    model_val_report = _enriched_model_val_report(
        config,
        base_report,
        source_type=source_type,
        checkpoint_sha256=checkpoint_sha,
        registration=False,
    )
    final_test_report = None
    if config.evaluate_final_test:
        final_test_report = evaluate_pd10_part_teacher_final_test(config)
    return _write_run_artifacts(
        config,
        source_metadata=source_metadata,
        model_val_report=model_val_report,
        final_test_report=final_test_report,
    )


def register_pd10_part_teacher_checkpoint(
    config: PD10PartTeacherTrainConfig,
    *,
    source_checkpoint: str | Path,
    source_model_val_report: str | Path | None = None,
    source_final_test_report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_checkpoint = Path(source_checkpoint)
    output_dir = Path(config.output_dir)
    target_checkpoint = config.checkpoint_path
    if source_final_test_report is not None and not bool(config.confirm_final_test):
        raise ValueError("Registering a source final-test report requires confirm_final_test=True")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_checkpoint.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite registered checkpoint: {target_checkpoint}")
    shutil.copy2(source_checkpoint, target_checkpoint)

    source_sha = sha256_file(source_checkpoint)
    checkpoint_sha = sha256_file(target_checkpoint)
    checkpoint_metadata = maybe_load_checkpoint_metadata(target_checkpoint)
    registration: dict[str, Any] = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "registered_checkpoint_sha256": checkpoint_sha,
        "source_model_val_report": None if source_model_val_report is None else str(source_model_val_report),
        "source_final_test_report": None if source_final_test_report is None else str(source_final_test_report),
        "checkpoint_metadata": checkpoint_metadata,
        "requires_user_trust_that_source_matches_pd10_splits_and_view": True,
    }
    if source_model_val_report is not None and Path(source_model_val_report).exists():
        registered = output_dir / "registered_source_model_val_report.json"
        shutil.copy2(source_model_val_report, registered)
        registration["registered_source_model_val_report"] = str(registered)
        registration["source_model_val_report_sha256"] = sha256_file(source_model_val_report)
    if source_final_test_report is not None and Path(source_final_test_report).exists():
        registered = output_dir / "registered_source_final_test_report.json"
        shutil.copy2(source_final_test_report, registered)
        registration["registered_source_final_test_report"] = str(registered)
        registration["source_final_test_report_sha256"] = sha256_file(source_final_test_report)

    source_metadata = _source_metadata(
        config,
        source_type="registered_existing_checkpoint",
        checkpoint_sha256=checkpoint_sha,
        registration=registration,
    )
    source_report = _read_json_if_exists(source_model_val_report)
    base_report = {
        "registration": True,
        "best_epoch": checkpoint_metadata.get("epoch"),
        "best_model_val_accuracy": checkpoint_metadata.get("model_val_accuracy"),
        "best_model_val_loss": checkpoint_metadata.get("model_val_loss"),
        "source_model_val_report": source_report,
    }
    model_val_report = _enriched_model_val_report(
        config,
        base_report,
        source_type="registered_existing_checkpoint",
        checkpoint_sha256=checkpoint_sha,
        registration=True,
    )
    final_test_report = None
    if config.evaluate_final_test:
        final_test_report = evaluate_pd10_part_teacher_final_test(config)
    elif source_final_test_report is not None and Path(source_final_test_report).exists():
        final_test_report = {
            "ok": True,
            "experiment_name": PD10_EXPERIMENT_NAME,
            "experiment_step": PD10_STEP3_REGISTER_EXPERIMENT_STEP,
            "teacher_target": config.teacher_target,
            "model_name": config.model_name,
            "registration": True,
            "source_final_test_report": _read_json_if_exists(source_final_test_report),
            "source_final_test_report_path": str(source_final_test_report),
            "checkpoint": str(target_checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
        save_json(output_dir / "final_test_report.json", final_test_report)

    if final_test_report is None:
        save_json(
            output_dir / "final_test_report.json",
            {
                "ok": False,
                "experiment_name": PD10_EXPERIMENT_NAME,
                "experiment_step": PD10_STEP3_REGISTER_EXPERIMENT_STEP,
                "teacher_target": config.teacher_target,
                "model_name": config.model_name,
                "skipped": True,
                "reason": "final_test evaluation was disabled and no source final-test report was supplied",
            },
        )
    run_report = _write_run_artifacts(
        config,
        source_metadata=source_metadata,
        model_val_report=model_val_report,
        final_test_report=final_test_report,
    )
    save_json(output_dir / "registration_report.json", run_report)
    return run_report


__all__ = [
    "PD10_HLT_TEACHER_SEED",
    "PD10_OFFLINE_TEACHER_SEED",
    "PD10_PART_TEACHER_TARGETS",
    "PD10_STEP3_EVALUATE_EXPERIMENT_STEP",
    "PD10_STEP3_EXPERIMENT_STEP",
    "PD10_STEP3_REGISTER_EXPERIMENT_STEP",
    "PD10_STEP3_TRAIN_EXPERIMENT_STEP",
    "PD10PartTeacherTrainConfig",
    "default_pd10_teacher_seed",
    "evaluate_pd10_part_teacher_final_test",
    "load_pd10_part_teacher_model_from_checkpoint",
    "maybe_load_checkpoint_metadata",
    "normalize_pd10_part_teacher_target",
    "pd10_part_teacher_checkpoint",
    "pd10_part_teacher_dir",
    "pd10_part_teacher_model_name",
    "register_pd10_part_teacher_checkpoint",
    "sha256_file",
    "train_pd10_part_teacher",
]
