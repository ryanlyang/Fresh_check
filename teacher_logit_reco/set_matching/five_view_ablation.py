"""Evaluation helpers for Step 11 five-view tagger ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json

from .experiment import (
    HLT_VIEW_NAME,
    RECONSTRUCTED_VIEW_NAMES,
    SPLIT_SIZES,
    SetMatchingMultiViewLayout,
    normalize_split_name,
    normalize_view_name,
)
from .five_view_data import FiveViewDatasetConfig, FiveViewJetDataset, make_five_view_loader
from .five_view_model import load_five_view_tagger_checkpoint
from .five_view_train import run_five_view_tagger_epoch


SET_MATCHING_FIVE_VIEW_ABLATION_STEP = "set_matching_multiview_step11_baseline_and_ablation_eval"

CANONICAL_ABLATION_NAMES: tuple[str, ...] = (
    "hlt_only",
    "hlt_plus_gt",
    "hlt_plus_pn",
    "hlt_plus_pfn",
    "hlt_plus_pcnn",
    "five_view_plain",
    "five_view_geometry",
    "five_view_no_confidence",
    "view_label_shuffle_control",
)


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


@dataclass(frozen=True)
class FiveViewAblationSpec:
    """One Step 11 ablation checkpoint plus its evaluation-time view setup."""

    name: str
    checkpoint: str
    description: str = ""
    drop_views: tuple[str, ...] = ()
    shuffle_view_labels: bool = False
    use_checkpoint_dataset_config: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("ablation spec name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "checkpoint", str(self.checkpoint))
        object.__setattr__(self, "drop_views", tuple(normalize_view_name(view) for view in self.drop_views))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "description": self.description,
            "drop_views": list(self.drop_views),
            "shuffle_view_labels": bool(self.shuffle_view_labels),
            "use_checkpoint_dataset_config": bool(self.use_checkpoint_dataset_config),
        }


@dataclass
class FiveViewAblationEvalConfig:
    """Configuration for evaluating Step 11 five-view tagger ablations."""

    output_dir: str
    experiment_dir: str
    hlt_cache_dir: str
    tagger_root: str | None = None
    reconstructed_view_dir: str | None = None
    checkpoint_specs: tuple[FiveViewAblationSpec, ...] = ()
    only: tuple[str, ...] = ()
    require_all_canonical: bool = False
    include_canonical: bool = True
    val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_final_test: bool = False
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_val_jets: int | None = None
    max_final_test_jets: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_tokens_per_view: int = 128
    min_tokens_per_view: int = 8
    confidence_threshold: float = 0.05
    selection_mode: str = "topk_or_threshold"
    verify_hlt_hash: bool = True
    seed: int = 1205

    def __post_init__(self) -> None:
        self.val_split = normalize_split_name(self.val_split)
        self.final_test_split = normalize_split_name(self.final_test_split)
        if self.val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 11 evaluates stack_val and optionally final_test")
        for field_name in ("batch_size", "max_tokens_per_view"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if int(self.min_tokens_per_view) < 0:
            raise ValueError("min_tokens_per_view cannot be negative")
        if int(self.min_tokens_per_view) > int(self.max_tokens_per_view):
            raise ValueError("min_tokens_per_view cannot exceed max_tokens_per_view")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_final_test_batches = _optional_nonnegative_int(self.max_final_test_batches, field_name="max_final_test_batches")
        self.checkpoint_specs = tuple(self.checkpoint_specs)
        self.only = tuple(str(value) for value in self.only)

    @property
    def layout(self) -> SetMatchingMultiViewLayout:
        experiment_path = Path(self.experiment_dir)
        return SetMatchingMultiViewLayout(output_root=experiment_path.parent, experiment_name=experiment_path.name)

    @property
    def resolved_tagger_root(self) -> Path:
        return Path(self.tagger_root) if self.tagger_root else self.layout.taggers_dir

    def evaluation_splits(self) -> tuple[str, ...]:
        if bool(self.confirm_final_test):
            return (self.val_split, self.final_test_split)
        return (self.val_split,)


def canonical_five_view_ablation_specs(tagger_root: str | Path) -> tuple[FiveViewAblationSpec, ...]:
    """Return the canonical Step 11 ablation checkpoint layout."""

    root = Path(tagger_root)
    all_reco = tuple(RECONSTRUCTED_VIEW_NAMES)
    return (
        FiveViewAblationSpec(
            name="hlt_only",
            checkpoint=str(root / "hlt_only" / "best_model_val.pt"),
            description="HLT-only transformer tagger using the five-view input path with all reconstructed views dropped.",
            drop_views=all_reco,
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="hlt_plus_gt",
            checkpoint=str(root / "hlt_plus_gt" / "best_model_val.pt"),
            description="HLT plus GT/ParT-style set-matching reconstructed view.",
            drop_views=tuple(view for view in all_reco if view != "gt_reco"),
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="hlt_plus_pn",
            checkpoint=str(root / "hlt_plus_pn" / "best_model_val.pt"),
            description="HLT plus ParticleNet-style set-matching reconstructed view.",
            drop_views=tuple(view for view in all_reco if view != "pn_reco"),
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="hlt_plus_pfn",
            checkpoint=str(root / "hlt_plus_pfn" / "best_model_val.pt"),
            description="HLT plus PFN-style set-matching reconstructed view.",
            drop_views=tuple(view for view in all_reco if view != "pfn_reco"),
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="hlt_plus_pcnn",
            checkpoint=str(root / "hlt_plus_pcnn" / "best_model_val.pt"),
            description="HLT plus PCNN-style set-matching reconstructed view.",
            drop_views=tuple(view for view in all_reco if view != "pcnn_reco"),
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="five_view_plain",
            checkpoint=str(root / "five_view_plain" / "best_model_val.pt"),
            description="All five views with plain attention.",
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="five_view_geometry",
            checkpoint=str(root / "five_view_geometry" / "best_model_val.pt"),
            description="All five views with geometry-aware attention.",
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="five_view_no_confidence",
            checkpoint=str(root / "five_view_no_confidence" / "best_model_val.pt"),
            description="All five views with confidence features disabled in the model.",
            use_checkpoint_dataset_config=False,
        ),
        FiveViewAblationSpec(
            name="view_label_shuffle_control",
            checkpoint=str(root / "view_label_shuffle_control" / "best_model_val.pt"),
            description="Negative control with non-HLT view labels shuffled.",
            shuffle_view_labels=True,
            use_checkpoint_dataset_config=False,
        ),
    )


def parse_ablation_checkpoint_spec(raw: str) -> FiveViewAblationSpec:
    """Parse ``name=path`` or ``name:path`` CLI checkpoint specs."""

    text = str(raw).strip()
    if not text:
        raise ValueError("checkpoint spec cannot be empty")
    if "=" in text:
        name, path = text.split("=", 1)
    elif ":" in text and not Path(text).drive:
        name, path = text.split(":", 1)
    else:
        path_obj = Path(text)
        name = path_obj.parent.name or path_obj.stem
        path = text
    return FiveViewAblationSpec(name=name, checkpoint=path)


def _checkpoint_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload.get("config")
    return config if isinstance(config, Mapping) else {}


def discover_five_view_ablation_specs(config: FiveViewAblationEvalConfig) -> tuple[list[FiveViewAblationSpec], list[dict[str, Any]]]:
    """Discover canonical and explicit specs, returning specs plus skipped rows."""

    specs_by_name: dict[str, FiveViewAblationSpec] = {}
    skipped: list[dict[str, Any]] = []
    if bool(config.include_canonical):
        for spec in canonical_five_view_ablation_specs(config.resolved_tagger_root):
            exists = Path(spec.checkpoint).exists()
            if exists:
                specs_by_name[spec.name] = spec
            else:
                payload = {"name": spec.name, "checkpoint": spec.checkpoint, "reason": "missing_checkpoint"}
                skipped.append(payload)
                if bool(config.require_all_canonical):
                    raise FileNotFoundError(f"missing canonical Step 11 checkpoint for {spec.name}: {spec.checkpoint}")
    for spec in config.checkpoint_specs:
        specs_by_name[spec.name] = spec
    if specs_by_name:
        skipped = [row for row in skipped if row.get("name") not in specs_by_name]
    if config.only:
        wanted = set(config.only)
        specs_by_name = {name: spec for name, spec in specs_by_name.items() if name in wanted}
        skipped = [row for row in skipped if row.get("name") in wanted]
        missing = sorted(wanted - set(specs_by_name))
        for name in missing:
            skipped.append({"name": name, "reason": "requested_name_not_found"})
            if bool(config.require_all_canonical):
                raise FileNotFoundError(f"requested ablation was not found: {name}")
    return list(specs_by_name.values()), skipped


def _dataset_config_for_spec(
    config: FiveViewAblationEvalConfig,
    spec: FiveViewAblationSpec,
    *,
    split: str,
    checkpoint_payload: Mapping[str, Any],
) -> FiveViewDatasetConfig:
    checkpoint_config = _checkpoint_config(checkpoint_payload)
    drop_views = tuple(spec.drop_views)
    shuffle_view_labels = bool(spec.shuffle_view_labels)
    if bool(spec.use_checkpoint_dataset_config):
        drop_views = tuple(checkpoint_config.get("drop_views", drop_views))
        shuffle_view_labels = bool(checkpoint_config.get("shuffle_view_labels", shuffle_view_labels))
    return FiveViewDatasetConfig(
        output_dir=config.experiment_dir,
        hlt_cache_dir=str(checkpoint_config.get("hlt_cache_dir") or config.hlt_cache_dir),
        reconstructed_view_dir=str(checkpoint_config.get("reconstructed_view_dir") or config.reconstructed_view_dir)
        if (checkpoint_config.get("reconstructed_view_dir") or config.reconstructed_view_dir)
        else None,
        split=split,
        max_tokens_per_view=int(checkpoint_config.get("max_tokens_per_view") or config.max_tokens_per_view),
        min_tokens_per_view=int(checkpoint_config.get("min_tokens_per_view") if checkpoint_config.get("min_tokens_per_view") is not None else config.min_tokens_per_view),
        confidence_threshold=float(checkpoint_config.get("confidence_threshold") if checkpoint_config.get("confidence_threshold") is not None else config.confidence_threshold),
        selection_mode=str(checkpoint_config.get("selection_mode") or config.selection_mode),
        drop_views=drop_views,
        shuffle_view_labels=shuffle_view_labels,
        view_label_shuffle_seed=int(checkpoint_config.get("view_label_shuffle_seed") or config.seed),
        verify_hlt_hash=bool(config.verify_hlt_hash),
    )


def _copy_dataset_with_ablations(
    dataset: FiveViewJetDataset,
    *,
    drop_views: Sequence[str] = (),
    shuffle_view_labels: bool = False,
    seed: int = 1205,
) -> FiveViewJetDataset:
    view_features = dataset.view_features.copy()
    view_masks = dataset.view_masks.copy()
    view_confidence = dataset.view_confidence.copy()
    source_indices = dataset.source_indices.copy()
    dropped = tuple(normalize_view_name(view) for view in drop_views)
    view_names = tuple(dataset.view_names)
    for view in dropped:
        if view not in view_names:
            raise ValueError(f"cannot drop unknown view {view!r}; available views are {view_names}")
        index = view_names.index(view)
        view_features[:, index] = 0.0
        view_masks[:, index] = False
        view_confidence[:, index] = 0.0
        source_indices[:, index] = -1

    view_ids = np.asarray(dataset.view_ids, dtype=np.int64).copy()
    if bool(shuffle_view_labels):
        rng = np.random.default_rng(int(seed))
        if HLT_VIEW_NAME in view_names:
            anchor = view_names.index(HLT_VIEW_NAME)
            movable = [index for index in range(len(view_names)) if index != anchor]
            shuffled = view_ids[movable].copy()
            rng.shuffle(shuffled)
            view_ids[movable] = shuffled
        else:
            rng.shuffle(view_ids)

    metadata = dict(dataset.metadata)
    metadata["step11_drop_views"] = list(dropped)
    metadata["step11_shuffle_view_labels"] = bool(shuffle_view_labels)
    return FiveViewJetDataset(
        view_features=view_features,
        view_masks=view_masks,
        view_confidence=view_confidence,
        labels=dataset.labels,
        jet_ids=dataset.jet_ids,
        split=dataset.split,
        view_names=dataset.view_names,
        source_types=dataset.source_types,
        view_ids=view_ids,
        source_type_ids=dataset.source_type_ids,
        source_indices=source_indices,
        metadata=metadata,
    )


def _slice_dataset(dataset: FiveViewJetDataset, max_jets: int | None) -> FiveViewJetDataset:
    if max_jets is None or int(max_jets) >= len(dataset):
        return dataset
    limit = int(max_jets)
    metadata = dict(dataset.metadata)
    metadata["limited_from_n_jets"] = len(dataset)
    metadata["n_jets"] = limit
    metadata["max_jets_limit"] = limit
    return FiveViewJetDataset(
        view_features=dataset.view_features[:limit],
        view_masks=dataset.view_masks[:limit],
        view_confidence=dataset.view_confidence[:limit],
        labels=dataset.labels[:limit],
        jet_ids=dataset.jet_ids[:limit],
        split=dataset.split,
        view_names=dataset.view_names,
        source_types=dataset.source_types,
        view_ids=dataset.view_ids,
        source_type_ids=dataset.source_type_ids,
        source_indices=dataset.source_indices[:limit],
        metadata=metadata,
    )


def _load_dataset_for_spec(
    config: FiveViewAblationEvalConfig,
    spec: FiveViewAblationSpec,
    *,
    split: str,
    checkpoint_payload: Mapping[str, Any],
    injected_datasets: Mapping[str, FiveViewJetDataset] | None,
) -> FiveViewJetDataset:
    max_jets = config.max_final_test_jets if split == "final_test" else config.max_val_jets
    if injected_datasets and split in injected_datasets:
        checkpoint_config = _checkpoint_config(checkpoint_payload)
        drop_views = tuple(spec.drop_views)
        shuffle_view_labels = bool(spec.shuffle_view_labels)
        if bool(spec.use_checkpoint_dataset_config):
            drop_views = tuple(checkpoint_config.get("drop_views", drop_views))
            shuffle_view_labels = bool(checkpoint_config.get("shuffle_view_labels", shuffle_view_labels))
        dataset = _copy_dataset_with_ablations(
            injected_datasets[split],
            drop_views=drop_views,
            shuffle_view_labels=shuffle_view_labels,
            seed=int(checkpoint_config.get("view_label_shuffle_seed") or config.seed),
        )
        return _slice_dataset(dataset, max_jets)
    dataset_config = _dataset_config_for_spec(config, spec, split=split, checkpoint_payload=checkpoint_payload)
    return _slice_dataset(FiveViewJetDataset.from_caches(dataset_config), max_jets)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "ablation",
        "split",
        "accuracy",
        "loss",
        "macro_per_class_accuracy",
        "n_jets",
        "checkpoint",
        "contract",
        "use_geometry_attention",
        "use_confidence",
        "drop_views",
        "shuffle_view_labels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_per_class_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["ablation", "split", "class_index", "class_name", "support", "correct", "accuracy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def evaluate_five_view_ablation_suite(
    config: FiveViewAblationEvalConfig,
    *,
    datasets_by_split: Mapping[str, FiveViewJetDataset] | None = None,
) -> dict[str, Any]:
    """Evaluate trained Step 11 ablation checkpoints and write summary files."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, skipped = discover_five_view_ablation_specs(config)
    if not specs:
        raise FileNotFoundError("no Step 11 ablation checkpoints were found or provided")

    criterion = torch.nn.CrossEntropyLoss()
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    result_by_ablation: dict[str, Any] = {}

    for spec in specs:
        checkpoint = Path(spec.checkpoint)
        if not checkpoint.exists():
            skipped.append({"name": spec.name, "checkpoint": str(checkpoint), "reason": "missing_explicit_checkpoint"})
            continue
        model, payload = load_five_view_tagger_checkpoint(checkpoint, device=device)
        model_config = model.to_config_dict()
        split_results: dict[str, Any] = {}
        for split in config.evaluation_splits():
            max_batches = config.max_final_test_batches if split == "final_test" else config.max_val_batches
            dataset = _load_dataset_for_spec(
                config,
                spec,
                split=split,
                checkpoint_payload=payload,
                injected_datasets=datasets_by_split,
            )
            loader = make_five_view_loader(
                dataset,
                batch_size=int(config.batch_size),
                shuffle=False,
                num_workers=int(config.num_workers),
                seed=int(config.seed),
            )
            metrics = run_five_view_tagger_epoch(
                model,
                loader,
                device=device,
                criterion=criterion,
                amp=False,
                max_batches=max_batches,
                collect_predictions=True,
            )
            split_results[split] = {
                "metrics": metrics,
                "dataset_metadata": dict(dataset.metadata),
            }
            summary_rows.append(
                {
                    "ablation": spec.name,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "loss": metrics.get("loss"),
                    "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                    "n_jets": metrics.get("n_jets"),
                    "checkpoint": str(checkpoint),
                    "contract": payload.get("output_contract") or model.output_contract,
                    "use_geometry_attention": bool(model_config.get("use_geometry_attention")),
                    "use_confidence": bool(model_config.get("use_confidence")),
                    "drop_views": " ".join(spec.drop_views),
                    "shuffle_view_labels": bool(spec.shuffle_view_labels),
                }
            )
            for row in metrics.get("per_class_accuracy", []):
                payload_row = {"ablation": spec.name, "split": split}
                payload_row.update(dict(row))
                per_class_rows.append(payload_row)

        result_by_ablation[spec.name] = {
            "spec": spec.to_dict(),
            "model_config": model_config,
            "checkpoint_payload_summary": {
                "experiment_step": payload.get("experiment_step"),
                "output_contract": payload.get("output_contract"),
                "epoch": payload.get("epoch"),
            },
            "splits": split_results,
        }

    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_per_class_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    report = {
        "experiment_step": SET_MATCHING_FIVE_VIEW_ABLATION_STEP,
        "config": asdict(config),
        "expected_split_sizes": dict(SPLIT_SIZES),
        "canonical_ablation_names": list(CANONICAL_ABLATION_NAMES),
        "evaluated_ablations": list(result_by_ablation),
        "skipped": skipped,
        "summary_rows": summary_rows,
        "summary_csv": str(output_dir / "summary.csv"),
        "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
        "results": result_by_ablation,
        "final_test_evaluated": bool(config.confirm_final_test),
        "leakage_rule": (
            "Step 11 only evaluates already-selected tagger checkpoints. stack_val is always allowed; "
            "final_test is loaded only with --confirm-final-test."
        ),
    }
    save_json(output_dir / "summary.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "CANONICAL_ABLATION_NAMES",
    "SET_MATCHING_FIVE_VIEW_ABLATION_STEP",
    "FiveViewAblationEvalConfig",
    "FiveViewAblationSpec",
    "canonical_five_view_ablation_specs",
    "discover_five_view_ablation_specs",
    "evaluate_five_view_ablation_suite",
    "parse_ablation_checkpoint_spec",
]
