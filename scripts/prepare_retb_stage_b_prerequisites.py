#!/usr/bin/env python3
"""Train Stage-A offline teachers and publish Stage-B teacher inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import tempfile
import time
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_offline_expert import (  # noqa: E402
    _checkpoint_state,
    _dataset,
    _load_npz,
    _load_trees,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    EXPERT_REGISTRATION_CONTRACT,
    OfflineExpertTrainingConfig,
    build_attachment_pretraining_record,
    build_teacher_logits_manifest,
    make_offline_expert_loader,
    train_offline_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    FAMILIES,
    OfflineClassifierAdapter,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
)

import torch  # noqa: E402


BASELINE_REGISTRY_CONTRACT = "retb_stage_a_offline_teacher_registry_v1"
BASELINE_BUNDLE_CONTRACT = "retb_stage_a_offline_teacher_bundle_v1"
TEACHER_SELECTION_CONTRACT = "retb_stage_a_strongest_teacher_selection_v1"
BASELINE_TRAINING_RECEIPT_CONTRACT = (
    "retb_stage_a_offline_teacher_training_receipt_v1"
)
CONTROL_ORDER = ("O_BASE", "O_FULLREL")
LOSS_TEACHERS = {
    "ELOSS_BASE_LOW": ("O_BASE",),
    "ELOSS_BASE": ("O_BASE",),
    "ELOSS_FULLREL": ("O_FULLREL",),
    "ELOSS_ENSEMBLE": ("O_BASE", "O_FULLREL"),
    "ELOSS_KD_DOMINANT": ("SELECTED_STRONGEST",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration(control_id: str) -> dict[str, Any]:
    full = control_id == "O_FULLREL"
    return {
        "expert_id": control_id,
        "relation_family": "ALL" if full else None,
        "topology": "B_CONCAT",
        "shape_id": "S8_128",
        "token_count": 8,
        "token_dimension": 128,
        "tokenizer_mode": "TOK_WEAVER_CLASS",
        "measurement_embedding": False,
        "initialization": "INIT_SCRATCH",
        "loss_id": "ELOSS_CE",
        "learning_rate": 1.0e-3,
        "particle_dropout": 0.0,
        "ordinary_weaver_head": True,
        "token_target_eligible": False,
        "stage_a_teacher_control": True,
    }


def _registry(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for control_id in CONTROL_ORDER:
        config = _configuration(control_id)
        rows.append(
            {
                "run_id": f"RETBA_{control_id}",
                "seed": 101,
                "control_id": control_id,
                "configuration": config,
            }
        )
    return bind_source(
        with_content_hash(
            {
                "contract": BASELINE_REGISTRY_CONTRACT,
                "schema_version": 1,
                "stage": "A",
                "rows": rows,
                "control_order": list(CONTROL_ORDER),
                "selection_split": "val_stop",
                "final_test_access": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source,
    )


def _model(
    control_id: str,
    *,
    configuration: Mapping[str, Any],
    weaver: Any,
    relation: Mapping[str, Any],
    region: Mapping[str, Any],
) -> Any:
    classifier = (
        RelationalParticleTransformer(weaver_module=weaver)
        if control_id == "O_BASE"
        else RelationalFamilyParticleTransformer(
            FAMILIES,
            normalization_artifact=relation,
            region_normalization_artifact=region,
            weaver_module=weaver,
        )
    )
    return OfflineClassifierAdapter(
        classifier, expert_configuration=configuration
    )


def _predict(model: Any, loader: Any, *, device: torch.device) -> np.ndarray:
    was_training = bool(model.training)
    rows = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                moved = {
                    name: value.to(device) if isinstance(value, torch.Tensor) else value
                    for name, value in batch.items()
                }
                logits = model(
                    features=moved["features"],
                    vectors=moved["vectors"],
                    mask=moved["mask"],
                    raw_tokens=moved["raw_tokens"],
                    region_trees=moved.get("region_trees"),
                )
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Stage-A teacher logits are nonfinite")
                rows.append(logits.detach().float().cpu().numpy())
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Stage-A teacher inference received no events")
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(
                    f"refusing to overwrite different teacher view: {path}"
                )
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _publish_checkpoint_alias(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256(source)
    if target.exists():
        if target.is_symlink() or _sha256(target) != digest:
            raise FileExistsError("strongest-teacher checkpoint alias differs")
    else:
        os.link(source, target)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-control", choices=CONTROL_ORDER)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    source = source_snapshot(REPO_ROOT)
    registry = _registry(source)
    write_immutable_json(
        root / "registry" / "retb_stage_a_offline_teachers.json", registry
    )
    step3 = load_hashed_json(
        root / "registry" / "retb_step3_architecture_bundle.json"
    )
    losses = load_hashed_json(root / "registry" / "retb_expert_losses.json")
    relation = load_hashed_json(
        root / "inputs/normalization/offline_500k/relation.json"
    )
    region = load_hashed_json(
        root / "inputs/normalization/offline_500k/region.json"
    )
    resource = load_hashed_json(root / "job_ledgers/resource_probes/gpu.json")
    arrays = {
        split: _load_npz(
            root / "inputs" / "offline" / split / "offline_inputs.npz"
        )
        for split in ("model_train", "val_stop")
    }
    needs_region_trees = bool(
        args.finalize or args.train_control == "O_FULLREL"
    )
    trees = (
        {
            split: _load_trees(
                root / "inputs/region_tree/offline",
                split=split,
                identities=[
                    str(value) for value in payload["identities"].tolist()
                ],
            )
            for split, payload in arrays.items()
        }
        if needs_region_trees
        else {split: None for split in arrays}
    )
    datasets = {
        split: _dataset(payload, region_trees=trees[split])
        for split, payload in arrays.items()
    }
    loaders = {
        split: make_offline_expert_loader(
            dataset,
            seed=101,
            training=split == "model_train",
            batch_size=64,
        )
        for split, dataset in datasets.items()
    }
    inference_loaders = {
        split: make_offline_expert_loader(
            dataset,
            seed=0,
            training=False,
            batch_size=64,
        )
        for split, dataset in datasets.items()
    }
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    registrations: dict[str, Mapping[str, Any]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    elapsed: dict[str, float] = {}
    training_rows = (
        []
        if args.finalize
        else [
            row
            for row in registry["rows"]
            if row["control_id"] == args.train_control
        ]
    )
    for row in training_rows:
        control_id = str(row["control_id"])
        model = _model(
            control_id,
            configuration=row["configuration"],
            weaver=weaver,
            relation=relation,
            region=region,
        )
        output = root / "runs/stage_a/offline_controls" / control_id / "seed_101"
        started = time.monotonic()
        registration = train_offline_expert(
            model=model,
            train_loader=loaders["model_train"],
            val_stop_loader=loaders["val_stop"],
            output_dir=output,
            run_record=row,
            run_registry_sha256=registry["content_hash"],
            step3_bundle_sha256=step3["content_hash"],
            global_determinism_sha256=campaign["parent_artifact_hashes"][
                "global_determinism"
            ],
            expert_loss_registry=losses,
            lineage_hashes={
                "campaign_spec": campaign["content_hash"],
                "model_train_inputs": _sha256(
                    root / "inputs/offline/model_train/offline_inputs.npz"
                ),
                "val_stop_inputs": _sha256(
                    root / "inputs/offline/val_stop/offline_inputs.npz"
                ),
                **(
                    {}
                    if control_id == "O_BASE"
                    else {
                        "relation_normalization": relation["content_hash"],
                        "region_normalization": region["content_hash"],
                    }
                ),
            },
            config=OfflineExpertTrainingConfig(
                seed=101,
                maximum_epochs=(
                    2 if campaign["campaign_profile"] == "miniature_test" else 40
                ),
                campaign_profile=(
                    "miniature_test"
                    if campaign["campaign_profile"] == "miniature_test"
                    else "production"
                ),
            ),
            device=device,
            resource_profile=resource,
        )
        elapsed[control_id] = time.monotonic() - started
        registrations[control_id] = registration

        receipt = bind_source(
            with_content_hash(
                {
                    "contract": BASELINE_TRAINING_RECEIPT_CONTRACT,
                    "schema_version": 1,
                    "control_id": control_id,
                    "run_id": row["run_id"],
                    "seed": int(row["seed"]),
                    "registry_sha256": registry["content_hash"],
                    "checkpoint_registration_sha256": registration[
                        "content_hash"
                    ],
                    "checkpoint_sha256": registration["checkpoint_sha256"],
                    "walltime_seconds": float(elapsed[control_id]),
                    "fixed_budget_completed": registration[
                        "fixed_epoch_budget_completed"
                    ],
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source,
        )
        write_immutable_json(output / "stage_a_training_receipt.json", receipt)

    if not args.finalize:
        return 0

    for row in registry["rows"]:
        control_id = str(row["control_id"])
        output = root / "runs/stage_a/offline_controls" / control_id / "seed_101"
        registration = load_hashed_json(
            output / "checkpoint_registration.json",
            expected_contract=EXPERT_REGISTRATION_CONTRACT,
        )
        receipt = load_hashed_json(
            output / "stage_a_training_receipt.json",
            expected_contract=BASELINE_TRAINING_RECEIPT_CONTRACT,
        )
        checkpoint = output / "best_model_val.pt"
        checkpoint_sha = _sha256(checkpoint)
        expected = {
            "control_id": control_id,
            "run_id": row["run_id"],
            "seed": int(row["seed"]),
            "registry_sha256": registry["content_hash"],
            "checkpoint_registration_sha256": registration["content_hash"],
            "checkpoint_sha256": checkpoint_sha,
            "fixed_budget_completed": True,
            "performance_based_termination": False,
            "source": campaign["source"],
        }
        drift = {
            name: (receipt.get(name), value)
            for name, value in expected.items()
            if receipt.get(name) != value
        }
        if drift:
            raise ValueError(
                f"Stage-A teacher training receipt drifted: {drift}"
            )
        if registration.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Stage-A teacher checkpoint registration drifted")
        expected_registration = {
            "run_id": row["run_id"],
            "seed": int(row["seed"]),
            "run_registry_sha256": registry["content_hash"],
            "step3_bundle_sha256": step3["content_hash"],
            "fixed_epoch_budget_completed": True,
            "stopped_early": False,
            "performance_based_termination": False,
            "resource_profile_sha256": resource["content_hash"],
        }
        registration_drift = {
            name: (registration.get(name), value)
            for name, value in expected_registration.items()
            if registration.get(name) != value
        }
        expected_lineage = {
            "campaign_spec": campaign["content_hash"],
            "model_train_inputs": _sha256(
                root / "inputs/offline/model_train/offline_inputs.npz"
            ),
            "val_stop_inputs": _sha256(
                root / "inputs/offline/val_stop/offline_inputs.npz"
            ),
        }
        if control_id == "O_FULLREL":
            expected_lineage.update(
                {
                    "relation_normalization": relation["content_hash"],
                    "region_normalization": region["content_hash"],
                }
            )
        if registration.get("lineage_hashes") != expected_lineage:
            registration_drift["lineage_hashes"] = (
                registration.get("lineage_hashes"),
                expected_lineage,
            )
        if registration_drift:
            raise ValueError(
                f"Stage-A teacher registration drifted: {registration_drift}"
            )
        model = _model(
            control_id,
            configuration=row["configuration"],
            weaver=weaver,
            relation=relation,
            region=region,
        )
        model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        model.to(device)
        registrations[control_id] = registration
        elapsed[control_id] = float(receipt["walltime_seconds"])
        predictions[control_id] = {
            split: _predict(model, loader, device=device)
            for split, loader in inference_loaders.items()
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    strongest = min(
        CONTROL_ORDER,
        key=lambda name: (
            -float(registrations[name]["selected_val_stop"]["accuracy"]),
            float(registrations[name]["selected_val_stop"]["cross_entropy"]),
            CONTROL_ORDER.index(name),
        ),
    )
    selection = bind_source(
        with_content_hash(
            {
                "contract": TEACHER_SELECTION_CONTRACT,
                "schema_version": 1,
                "selection_split": "val_stop",
                "candidate_order": list(CONTROL_ORDER),
                "selected_teacher": strongest,
                "candidate_checkpoint_sha256": {
                    name: registrations[name]["checkpoint_sha256"]
                    for name in CONTROL_ORDER
                },
                "candidate_metrics": {
                    name: registrations[name]["selected_val_stop"]
                    for name in CONTROL_ORDER
                },
                "tie_break": "accuracy_desc_cross_entropy_asc_control_order",
                "final_test_access": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root / "selection/stage_a_strongest_teacher.json", selection
    )
    selected_checkpoint = (
        root
        / "runs/stage_a/offline_controls"
        / strongest
        / "seed_101/best_model_val.pt"
    )
    selected_alias = (
        root
        / "runs/stage_a/offline_controls/SELECTED_STRONGEST/seed_101/"
        / "best_model_val.pt"
    )
    _publish_checkpoint_alias(selected_checkpoint, selected_alias)

    obase = registrations["O_BASE"]
    attachment = bind_source(
        build_attachment_pretraining_record(
            checkpoint_sha256=obase["checkpoint_sha256"],
            epochs=int(obase["epochs_completed"]),
            label_presentations=(
                len(datasets["model_train"]) * int(obase["epochs_completed"])
            ),
            optimizer_updates=int(obase["optimizer_updates_completed"]),
            walltime_seconds=float(elapsed["O_BASE"]),
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root
        / "runs/stage_a/offline_controls/O_BASE/seed_101/"
        / "attachment_pretraining_record.json",
        attachment,
    )

    teacher_root = root / "inputs/teacher_logits"
    cache_hashes = {}
    for loss_id, fields in LOSS_TEACHERS.items():
        loss_hashes = {}
        for split, original in arrays.items():
            augmented = dict(original)
            for field in fields:
                source_name = strongest if field == "SELECTED_STRONGEST" else field
                augmented[f"teacher_logits_{field}"] = predictions[source_name][split]
            loss_hashes[split] = _publish_npz(
                teacher_root / loss_id / f"{split}.npz", augmented
            )
        checkpoint_hashes = {
            field: (
                registrations[strongest]["checkpoint_sha256"]
                if field == "SELECTED_STRONGEST"
                else registrations[field]["checkpoint_sha256"]
            )
            for field in fields
        }
        manifest = bind_source(
            build_teacher_logits_manifest(
                model_train_npz_sha256=loss_hashes["model_train"],
                val_stop_npz_sha256=loss_hashes["val_stop"],
                teacher_checkpoint_hashes=checkpoint_hashes,
                teacher_fields=fields,
            ),
            source_snapshot=source,
        )
        write_immutable_json(teacher_root / f"{loss_id}.json", manifest)
        cache_hashes[loss_id] = {
            "manifest_sha256": manifest["content_hash"],
            "model_train_npz_sha256": loss_hashes["model_train"],
            "val_stop_npz_sha256": loss_hashes["val_stop"],
        }

    bundle = bind_source(
        with_content_hash(
            {
                "contract": BASELINE_BUNDLE_CONTRACT,
                "schema_version": 1,
                "registry_sha256": registry["content_hash"],
                "teacher_selection_sha256": selection["content_hash"],
                "attachment_pretraining_sha256": attachment["content_hash"],
                "checkpoint_sha256": {
                    name: registrations[name]["checkpoint_sha256"]
                    for name in CONTROL_ORDER
                },
                "teacher_caches": cache_hashes,
                "model_train_and_val_stop_only": True,
                "final_test_access": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(
        root / "registry/retb_stage_a_offline_teacher_bundle.json", bundle
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
