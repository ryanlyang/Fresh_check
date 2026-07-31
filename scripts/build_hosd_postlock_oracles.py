#!/usr/bin/env python3
"""Run post-lock offline-teacher diagnostics on the sealed stack population."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_postlock_oracle_manifest,
    infer_teacher_batch,
    load_and_validate_campaign,
    validate_teacher_lock,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FINALIST_LOCK_CONTRACT,
    STACK_PREDICTION_MANIFEST_CONTRACT,
    TEACHER_LOCK_CONTRACT,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _factory(locator: str):
    module_name, separator, function_name = locator.partition(":")
    if not separator:
        raise ValueError("--adapter-factory must be module:function")
    value = getattr(importlib.import_module(module_name), function_name)
    if not callable(value):
        raise TypeError("post-lock teacher adapter factory is not callable")
    return value


def _teacher_logits(
    *,
    teacher_id: str,
    teacher_lock,
    config_path: Path,
    adapter_factory: str,
    batch_size: int,
) -> tuple[tuple[str, ...], np.ndarray, str]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    adapter, identities, batch_provider = _factory(adapter_factory)(
        teacher_id=teacher_id,
        teacher_lock=teacher_lock,
        config=config,
    )
    identities = tuple(str(value) for value in identities)
    if (
        not identities
        or identities != tuple(sorted(identities))
        or len(identities) != len(set(identities))
    ):
        raise ValueError("post-lock teacher identity population differs")
    chunks = []
    key = f"T_OFFLINE_LOGITS_{teacher_id}"
    for start in range(0, len(identities), int(batch_size)):
        indices = np.arange(
            start,
            min(start + int(batch_size), len(identities)),
            dtype=np.int64,
        )
        batch = batch_provider(indices)
        if any(
            name.lower() in {"label", "labels", "class", "classes", "y"}
            for name in batch
        ):
            raise ValueError("post-lock teacher inference exposed labels")
        chunks.append(infer_teacher_batch(adapter, batch)[key].values.numpy())
    values = np.concatenate(chunks).astype(np.float32, copy=False)
    if values.shape != (len(identities), 10) or not np.isfinite(values).all():
        raise ValueError("post-lock teacher logits differ")
    return identities, values, hashlib.sha256(config_bytes).hexdigest()


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(
        logits, axis=1, keepdims=True
    )
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _agreement(student: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    student_probability = _probabilities(student)
    teacher_probability = _probabilities(teacher)
    epsilon = np.finfo(np.float64).tiny
    kl = np.sum(
        teacher_probability
        * (
            np.log(np.maximum(teacher_probability, epsilon))
            - np.log(np.maximum(student_probability, epsilon))
        ),
        axis=1,
    )
    cosine = np.sum(
        student_probability * teacher_probability, axis=1
    ) / (
        np.linalg.norm(student_probability, axis=1)
        * np.linalg.norm(teacher_probability, axis=1)
    ).clip(min=epsilon)
    return {
        "top1_agreement": float(
            np.mean(
                np.argmax(student, axis=1)
                == np.argmax(teacher, axis=1)
            )
        ),
        "teacher_to_student_kl": float(np.mean(kl)),
        "probability_cosine": float(np.mean(cosine)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--teacher-lock", required=True, type=Path)
    parser.add_argument(
        "--adapter-config-o-base", required=True, type=Path
    )
    parser.add_argument(
        "--adapter-config-o-fullrel", required=True, type=Path
    )
    parser.add_argument(
        "--adapter-factory",
        default=(
            "teacher_logit_reco.hlt_offline_structure_distillation."
            "teacher_inference_runtime:build_label_blind_relational_adapter"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("post-lock batch size must be positive")
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    for resource in ("locked_hosd_finalists", "stack_val_offline"):
        authorize_access(
            worker_role="postlock_oracle_diagnostic",
            requested_resource=resource,
        )
    lock = load_hashed_json(
        root / "selection" / "locked_hosd_finalists.json",
        expected_contract=FINALIST_LOCK_CONTRACT,
    )
    teacher_lock = load_hashed_json(
        args.teacher_lock, expected_contract=TEACHER_LOCK_CONTRACT
    )
    validate_teacher_lock(
        teacher_lock,
        source=campaign["source"],
        verify_checkpoint_bytes=True,
    )
    teacher_values = {}
    identity_population = None
    config_hashes = {}
    for teacher_id, config_path in (
        ("O_BASE", args.adapter_config_o_base),
        ("O_FULLREL", args.adapter_config_o_fullrel),
    ):
        identities, logits, config_hash = _teacher_logits(
            teacher_id=teacher_id,
            teacher_lock=teacher_lock,
            config_path=config_path.resolve(),
            adapter_factory=args.adapter_factory,
            batch_size=args.batch_size,
        )
        if identity_population is None:
            identity_population = identities
        elif identities != identity_population:
            raise ValueError("post-lock teacher populations differ")
        teacher_values[teacher_id] = logits
        config_hashes[teacher_id] = config_hash
    assert identity_population is not None
    rows = []
    for graph_id in lock["unique_finalist_graph_ids"]:
        evidence = lock["locked_selection_artifacts"][graph_id]
        for ordinal, seed in enumerate((202, 303, 404)):
            prediction_path = (
                root
                / "selection_predictions"
                / "stack_val"
                / "rows"
                / f"{graph_id}__seed_{seed}.json"
            )
            prediction = load_hashed_json(
                prediction_path,
                expected_contract=STACK_PREDICTION_MANIFEST_CONTRACT,
            )
            if (
                prediction["content_hash"]
                != evidence["prediction_hashes"][ordinal]
                or tuple(prediction["identities"]) != identity_population
                or prediction.get("source") != campaign["source"]
                or prediction.get("contains_labels")
                or prediction.get("contains_targets")
            ):
                raise ValueError(
                    "post-lock finalist prediction lineage differs"
                )
            student = np.asarray(prediction["logits"], dtype=np.float32)
            rows.append(
                {
                    "graph_id": graph_id,
                    "seed": seed,
                    "prediction_sha256": prediction["content_hash"],
                    "teacher_agreement": {
                        teacher_id: _agreement(student, logits)
                        for teacher_id, logits in sorted(
                            teacher_values.items()
                        )
                    },
                }
            )
    diagnostic = with_content_hash(
        {
            "contract": "hosd_postlock_teacher_agreement_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "finalist_lock_sha256": lock["content_hash"],
            "teacher_lock_sha256": teacher_lock["content_hash"],
            "adapter_config_hashes": config_hashes,
            "split": "stack_val",
            "identity_order_sha256": canonical_sha256(
                list(identity_population)
            ),
            "event_count": len(identity_population),
            "finalist_graph_ids": sorted(
                lock["unique_finalist_graph_ids"]
            ),
            "seed_order": [202, 303, 404],
            "rows": rows,
            "row_count": len(rows),
            "finalist_seed_coverage_exact": True,
            "offline_teacher_inference_fp32": True,
            "labels_consumed": False,
            "oracle_outputs_persisted": False,
            "selection_eligible": False,
        }
    )
    diagnostic_path = (
        root
        / "postlock_oracle_diagnostics"
        / "stack_val_teacher_agreement.json"
    )
    write_immutable_json(diagnostic_path, diagnostic)
    artifact = build_postlock_oracle_manifest(
        finalist_lock=lock,
        diagnostic_hashes={
            "stack_val_offline_teacher_agreement": diagnostic[
                "content_hash"
            ]
        },
        source=campaign["source"],
    )
    output = args.output or (
        root / "postlock_oracle_diagnostics" / "completion.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "diagnostic_sha256": diagnostic["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
