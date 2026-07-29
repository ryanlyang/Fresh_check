#!/usr/bin/env python3
"""Select and lock the complete seven-expert RETB predictor tuple."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    FUSION_CHECKPOINT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PIPELINE_SEEDS,
    build_predictor_cache_index,
    score_frozen_bundle,
    select_joint_predictor_bundle,
    validate_locked_target_coordinate,
    validate_predictor_bundle_selection,
    validate_predictor_candidate,
    validate_predictor_cache_index,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_losses import (  # noqa: E402
    validate_uncertainty_calibration,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    load_predictor_inference_cache,
    predictor_identity_order_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (  # noqa: E402
    PREDICTOR_CAPACITY_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (  # noqa: E402
    validate_materialized_predictor_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_labels(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"identities", "labels"}:
            raise ValueError("joint selector label NPZ fields differ")
        identities = [str(value) for value in payload["identities"].tolist()]
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if (
        not identities
        or len(identities) != len(set(identities))
        or labels.shape != (len(identities),)
        or bool(((labels < 0) | (labels >= 10)).any())
    ):
        raise ValueError("joint selector label population differs")
    return identities, labels


def _classification_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(logits, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int64)
    if values.shape != (len(truth), 10) or not np.isfinite(values).all():
        raise ValueError("joint selector candidate hybrid logits differ")
    shifted = values - values.max(axis=1, keepdims=True)
    accuracy = float((values.argmax(axis=1) == truth).mean())
    cross_entropy = float(
        (
            np.log(np.exp(shifted).sum(axis=1))
            - shifted[np.arange(len(truth)), truth]
        ).mean(dtype=np.float64)
    )
    return accuracy, cross_entropy


def _load_fusion(
    path: Path, coordinate: dict[str, Any], seed: int, *, device: str
):
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256(path)
        != coordinate["fusion_checkpoint_hashes"][str(seed)]
    ):
        raise ValueError("joint selector fusion checkpoint bytes differ")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("contract") != FUSION_CHECKPOINT_CONTRACT
        or payload.get("allocation") != coordinate["allocation"]
        or not isinstance(payload.get("model_state_dict"), dict)
    ):
        raise ValueError("joint selector fusion checkpoint semantics differ")
    model = build_fusion_model(
        "F_TOKEN_TRANSFORMER",
        bank_dimensions={
            expert: int(shape[1])
            for expert, shape in coordinate["allocation"].items()
        },
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "candidate_manifest_paths",
        "coordinate_manifest_paths",
        "inference_manifest_paths",
        "calibration_artifact_paths",
        "capacity_report_paths",
        "materialized_run_paths",
        "fusion_checkpoint_paths",
        "label_npz_paths",
        "label_manifest_paths_by_seed",
        "label_manifest_hashes_by_seed",
        "label_npz_hashes_by_seed",
    }
    if set(configuration) != required:
        raise ValueError("joint selector configuration fields differ")
    candidates = [
        load_hashed_json(Path(path))
        for path in configuration["candidate_manifest_paths"]
    ]
    coordinates = [
        load_hashed_json(Path(path))
        for path in configuration["coordinate_manifest_paths"]
    ]
    for row in candidates:
        validate_predictor_candidate(row)
    for row in coordinates:
        validate_locked_target_coordinate(row)
    if any(row.get("source") != campaign.get("source") for row in [
        *candidates,
        *coordinates,
    ]):
        raise ValueError("joint selector candidate/coordinate source differs")
    step9 = load_hashed_json(
        args.campaign_root / "registry" / "retb_step9_predictor_bundle.json"
    )
    snapshot = source_snapshot(REPO_ROOT)
    cache_index = bind_source(
        build_predictor_cache_index(
            candidates=candidates,
            coordinates=coordinates,
            step9_bundle_sha256=step9["content_hash"],
        ),
        source_snapshot=snapshot,
    )
    validate_predictor_cache_index(
        cache_index, candidates=candidates, coordinates=coordinates
    )
    candidate_map = {row["candidate_id"]: row for row in candidates}
    coordinate_map = {row["coordinate_id"]: row for row in coordinates}
    labels_by_seed, identities_by_seed = {}, {}
    for seed in PIPELINE_SEEDS:
        label_manifest_path = Path(
            configuration["label_manifest_paths_by_seed"].get(
                str(seed),
                configuration["label_manifest_paths_by_seed"].get(seed),
            )
        )
        label_manifest = load_hashed_json(label_manifest_path)
        expected_label_manifest = configuration[
            "label_manifest_hashes_by_seed"
        ].get(
            str(seed),
            configuration["label_manifest_hashes_by_seed"].get(seed),
        )
        if (
            label_manifest["content_hash"] != expected_label_manifest
            or label_manifest.get("source") != campaign.get("source")
        ):
            raise ValueError("joint selector label-manifest lineage differs")
        path = Path(
            configuration["label_npz_paths"].get(
                str(seed), configuration["label_npz_paths"].get(seed)
            )
        )
        expected_label_payload = configuration[
            "label_npz_hashes_by_seed"
        ].get(str(seed), configuration["label_npz_hashes_by_seed"].get(seed))
        if _sha256(path) != expected_label_payload:
            raise ValueError("joint selector label payload bytes differ")
        identities, labels = _load_labels(path)
        identities_by_seed[seed], labels_by_seed[seed] = identities, labels
        if predictor_identity_order_sha256(identities) != cache_index[
            "identity_order_hashes_by_seed"
        ][str(seed)]:
            raise ValueError("joint selector label identity order differs")
    predicted_banks = {}
    for candidate in candidates:
        candidate_paths = configuration["inference_manifest_paths"].get(
            candidate["candidate_id"]
        )
        if candidate_paths is None:
            raise ValueError("joint selector lacks candidate inference paths")
        for seed in PIPELINE_SEEDS:
            path = Path(
                candidate_paths.get(str(seed), candidate_paths.get(seed))
            )
            manifest, arrays = load_predictor_inference_cache(
                path,
                expected_pipeline_seed=seed,
                expected_registration_sha256=candidate["seed_artifacts"][
                    str(seed)
                ]["predictor_registration"],
            )
            expected = candidate["seed_artifacts"][str(seed)]
            if (
                manifest["content_hash"] != expected["inference_manifest"]
                or manifest.get("source") != campaign.get("source")
                or manifest["identity_order_sha256"]
                != expected["identity_order_sha256"]
                or manifest["expert_id"] != candidate["expert_id"]
                or manifest["parents"]["predictor_registration"]
                != expected["predictor_registration"]
                or manifest["parents"]["predictor_checkpoint"]
                != expected["predictor_checkpoint"]
                or [*arrays["predicted_tokens"].shape[1:]]
                != candidate["token_shape"]
                or arrays["identities"].tolist() != identities_by_seed[seed]
            ):
                raise ValueError("joint selector inference cache differs")
            hybrid_accuracy, hybrid_cross_entropy = _classification_metrics(
                arrays["hybrid_logits"], labels_by_seed[seed]
            )
            if (
                abs(
                    hybrid_accuracy
                    - candidate["metrics_by_seed"]["hybrid_accuracy"][
                        str(seed)
                    ]
                )
                > 1.0e-12
                or abs(
                    hybrid_cross_entropy
                    - candidate["metrics_by_seed"][
                        "hybrid_cross_entropy"
                    ][str(seed)]
                )
                > 1.0e-12
            ):
                raise ValueError(
                    "joint selector candidate hybrid metrics differ from cache"
                )
            calibration_path = Path(
                configuration["calibration_artifact_paths"][
                    candidate["candidate_id"]
                ].get(
                    str(seed),
                    configuration["calibration_artifact_paths"][
                        candidate["candidate_id"]
                    ].get(seed),
                )
            )
            calibration = load_hashed_json(calibration_path)
            validate_uncertainty_calibration(calibration)
            if (
                calibration["content_hash"]
                != expected["uncertainty_calibration"]
                or calibration.get("source") != campaign.get("source")
                or calibration["parents"]["predictor_inference_manifest"]
                != manifest["content_hash"]
                or calibration["parents"]["predictor_registration"]
                != expected["predictor_registration"]
                or calibration["parents"]["predictor_checkpoint"]
                != expected["predictor_checkpoint"]
                or abs(
                    float(
                        calibration["coverage_error_curve"][-1][
                            "observed_rmse"
                        ]
                    )
                    - candidate["metrics_by_seed"][
                        "normalized_token_error"
                    ][str(seed)]
                )
                > 1.0e-12
            ):
                raise ValueError("joint selector uncertainty calibration differs")
            capacity_path = Path(
                configuration["capacity_report_paths"][
                    candidate["candidate_id"]
                ].get(
                    str(seed),
                    configuration["capacity_report_paths"][
                        candidate["candidate_id"]
                    ].get(seed),
                )
            )
            capacity = load_hashed_json(
                capacity_path,
                expected_contract=PREDICTOR_CAPACITY_CONTRACT,
            )
            if (
                capacity["content_hash"] != expected["capacity_report"]
                or capacity.get("source") != campaign.get("source")
                or capacity["selected_predictor"]["analytical_flops"]
                != candidate["inference_flops"]
                or capacity["selected_predictor"]["parameter_count"]
                != candidate["parameter_count"]
            ):
                raise ValueError("joint selector predictor capacity differs")
            run_path = Path(
                configuration["materialized_run_paths"][
                    candidate["candidate_id"]
                ].get(
                    str(seed),
                    configuration["materialized_run_paths"][
                        candidate["candidate_id"]
                    ].get(seed),
                )
            )
            run = load_hashed_json(run_path)
            validate_materialized_predictor_run(run)
            if (
                run["content_hash"]
                != candidate["materialized_run_hashes"][str(seed)]
                or run.get("source") != campaign.get("source")
                or run["pipeline_seed"] != seed
                or run["expert_id"] != candidate["expert_id"]
                or run["target_mode"] != candidate["target_mode"]
                or [
                    int(run["token_count"]),
                    int(run["token_dimension"]),
                ]
                != candidate["token_shape"]
            ):
                raise ValueError("joint selector materialized run differs")
            predicted_banks[(candidate["candidate_id"], seed)] = arrays[
                "predicted_tokens"
            ]
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    fusion_by_coordinate_seed = {}
    for coordinate_id, coordinate in coordinate_map.items():
        paths = configuration["fusion_checkpoint_paths"].get(coordinate_id)
        if paths is None:
            raise ValueError("joint selector lacks coordinate fusion paths")
        for seed in PIPELINE_SEEDS:
            fusion_by_coordinate_seed[(coordinate_id, seed)] = _load_fusion(
                Path(paths.get(str(seed), paths.get(seed))),
                coordinate,
                seed,
                device=resolved_device,
            )

    def score_tuple(names):
        coordinate_id = candidate_map[names[0]]["coordinate_id"]
        return score_frozen_bundle(
            names,
            candidates_by_id=candidate_map,
            predicted_banks=predicted_banks,
            labels_by_seed=labels_by_seed,
            fusion_by_seed={
                seed: fusion_by_coordinate_seed[(coordinate_id, seed)]
                for seed in PIPELINE_SEEDS
            },
        )

    result = select_joint_predictor_bundle(
        candidates=candidates,
        coordinates=coordinates,
        score_tuple=score_tuple,
        predictor_cache_index_sha256=cache_index["content_hash"],
        label_manifest_hashes_by_seed=configuration[
            "label_manifest_hashes_by_seed"
        ],
        label_payload_hashes_by_seed=configuration[
            "label_npz_hashes_by_seed"
        ],
        source_snapshot=snapshot,
    )
    validate_predictor_bundle_selection(result)
    summary = {
        "dry_run": args.dry_run,
        "selected_tuple": result["search"]["selected_tuple"],
        "coordinate_id": result["predictor_bundle_lock"]["coordinate_id"],
        "predictor_bundle_lock_sha256": result["predictor_bundle_lock"][
            "content_hash"
        ],
    }
    if not args.dry_run:
        publications = {
            "cache_index": write_immutable_json(
                args.output_dir / "predictor_cache_index.json", cache_index
            ),
            "policy": write_immutable_json(
                args.output_dir / "bundle_search_policy.json", result["policy"]
            ),
            "search": write_immutable_json(
                args.output_dir / "bundle_search.json", result["search"]
            ),
            "lock": write_immutable_json(
                args.output_dir / "predictor_bundle_lock.json",
                result["predictor_bundle_lock"],
            ),
        }
        summary["publications"] = publications
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
