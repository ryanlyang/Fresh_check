#!/usr/bin/env python3
"""Materialize complete H candidates and run the joint predictor beam selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_retb_joint_predictor_bundle import main as select_main  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    phase_plan_path,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    build_locked_target_coordinate,
    build_predictor_candidate,
    select_carried_predictor_bundles,
    shared_predictor_configuration_id,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    STAGE_E_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    load_predictor_inference_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


BUNDLE_INPUT_INDEX_CONTRACT = "retb_predictor_bundle_input_index_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError("predictor bundle label payload differs")
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _classification(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(logits, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    return (
        float((values.argmax(axis=1) == truth).mean()),
        float(
            (
                np.log(np.exp(shifted).sum(axis=1))
                - shifted[np.arange(len(truth)), truth]
            ).mean(dtype=np.float64)
        ),
    )


def _coordinate_index(
    selection: Mapping[str, Any], *, expert: str, mode: str
) -> int:
    position = EXPERT_ORDER.index(expert)
    candidates = [
        (index, row)
        for index, row in enumerate(selection["locked_coordinate_systems"])
        if row["target_tuple"][position] == mode
    ]
    homogeneous = [
        pair for pair in candidates if len(set(pair[1]["target_tuple"])) == 1
    ]
    retained = homogeneous or candidates
    if not retained:
        raise ValueError("predictor candidate has no locked coordinate")
    return min(
        retained, key=lambda pair: pair[1]["coordinate_contract_sha256"]
    )[0]


def _group_key(run: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        run["expert_id"],
        run["shape_id"],
        run["target_mode"],
        run["architecture"],
        run["context"],
        run["objective_id"],
        run["uncertainty_head"],
        run["normalization_mode"],
        float(run["learning_rate"]),
        float(run["dropout"]),
        run["hlt_evidence_mode"],
        run["control_variant"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    selection = load_hashed_json(
        args.campaign_root / "selection" / "locked_bridge_coordinates.json"
    )
    plan = load_hashed_json(
        phase_plan_path(
            args.campaign_root,
            controller_id="predictor_campaign",
            phase_id="H_CONFIRMATION",
        )
    )
    grouped: dict[tuple[Any, ...], dict[int, tuple[Path, dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for row in plan["rows"]:
        run_path = Path(row["argv"][row["argv"].index("--run") + 1])
        run = load_hashed_json(run_path)
        if run["control_variant"] != "STANDARD":
            continue
        seed = int(run["pipeline_seed"])
        key = _group_key(run)
        if seed in grouped[key]:
            raise ValueError("predictor bundle candidate seed is duplicated")
        grouped[key][seed] = (run_path, run)
    if not grouped or any(set(rows) != {101, 202, 303} for rows in grouped.values()):
        raise ValueError("predictor bundle candidate seed coverage differs")

    snapshot = source_snapshot(REPO_ROOT)
    candidate_paths = []
    inference_paths: dict[str, dict[str, str]] = {}
    calibration_paths: dict[str, dict[str, str]] = {}
    capacity_paths: dict[str, dict[str, str]] = {}
    run_paths: dict[str, dict[str, str]] = {}
    label_rows: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    used_coordinates: set[tuple[int, str]] = set()
    for key, seed_rows in sorted(grouped.items(), key=lambda item: item[0]):
        (
            expert,
            shape,
            target_mode,
            architecture,
            context,
            objective,
            uncertainty,
            normalization,
            learning_rate,
            dropout,
            evidence_mode,
            _,
        ) = key
        coordinate_index = _coordinate_index(
            selection, expert=expert, mode=target_mode
        )
        coordinate_id = f"coordinate_{coordinate_index:03d}:{shape}"
        used_coordinates.add((coordinate_index, shape))
        candidate_id = "candidate_" + canonical_sha256(
            {"coordinate_id": coordinate_id, "configuration": list(key[:-1])}
        )[:24]
        seed_artifacts, run_hashes = {}, {}
        errors, accuracies, cross_entropies = {}, {}, {}
        inference_paths[candidate_id] = {}
        calibration_paths[candidate_id] = {}
        capacity_paths[candidate_id] = {}
        run_paths[candidate_id] = {}
        flops = parameters = None
        token_count = token_dimension = None
        for seed, (run_path, run) in sorted(seed_rows.items()):
            root = args.campaign_root / "runs" / "predictors" / run["run_id"]
            registration = load_hashed_json(
                root / "training" / "worker_registration.json"
            )
            inference_path = (
                root / "val_design" / "predictor_outputs_manifest.json"
            )
            inference, arrays = load_predictor_inference_cache(
                inference_path,
                expected_pipeline_seed=seed,
                expected_registration_sha256=registration["content_hash"],
            )
            calibration_path = (
                root / "val_design" / "uncertainty_calibration.json"
            )
            calibration = load_hashed_json(calibration_path)
            capacity_path = root / "training" / "capacity_report.json"
            capacity = load_hashed_json(capacity_path)
            metrics = load_hashed_json(
                root / "val_design" / "val_design_metrics.json"
            )
            with np.load(
                root / "prepared" / "val_design.npz",
                allow_pickle=False,
            ) as prepared:
                prepared_identities = np.asarray(
                    prepared["identities"]
                ).astype(str)
                labels = np.asarray(prepared["labels"], dtype=np.int64)
            identities = np.asarray(arrays["identities"]).astype(str)
            if not np.array_equal(identities, prepared_identities):
                raise ValueError(
                    "predictor inference/prepared identity order differs"
                )
            accuracy, cross_entropy = _classification(
                arrays["hybrid_logits"], labels
            )
            observed = float(
                calibration["coverage_error_curve"][-1]["observed_rmse"]
            )
            if (
                abs(accuracy - float(metrics["val_design_accuracy"])) > 1e-12
                or abs(cross_entropy - float(metrics["cross_entropy"])) > 1e-12
                or abs(observed - float(metrics["normalized_token_error"]))
                > 1e-12
            ):
                raise ValueError("predictor bundle candidate metrics differ")
            current_flops = int(
                capacity["selected_predictor"]["analytical_flops"]
            )
            current_parameters = int(
                capacity["selected_predictor"]["parameter_count"]
            )
            if flops is None:
                flops, parameters = current_flops, current_parameters
                token_count = int(run["token_count"])
                token_dimension = int(run["token_dimension"])
            elif (
                flops != current_flops
                or parameters != current_parameters
                or token_count != int(run["token_count"])
                or token_dimension != int(run["token_dimension"])
            ):
                raise ValueError("predictor candidate capacity drifts by seed")
            seed_artifacts[seed] = {
                "predictor_registration": registration["content_hash"],
                "predictor_checkpoint": registration["checkpoint_sha256"],
                "inference_manifest": inference["content_hash"],
                "uncertainty_calibration": calibration["content_hash"],
                "capacity_report": capacity["content_hash"],
                "identity_order_sha256": inference["identity_order_sha256"],
            }
            run_hashes[seed] = run["content_hash"]
            errors[seed], accuracies[seed], cross_entropies[seed] = (
                observed,
                accuracy,
                cross_entropy,
            )
            inference_paths[candidate_id][str(seed)] = str(inference_path)
            calibration_paths[candidate_id][str(seed)] = str(calibration_path)
            capacity_paths[candidate_id][str(seed)] = str(capacity_path)
            run_paths[candidate_id][str(seed)] = str(run_path)
            order = inference["identity_order_sha256"]
            if seed in label_rows:
                old_ids, old_labels, old_order = label_rows[seed]
                if (
                    not np.array_equal(old_ids, identities)
                    or not np.array_equal(old_labels, labels)
                    or old_order != order
                ):
                    raise ValueError("predictor bundle labels differ by candidate")
            else:
                label_rows[seed] = (identities, labels, order)
        candidate = bind_source(
            build_predictor_candidate(
                candidate_id=candidate_id,
                expert_id=expert,
                coordinate_id=coordinate_id,
                target_mode=target_mode,
                shape_id=shape,
                token_count=token_count,
                token_dimension=token_dimension,
                architecture=architecture,
                context=context,
                objective_id=objective,
                uncertainty_head=uncertainty,
                normalization_mode=normalization,
                learning_rate=learning_rate,
                dropout=dropout,
                hlt_evidence_mode=evidence_mode,
                shared_configuration_id=shared_predictor_configuration_id(
                    architecture=architecture,
                    context=context,
                    objective_id=objective,
                    uncertainty_head=uncertainty,
                    normalization_mode=normalization,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    hlt_evidence_mode=evidence_mode,
                ),
                normalized_token_error_by_seed=errors,
                hybrid_accuracy_by_seed=accuracies,
                hybrid_cross_entropy_by_seed=cross_entropies,
                inference_flops=flops,
                parameter_count=parameters,
                seed_artifacts=seed_artifacts,
                materialized_run_hashes=run_hashes,
            ),
            source_snapshot=snapshot,
        )
        path = args.output_dir / "inputs" / "candidates" / f"{candidate_id}.json"
        write_immutable_json(path, candidate)
        candidate_paths.append(str(path))

    coordinate_paths = []
    fusion_paths: dict[str, dict[str, str]] = {}
    for coordinate_index, shape in sorted(used_coordinates):
        system = selection["locked_coordinate_systems"][coordinate_index]
        coordinate_id = f"coordinate_{coordinate_index:03d}:{shape}"
        hashes, registrations, paths = {}, {}, {}
        allocation = None
        for seed in (101, 202, 303):
            root = (
                args.campaign_root
                / "inputs"
                / "target_caches"
                / f"coordinate_{coordinate_index:03d}"
                / shape
                / f"seed_{seed}"
                / "model_train"
            )
            manifest = load_hashed_json(root / "target_cache_manifest.json")
            registration = load_hashed_json(root / "fusion_registration.json")
            if allocation is None:
                allocation = manifest["allocation"]
            elif allocation != manifest["allocation"]:
                raise ValueError("coordinate allocation drifts by seed")
            hashes[seed] = registration["checkpoint_sha256"]
            registrations[seed] = registration["content_hash"]
            paths[str(seed)] = registration["checkpoint_path"]
        coordinate = bind_source(
            build_locked_target_coordinate(
                coordinate_id=coordinate_id,
                target_modes=dict(
                    zip(
                        EXPERT_ORDER,
                        system["target_tuple"],
                        strict=True,
                    )
                ),
                allocation=allocation,
                fusion_checkpoint_hashes=hashes,
                fusion_registration_hashes=registrations,
                stage_e_coordinate_sha256=system[
                    "coordinate_contract_sha256"
                ],
            ),
            source_snapshot=snapshot,
        )
        path = (
            args.output_dir
            / "inputs"
            / "coordinates"
            / f"{canonical_sha256(coordinate_id)[:24]}.json"
        )
        write_immutable_json(path, coordinate)
        coordinate_paths.append(str(path))
        fusion_paths[coordinate_id] = paths

    label_npz_paths, label_npz_hashes = {}, {}
    label_manifest_paths, label_manifest_hashes = {}, {}
    canonical_seed = min(label_rows)
    identities, labels, order = label_rows[canonical_seed]
    if any(
        tuple(other_ids.tolist()) != tuple(identities.tolist())
        or not np.array_equal(other_labels, labels)
        or other_order != order
        for other_ids, other_labels, other_order in label_rows.values()
    ):
        raise ValueError(
            "val_design label population unexpectedly differs by pipeline seed"
        )
    npz_path = args.output_dir / "inputs" / "labels" / "val_design.npz"
    npz_sha = _publish_npz(
        npz_path, {"identities": identities, "labels": labels}
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": "retb_predictor_bundle_label_manifest_v2",
                "schema_version": 2,
                "logical_role": "val_design",
                "pipeline_seed_independent": True,
                "applicable_pipeline_seeds": list(PIPELINE_SEEDS),
                "event_count": len(labels),
                "identity_order_sha256": order,
                "label_npz_sha256": npz_sha,
            }
        ),
        source_snapshot=snapshot,
    )
    manifest_path = npz_path.with_suffix(".json")
    write_immutable_json(manifest_path, manifest)
    for seed in PIPELINE_SEEDS:
        label_npz_paths[str(seed)] = str(npz_path)
        label_npz_hashes[str(seed)] = npz_sha
        label_manifest_paths[str(seed)] = str(manifest_path)
        label_manifest_hashes[str(seed)] = manifest["content_hash"]

    configuration = {
        "candidate_manifest_paths": candidate_paths,
        "coordinate_manifest_paths": coordinate_paths,
        "inference_manifest_paths": inference_paths,
        "calibration_artifact_paths": calibration_paths,
        "capacity_report_paths": capacity_paths,
        "materialized_run_paths": run_paths,
        "fusion_checkpoint_paths": fusion_paths,
        "label_npz_paths": label_npz_paths,
        "label_manifest_paths_by_seed": label_manifest_paths,
        "label_manifest_hashes_by_seed": label_manifest_hashes,
        "label_npz_hashes_by_seed": label_npz_hashes,
    }
    configuration_path = args.output_dir / "inputs" / "selector_configuration.json"
    configuration_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    if configuration_path.exists():
        if configuration_path.read_text("utf-8") != encoded:
            raise FileExistsError("predictor selector configuration differs")
    else:
        configuration_path.write_text(encoded, encoding="utf-8")
    index = bind_source(
        with_content_hash(
            {
                "contract": BUNDLE_INPUT_INDEX_CONTRACT,
                "schema_version": 1,
                "candidate_count": len(candidate_paths),
                "coordinate_count": len(coordinate_paths),
                "candidate_manifest_hashes": [
                    load_hashed_json(path)["content_hash"]
                    for path in candidate_paths
                ],
                "coordinate_manifest_hashes": [
                    load_hashed_json(path)["content_hash"]
                    for path in coordinate_paths
                ],
                "selector_configuration_sha256": _sha256(configuration_path),
                "predictor_phase_plan_sha256": plan["content_hash"],
                "scientific_underperformance_blocks_continuation": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(args.output_dir / "bundle_input_index.json", index)
    select_main(
        [
            "--campaign-root",
            str(args.campaign_root),
            "--configuration",
            str(configuration_path),
            "--output-dir",
            str(args.output_dir),
            "--device",
            args.device,
        ]
    )
    search = load_hashed_json(args.output_dir / "bundle_search.json")
    carried = select_carried_predictor_bundles(
        search=search,
        candidates=[load_hashed_json(path) for path in candidate_paths],
        coordinates=[load_hashed_json(path) for path in coordinate_paths],
        carried_shape_roles=STAGE_E_SHAPES,
        source_snapshot=snapshot,
    )
    carried_root = args.output_dir / "carried"
    for role, lock in carried["locks"].items():
        write_immutable_json(carried_root / f"{role}.json", lock)
    write_immutable_json(
        args.output_dir / "carried_predictor_bundle_index.json",
        carried["index"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
