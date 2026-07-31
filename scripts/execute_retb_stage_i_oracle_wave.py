#!/usr/bin/env python3
"""Prepare and evaluate every locked RETB Stage-I substitution seed."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_retb_stage_i_substitutions import main as evaluate_main  # noqa: E402
from scripts.train_retb_native_hlt_fusion import _ensure_cache  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.native_fusion import (  # noqa: E402
    build_native_fusion_model,
    load_native_fusion_cache,
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
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    load_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


STAGE_I_INPUT_INDEX_CONTRACT = "retb_stage_i_input_index_v1"
IDENTITY_PROJECTED_HLT_CONTRACT = (
    "retb_stage_i_identity_projected_hlt_cache_v1"
)
NO_RECONSTRUCTION_PREDICTION_CONTRACT = (
    "retb_stage_i_no_reconstruction_prediction_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable Stage-I payload differs: {path}")
    else:
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _value(mapping: Mapping[str, Any], key: int) -> Any:
    return mapping.get(str(key), mapping.get(key))


def _native_fusion_row(
    confirmation: Mapping[str, Any], *, shape: str, seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in confirmation["rows"]
        if row["component"] == "NATIVE_HLT_FUSION"
        and int(row["seed"]) == seed
        and row["configuration"]["shape_id"] == shape
        and row["configuration"]["fusion_variant"] == "HF_NATIVE"
    ]
    if len(rows) != 1:
        raise ValueError("Stage-I native HLT fusion parent differs")
    return rows[0]


def _native_logits(
    *,
    root: Path,
    registry: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    shape: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    row = _native_fusion_row(confirmation, shape=shape, seed=seed)
    run_id = str(row["run_id"])
    cache_root = (
        root
        / "inputs"
        / "selected_native_fusion"
        / shape
        / "HF_NATIVE"
        / f"seed_{seed}"
    )
    manifest_path = cache_root / "val_design_native_hlt_tokens.json"
    _ensure_cache(
        path=manifest_path,
        root=root,
        registry=dict(registry),
        confirmation=dict(confirmation),
        shape=shape,
        seed=seed,
        split="val_design",
    )
    manifest, arrays = load_native_fusion_cache(manifest_path)
    output = (
        root
        / "runs"
        / "stage_d"
        / "native_fusions"
        / run_id
        / f"seed_{seed}"
    )
    registration = load_hashed_json(output / "fusion_registration.json")
    checkpoint_path = output / "best_model_val.pt"
    if _sha256(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("Stage-I native fusion checkpoint bytes differ")
    model = build_native_fusion_model(
        "HF_NATIVE",
        bank_dimensions={
            expert: int(manifest["allocation"][expert][1])
            for expert in EXPERT_ORDER
        },
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, manifest["event_count"], 512):
            stop = min(start + 512, manifest["event_count"])
            banks = {
                expert: torch.from_numpy(
                    arrays[f"tokens_r0_{expert}"][start:stop]
                ).to(device)
                for expert in EXPERT_ORDER
            }
            expert_logits = {
                expert: torch.from_numpy(
                    arrays[f"logits_r0_{expert}"][start:stop]
                ).to(device)
                for expert in EXPERT_ORDER
            }
            pieces.append(
                model(
                    token_banks=banks, expert_logits=expert_logits
                )
                .float()
                .cpu()
                .numpy()
            )
    return (
        np.asarray(arrays["identities"]),
        np.asarray(arrays["labels"], dtype=np.int64),
        np.concatenate(pieces).astype(np.float32),
        manifest,
        registration,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--bundle-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    lock = load_hashed_json(args.bundle_lock)
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    confirmation = load_hashed_json(
        args.campaign_root
        / "selection"
        / "predictor_phases"
        / "stage_d_evidence_confirmations.json"
    )
    selector_config = json.loads(
        (
            args.campaign_root
            / "selection"
            / "predictor_bundle"
            / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    coordinate_text = str(lock["coordinate_id"])
    coordinate_name, shape = coordinate_text.split(":", 1)
    coordinate_index = int(coordinate_name.rsplit("_", 1)[1])
    policy = (
        args.campaign_root
        / "registry"
        / "retb_stage_i_oracle_substitution_policy.json"
    )
    resolved = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    snapshot = source_snapshot(REPO_ROOT)
    records = []
    for seed in (101, 202, 303):
        seed_root = args.output_dir / f"seed_{seed}"
        target_root = (
            args.campaign_root
            / "inputs"
            / "target_caches"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
            / "val_design"
        )
        target_manifest_path = target_root / "target_cache_manifest.json"
        target_spec = load_hashed_json(
            target_root / "target_cache_specification.json"
        )
        target_manifest, target = load_offline_target_cache(
            target_manifest_path,
            expected_pipeline_seed=seed,
            expected_specification_sha256=target_spec["content_hash"],
        )
        fusion_registration = load_hashed_json(
            target_root / "fusion_registration.json"
        )
        predicted, predicted_paths = {}, {}
        target_identities = None
        for expert in EXPERT_ORDER:
            candidate_id = lock["selected_candidate_descriptors"][expert][
                "candidate_id"
            ]
            path = Path(
                _value(
                    selector_config["inference_manifest_paths"][
                        candidate_id
                    ],
                    seed,
                )
            )
            manifest, arrays = load_predictor_inference_cache(
                path,
                expected_pipeline_seed=seed,
                expected_registration_sha256=lock[
                    "seed_specific_artifacts"
                ][str(seed)][expert]["predictor_registration"],
            )
            if (
                manifest["content_hash"]
                != lock["seed_specific_artifacts"][str(seed)][expert][
                    "inference_manifest"
                ]
            ):
                raise ValueError("Stage-I predicted-cache lock differs")
            predicted[expert] = arrays["predicted_tokens"]
            current_identities = np.asarray(arrays["identities"])
            if target_identities is None:
                target_identities = current_identities
            elif not np.array_equal(
                target_identities, current_identities
            ):
                raise ValueError(
                    "Stage-I predicted-cache identity orders differ"
                )
            predicted_paths[expert] = str(path)
        assert target_identities is not None

        identity_hlt, evidence_hashes = {}, {}
        for expert in EXPERT_ORDER:
            mode = lock["selected_candidate_descriptors"][expert][
                "configuration"
            ]["hlt_evidence_mode"]
            evidence_path = (
                args.campaign_root
                / "inputs"
                / "selected_hlt_evidence"
                / shape
                / mode
                / f"seed_{seed}"
                / "val_design_evidence.npz"
            )
            with np.load(evidence_path, allow_pickle=False) as evidence:
                if not np.array_equal(
                    evidence["identities"], target_identities
                ):
                    raise ValueError(
                        "Stage-I HLT/target identity order differs"
                    )
                identity_hlt[expert] = np.asarray(
                    evidence[f"hlt_tokens_{expert}"], dtype=np.float32
                )
            evidence_hashes[expert] = _sha256(evidence_path)
        identity_manifest = bind_source(
            with_content_hash(
                {
                    "contract": IDENTITY_PROJECTED_HLT_CONTRACT,
                    "schema_version": 1,
                    "pipeline_seed": seed,
                    "shape_id": shape,
                    "identity_manifest_sha256": target_manifest[
                        "identity_manifest_sha256"
                    ],
                    "expert_payload_hashes": evidence_hashes,
                    "learned_or_statistical_projection_applied": False,
                }
            ),
            source_snapshot=snapshot,
        )
        identity_manifest_path = seed_root / "identity_projected_hlt.json"
        write_immutable_json(identity_manifest_path, identity_manifest)

        native_ids, native_labels, native_logits, native_cache, native_reg = (
            _native_logits(
                root=args.campaign_root,
                registry=registry,
                confirmation=confirmation,
                shape=shape,
                seed=seed,
                device=resolved,
            )
        )
        if not np.array_equal(native_ids, target_identities) or not np.array_equal(
            native_labels, target["labels"]
        ):
            raise ValueError("Stage-I native baseline population differs")
        no_reco_npz = seed_root / "no_reconstruction_predictions.npz"
        no_reco_npz_sha = _publish_npz(
            no_reco_npz,
            {
                "identities": native_ids,
                "logits": native_logits,
            },
        )
        no_reco_manifest = bind_source(
            with_content_hash(
                {
                    "contract": NO_RECONSTRUCTION_PREDICTION_CONTRACT,
                    "schema_version": 1,
                    "pipeline_seed": seed,
                    "split": "val_design",
                    "native_fusion_registration_sha256": native_reg[
                        "content_hash"
                    ],
                    "native_fusion_cache_sha256": native_cache[
                        "content_hash"
                    ],
                    "prediction_npz_sha256": no_reco_npz_sha,
                    "offline_inputs_or_targets_consumed": False,
                }
            ),
            source_snapshot=snapshot,
        )
        no_reco_manifest_path = seed_root / "no_reconstruction.json"
        write_immutable_json(no_reco_manifest_path, no_reco_manifest)

        arrays = {
            "identities": target_identities,
            "labels": np.asarray(target["labels"], dtype=np.int64),
            "no_reconstruction_logits": native_logits,
        }
        for expert in EXPERT_ORDER:
            arrays[f"predicted_{expert}"] = predicted[expert]
            arrays[f"oracle_{expert}"] = target["tokens"][expert]
            arrays[f"identity_hlt_{expert}"] = identity_hlt[expert]
        input_path = seed_root / "stage_i_inputs.npz"
        input_sha = _publish_npz(input_path, arrays)
        configuration = {
            "pipeline_seed": seed,
            "input_npz_sha256": input_sha,
            "identity_manifest_sha256": target_manifest[
                "identity_manifest_sha256"
            ],
            "label_manifest_sha256": lock["selection_data_hashes"][
                "label_manifests"
            ][str(seed)],
            "oracle_target_cache_sha256": target_manifest["content_hash"],
            "hlt_cache_sha256": native_cache["content_hash"],
            "identity_projected_hlt_cache_sha256": identity_manifest[
                "content_hash"
            ],
            "identity_projected_hlt_cache_manifest_path": str(
                identity_manifest_path
            ),
            "no_reconstruction_prediction_sha256": no_reco_manifest[
                "content_hash"
            ],
            "no_reconstruction_prediction_manifest_path": str(
                no_reco_manifest_path
            ),
            "predicted_cache_hashes": {
                expert: lock["seed_specific_artifacts"][str(seed)][expert][
                    "inference_manifest"
                ]
                for expert in EXPERT_ORDER
            },
            "predicted_cache_manifest_paths": predicted_paths,
            "oracle_target_cache_specification_sha256": target_spec[
                "content_hash"
            ],
        }
        config_path = seed_root / "configuration.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(configuration, indent=2, sort_keys=True) + "\n"
        if config_path.exists():
            if config_path.read_text("utf-8") != encoded:
                raise FileExistsError(
                    "immutable Stage-I configuration differs"
                )
        else:
            config_path.write_text(encoded, encoding="utf-8")
        output = seed_root / "evaluation.json"
        evaluate_main(
            [
                "--campaign-root",
                str(args.campaign_root),
                "--bundle-lock",
                str(args.bundle_lock),
                "--stage-i-policy",
                str(policy),
                "--input-npz",
                str(input_path),
                "--configuration",
                str(config_path),
                "--fusion-checkpoint",
                str(fusion_registration["checkpoint_path"]),
                "--oracle-target-cache",
                str(target_manifest_path),
                "--output",
                str(output),
                "--device",
                str(resolved),
            ]
        )
        records.append(
            {
                "pipeline_seed": seed,
                "evaluation_sha256": load_hashed_json(output)[
                    "content_hash"
                ],
                "input_npz_sha256": input_sha,
                "configuration_sha256": _sha256(config_path),
            }
        )
    index = bind_source(
        with_content_hash(
            {
                "contract": STAGE_I_INPUT_INDEX_CONTRACT,
                "schema_version": 1,
                "predictor_bundle_lock_sha256": lock["content_hash"],
                "shape_id": shape,
                "pipeline_seed_records": records,
                "complete_seed_coverage": True,
                "selection_use_permitted": False,
                "scientific_underperformance_blocks_continuation": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(args.output_dir / "stage_i_index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
