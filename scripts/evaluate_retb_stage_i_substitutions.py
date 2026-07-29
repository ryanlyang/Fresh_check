#!/usr/bin/env python3
"""Evaluate locked RETB Stage-I oracle substitutions on val_design only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

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
from teacher_logit_reco.relation_expert_token_bridge.oracle_substitutions import (  # noqa: E402
    evaluate_stage_i_substitutions,
    validate_stage_i_evaluation,
    validate_stage_i_policy,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    load_predictor_inference_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    identity_order_sha256,
    load_offline_target_cache,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--bundle-lock", required=True, type=Path)
    parser.add_argument("--stage-i-policy", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--oracle-target-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    lock = load_hashed_json(
        args.bundle_lock, expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT
    )
    policy = load_hashed_json(args.stage_i_policy)
    validate_stage_i_policy(policy)
    if lock.get("source") != campaign.get("source") or policy.get(
        "source"
    ) != campaign.get("source"):
        raise ValueError("Stage-I source lineage differs")
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "pipeline_seed",
        "input_npz_sha256",
        "identity_manifest_sha256",
        "label_manifest_sha256",
        "oracle_target_cache_sha256",
        "hlt_cache_sha256",
        "identity_projected_hlt_cache_sha256",
        "identity_projected_hlt_cache_manifest_path",
        "no_reconstruction_prediction_sha256",
        "no_reconstruction_prediction_manifest_path",
        "predicted_cache_hashes",
        "predicted_cache_manifest_paths",
        "oracle_target_cache_specification_sha256",
    }
    if set(configuration) != required:
        raise ValueError("Stage-I configuration fields differ")
    seed = int(configuration["pipeline_seed"])
    for hash_name, path_name in (
        (
            "identity_projected_hlt_cache_sha256",
            "identity_projected_hlt_cache_manifest_path",
        ),
        (
            "no_reconstruction_prediction_sha256",
            "no_reconstruction_prediction_manifest_path",
        ),
    ):
        parent = load_hashed_json(Path(configuration[path_name]))
        if (
            parent["content_hash"] != configuration[hash_name]
            or parent.get("source") != campaign.get("source")
        ):
            raise ValueError(f"Stage-I {hash_name} lineage differs")
    if _sha256(args.input_npz) != configuration["input_npz_sha256"]:
        raise ValueError("Stage-I input NPZ bytes differ")
    expected_fields = {"identities", "labels", "no_reconstruction_logits"} | {
        f"{kind}_{expert}"
        for kind in ("predicted", "oracle", "identity_hlt")
        for expert in EXPERT_ORDER
    }
    with np.load(args.input_npz, allow_pickle=False) as payload:
        if set(payload.files) != expected_fields:
            raise ValueError("Stage-I input NPZ fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    fusion_hash = lock["fusion_checkpoint_hashes"].get(str(seed))
    if (
        fusion_hash is None
        or _sha256(args.fusion_checkpoint) != fusion_hash
    ):
        raise ValueError("Stage-I fusion checkpoint bytes differ")
    fusion_payload = torch.load(
        args.fusion_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        fusion_payload.get("contract") != FUSION_CHECKPOINT_CONTRACT
        or fusion_payload.get("allocation") != lock["allocation"]
    ):
        raise ValueError("Stage-I fusion checkpoint semantics differ")
    fusion = build_fusion_model(
        "F_TOKEN_TRANSFORMER",
        bank_dimensions={
            expert: int(lock["allocation"][expert][1])
            for expert in EXPERT_ORDER
        },
    )
    fusion.load_state_dict(fusion_payload["model_state_dict"], strict=True)
    resolved = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    fusion.to(resolved).eval()
    expected_cache_hashes = {
        expert: lock["seed_specific_artifacts"][str(seed)][expert][
            "inference_manifest"
        ]
        for expert in EXPERT_ORDER
    }
    if configuration["predicted_cache_hashes"] != expected_cache_hashes:
        raise ValueError("Stage-I selected predicted-cache hashes differ")
    for expert in EXPERT_ORDER:
        candidate_artifacts = lock["seed_specific_artifacts"][str(seed)][
            expert
        ]
        cache_manifest, cache_arrays = load_predictor_inference_cache(
            Path(configuration["predicted_cache_manifest_paths"][expert]),
            expected_pipeline_seed=seed,
            expected_registration_sha256=candidate_artifacts[
                "predictor_registration"
            ],
        )
        if (
            cache_manifest["content_hash"] != expected_cache_hashes[expert]
            or cache_manifest.get("source") != campaign.get("source")
            or cache_manifest["parents"]["identity_manifest"]
            != configuration["identity_manifest_sha256"]
            or cache_arrays["identities"].tolist()
            != arrays["identities"].tolist()
            or not np.array_equal(
                cache_arrays["predicted_tokens"],
                arrays[f"predicted_{expert}"],
            )
        ):
            raise ValueError("Stage-I predicted bank differs from locked cache")
    target_manifest, target_arrays = load_offline_target_cache(
        args.oracle_target_cache,
        expected_pipeline_seed=seed,
        expected_specification_sha256=configuration[
            "oracle_target_cache_specification_sha256"
        ],
    )
    if (
        target_manifest["content_hash"]
        != configuration["oracle_target_cache_sha256"]
        or target_manifest.get("source") != campaign.get("source")
        or target_manifest["identity_manifest_sha256"]
        != configuration["identity_manifest_sha256"]
        or target_manifest["identity_order_sha256"]
        != identity_order_sha256(
            arrays["identities"].tolist(), arrays["labels"]
        )
        or not np.array_equal(target_arrays["labels"], arrays["labels"])
        or any(
            not np.array_equal(
                target_arrays["tokens"][expert],
                arrays[f"oracle_{expert}"],
            )
            for expert in EXPERT_ORDER
        )
    ):
        raise ValueError("Stage-I oracle banks differ from locked target cache")
    artifact = bind_source(
        evaluate_stage_i_substitutions(
            identities=arrays["identities"].tolist(),
            labels=arrays["labels"],
            predicted_banks={
                expert: arrays[f"predicted_{expert}"]
                for expert in EXPERT_ORDER
            },
            oracle_banks={
                expert: arrays[f"oracle_{expert}"]
                for expert in EXPERT_ORDER
            },
            identity_projected_hlt_banks={
                expert: arrays[f"identity_hlt_{expert}"]
                for expert in EXPERT_ORDER
            },
            no_reconstruction_logits=arrays["no_reconstruction_logits"],
            frozen_offline_fusion=fusion,
            pipeline_seed=seed,
            stage_i_policy_sha256=policy["content_hash"],
            stage_i_input_payload_sha256=configuration[
                "input_npz_sha256"
            ],
            predictor_bundle_lock_sha256=lock["content_hash"],
            frozen_fusion_checkpoint_sha256=fusion_hash,
            identity_manifest_sha256=configuration[
                "identity_manifest_sha256"
            ],
            label_manifest_sha256=configuration["label_manifest_sha256"],
            predicted_cache_hashes=expected_cache_hashes,
            oracle_target_cache_sha256=configuration[
                "oracle_target_cache_sha256"
            ],
            hlt_cache_sha256=configuration["hlt_cache_sha256"],
            identity_projected_hlt_cache_sha256=configuration[
                "identity_projected_hlt_cache_sha256"
            ],
            no_reconstruction_prediction_sha256=configuration[
                "no_reconstruction_prediction_sha256"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_stage_i_evaluation(artifact)
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "stage_i_evaluation_sha256": artifact["content_hash"],
                "condition_count": artifact["condition_count"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
