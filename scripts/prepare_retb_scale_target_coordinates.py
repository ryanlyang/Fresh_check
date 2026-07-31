#!/usr/bin/env python3
"""Materialize scale-teacher token coordinates before sealing target caches.

This is the non-circular first half of Stage-M target production.  It applies
the already locked Stage-E coordinate transform to newly scale-trained
offline experts, writes a frozen coordinate cache, and publishes hybrid
target checkpoints whose learned coordinate-only parameters remain locked
while every offline particle encoder comes from ``scale_train``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_bridge_target import _model  # noqa: E402
from scripts.train_retb_offline_expert import (  # noqa: E402
    _dataset,
    _load_npz,
    _load_trees,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (  # noqa: E402
    BridgeProjection,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (  # noqa: E402
    TokenOnlyExpertHead,
)
from teacher_logit_reco.relation_expert_token_bridge.target_coordinates import (  # noqa: E402
    target_slot_query_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


CONFIGURATION_CONTRACT = "retb_scale_target_coordinate_configuration_v1"
INDEX_CONTRACT = "retb_scale_target_coordinate_index_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("offline_target_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("locked bridge target lacks offline-target state")
    return state


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        if path.exists():
            if path.is_symlink() or _sha256(path) != _sha256(temporary):
                raise FileExistsError(
                    f"scale target checkpoint differs: {path}"
                )
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hybrid_model(
    *,
    mode: str,
    expert: str,
    scale_checkpoint: Path,
    locked_checkpoint: Path,
    relation: Mapping[str, Any],
    region: Mapping[str, Any],
    output_checkpoint: Path,
) -> tuple[Any, Any, Path]:
    model = _model(
        checkpoint=scale_checkpoint, relation=relation, region=region
    )
    projection = None
    projected_head = None
    if mode in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE"}:
        locked = _state(locked_checkpoint)
        target_specific = {
            name.removeprefix("expert_model."): value
            for name, value in locked.items()
            if name.startswith("expert_model.tokenizer.")
            or name.startswith("expert_model.head.")
        }
        updated = model.state_dict()
        if not target_specific or any(
            name not in updated
            or tuple(updated[name].shape) != tuple(value.shape)
            for name, value in target_specific.items()
        ):
            raise ValueError(
                f"locked {mode} coordinate is incompatible with scale {expert}"
            )
        updated.update(target_specific)
        model.load_state_dict(updated, strict=True)
    elif mode == "T2_PROJECT":
        locked = _state(locked_checkpoint)
        projection_state = {
            name.removeprefix("projection."): value
            for name, value in locked.items()
            if name.startswith("projection.")
        }
        head_state = {
            name.removeprefix("projected_expert_head."): value
            for name, value in locked.items()
            if name.startswith("projected_expert_head.")
        }
        up = projection_state.get("up.weight")
        if not isinstance(up, torch.Tensor) or up.ndim != 2:
            raise ValueError("locked T2 projection is incomplete")
        source_dimension = int(model.token_dimension)
        bridge_dimension = int(up.shape[1])
        projection = BridgeProjection(source_dimension, bridge_dimension)
        projection.load_state_dict(projection_state, strict=True)
        projected_head = TokenOnlyExpertHead(
            token_dimension=bridge_dimension, num_classes=10
        )
        projected_head.load_state_dict(head_state, strict=True)
    elif mode != "T0_PURE":
        raise ValueError("scale target mode is not token-valued")

    if mode == "T0_PURE":
        return model, None, scale_checkpoint
    offline_state = {
        f"expert_model.{name}": value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    if projection is not None:
        offline_state.update(
            {
                f"projection.{name}": value.detach().cpu()
                for name, value in projection.state_dict().items()
            }
        )
        offline_state.update(
            {
                f"projected_expert_head.{name}": value.detach().cpu()
                for name, value in projected_head.state_dict().items()
            }
        )
    _atomic_torch_save(
        output_checkpoint,
        {
            "contract": "retb_scale_target_checkpoint_v1",
            "schema_version": 1,
            "target_mode": mode,
            "expert_id": expert,
            "scale_expert_checkpoint_sha256": _sha256(scale_checkpoint),
            "locked_coordinate_checkpoint_sha256": _sha256(
                locked_checkpoint
            ),
            "offline_target_state_dict": offline_state,
        },
    )
    return model, (projection, projected_head), output_checkpoint


@torch.no_grad()
def _infer(
    *,
    model: Any,
    transform: Any,
    loader: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.to(device).eval()
    if transform is not None:
        projection, head = transform
        projection.to(device).eval()
        head.to(device).eval()
    token_rows, logit_rows, label_rows = [], [], []
    identities: list[str] = []
    for raw in loader:
        batch = {
            name: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in raw.items()
        }
        details = model(
            return_details=True,
            **{
                name: batch[name]
                for name in (
                    "features",
                    "vectors",
                    "mask",
                    "raw_tokens",
                    "region_trees",
                )
                if name in batch
            },
        )
        tokens = details["tokens"]
        logits = details["logits"]
        if transform is not None:
            projection, head = transform
            tokens = projection(tokens)
            logits = head(tokens)
        token_rows.append(tokens.detach().float().cpu().numpy())
        logit_rows.append(logits.detach().float().cpu().numpy())
        label_rows.append(batch["labels"].detach().cpu().numpy())
        identities.extend(str(value) for value in raw["identities"])
    return (
        np.asarray(identities),
        np.concatenate(label_rows).astype(np.int64, copy=False),
        np.concatenate(token_rows).astype(np.float32, copy=False),
        np.concatenate(logit_rows).astype(np.float32, copy=False),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=(
            "scale_train",
            "val_stop",
            "val_design",
            "stack_val",
            "final_test",
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    config = load_hashed_json(
        args.configuration, expected_contract=CONFIGURATION_CONTRACT
    )
    if (
        config.get("source") != campaign.get("source")
        or int(config.get("pipeline_seed", -1)) not in {101, 202, 303}
        or set(config.get("scale_experts", {})) != set(EXPERT_ORDER)
        or set(config.get("locked_targets", {})) != set(EXPERT_ORDER)
        or set(config.get("target_modes", {})) != set(EXPERT_ORDER)
    ):
        raise ValueError("scale target-coordinate configuration differs")
    authorize_dataset_access(
        worker_role=(
            "scale_training_worker"
            if args.split == "scale_train"
            else "training_worker"
            if args.split == "val_stop"
            else "design_worker"
            if args.split == "val_design"
            else "postlock_stack_diagnostic"
            if args.split == "stack_val"
            else "final_test_worker"
        ),
        requested_resource=(
            "stack_val_oracle_targets"
            if args.split == "stack_val"
            else "final_test_targets"
            if args.split == "final_test"
            else args.split
        ),
    )
    relation = load_hashed_json(config["offline_relation_normalizer"])
    region = load_hashed_json(config["offline_region_normalizer"])
    raw_path = (
        root / "inputs" / "offline" / args.split / "offline_inputs.npz"
    )
    arrays = _load_npz(raw_path)
    identities = [str(value) for value in arrays["identities"].tolist()]
    trees = _load_trees(
        root / "inputs" / "region_tree" / "offline",
        split=args.split,
        identities=identities,
    )
    dataset = _dataset(arrays, region_trees=trees)
    loader = make_offline_expert_loader(
        dataset,
        seed=int(config["pipeline_seed"]),
        training=False,
        batch_size=args.batch_size,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    token_banks, expert_logits = {}, {}
    checkpoint_hashes, registration_hashes, target_rows = {}, {}, {}
    expected_identities = expected_labels = None
    checkpoint_root = args.output_dir.parent / "target_checkpoints"
    snapshot = source_snapshot(REPO_ROOT)
    for expert in EXPERT_ORDER:
        mode = str(config["target_modes"][expert])
        scale = config["scale_experts"][expert]
        locked = config["locked_targets"][expert]
        scale_checkpoint = Path(scale["checkpoint"])
        locked_checkpoint = Path(locked["checkpoint"])
        output_checkpoint = (
            checkpoint_root / expert / "scale_target.pt"
        )
        model, transform, checkpoint = _hybrid_model(
            mode=mode,
            expert=expert,
            scale_checkpoint=scale_checkpoint,
            locked_checkpoint=locked_checkpoint,
            relation=relation,
            region=region,
            output_checkpoint=output_checkpoint,
        )
        current_ids, labels, tokens, logits = _infer(
            model=model,
            transform=transform,
            loader=loader,
            device=device,
        )
        if expected_identities is None:
            expected_identities, expected_labels = current_ids, labels
        elif not np.array_equal(current_ids, expected_identities) or not np.array_equal(
            labels, expected_labels
        ):
            raise ValueError("scale target expert identity order differs")
        token_banks[expert], expert_logits[expert] = tokens, logits
        checkpoint_sha = _sha256(checkpoint)
        registration = bind_source(
            with_content_hash(
                {
                    "contract": "retb_scale_target_registration_v1",
                    "schema_version": 1,
                    "expert_id": expert,
                    "pipeline_seed": int(config["pipeline_seed"]),
                    "shape_role": config["shape_role"],
                    "target_mode": mode,
                    "checkpoint_sha256": checkpoint_sha,
                    "scale_expert_checkpoint_sha256": scale[
                        "checkpoint_sha256"
                    ],
                    "scale_expert_registration_sha256": scale[
                        "registration_sha256"
                    ],
                    "locked_coordinate_checkpoint_sha256": locked[
                        "checkpoint_sha256"
                    ],
                    "locked_coordinate_registration_sha256": locked[
                        "registration_sha256"
                    ],
                    "slot_query_sha256": target_slot_query_sha256(
                        checkpoint, target_mode=mode
                    ),
                    "training_population": "scale_train",
                    "coordinate_only_weights_carried_without_reselection": (
                        mode != "T0_PURE"
                    ),
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=snapshot,
        )
        registration_path = (
            checkpoint_root / expert / "registration.json"
        )
        write_immutable_json(registration_path, registration)
        checkpoint_hashes[expert] = checkpoint_sha
        registration_hashes[expert] = registration["content_hash"]
        target_rows[expert] = {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "registration_path": str(registration_path.resolve()),
            "registration_sha256": registration["content_hash"],
            "target_mode": mode,
            "slot_query_sha256": registration["slot_query_sha256"],
        }
    input_manifest = load_hashed_json(
        raw_path.with_name("offline_input_manifest.json")
    )
    cache = publish_frozen_token_cache(
        output_dir=args.output_dir,
        split=args.split,
        pipeline_seed=int(config["pipeline_seed"]),
        shape_id=str(config["shape_role"]),
        identities=expected_identities,
        labels=expected_labels,
        token_banks=token_banks,
        expert_logits=expert_logits,
        expert_checkpoint_hashes=checkpoint_hashes,
        expert_registration_hashes=registration_hashes,
        identity_manifest_sha256=input_manifest[
            "identity_manifest_sha256"
        ],
        label_manifest_sha256=input_manifest["content_hash"],
        source_snapshot=snapshot,
    )
    index = bind_source(
        with_content_hash(
            {
                "contract": INDEX_CONTRACT,
                "schema_version": 1,
                "configuration_sha256": validate_content_hash(config),
                "shape_role": config["shape_role"],
                "pipeline_seed": int(config["pipeline_seed"]),
                "split": args.split,
                "coordinate_cache_sha256": cache["content_hash"],
                "target_checkpoints": target_rows,
                "allocation": cache["allocation"],
                "event_count": cache["event_count"],
                "training_population": (
                    "scale_train"
                    if args.split == "scale_train"
                    else args.split
                ),
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    publication = write_immutable_json(
        args.output_dir / "scale_target_coordinate_index.json", index
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
