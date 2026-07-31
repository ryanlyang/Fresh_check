#!/usr/bin/env python3
"""Materialize and build one selector-approved Stage-F target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step8 import (  # noqa: E402
    build_locked_target_cache_specification,
    build_selected_target_lineage,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    identity_order_sha256,
    load_frozen_token_head_reproducer,
    publish_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.target_coordinates import (  # noqa: E402
    target_slot_query_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from scripts.execute_retb_bridge_certification_wave import (  # noqa: E402
    _fit_readout,
)

import torch  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _template_slug(template: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(template))[:16]


def _pilot_root(
    root: Path, *, shape: str, expert: str, seed: int
) -> Path:
    identity = f"pilot_t0:{shape}:{expert}:seed_{seed}"
    return (
        root
        / "runs"
        / "stage_e"
        / "pilots"
        / canonical_sha256(identity)[:20]
    )


def _candidate_root(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    template: Mapping[str, Any],
) -> Path:
    return (
        root
        / "runs"
        / "stage_e"
        / "targets"
        / shape
        / expert
        / f"seed_{seed}"
        / _template_slug(template)
    )


def _selected_templates(
    certification_index: Mapping[str, Any],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in certification_index["records"]:
        mode = str(record["target_mode"])
        if mode == "T0_PURE":
            continue
        key = (
            str(record["shape_id"]),
            str(record["expert_id"]),
            mode,
        )
        template = record.get("selected_template")
        if not isinstance(template, Mapping) or key in rows:
            raise ValueError("Stage-F selected target template differs")
        rows[key] = dict(template)
    return rows


def _coordinate_root(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> Path:
    if mode == "T0_PURE":
        return _pilot_root(root, shape=shape, expert=expert, seed=seed)
    return _candidate_root(
        root,
        shape=shape,
        expert=expert,
        seed=seed,
        template=templates[(shape, expert, mode)],
    )


def _target_registration(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    if mode == "T0_PURE":
        path = (
            root
            / "selection"
            / "stage_e_parents"
            / shape
            / expert
            / f"seed_{seed}"
            / "t0_registration.json"
        )
    else:
        path = (
            _coordinate_root(
                root,
                shape=shape,
                expert=expert,
                seed=seed,
                mode=mode,
                templates=templates,
            )
            / "checkpoint_registration.json"
        )
    return path, load_hashed_json(path)


def _checkpoint_path(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> Path:
    if mode == "T0_PURE":
        return (
            root
            / "selection"
            / "stage_e_parents"
            / shape
            / expert
            / f"seed_{seed}"
            / "t0_best_model_val.pt"
        )
    return (
        _coordinate_root(
            root,
            shape=shape,
            expert=expert,
            seed=seed,
            mode=mode,
            templates=templates,
        )
        / "best_model_val.pt"
    )


def _find_readout(
    root: Path, *,
    fusion_sha256: str,
    normalizer_set_sha256: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    matches = []
    readout_root = root / "runs" / "stage_e" / "coordinate_readouts"
    for score_path in sorted(readout_root.glob("*/score.json")):
        score = load_hashed_json(score_path)
        if (
            score.get("fusion_sha256") == fusion_sha256
            and score.get("normalizer_set_sha256")
            == normalizer_set_sha256
        ):
            matches.append((score_path, score))
    if len(matches) != 1:
        raise ValueError("locked coordinate readout is absent or duplicated")
    score_path, score = matches[0]
    checkpoint = Path(score["checkpoint_path"]).resolve()
    normalizer_path = score_path.with_name("normalizer_set.json")
    normalizer = load_hashed_json(normalizer_path)
    if (
        _sha256(checkpoint) != fusion_sha256
        or normalizer["content_hash"] != normalizer_set_sha256
    ):
        raise ValueError("locked coordinate readout bytes drifted")
    return checkpoint, score, normalizer_path, normalizer


def _identity_artifact(
    *,
    split: str,
    identities: Sequence[str],
    labels: np.ndarray,
    parent_identity_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_f_target_identity_manifest_v1",
                "schema_version": 1,
                "split": split,
                "event_count": len(identities),
                "identity_order_sha256": identity_order_sha256(
                    identities, labels
                ),
                "parent_identity_manifest_sha256": parent_identity_sha256,
            }
        ),
        source_snapshot=snapshot,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--coordinate-index", required=True, type=int)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument(
        "--split",
        required=True,
        choices=("model_train", "val_stop", "val_design", "all"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split == "all":
        for split in ("model_train", "val_stop", "val_design"):
            child = [
                "--campaign-root",
                str(args.campaign_root),
                "--coordinate-index",
                str(args.coordinate_index),
                "--shape-id",
                args.shape_id,
                "--pipeline-seed",
                str(args.pipeline_seed),
                "--split",
                split,
                "--output-dir",
                str(args.output_dir / split),
                "--shard-size",
                str(args.shard_size),
            ]
            main(child)
        return 0
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    if args.pipeline_seed not in {101, 202, 303}:
        raise ValueError("Stage-F target-cache seed differs")
    selection = load_hashed_json(
        args.campaign_root
        / "selection"
        / "locked_bridge_coordinates.json"
    )
    index = load_hashed_json(
        args.campaign_root
        / "selection"
        / "stage_e"
        / "bridge_certification_index.json"
    )
    systems = selection["locked_coordinate_systems"]
    if not 0 <= args.coordinate_index < len(systems):
        raise ValueError("locked coordinate index is outside the selection")
    system = systems[args.coordinate_index]
    shape = str(args.shape_id)
    stage_e_registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    if shape not in stage_e_registry["shapes"]:
        raise ValueError("Stage-F target-cache shape is unregistered")
    modes = tuple(str(value) for value in system["target_tuple"])
    templates = _selected_templates(index)
    snapshot = source_snapshot(REPO_ROOT)

    arrays_by_expert: dict[str, dict[str, np.ndarray]] = {}
    registrations: dict[str, dict[str, Any]] = {}
    eligibility: dict[str, dict[str, Any]] = {}
    content: dict[str, dict[str, Any] | None] = {}
    noninferiority: dict[str, dict[str, Any] | None] = {}
    slot_hashes: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    identities: tuple[str, ...] | None = None
    labels: np.ndarray | None = None
    for expert, mode in zip(EXPERT_ORDER, modes, strict=True):
        coordinate_root = _coordinate_root(
            args.campaign_root,
            shape=shape,
            expert=expert,
            seed=args.pipeline_seed,
            mode=mode,
            templates=templates,
        )
        current = _npz(
            coordinate_root / f"{args.split}_coordinate_arrays.npz"
        )
        current_ids = tuple(str(value) for value in current["identities"])
        current_labels = np.asarray(current["labels"], dtype=np.int64)
        if identities is None:
            identities, labels = current_ids, current_labels
        elif (
            current_ids != identities
            or not np.array_equal(current_labels, labels)
        ):
            raise ValueError("selected target coordinate populations differ")
        arrays_by_expert[expert] = current
        _, registrations[expert] = _target_registration(
            args.campaign_root,
            shape=shape,
            expert=expert,
            seed=args.pipeline_seed,
            mode=mode,
            templates=templates,
        )
        eligibility[expert] = load_hashed_json(
            args.campaign_root
            / "selection"
            / "stage_e"
            / shape
            / expert
            / mode
            / "eligibility.json"
        )
        if mode == "T0_PURE":
            content[expert] = None
            noninferiority[expert] = None
        else:
            content[expert] = load_hashed_json(
                args.campaign_root
                / "selection"
                / "stage_e"
                / shape
                / expert
                / mode
                / f"seed_{args.pipeline_seed}_content.json"
            )
            noninferiority[expert] = load_hashed_json(
                args.campaign_root
                / "selection"
                / "stage_e"
                / shape
                / expert
                / mode
                / "noninferiority.json"
            )
        checkpoint_paths[expert] = _checkpoint_path(
            args.campaign_root,
            shape=shape,
            expert=expert,
            seed=args.pipeline_seed,
            mode=mode,
            templates=templates,
        )
        slot_hashes[expert] = target_slot_query_sha256(
            checkpoint_paths[expert], target_mode=mode
        )
    assert identities is not None and labels is not None

    parent_bundle = load_hashed_json(
        args.campaign_root
        / "selection"
        / "stage_e_parents"
        / shape
        / EXPERT_ORDER[0]
        / f"seed_{args.pipeline_seed}"
        / "parent_bundle.json"
    )
    parent_split = parent_bundle["dataset_evidence"][args.split]
    identity = _identity_artifact(
        split=args.split,
        identities=identities,
        labels=labels,
        parent_identity_sha256=parent_split[
            "identity_manifest_sha256"
        ],
        snapshot=snapshot,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "identity_manifest.json"
    write_immutable_json(identity_path, identity)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    readout = _fit_readout(
        root=args.campaign_root,
        shape=shape,
        target_tuple=tuple(modes),
        variant="F_TOKEN_TRANSFORMER",
        pipeline_seed=args.pipeline_seed,
        selected_templates=templates,
        miniature=campaign["campaign_profile"] == "miniature_test",
        batch_size=512,
        device=device,
    )
    fusion_checkpoint = Path(readout["checkpoint_path"]).resolve()
    normalizer = load_hashed_json(
        fusion_checkpoint.with_name("normalizer_set.json")
    )
    if (
        _sha256(fusion_checkpoint) != readout["fusion_sha256"]
        or normalizer["content_hash"]
        != readout["normalizer_set_sha256"]
    ):
        raise ValueError("shape/seed coordinate readout lineage differs")
    fusion_registration = bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_f_coordinate_fusion_registration_v2",
                "schema_version": 2,
                "checkpoint_sha256": _sha256(fusion_checkpoint),
                "checkpoint_path": str(fusion_checkpoint),
                "coordinate_contract_sha256": system[
                    "coordinate_contract_sha256"
                ],
                "selector_parent_fusion_sha256": system["fusion_sha256"],
                "selector_parent_normalizer_set_sha256": system[
                    "normalizer_set_sha256"
                ],
                "shape_id": shape,
                "pipeline_seed": args.pipeline_seed,
                "target_tuple": modes,
                "readout_score_sha256": readout["content_hash"],
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "fusion_registration.json",
        fusion_registration,
    )
    lineage = bind_source(
        build_selected_target_lineage(
            pipeline_seed=args.pipeline_seed,
            shape_id=shape,
            target_tuple=modes,
            target_registrations=registrations,
            slot_query_hashes=slot_hashes,
            eligibility_artifacts=eligibility,
            content_certifications=content,
            noninferiority_artifacts=noninferiority,
        ),
        source_snapshot=snapshot,
    )
    specification = bind_source(
        build_locked_target_cache_specification(
            split=args.split,
            pipeline_seed=args.pipeline_seed,
            shape_id=shape,
            allocation=parent_bundle["allocation"],
            coordinate_selection=selection,
            coordinate_contract_sha256=system[
                "coordinate_contract_sha256"
            ],
            target_lineage=lineage,
            fusion_registration=fusion_registration,
            normalizer_set=normalizer,
            identity_manifest_sha256=identity["content_hash"],
            identity_order_sha256=identity["identity_order_sha256"],
            event_count=len(identities),
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(args.output_dir / "target_lineage.json", lineage)
    write_immutable_json(
        args.output_dir / "target_cache_specification.json",
        specification,
    )
    reproducers = {
        expert: load_frozen_token_head_reproducer(
            checkpoint_path=checkpoint_paths[expert],
            expected_checkpoint_sha256=registrations[expert][
                "checkpoint_sha256"
            ],
            target_mode=modes[index],
            token_dimension=int(parent_bundle["allocation"][expert][1]),
        )
        for index, expert in enumerate(EXPERT_ORDER)
    }

    def generate(start: int, stop: int) -> dict[str, Any]:
        return {
            "tokens": {
                expert: arrays_by_expert[expert]["moving_tokens"][
                    start:stop
                ]
                for expert in EXPERT_ORDER
            },
            "expert_logits": {
                expert: arrays_by_expert[expert][
                    "moving_expert_logits"
                ][start:stop]
                for expert in EXPERT_ORDER
            },
        }

    manifest = publish_offline_target_cache(
        output_dir=args.output_dir,
        specification=specification,
        identities=identities,
        labels=labels,
        generator=generate,
        logit_reproducers=reproducers,
        source_snapshot=snapshot,
        shard_size=args.shard_size,
    )
    print(
        json.dumps(
            {
                "target_cache_manifest_sha256": manifest["content_hash"],
                "coordinate_index": args.coordinate_index,
                "shape_id": shape,
                "pipeline_seed": args.pipeline_seed,
                "split": args.split,
                "event_count": len(identities),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
