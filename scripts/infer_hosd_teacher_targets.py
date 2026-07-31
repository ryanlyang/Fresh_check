#!/usr/bin/env python3
"""Run a locked HOSD teacher adapter and publish label-blind output caches.

The adapter factory is an importable ``module:function`` returning
``(TeacherInferenceAdapter, identities, batches)``.  Each batch is a mapping
with ``source_indices`` and the tensors accepted by the adapter.
"""

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
    build_target_cache_spec,
    infer_teacher_batch,
    load_and_validate_campaign,
    load_hashed_json,
    publish_target_cache,
    validate_teacher_lock,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TEACHER_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    CLASS_NAMES,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_b_runtime import (  # noqa: E402
    try_finalize_stage_b_wave,
)


def _factory(locator: str):
    module_name, separator, function_name = locator.partition(":")
    if not separator:
        raise ValueError("--adapter-factory must be module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError("teacher adapter factory is not callable")
    return function


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--teacher-lock", required=True, type=Path)
    parser.add_argument("--teacher-id", required=True, choices=("O_BASE", "O_FULLREL"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--adapter-factory",
        default=(
            "teacher_logit_reco.hlt_offline_structure_distillation."
            "teacher_inference_runtime:build_label_blind_relational_adapter"
        ),
    )
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--access-authorization-sha256")
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    lock = load_hashed_json(args.teacher_lock, expected_contract=TEACHER_LOCK_CONTRACT)
    validate_teacher_lock(lock, source=campaign["source"])
    target_registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    if target_registry.get("source") != campaign["source"]:
        raise ValueError("target registry source differs from active campaign")
    config = (
        json.loads(args.adapter_config.read_text(encoding="utf-8"))
        if args.adapter_config is not None
        else {}
    )
    adapter, identities, batch_provider = _factory(args.adapter_factory)(
        teacher_id=args.teacher_id,
        teacher_lock=lock,
        config=config,
    )
    teacher_row = next(
        row for row in lock["teachers"] if row["teacher_id"] == args.teacher_id
    )
    parent_hashes = {
        "campaign_spec": campaign["content_hash"],
        "teacher_lock": lock["content_hash"],
        "teacher_checkpoint": teacher_row["checkpoint_sha256"],
        "target_registry": target_registry["content_hash"],
    }
    if args.adapter_config is not None:
        parent_hashes["adapter_config"] = _sha256(args.adapter_config)
    if "input_npz" in config:
        parent_hashes["offline_input_view"] = _sha256(
            Path(config["input_npz"])
        )
    runtime_normalizer_hashes = set()
    for key in ("relation_normalizer", "region_normalizer"):
        if config.get(key) is not None:
            artifact = load_hashed_json(config[key])
            parent_hashes[key] = artifact["content_hash"]
            runtime_normalizer_hashes.add(artifact["content_hash"])
    if not runtime_normalizer_hashes.issubset(
        set(teacher_row["normalizer_hashes"].values())
    ):
        raise ValueError("teacher inference normalizer differs from its checkpoint lock")
    if config.get("tree_cache_dir") is not None:
        tree_manifest = load_hashed_json(
            Path(config["tree_cache_dir"]) / "manifest.json"
        )
        parent_hashes["region_tree_resource"] = tree_manifest["content_hash"]
    identities = tuple(str(value) for value in identities)
    output_ids = [f"T_OFFLINE_LOGITS_{args.teacher_id}"]
    if args.teacher_id == "O_BASE":
        output_ids.append("T_OFFLINE_POOLED_LATENT")
    components = {
        output_ids[0]: tuple(CLASS_NAMES)
    }
    if args.teacher_id == "O_BASE":
        components["T_OFFLINE_POOLED_LATENT"] = tuple(
            f"latent_{index:03d}" for index in range(128)
        )
    spec = build_target_cache_spec(
        cache_id=f"teacher_{args.teacher_id}_{args.split}",
        split=args.split,
        artifact_kind="teacher_output",
        identities=identities,
        target_components=components,
        parent_hashes=parent_hashes,
        source=campaign["source"],
        shard_size=args.shard_size,
        access_authorization_hash=args.access_authorization_sha256,
    )

    def generate(indices: np.ndarray):
        batch = batch_provider(indices)
        if any(key.lower() in {"label", "labels", "class", "classes", "y"} for key in batch):
            raise ValueError("teacher inference batch exposed labels")
        return infer_teacher_batch(adapter, batch)

    manifest = publish_target_cache(
        args.output_dir,
        cache_spec=spec,
        identities=identities,
        generator=generate,
    )
    wave = try_finalize_stage_b_wave(
        campaign_root=args.campaign_root,
        wave_kind="teacher_output",
        target_registry=target_registry,
        source=campaign["source"],
    )
    print(json.dumps({"cache_manifest_sha256": manifest["content_hash"], "teacher_id": args.teacher_id, "wave_completion_sha256": None if wave is None else wave["content_hash"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
