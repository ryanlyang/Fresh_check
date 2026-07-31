#!/usr/bin/env python3
"""Assemble selected native-HLT expert outputs into predictor evidence caches."""

from __future__ import annotations

import argparse
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

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
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
from teacher_logit_reco.relation_expert_token_bridge.stage_d_selection import (  # noqa: E402
    EVIDENCE_MODES,
    STAGE_D_EVIDENCE_SELECTION_CONTRACT,
    validate_stage_d_evidence_selection,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    validate_stage_d_confirmation_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


EVIDENCE_CACHE_MANIFEST_CONTRACT = (
    "retb_selected_hlt_evidence_cache_manifest_v1"
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
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError("selected HLT evidence cache differs")
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _selected_configuration(
    selection: Mapping[str, Any],
    *,
    shape: str,
    expert: str,
    mode: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in selection["selected_rows"]
        if (
            row["shape_id"],
            row["expert_id"],
            row["mode"],
        )
        == (shape, expert, mode)
    ]
    if len(matches) != 1:
        raise ValueError("selected HLT evidence coordinate is absent/duplicated")
    return matches[0]["configuration"]


def _confirmation_row(
    confirmation: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in confirmation["rows"]
        if row["component"] == "HLT_EXPERT"
        and int(row["seed"]) == int(seed)
        and row["configuration"] == configuration
    ]
    if len(matches) != 1:
        raise ValueError("selected HLT confirmation row is absent/duplicated")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--confirmation-registry", required=True, type=Path)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--evidence-mode", required=True, choices=EVIDENCE_MODES)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    selection = load_hashed_json(
        args.selection, expected_contract=STAGE_D_EVIDENCE_SELECTION_CONTRACT
    )
    validate_stage_d_evidence_selection(selection)
    confirmation = load_hashed_json(args.confirmation_registry)
    validate_stage_d_confirmation_registry(confirmation)
    if (
        selection.get("source") != campaign.get("source")
        or confirmation.get("source") != campaign.get("source")
        or args.shape_id not in selection["shapes"]
        or args.pipeline_seed not in {101, 202, 303}
    ):
        raise ValueError("selected HLT evidence lineage differs")

    parents: dict[str, dict[str, str]] = {}
    expert_sources: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for expert in EXPERT_ORDER:
        config = _selected_configuration(
            selection,
            shape=args.shape_id,
            expert=expert,
            mode=args.evidence_mode,
        )
        row = _confirmation_row(
            confirmation, configuration=config, seed=args.pipeline_seed
        )
        root = (
            args.campaign_root
            / "runs"
            / "stage_d"
            / "hlt_experts"
            / row["run_id"]
            / f"seed_{args.pipeline_seed}"
        )
        manifest = load_hashed_json(root / "native_output_manifest.json")
        registration = load_hashed_json(
            root / "checkpoint_registration.json"
        )
        if (
            manifest.get("source") != campaign.get("source")
            or registration.get("source") != campaign.get("source")
            or manifest["expert_registration_sha256"]
            != registration["content_hash"]
            or manifest["expert_id"] != expert
        ):
            raise ValueError("selected HLT expert output lineage differs")
        parents[expert] = {
            "run_id": row["run_id"],
            "registration_sha256": registration["content_hash"],
            "output_manifest_sha256": manifest["content_hash"],
            "registration_path": str(
                (root / "checkpoint_registration.json").resolve()
            ),
        }
        expert_sources[expert] = (root, manifest)

    split_records = {}
    for split in ("model_train", "val_stop", "val_design"):
        replicas = (0, 1, 2, 3) if split == "model_train" else (0,)
        identities = labels = None
        banks: dict[str, list[np.ndarray]] = {}
        states: dict[str, list[np.ndarray]] = {}
        masks: dict[str, list[np.ndarray]] = {}
        for expert in EXPERT_ORDER:
            root, manifest = expert_sources[expert]
            banks[expert], states[expert], masks[expert] = [], [], []
            for replica in replicas:
                record = manifest["files"][f"{split}_replica_{replica}"]
                path = root / record["relative_path"]
                if _sha256(path) != record["file_sha256"]:
                    raise ValueError("selected HLT expert output bytes differ")
                with np.load(path, allow_pickle=False) as payload:
                    current_ids = np.asarray(payload["identities"])
                    current_labels = np.asarray(payload["labels"], dtype=np.int64)
                    banks[expert].append(
                        np.asarray(payload["tokens"], dtype=np.float32)
                    )
                    states[expert].append(
                        np.asarray(payload["particle_states"], dtype=np.float32)
                    )
                    masks[expert].append(
                        np.asarray(payload["particle_mask"], dtype=bool)
                    )
                if identities is None:
                    identities, labels = current_ids, current_labels
                elif not np.array_equal(
                    identities, current_ids
                ) or not np.array_equal(labels, current_labels):
                    raise ValueError(
                        "selected HLT evidence populations differ"
                    )

        def joined(values: Sequence[np.ndarray]) -> np.ndarray:
            return values[0] if len(values) == 1 else np.stack(values)

        arrays = {
            "identities": identities,
            "labels": labels,
            "unbiased_particle_states": joined(states["BASE4"]),
            "particle_mask": joined(masks["BASE4"]),
            **{
                f"hlt_tokens_{expert}": joined(banks[expert])
                for expert in EXPERT_ORDER
            },
            **{
                f"relation_particle_states_{expert}": joined(states[expert])
                for expert in ("PT", "TRACK", "REGION")
            },
            **{
                f"relation_particle_mask_{expert}": joined(masks[expert])
                for expert in ("PT", "TRACK", "REGION")
            },
        }
        path = args.output_dir / f"{split}_evidence.npz"
        split_records[split] = {
            "relative_path": path.name,
            "file_sha256": _publish_npz(path, arrays),
            "event_count": int(len(labels)),
            "replica_count": len(replicas),
        }
    manifest = bind_source(
        with_content_hash(
            {
                "contract": EVIDENCE_CACHE_MANIFEST_CONTRACT,
                "schema_version": 1,
                "shape_id": args.shape_id,
                "evidence_mode": args.evidence_mode,
                "pipeline_seed": args.pipeline_seed,
                "selection_sha256": selection["content_hash"],
                "confirmation_registry_sha256": confirmation["content_hash"],
                "expert_parents": parents,
                "splits": split_records,
                "offline_targets_included": False,
                "scientific_underperformance_blocks_continuation": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(
        args.output_dir / "evidence_manifest.json", manifest
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
