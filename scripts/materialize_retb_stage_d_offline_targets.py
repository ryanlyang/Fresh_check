#!/usr/bin/env python3
"""Publish the frozen offline expert targets consumed by Stage-D dual rows."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

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
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    load_frozen_token_cache,
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


TARGET_INDEX_CONTRACT = "retb_stage_d_offline_target_index_v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError(f"Stage-D offline target differs: {path}")
    else:
        path.write_bytes(data)
    return _sha256_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    selection = load_hashed_json(
        args.campaign_root / "selection" / "retb_offline_shapes.json"
    )
    if selection.get("source") != campaign.get("source"):
        raise ValueError("Stage-D target shape selection source differs")
    output_root = args.campaign_root / "inputs" / "stage_d_offline_targets"
    records = []
    for alias in ("SHAPE_COMPACT", "SHAPE_HIGH"):
        selected = selection[alias]
        shape_id = str(selected["shape_id"])
        for seed in (101, 202, 303):
            cache_path = (
                args.campaign_root
                / "inputs"
                / "fusion_cache"
                / "offline"
                / shape_id
                / f"seed_{seed}"
                / "model_train"
                / "model_train_frozen_tokens.json"
            )
            manifest, arrays = load_frozen_token_cache(cache_path)
            if manifest.get("source") != campaign.get("source"):
                raise ValueError("Stage-D target cache source differs")
            for expert in EXPERT_ORDER:
                path = output_root / alias / expert / f"seed_{seed}.npz"
                digest = _publish_npz(
                    path,
                    {
                        "identities": np.asarray(arrays["identities"]),
                        "tokens": np.asarray(
                            arrays["token_banks"][expert], dtype=np.float32
                        ),
                        "logits": np.asarray(
                            arrays["expert_logits"][expert], dtype=np.float32
                        ),
                    },
                )
                records.append(
                    {
                        "shape_alias": alias,
                        "resolved_shape_id": shape_id,
                        "expert_id": expert,
                        "pipeline_seed": seed,
                        "relative_path": path.relative_to(
                            args.campaign_root
                        ).as_posix(),
                        "file_sha256": digest,
                        "source_cache_manifest_sha256": manifest[
                            "content_hash"
                        ],
                        "event_count": int(manifest["event_count"]),
                    }
                )
    index = bind_source(
        with_content_hash(
            {
                "contract": TARGET_INDEX_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "shape_selection_sha256": selection["content_hash"],
                "record_count": len(records),
                "records": records,
                "offline_targets_persisted_only_for_privileged_training": True,
                "final_test_included": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = args.output or output_root / "index.json"
    publication = write_immutable_json(output, index)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
