#!/usr/bin/env python3
"""Measure real label-blind target outputs and freeze HOSD storage policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    ExtractorResources,
    PHYSICAL_TARGET_IDS,
    build_storage_measurements,
    extract_registered_target,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STORAGE_PROBE_EVIDENCE_CONTRACT,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.target_cache import (  # noqa: E402
    deterministic_npz_bytes,
)


SAMPLE_EVENT_LIMIT = 128


def _load_probe_input(path: Path):
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"storage probe input is absent or unsafe: {path}"
        )
    with np.load(path, allow_pickle=False) as archive:
        forbidden = {
            name
            for name in archive.files
            if name.lower() in {"label", "labels", "class", "classes", "y"}
        }
        if forbidden:
            raise ValueError(
                "storage probe must remain label blind; forbidden arrays: "
                f"{sorted(forbidden)}"
            )
        required = {"identity", "raw_tokens", "mask"}
        if not required.issubset(archive.files):
            raise ValueError(
                "storage probe input lacks "
                f"{sorted(required - set(archive.files))}"
            )
        identities = tuple(
            str(value) for value in archive["identity"].tolist()
        )
        raw_tokens = np.asarray(archive["raw_tokens"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
        vectors = (
            np.asarray(archive["vectors"], dtype=np.float32)
            if "vectors" in archive.files
            else None
        )
    if (
        not identities
        or len(identities) != raw_tokens.shape[0]
        or mask.shape[0] != len(identities)
        or len(set(identities)) != len(identities)
        or not np.isfinite(raw_tokens).all()
    ):
        raise ValueError("storage probe input arrays are malformed")
    selected = np.linspace(
        0,
        len(identities) - 1,
        num=min(SAMPLE_EVENT_LIMIT, len(identities)),
        dtype=np.int64,
    )
    return (
        tuple(identities[int(index)] for index in selected),
        raw_tokens[selected],
        mask[selected],
        None if vectors is None else vectors[selected],
    )


def _measure_families(
    *,
    target_ids,
    raw_tokens,
    mask,
    vectors,
    resources,
):
    families = {}
    for target_id in target_ids:
        started = time.perf_counter()
        result = extract_registered_target(
            target_id,
            raw_tokens,
            mask,
            vectors=vectors,
            resources=resources,
        )
        values = np.asarray(
            result.values.detach().cpu().numpy(), dtype=np.float32
        )
        available = np.asarray(
            result.loss_mask.detach().cpu().numpy(), dtype=np.bool_
        )
        encoded = deterministic_npz_bytes(
            {"values": values, "availability_mask": available}
        )
        elapsed = time.perf_counter() - started
        if elapsed <= 0 or not np.isfinite(values).all():
            raise RuntimeError(
                f"invalid measured extraction for {target_id}"
            )
        families[target_id] = {
            "storage_class": (
                "stream_same_view_pair"
                if values.ndim == 4
                else "compact_jet"
            ),
            "sample_events": int(values.shape[0]),
            "bytes_written": len(encoded),
            "elapsed_seconds": elapsed,
            "valid_components": int(available.sum()),
            "total_components": int(available.size),
            "maximum_shard_rebuild_seconds": elapsed,
            "component_count": int(values.shape[1]),
            "tensor_shape": list(values.shape),
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return families


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--probe-input", required=True, type=Path)
    parser.add_argument("--relation-normalizer", required=True, type=Path)
    parser.add_argument(
        "--available-storage-bytes", required=True, type=int
    )
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    relation = load_hashed_json(args.relation_normalizer)
    if (
        registry.get("source") != campaign["source"]
        or relation.get("source") != campaign["source"]
    ):
        raise ValueError(
            "storage probe parent source differs from campaign"
        )
    target_ids = tuple(
        sorted(
            row["target_id"]
            for row in registry["targets"]
            if row.get("executable_current_source")
            and row["target_id"] in PHYSICAL_TARGET_IDS
        )
    )
    if set(target_ids) != set(PHYSICAL_TARGET_IDS):
        raise ValueError(
            "storage probe lacks complete current physical target coverage"
        )
    identities, raw_tokens, mask, vectors = _load_probe_input(
        args.probe_input
    )
    floors = relation.get("track_uncertainty_floors", {})
    resources = ExtractorResources(
        d0_uncertainty_floor=float(
            floors.get("d0", {}).get("floor", 0.0)
        ),
        dz_uncertainty_floor=float(
            floors.get("dz", {}).get("floor", 0.0)
        ),
        sentinel_policy=relation.get("track_sentinel_policy"),
    )
    families = _measure_families(
        target_ids=target_ids,
        raw_tokens=raw_tokens,
        mask=mask,
        vectors=vectors,
        resources=resources,
    )
    evidence = with_content_hash(
        {
            "contract": STORAGE_PROBE_EVIDENCE_CONTRACT,
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "target_registry_sha256": registry["content_hash"],
            "relation_normalizer_sha256": relation["content_hash"],
            "probe_input_sha256": hashlib.sha256(
                args.probe_input.read_bytes()
            ).hexdigest(),
            "sample_event_limit": SAMPLE_EVENT_LIMIT,
            "sample_event_count": len(identities),
            "sample_identity_sha256": canonical_sha256(list(identities)),
            "families": families,
            "target_ids": list(target_ids),
            "label_access": False,
            "scientific_metrics_inspected": False,
            "measurements_hand_authored": False,
        }
    )
    evidence_output = args.evidence_output or (
        args.campaign_root
        / "job_ledgers"
        / "storage_probe_evidence.json"
    )
    evidence_publication = write_immutable_json(evidence_output, evidence)
    artifact = build_storage_measurements(
        family_measurements=families,
        available_storage_bytes=args.available_storage_bytes,
        parent_hashes={
            "campaign_spec": campaign["content_hash"],
            "measurement_evidence": evidence["content_hash"],
        },
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root
        / "job_ledgers"
        / "runtime_storage_measurements.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "evidence": evidence_publication,
                "measurement": publication,
                "projections": artifact["projected_storage_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
