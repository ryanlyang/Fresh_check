#!/usr/bin/env python3
"""Audit HOSD target caches without redefining targets from observed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
    normalize_target,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_AUDIT_CONTRACT,
    TARGET_CACHE_SPEC_CONTRACT,
    TARGET_NORMALIZER_CONTRACT,
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)


def _named_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in output:
            raise ValueError("--cache entries must be unique NAME=PATH")
        output[name] = Path(path)
    return output


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0 or float(right.std()) == 0:
        return None
    return float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--label-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    caches = {}
    for name, path in _named_paths(args.cache).items():
        spec = load_hashed_json(
            path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
        )
        caches[name] = load_target_cache(path, cache_spec=spec)
    normalizer = (
        load_hashed_json(args.normalizer, expected_contract=TARGET_NORMALIZER_CONTRACT)
        if args.normalizer is not None
        else None
    )
    labels = None
    label_hash = None
    if args.label_manifest is not None:
        raw_labels = json.loads(args.label_manifest.read_text(encoding="utf-8"))
        if not isinstance(raw_labels, dict) or not isinstance(
            raw_labels.get("identity_to_label"), dict
        ):
            raise ValueError("label manifest requires identity_to_label")
        label_hash = raw_labels.get("content_hash") or canonical_sha256(raw_labels)
        labels = raw_labels["identity_to_label"]
    cache_rows = []
    unusual = []
    for cache_name, cache in sorted(caches.items()):
        targets = []
        joined_labels = None
        if labels is not None:
            if set(labels) != set(cache.identities):
                raise ValueError("label-audit identity join is not exact")
            joined_labels = np.asarray([labels[item] for item in cache.identities])
        for target_id in cache.manifest["persisted_target_ids"]:
            values = cache.values[target_id]
            masks = cache.masks[target_id]
            normalized = (
                normalize_target(
                    values,
                    masks,
                    target_id=target_id,
                    normalizer=normalizer,
                )
                if normalizer is not None
                and any(row["target_id"] == target_id for row in normalizer["targets"])
                else None
            )
            components = []
            for index in range(values.shape[1]):
                selected = masks[:, index]
                observed = values[selected, index].astype(np.float64)
                class_correlations = {}
                if joined_labels is not None:
                    for label in sorted(set(joined_labels.tolist())):
                        correlation = _correlation(
                            observed,
                            (joined_labels[selected] == label).astype(np.float64),
                        )
                        class_correlations[str(label)] = correlation
                        if correlation is not None and abs(correlation) >= 0.995:
                            unusual.append(
                                {
                                    "cache": cache_name,
                                    "target_id": target_id,
                                    "component_index": index,
                                    "class": int(label),
                                    "correlation": correlation,
                                }
                            )
                components.append(
                    {
                        "component_index": index,
                        "valid_count": int(selected.sum()),
                        "mask_sparsity": float(1.0 - selected.mean()),
                        "minimum": float(observed.min()) if observed.size else None,
                        "maximum": float(observed.max()) if observed.size else None,
                        "mean": float(observed.mean()) if observed.size else None,
                        "population_std": float(observed.std()) if observed.size else None,
                        "normalized_clip_fraction": (
                            float(
                                np.mean(np.abs(normalized[selected, index]) >= 12.0)
                            )
                            if normalized is not None and selected.any()
                            else None
                        ),
                        "one_vs_rest_class_correlations": class_correlations,
                    }
                )
            targets.append({"target_id": target_id, "components": components})
        cache_rows.append(
            {
                "cache_name": cache_name,
                "manifest_sha256": cache.manifest["content_hash"],
                "split": cache.manifest["split"],
                "event_count": cache.manifest["event_count"],
                "targets": targets,
            }
        )
    artifact = with_content_hash(
        {
            "contract": TARGET_AUDIT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "caches": cache_rows,
            "target_normalizer_sha256": (
                None if normalizer is None else normalizer["content_hash"]
            ),
            "label_manifest_sha256": label_hash,
            "labels_stored": False,
            "unusual_correlation_reports": unusual,
            "unusual_correlations_change_target_semantics": False,
            "scientific_underperformance_can_fail_or_cancel": False,
            "all_values_finite_and_masks_valid": True,
            "source": dict(campaign["source"]),
        }
    )
    output = args.output or args.campaign_root / "targets" / "target_audit.json"
    publication = write_immutable_json(output, artifact)
    print(json.dumps({**publication, "target_audit_sha256": artifact["content_hash"], "unusual_correlation_count": len(unusual)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
