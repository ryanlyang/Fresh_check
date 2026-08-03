#!/usr/bin/env python3
"""Build all predeclared target shuffle plans and immutable control caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_hosd_target_derivatives import main as derivative_main  # noqa: E402
from scripts.build_hosd_target_shuffle_plans import main as shuffle_main  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_target_shuffle_plan,
    identity_order_sha256,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_MANIFEST_CONTRACT,
    TARGET_CACHE_SPEC_CONTRACT,
    TARGET_CONTROL_WAVE_CONTRACT,
    TARGET_SHUFFLE_PLAN_CONTRACT,
    RUNTIME_LABEL_MANIFEST_CONTRACT,
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)


SOURCE_SPLITS = ("model_train", "val_stop", "val_design")
DESIGN_SUBROLES = ("design_select", "design_confirm")
LABEL_ROLES = (*SOURCE_SPLITS, *DESIGN_SUBROLES)
CONTROL_SOURCES = ("physical", "O_BASE", "O_FULLREL")


def _ordered_label_population(
    raw: object, *, split: str
) -> tuple[tuple[str, ...], list[int], str]:
    """Return the authenticated positional population for a label role."""

    if not isinstance(raw, dict):
        raise ValueError(f"{split} label manifest is not an object")
    if (
        raw.get("contract") != RUNTIME_LABEL_MANIFEST_CONTRACT
        or raw.get("schema_version") != 2
        or raw.get("split") != split
    ):
        raise ValueError(f"{split} label-manifest contract differs")
    identity_to_label = raw.get("identity_to_label")
    order = raw.get("identity_order")
    if not isinstance(identity_to_label, dict) or not isinstance(order, list):
        raise ValueError(f"{split} label manifest lacks ordered identities")
    identities = tuple(str(value) for value in order)
    if (
        not identities
        or len(identities) != len(set(identities))
        or set(identities) != set(identity_to_label)
        or raw.get("identity_order_sha256")
        != identity_order_sha256(identities)
    ):
        raise ValueError(f"{split} ordered label population differs")
    labels = [int(identity_to_label[identity]) for identity in identities]
    if any(value < 0 or value >= 10 for value in labels):
        raise ValueError(f"{split} labels differ from the ten-class contract")
    return identities, labels, str(raw.get("content_hash") or canonical_sha256(raw))


def _labels(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        split, separator, raw = value.partition("=")
        if not separator or split in result:
            raise ValueError("--label-manifest requires unique SPLIT=PATH")
        result[split] = Path(raw)
    if set(result) != set(LABEL_ROLES):
        raise ValueError("--label-manifest split coverage differs")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--label-manifest", action="append", default=[], required=True
    )
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    labels = _labels(args.label_manifest)
    normalizer = (
        root / "normalization" / "target_500k" / "normalizer_manifest.json"
    )
    hashes = {}
    for split in SOURCE_SPLITS:
        source_roots = {
            "physical": root / "targets" / "canonical" / split,
            "O_BASE": root / "teachers" / "outputs" / split / "O_BASE",
            "O_FULLREL": (
                root / "teachers" / "outputs" / split / "O_FULLREL"
            ),
        }
        source_caches = {}
        for source_name, source_root in source_roots.items():
            source_spec = load_hashed_json(
                source_root / "cache_spec.json",
                expected_contract=TARGET_CACHE_SPEC_CONTRACT,
            )
            source_caches[source_name] = load_target_cache(
                source_root, cache_spec=source_spec
            )
        canonical_cache = source_caches["physical"]
        raw_labels = load_hashed_json(
            labels[split], expected_contract=RUNTIME_LABEL_MANIFEST_CONTRACT
        )
        manifest_identities, _, label_hash = _ordered_label_population(
            raw_labels, split=split
        )
        if set(manifest_identities) != set(canonical_cache.identities):
            raise ValueError("control-wave label identities differ")
        label_values = [
            int(raw_labels["identity_to_label"][identity])
            for identity in canonical_cache.identities
        ]
        for shuffle_kind in ("global", "within_class"):
            plan_dir = (
                root
                / "targets"
                / "controls"
                / "plans"
                / split
                / shuffle_kind
            )
            for source_name, source_root in source_roots.items():
                shuffle_main(
                    [
                        "--campaign-root",
                        str(root),
                        "--canonical-cache",
                        str(source_root),
                        "--label-manifest",
                        str(labels[split]),
                        "--output-dir",
                        str(plan_dir),
                        "--shuffle-kind",
                        shuffle_kind,
                    ]
                )
            for target_id in (
                "T_HLT_TRACK_PAIR_13",
                "T_HLT_REGION_PAIR_8",
            ):
                plan = build_target_shuffle_plan(
                    canonical_cache.identities,
                    labels=label_values,
                    target_id=target_id,
                    split=split,
                    shuffle_kind=shuffle_kind,
                    label_manifest_sha256=label_hash,
                    canonical_cache_manifest_sha256=canonical_cache.manifest[
                        "content_hash"
                    ],
                    source=campaign["source"],
                )
                write_immutable_json(
                    plan_dir / f"{target_id}.json", plan
                )
            for path in sorted(plan_dir.glob("*.json")):
                artifact = load_hashed_json(
                    path, expected_contract=TARGET_SHUFFLE_PLAN_CONTRACT
                )
                hashes[
                    f"plan::{split}::{shuffle_kind}::{path.stem}"
                ] = artifact["content_hash"]
        for control in (
            "target_mean",
            "global_shuffle",
            "within_class_shuffle",
        ):
            for source_name, source_root in source_roots.items():
                output_dir = (
                    root
                    / "targets"
                    / "controls"
                    / control
                    / split
                    / source_name
                )
                command = [
                    "--campaign-root",
                    str(root),
                    "--canonical-cache",
                    str(source_root),
                    "--output-dir",
                    str(output_dir),
                    "--kind",
                    control,
                    "--normalizer",
                    str(normalizer),
                    "--cache-id",
                    f"CONTROL__{control}__{split}__{source_name}",
                ]
                if control != "target_mean":
                    command.extend(
                        [
                            "--shuffle-plan-dir",
                            str(
                                root
                                / "targets"
                                / "controls"
                                / "plans"
                                / split
                                / control.removesuffix("_shuffle")
                            ),
                        ]
                    )
                derivative_main(command)
                manifest = load_hashed_json(
                    output_dir / "target_manifest.json",
                    expected_contract=TARGET_CACHE_MANIFEST_CONTRACT,
                )
                hashes[
                    f"cache::{split}::{control}::{source_name}"
                ] = manifest["content_hash"]
    val_design_cache = load_target_cache(
        root / "targets" / "canonical" / "val_design",
        cache_spec=load_hashed_json(
            root / "targets" / "canonical" / "val_design" / "cache_spec.json",
            expected_contract=TARGET_CACHE_SPEC_CONTRACT,
        ),
    )
    for split in DESIGN_SUBROLES:
        raw_labels = load_hashed_json(
            labels[split], expected_contract=RUNTIME_LABEL_MANIFEST_CONTRACT
        )
        identities, label_values, label_hash = _ordered_label_population(
            raw_labels, split=split
        )
        if not set(identities).issubset(set(val_design_cache.identities)):
            raise ValueError("design-subrole control identity coverage differs")
        for shuffle_kind in ("global", "within_class"):
            plan_dir = (
                root
                / "targets"
                / "controls"
                / "plans"
                / split
                / shuffle_kind
            )
            for target_id in (
                "T_HLT_TRACK_PAIR_13",
                "T_HLT_REGION_PAIR_8",
            ):
                plan = build_target_shuffle_plan(
                    identities,
                    labels=label_values,
                    target_id=target_id,
                    split=split,
                    shuffle_kind=shuffle_kind,
                    label_manifest_sha256=label_hash,
                    canonical_cache_manifest_sha256=(
                        val_design_cache.manifest["content_hash"]
                    ),
                    source=campaign["source"],
                )
                path = plan_dir / f"{target_id}.json"
                write_immutable_json(path, plan)
                hashes[
                    f"plan::{split}::{shuffle_kind}::{target_id}"
                ] = plan["content_hash"]
    artifact = with_content_hash(
        {
            "contract": TARGET_CONTROL_WAVE_CONTRACT,
            "schema_version": 2,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "source_splits": list(SOURCE_SPLITS),
            "design_subroles": list(DESIGN_SUBROLES),
            "splits": list(LABEL_ROLES),
            "control_kinds": [
                "target_mean",
                "global_shuffle",
                "within_class_shuffle",
            ],
            "control_sources": list(CONTROL_SOURCES),
            "control_source_semantics": {
                "physical": "canonical_offline_physical_targets",
                "O_BASE": (
                    "O_BASE_logits_and_canonical_128d_pooled_latent"
                ),
                "O_FULLREL": "O_FULLREL_logits",
            },
            "artifact_hashes": dict(sorted(hashes.items())),
            "all_controls_complete": True,
            "canonical_caches_mutated": False,
            "performance_results_read": False,
        }
    )
    output = root / "targets" / "controls" / "control_plan_completion.json"
    write_immutable_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
