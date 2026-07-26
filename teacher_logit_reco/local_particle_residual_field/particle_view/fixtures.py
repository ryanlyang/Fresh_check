"""Small deterministic fixtures for particle-view contract tests and rehearsals."""

from __future__ import annotations

from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
)

from .splits import ParticleViewSplitConfig


def miniature_parent_manifest(*, rows_per_class: int = 4) -> SplitManifest:
    """Return a balanced, globally disjoint five-way synthetic manifest."""

    if (
        not isinstance(rows_per_class, int)
        or isinstance(rows_per_class, bool)
        or rows_per_class <= 0
        or rows_per_class % 2
    ):
        raise ValueError("rows_per_class must be a positive even integer")
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        rows = [
            JetIdentity(
                file=f"/synthetic/{split}/class-{label}.root",
                entry=split_index * 1_000_000 + label * 10_000 + local_index,
                label=label,
            )
            for label in range(len(LABEL_NAMES))
            for local_index in range(rows_per_class)
        ]
        splits[split] = rows
    count = rows_per_class * len(LABEL_NAMES)
    return SplitManifest(
        data_dir="/synthetic/particle_view",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: count for split in SPLIT_ORDER},
        split_seeds={split: 70_000 + index for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits=splits,
        metadata={
            "fixture": "particle_view_step1_miniature_v1",
            "file_level_separation_claimed": False,
        },
    )


def miniature_split_config(*, rows_per_class: int = 4) -> ParticleViewSplitConfig:
    if (
        not isinstance(rows_per_class, int)
        or isinstance(rows_per_class, bool)
        or rows_per_class <= 0
        or rows_per_class % 2
    ):
        raise ValueError("rows_per_class must be a positive even integer")
    count = rows_per_class * len(LABEL_NAMES)
    return ParticleViewSplitConfig(
        contract="particle_view_split_config_v1_miniature",
        train_count=count,
        model_val_count=count,
        stack_val_count=count,
        final_test_count=count,
        unused_parent_split_counts=(("stack_train", count),),
        model_val_partition_seed=9_120_202,
    )


__all__ = ["miniature_parent_manifest", "miniature_split_config"]
