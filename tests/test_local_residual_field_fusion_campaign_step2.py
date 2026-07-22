from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from scripts.prepare_local_residual_field_a0_seed1 import main as prepare_a0_seed1_main

from teacher_logit_reco.local_particle_residual_field import (
    A0_SEED1_ALLOWED_CONFIG_DIFFERENCES,
    A0_SEED1_TRAINING_SEED,
    A0_TRAINING_SEED,
    LocalResidualFieldTaggerTrainConfig,
    audit_a0_seed1_recipe,
    build_a0_seed1_recipe,
    build_a0_seed1_train_config,
    load_a0_seed1_recipe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _a0_config(tmp_path: Path) -> LocalResidualFieldTaggerTrainConfig:
    return LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "A0"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        target_cache_dir=str(tmp_path / "targets"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        field_source="hlt_only",
        seed=A0_TRAINING_SEED,
        baseline_checkpoint=None,
        require_baseline_warm_start=False,
        reconstructor_checkpoint=None,
        teacher_logits_dir=None,
        kd_loss_weight=0.0,
        reconstructor_loss_weight=0.0,
    )


def test_step2_candidate_changes_only_seed_and_output_dir(tmp_path: Path) -> None:
    a0 = _a0_config(tmp_path)
    candidate = build_a0_seed1_train_config(a0, output_dir=tmp_path / "A0_seed1")
    audit = audit_a0_seed1_recipe(a0, candidate)

    assert audit["ok"] is True
    assert audit["problems"] == []
    assert candidate["seed"] == A0_SEED1_TRAINING_SEED
    assert candidate["field_source"] == "hlt_only"
    assert candidate["baseline_checkpoint"] is None
    assert candidate["require_baseline_warm_start"] is False
    assert candidate["teacher_logits_dir"] is None
    assert {row["path"] for row in audit["differences"]} == A0_SEED1_ALLOWED_CONFIG_DIFFERENCES


def test_step2_inactive_legacy_teacher_paths_are_scrubbed_from_seed_control(tmp_path: Path) -> None:
    a0 = asdict(_a0_config(tmp_path))
    a0["teacher_logits_dir"] = "/legacy/globally_propagated/teacher_logits"
    a0["teacher_logits_train_path"] = "/legacy/train.npz"
    assert a0["kd_loss_weight"] == 0.0

    candidate = build_a0_seed1_train_config(a0, output_dir=tmp_path / "A0_seed1")
    audit = audit_a0_seed1_recipe(a0, candidate)

    assert audit["ok"] is True
    assert candidate["teacher_logits_dir"] is None
    assert candidate["teacher_logits_train_path"] is None
    assert {row["path"] for row in audit["differences"]} == A0_SEED1_ALLOWED_CONFIG_DIFFERENCES


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("model_size", "large"),
        ("part_lr", 9.0e-5),
        ("hlt_cache_dir", "/different/cache"),
        ("selection_metric", "cross_entropy"),
    ),
)
def test_step2_audit_rejects_non_allowlisted_recipe_changes(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    a0 = _a0_config(tmp_path)
    candidate = build_a0_seed1_train_config(a0, output_dir=tmp_path / "A0_seed1")
    candidate[field_name] = bad_value

    audit = audit_a0_seed1_recipe(a0, candidate)

    assert audit["ok"] is False
    assert field_name in " ".join(audit["problems"])


def test_step2_audit_rejects_warm_start_or_teacher_inputs(tmp_path: Path) -> None:
    a0 = _a0_config(tmp_path)
    candidate = build_a0_seed1_train_config(a0, output_dir=tmp_path / "A0_seed1")
    candidate["baseline_checkpoint"] = "/forbidden/A0.pt"
    candidate["require_baseline_warm_start"] = True
    candidate["teacher_logits_dir"] = "/forbidden/teacher_logits"

    audit = audit_a0_seed1_recipe(a0, candidate)

    assert audit["ok"] is False
    joined = " ".join(audit["problems"])
    assert "warm start" in joined
    assert "baseline_checkpoint" in joined
    assert "teacher_logits_dir" in joined


def test_step2_audit_rejects_training_source_drift(tmp_path: Path) -> None:
    a0 = _a0_config(tmp_path)
    candidate = build_a0_seed1_train_config(a0, output_dir=tmp_path / "A0_seed1")

    audit = audit_a0_seed1_recipe(a0, candidate, provenance={"training_source_match": False})

    assert audit["ok"] is False
    assert "training source" in " ".join(audit["problems"])


def test_step2_recipe_round_trip_is_hash_bound_to_audit(tmp_path: Path) -> None:
    recipe, audit = build_a0_seed1_recipe(
        _a0_config(tmp_path),
        output_dir=tmp_path / "A0_seed1",
        provenance={"source_commit": "abc"},
    )

    loaded_recipe, loaded_config = load_a0_seed1_recipe(recipe.to_dict(), audit)

    assert loaded_recipe.config_hash == recipe.config_hash
    assert loaded_config.seed == A0_SEED1_TRAINING_SEED
    assert loaded_config.field_source == "hlt_only"
    assert loaded_config.baseline_checkpoint is None
    assert loaded_config.teacher_logits_dir is None

    tampered = recipe.to_dict()
    tampered["config"] = dict(tampered["config"])
    tampered["config"]["epochs"] += 1
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_a0_seed1_recipe(tampered, audit)


def test_step2_prepare_cli_writes_hash_bound_recipe_and_audit(tmp_path: Path) -> None:
    a0 = _a0_config(tmp_path)
    source_metadata = tmp_path / "A0_source_metadata.json"
    checkpoint = tmp_path / "A0.pt"
    run_config = tmp_path / "A0_slurm_run_config.json"
    output_dir = tmp_path / "A0_seed1"
    recipe_path = output_dir / "a0_seed1_train_config.json"
    audit_path = output_dir / "a0_seed_recipe_audit.json"
    source_metadata.write_text(json.dumps({"config": asdict(a0)}), encoding="utf-8")
    checkpoint.write_bytes(b"synthetic-checkpoint")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_config.write_text(json.dumps({"source_commit": commit, "source_status_hash": "status"}), encoding="utf-8")

    return_code = prepare_a0_seed1_main(
        [
            "--a0-source-metadata",
            str(source_metadata),
            "--a0-checkpoint",
            str(checkpoint),
            "--a0-run-config",
            str(run_config),
            "--output-dir",
            str(output_dir),
            "--recipe-output",
            str(recipe_path),
            "--audit-output",
            str(audit_path),
        ]
    )

    assert return_code == 0
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["ok"] is True
    assert audit["provenance"]["training_source_match"] is True
    loaded, config = load_a0_seed1_recipe(recipe, audit)
    assert loaded.run_id == "A0_seed1"
    assert config.output_dir == str(output_dir)


def test_step2_dedicated_slurm_wrapper_is_tigris_safe_and_audited() -> None:
    text = (REPO_ROOT / "sbatch" / "run_train_local_residual_field_a0_seed1.sh").read_text(encoding="utf-8")

    assert "#SBATCH --account=reu-aisocial" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "prepare_local_residual_field_a0_seed1.py" in text
    assert "train_local_residual_field_a0_seed1.py" in text
    assert "a0_seed_recipe_audit.json" in text
    assert "--baseline-checkpoint" not in text
    assert "afterok" not in text  # Step 12 owns campaign-level dependencies.


def test_step2_existing_a1_warm_start_semantics_remain_unchanged() -> None:
    text = (REPO_ROOT / "sbatch" / "run_train_local_residual_field_tagger.sh").read_text(encoding="utf-8")

    assert 'A1)\n    field_source="hlt_only"\n    require_baseline_checkpoint' in text
