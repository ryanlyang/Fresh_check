from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity
from jetclass_fresh.jetclass_data import LABEL_NAMES
from scripts.cache_local_residual_field_a0_seed1_predictions import parse_args
from teacher_logit_reco.local_particle_residual_field import (
    FUSION_DEVELOPMENT_SPLITS,
    LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_PAIRS,
    LocalResidualFieldPredictionConfig,
    cache_a0_seed1_development_predictions,
    sha256_file,
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field import fusion as fusion_module
from teacher_logit_reco.local_particle_residual_field import fusion_sources as source_module

from tests.test_local_residual_field_fusion_campaign_step3 import (
    _OracleFreeModel,
    build_source_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_step4_prediction_metrics_include_all_predeclared_qcd_projections() -> None:
    labels = np.asarray(
        [value for index in range(len(LABEL_NAMES)) for value in (0, index) if index != 0],
        dtype=np.int64,
    )
    logits = np.zeros((len(labels), len(LABEL_NAMES)), dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0

    metrics = fusion_module._metrics_from_logits(logits, labels, label_names=LABEL_NAMES)

    assert set(metrics["binary_projection_metrics"]) == {
        f"{negative}_vs_{positive}" for negative, positive in LOCAL_RESIDUAL_FIELD_BINARY_PROJECTION_PAIRS
    }
    assert all(row["available"] for row in metrics["binary_projection_metrics"].values())
    assert all(
        set(row["operating_points"]) == {"signal_efficiency_0.30", "signal_efficiency_0.50"}
        for row in metrics["binary_projection_metrics"].values()
    )


def test_step4_prediction_block_persists_binary_metrics_inside_metrics_metadata(tmp_path: Path) -> None:
    labels = np.asarray([0, 3, 0, 3], dtype=np.int64)
    logits = np.zeros((4, len(LABEL_NAMES)), dtype=np.float32)
    logits[np.arange(4), labels] = 2.0
    metrics = fusion_module._metrics_from_logits(logits, labels, label_names=LABEL_NAMES)
    save_prediction_block(
        PredictionBlock(
            model_name="A0",
            split="stack_val",
            logits=logits,
            probs=np.zeros_like(logits),
            labels=labels,
            jet_ids=[JetIdentity(file="stack_val.root", entry=index, label=int(label)) for index, label in enumerate(labels)],
            metadata={"metrics": metrics},
        ),
        tmp_path,
    )

    metadata = load_prediction_block(tmp_path, "A0", "stack_val").metadata
    assert "binary_projection_metrics" in metadata["metrics"]
    assert metadata["metrics"]["binary_projection_metrics"]["QCD_vs_Hgg"]["available"] is True


def test_step4_preflight_backfills_legacy_binary_metrics_without_mutating_sources(tmp_path: Path) -> None:
    audit_config, payload = build_source_fixture(tmp_path)
    source_metadata = Path(audit_config.a0_prediction_dir) / "A0" / "stack_train_predictions_metadata.json"
    original_bytes = source_metadata.read_bytes()

    audit = source_module.audit_fusion_source_artifacts(
        audit_config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), payload),
    )

    row = audit["reusable_predictions"]["A0"]["splits"]["stack_train"]
    assert row["binary_projection_metrics"]["contract"].endswith("_v2")
    assert row["binary_projection_metrics_source"] == "computed_from_hash_validated_legacy_predictions"
    assert row["binary_projection_metrics_hash"]
    assert source_metadata.read_bytes() == original_bytes


def test_step4_development_cli_has_no_split_or_final_test_override() -> None:
    args = parse_args(
        [
            "--source-artifact-audit", "audit.json",
            "--checkpoint", "A0_seed1.pt",
            "--prediction-dir", "predictions",
            "--hlt-cache-dir", "hlt",
            "--manifest-path", "manifest.json.gz",
        ]
    )

    assert not hasattr(args, "splits")
    assert not hasattr(args, "confirm_final_test")


def test_step4_cache_reuses_audited_members_and_registers_seed_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_config, payload = build_source_fixture(tmp_path)
    audit = source_module.audit_fusion_source_artifacts(
        audit_config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), payload),
    )
    assert audit["ok"] is True
    seed_checkpoint = tmp_path / "A0_seed1" / "best_model_val.pt"
    seed_checkpoint.parent.mkdir(parents=True)
    seed_checkpoint.write_bytes(b"synthetic-a0-seed1-checkpoint")
    seed_hash = sha256_file(seed_checkpoint)
    completion = {
        "ok": True, "contract": "local_residual_field_a0_seed1_completion_v1",
        "checkpoint_path": str(seed_checkpoint.resolve()), "checkpoint_sha256": seed_hash,
    }
    completion["artifact_hash"] = stable_fusion_json_hash(completion)
    (seed_checkpoint.parent / "seed_control_completion.json").write_text(
        json.dumps(completion), encoding="utf-8",
    )
    output = tmp_path / "campaign_predictions"

    def fake_cache(config: LocalResidualFieldPredictionConfig) -> dict:
        for split in FUSION_DEVELOPMENT_SPLITS:
            reusable = source_module.load_prediction_block(
                audit["reusable_predictions"]["A0"]["prediction_root"], "A0", split
            )
            metadata = dict(reusable.metadata)
            metadata["checkpoint_hash"] = seed_hash
            metadata["student_checkpoint_hash"] = seed_hash
            save_prediction_block(
                PredictionBlock(
                    model_name="A0_seed1",
                    split=split,
                    logits=reusable.logits.copy(),
                    probs=reusable.probs.copy(),
                    labels=reusable.labels.copy(),
                    jet_ids=list(reusable.jet_ids),
                    metadata=metadata,
                ),
                output,
            )
        manifest = output / "A0_seed1" / "prediction_manifest.json"
        manifest.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
        return {"ok": True, "splits": list(config.splits)}

    monkeypatch.setattr(source_module, "cache_local_residual_field_tagger_predictions", fake_cache)
    report = cache_a0_seed1_development_predictions(
        LocalResidualFieldPredictionConfig(
            checkpoint=str(seed_checkpoint),
            prediction_dir=str(output),
            model_name="A0_seed1",
            hlt_cache_dir="unused-by-fake-cache",
            manifest_path=audit_config.manifest_path,
            splits=FUSION_DEVELOPMENT_SPLITS,
            amp=False,
        ),
        source_artifact_audit=audit_config.output_path,
    )

    registry = json.loads(Path(report["prediction_sources"]).read_text(encoding="utf-8"))
    assert registry["ok"] is True
    assert set(registry["members"]) == {"A0", "A0_seed1", "P7b"}
    assert registry["development_splits"] == ["stack_train", "stack_val"]
    assert registry["final_test_opened"] is False
    assert registry["members"]["A0"]["prediction_root"] != registry["members"]["A0_seed1"]["prediction_root"]


def test_step4_cache_rejects_final_test_even_before_inference(tmp_path: Path) -> None:
    audit_config, _ = build_source_fixture(tmp_path)
    with pytest.raises(ValueError, match="final_test prediction caching"):
        LocalResidualFieldPredictionConfig(
            checkpoint="seed.pt",
            prediction_dir="predictions",
            model_name="A0_seed1",
            hlt_cache_dir="hlt",
            splits=("stack_train", "stack_val", "final_test"),
        )


def test_step4_slurm_wrappers_are_tigris_safe_stack_only_and_hash_gated() -> None:
    audit_wrapper = (REPO_ROOT / "sbatch" / "run_audit_local_residual_field_fusion_sources.sh").read_text(encoding="utf-8")
    cache_wrapper = (REPO_ROOT / "sbatch" / "run_cache_local_residual_field_a0_seed1_predictions.sh").read_text(encoding="utf-8")
    seed_wrapper = (REPO_ROOT / "sbatch" / "run_train_local_residual_field_a0_seed1.sh").read_text(encoding="utf-8")

    for text in (audit_wrapper, cache_wrapper, seed_wrapper):
        assert "#SBATCH --account=reu-aisocial" in text
        assert "export PYTHONNOUSERSITE=1" in text
    assert "require_local_residual_field_fusion_source_audit.py" in cache_wrapper
    assert "require_local_residual_field_fusion_source_audit.py" in seed_wrapper
    assert "for split in stack_train stack_val" in cache_wrapper
    assert "confirm-final-test" not in cache_wrapper
