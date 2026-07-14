from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.constrained_coarse_to_fine import (
    C5_B1,
    COARSE_TO_FINE_FUSION_CONTRACT,
    COARSE_TO_FINE_TRAIN_CONTRACT,
    D8_MULTIDEPTH,
    CampaignReportConfig,
    EndToEndPredictionConfig,
    FusionGroupSpec,
    ParticleStreamInput,
    ReconstructorSourceSpec,
    Step9FusionConfig,
    build_c_tier_reconstructor,
    build_end_to_end_tagger,
    cache_end_to_end_predictions,
    cache_prediction_alias,
    default_hierarchy_target_layout,
    load_end_to_end_tagger_checkpoint,
    run_step9_fusion,
    write_campaign_report,
)


def _small_reconstructor():
    layout = default_hierarchy_target_layout(radial_boundary=0.16, coordinate_extent=0.8)
    return build_c_tier_reconstructor(
        C5_B1,
        hierarchy_overrides={
            "d_model": 32,
            "num_heads": 4,
            "encoder_layers": 1,
            "pool_layers": 1,
            "decoder_layers_per_level": 1,
            "ffn_multiplier": 2.0,
            "pair_hidden_dim": 8,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        slot_overrides={"ffn_multiplier": 2.0, "dropout": 0.0, "attention_dropout": 0.0},
        layout=layout,
    )


def _write_reconstructor(path: Path, model, *, hlt_val: str = "hlt-val") -> None:
    provenance = {
        "model_train": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt-train",
            "layout": model.hierarchy.layout.to_dict(),
        },
        "model_val": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": hlt_val,
            "layout": model.hierarchy.layout.to_dict(),
        },
    }
    torch.save(
        {
            "checkpoint_contract": COARSE_TO_FINE_TRAIN_CONTRACT,
            "checkpoint_role": "best_model_val",
            "model_state_dict": model.state_dict(),
            "model": {
                "family": "C",
                "variant": model.slot_decoder.config.variant,
                "hierarchy_config": model.hierarchy.config.to_dict(),
                "slot_config": model.slot_decoder.config.to_dict(),
            },
            "provenance": provenance,
        },
        path,
    )


def _toy_hlt() -> tuple[ParticleStreamInput, torch.Tensor, torch.Tensor]:
    torch.manual_seed(901)
    batch, particles = 2, 6
    points = 0.1 * torch.randn(batch, 2, particles)
    features = torch.randn(batch, 17, particles)
    features[:, 6:11] = 0
    features[:, 6] = 1
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    pt = torch.linspace(5, 20, particles).expand(batch, -1)
    eta, phi = points[:, 0], points[:, 1]
    vectors = torch.stack(
        (pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta), pt * torch.cosh(eta)), dim=1
    )
    return ParticleStreamInput(points, features, vectors, mask), torch.zeros(batch), torch.zeros(batch)


def _prediction_block(name: str, split: str, *, shift: float = 0.0, offline: bool = False) -> PredictionBlock:
    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    logits = np.full((len(labels), 10), -0.5, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 1.5 + shift
    ids = [JetIdentity(file=f"{split}.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
    metadata = {
        "ok": True,
        "source_manifest_hash": "manifest",
        "hlt_content_hash": None if offline else f"hlt-{split}",
        "offline_content_hash": f"offline-{split}" if offline else None,
        "checkpoint_sha256": "checkpoint-" + name,
        "deployable_hlt_only": not offline,
        "hlt_profile": "fixed_hlt_v2_realistic",
        "hlt_profile_version": "v1",
        "hlt_degradation_strength": 2.5,
        "final_test_confirmed": split == "final_test",
        "source_state": {"source_commit": "commit", "source_status_hash": "status"},
    }
    return PredictionBlock(name, split, logits, np.empty_like(logits), labels, ids, metadata)


def _save_prediction_with_representation(
    block: PredictionBlock,
    prediction_dir: Path,
    *,
    shift: float = 0.0,
) -> None:
    metadata = save_prediction_block(block, prediction_dir)
    path = prediction_dir / block.model_name / f"{block.split}_representations.npz"
    representation = block.logits + np.float32(shift)
    np.savez_compressed(path, representation=representation.astype(np.float16), labels=block.labels)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["fusion_representation_path"] = str(path)
    metadata["fusion_representation_sha256"] = digest
    (prediction_dir / block.model_name / f"{block.split}_predictions_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


class ConstrainedCoarseToFineStep9Tests(unittest.TestCase):
    def test_prediction_report_runtime_path_merges_split_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = EndToEndPredictionConfig(
                prediction_dir=str(root),
                model_name="D5",
                manifest_path="manifest.json",
                hlt_cache_dir="hlt",
                checkpoint_path="checkpoint.pt",
                splits=("model_val",),
                device="cpu",
            )
            second = EndToEndPredictionConfig(**{**first.__dict__, "splits": ("stack_val",)})
            with patch(
                "teacher_logit_reco.constrained_coarse_to_fine.evaluation.load_end_to_end_tagger_checkpoint",
                return_value=object(),
            ), patch(
                "teacher_logit_reco.constrained_coarse_to_fine.evaluation.cache_end_to_end_prediction_split",
                side_effect=lambda _config, split, model_bundle: {"split": split, "metrics": {"accuracy": 0.5}},
            ):
                cache_end_to_end_predictions(first)
                report = cache_end_to_end_predictions(second)
            self.assertEqual(set(report["splits"]), {"model_val", "stack_val"})
            saved = json.loads((root / "D5" / "prediction_run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(saved["splits"]), {"model_val", "stack_val"})

    def test_prediction_alias_runtime_path_writes_complete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("model_val", "stack_val"):
                block = _prediction_block("D5", split)
                save_prediction_block(block, root)
                representation = root / "D5" / f"{split}_representations.npz"
                np.savez_compressed(representation, representation=block.logits, labels=block.labels)
            cache_prediction_alias(
                root,
                source_name="D5",
                alias_name="D5-B3",
                splits=("model_val",),
            )
            report = cache_prediction_alias(
                root,
                source_name="D5",
                alias_name="D5-B3",
                splits=("stack_val",),
            )
            self.assertEqual(set(report["splits"]), {"model_val", "stack_val"})
            self.assertEqual(report["alias_of"], "D5")
            self.assertTrue((root / "D5-B3" / "model_val_representations.npz").is_file())
            saved = json.loads((root / "D5-B3" / "prediction_run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["run_id"], "D5-B3")

    def test_selected_checkpoint_is_self_contained_and_d8_masks_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.pt", root / "right.pt"
            _write_reconstructor(left, _small_reconstructor())
            right_model = _small_reconstructor()
            with torch.no_grad():
                next(right_model.parameters()).add_(0.001)
            _write_reconstructor(right, right_model)
            model, resolved = build_end_to_end_tagger(
                D8_MULTIDEPTH,
                (
                    ReconstructorSourceSpec("b1", str(left), ("b1",)),
                    ReconstructorSourceSpec("b2", str(right), ("b2",)),
                ),
                fusion_overrides={
                    "d_model": 32,
                    "num_heads": 4,
                    "hlt_encoder_layers": 1,
                    "hlt_pool_layers": 1,
                    "pseudo_local_layers": 1,
                    "pseudo_global_layers": 1,
                    "fusion_layers": 1,
                    "pair_hidden_dim": 8,
                    "ffn_multiplier": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "pseudo_view_dropout": 0.0,
                },
            )
            selected = root / "selected.pt"
            torch.save(
                {
                    "checkpoint_contract": "constrained_coarse_to_fine_end_to_end_training_v1",
                    "checkpoint_role": "best_model_val",
                    "model_state_dict": model.state_dict(),
                    "variant": D8_MULTIDEPTH,
                    "fusion_config": model.tagger.config.to_dict(),
                    "reconstructors": resolved.to_dict(),
                    "provenance": {
                        "model_train": {"source_manifest_hash": "manifest", "hlt_content_hash": "hlt-train"},
                        "model_val": {"source_manifest_hash": "manifest", "hlt_content_hash": "hlt-val"},
                    },
                },
                selected,
            )
            left.unlink()
            right.unlink()
            loaded, _, _ = load_end_to_end_tagger_checkpoint(selected)
            hlt, eta, phi = _toy_hlt()
            output = loaded.forward_detailed(hlt, reference_eta=eta, reference_phi=phi)
            masked = loaded.tagger.forward_detailed(
                hlt,
                output.pseudo_views,
                view_availability_override=torch.zeros(2, 2, dtype=torch.bool),
            )
            self.assertEqual(tuple(output.logits.shape), (2, 10))
            self.assertTrue(torch.equal(masked.pooled_gates, torch.zeros_like(masked.pooled_gates)))

    def test_fusion_requires_real_f2_and_fits_simplex_on_stack_train(self):
        with self.assertRaisesRegex(ValueError, "F2"):
            FusionGroupSpec("F2", ("a", "b"), "mean_logits")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions"
            for name, shift in (
                ("A0", 0.0),
                ("D3", 0.1),
                ("D4", 0.2),
                ("D3-seed1", 0.08),
                ("D3-seed2", 0.09),
                ("D4-seed1", 0.18),
                ("D4-seed2", 0.19),
                ("D6", 0.12),
                ("D8", 0.13),
            ):
                for split in ("model_val", "stack_train", "stack_val", "final_test"):
                    _save_prediction_with_representation(
                        _prediction_block(name, split, shift=shift), predictions, shift=shift
                    )
            report = run_step9_fusion(
                Step9FusionConfig(
                    prediction_dir=str(predictions),
                    output_dir=str(root / "fusion"),
                    groups=(
                        FusionGroupSpec("F0", ("A0", "BEST_D"), "mean_logits"),
                        FusionGroupSpec("F2", ("D3", "D4"), "representation_stacker"),
                        FusionGroupSpec("F4", ("BEST_D", "BEST_D_SEED1", "BEST_D_SEED2"), "mean_logits"),
                        FusionGroupSpec(
                            "F5",
                            ("D8", "D6", "BEST_D", "BEST_D_SEED1", "BEST_D_SEED2"),
                            "linear_stacker",
                        ),
                    ),
                    required_groups=("F0", "F2", "F4", "F5"),
                    simplex_samples=16,
                    c_grid=(1.0,),
                    max_iter=100,
                    confirm_final_test=True,
                    best_d_candidates=("D3", "D4"),
                )
            )
            self.assertTrue(report["ok"])
            self.assertIsNone(report["groups"]["F0"]["fit"]["fit_split"])
            self.assertEqual(report["groups"]["F2"]["spec"]["method"], "representation_stacker")
            self.assertEqual(report["groups"]["F2"]["fit"]["fit_split"], "stack_train")
            self.assertEqual(report["best_d_selection_metric"], "model_val.cross_entropy")
            self.assertEqual(report["selected_best_d"], "D4")
            self.assertEqual(
                report["groups"]["F4"]["spec"]["members"],
                ("D4", "D4-seed1", "D4-seed2"),
            )
            self.assertEqual(
                report["groups"]["F5"]["spec"]["members"],
                ("D8", "D6", "D4", "D4-seed1", "D4-seed2"),
            )

    def test_strict_report_preserves_alias_and_d8_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions"
            report_root = root / "campaign"
            recon = report_root / "reconstructors" / "B0"
            recon.mkdir(parents=True)
            provenance = {
                split: {
                    "source_manifest_hash": "manifest",
                    "hlt_content_hash": f"hlt-{split}",
                    "offline_content_hash": f"offline-{split}",
                    "target_content_hash": f"target-{split}",
                    "target_builder_version": "builder",
                    "jet_identity_hash": f"identity-{split}",
                    "hlt_profile": "fixed_hlt_v2_realistic",
                    "hlt_profile_version": "v1",
                    "hlt_degradation_strength": 2.5,
                }
                for split in ("model_train", "model_val")
            }
            (recon / "run_report.json").write_text(
                json.dumps({
                    "ok": True,
                    "best_model_val": {"loss": 1.0},
                    "checkpoint_sha256": "b0",
                    "provenance": provenance,
                    "source_state": {"source_commit": "commit", "source_status_hash": "status"},
                }),
                encoding="utf-8",
            )
            taggers = ("A0", "A1", "A2", "D5", "D5-B3", "D8")
            for name in taggers:
                run = report_root / "taggers" / name
                run.mkdir(parents=True)
                (run / "run_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
                for split in ("model_val", "stack_val", "final_test"):
                    block = _prediction_block(name, split, offline=name == "A2")
                    if name in {"D5", "D5-B3"}:
                        block.metadata["checkpoint_sha256"] = "shared"
                        block.metadata["configuration_hash"] = "shared-config"
                    if name == "D5":
                        block.metadata["alias_of"] = "D5-B3"
                    if name == "D8" and split == "model_val":
                        block.metadata["d8_model_val_view_ablations"] = {
                            "hlt_only": {"accuracy": 0.5, "cross_entropy": 1.0, "n_jets": 20},
                            "all_views": {"accuracy": 1.0, "cross_entropy": 0.1, "n_jets": 20},
                        }
                    save_prediction_block(block, predictions)
            fusion_path = root / "fusion_report.json"
            fusion_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": COARSE_TO_FINE_FUSION_CONTRACT,
                        "groups": {
                            "F2": {
                                "spec": {"method": "representation_stacker"},
                                "splits": {"final_test": {"metrics": {"accuracy": 1.0}}},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = write_campaign_report(
                CampaignReportConfig(
                    campaign_root=str(report_root),
                    prediction_dir=str(predictions),
                    output_dir=str(root / "report"),
                    fusion_report_path=str(fusion_path),
                    reconstructor_runs=("B0",),
                    tagger_runs=taggers,
                    required_fusion_groups=("F2",),
                    confirm_final_test=True,
                )
            )
            self.assertTrue(report["ok"], report["problems"])
            self.assertTrue(report["d5_d5_b3_alias_audit"]["shared_configuration"])
            self.assertTrue((root / "report" / "d8_view_ablations.csv").exists())


if __name__ == "__main__":
    unittest.main()
