import json
import math
import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT,
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC,
    LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    LocalCompressionFeatureAdapterParT,
    LocalCompressionFeatureConfig,
    LocalCompressionPartConfig,
    LocalCompressionPartProtocol,
    LocalCompressionPartReportConfig,
    LocalCompressionTaggerTrainConfig,
    build_local_compression_part_report,
    default_local_compression_modality_specs,
    load_hlt_part_baseline_checkpoint,
)


torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    """Tiny is-a ParticleTransformerHLTClassifier for Step 16 smoke checks."""

    def __init__(self, num_classes: int = 2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(num_classes))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 0.4
            self.linear.weight[0, 4] = -0.3
            self.linear.weight[1, 1] = -0.2
            self.linear.weight[1, 7] = 0.5

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


def small_config(**kwargs) -> LocalCompressionPartConfig:
    payload = {
        "embed_dim": 16,
        "local_layers": 1,
        "local_heads": 4,
        "context_layers": 1,
        "context_heads": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(kwargs)
    return LocalCompressionPartConfig(**payload)


def make_tokens(num_particles: int = 6):
    tokens = torch.zeros((2, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :3] = True
    for batch_index in range(2):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 15.0 + 2.0 * particle_index + 0.5 * batch_index
            eta = -0.25 + 0.07 * particle_index
            phi = -math.pi + 0.18 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.4
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.015 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.08
    return tokens, mask


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_checkpoint(path: Path, model: torch.nn.Module, *, selection_metric: str, split_hash: str = "split-ok") -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_metric": str(selection_metric),
            "hlt_degradation_strength": 0.6,
            "split_manifest_hash": str(split_hash),
            "model_config": dict(getattr(model, "config", {})),
        },
        path,
    )


def metrics(*, fpr50: float, auc: float = 0.8, arrays: dict | None = None) -> dict:
    payload = {
        "accuracy": 0.75,
        "loss": 0.45,
        "binary_metrics": {
            "auc": float(auc),
            "fpr_at_signal_eff_0p50": float(fpr50),
            "background_rejection_at_signal_eff_0p50": 1.0 / float(fpr50),
        },
    }
    if arrays is not None:
        payload["_prediction_arrays"] = arrays
    return payload


def child_report(variant: str, *, final_fpr50: float, val_fpr50: float, arrays: bool = False) -> dict:
    model_val_arrays = None
    final_test_arrays = None
    if arrays:
        model_val_arrays = {
            "labels": [1, 1, 1, 1, 0, 0, 0, 0],
            "scores": [0.95, 0.80, 0.45, 0.10, 0.70, 0.60, 0.30, 0.05],
        }
        final_test_arrays = {
            "labels": [1, 1, 1, 1, 0, 0, 0, 0],
            "scores": [0.90, 0.55, 0.35, 0.15, 0.65, 0.40, 0.20, 0.02],
        }
    return {
        "experiment_step": "local_compression_part_step12_train",
        "variant": variant,
        "checkpoint": "best_model_val.pt",
        "best_epoch": 1,
        "epochs_completed": 2,
        "selection_metric": LOCAL_COMPRESSION_PRIMARY_METRIC,
        "selection_metric_direction": "minimize",
        "best_model_selection_metric_value": float(val_fpr50),
        "best_model_val_metrics": metrics(fpr50=val_fpr50, arrays=model_val_arrays),
        "stack_val_metrics": metrics(fpr50=val_fpr50 + 0.01),
        "final_test_metrics": metrics(fpr50=final_fpr50, auc=0.85, arrays=final_test_arrays),
        "final_test_evaluated": True,
        "num_classes": 2,
        "label_names": ["QCD", "Hgg"],
        "label_filter": [0, 1],
        "inference_consumes_hlt_only": True,
    }


class LocalCompressionStep16SmokeTests(unittest.TestCase):
    def test_protocol_and_config_guardrails_match_smoke_contract(self):
        protocol = LocalCompressionPartProtocol()
        training = LocalCompressionTaggerTrainConfig(
            output_dir="out",
            manifest_path="manifest.json",
            hlt_cache_dir="cache",
            baseline_checkpoint="baseline.pt",
            confirm_split_settings=True,
        )

        self.assertEqual(protocol.primary_metric, LOCAL_COMPRESSION_PRIMARY_METRIC)
        self.assertEqual(protocol.loss_name, LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT)
        self.assertEqual(protocol.selection_metric_direction, "minimize")
        self.assertEqual(training.selection_metric, LOCAL_COMPRESSION_PRIMARY_METRIC)
        self.assertEqual(tuple(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), tuple(PF_FEATURE_NAMES))

        specs = tuple(reversed(default_local_compression_modality_specs()))
        with self.assertRaisesRegex(ValueError, "modality order"):
            LocalCompressionFeatureConfig(modalities=specs)
        with self.assertRaisesRegex(ValueError, "CrossEntropyLoss"):
            LocalCompressionPartConfig(loss_name="bce_with_logits")
        with self.assertRaisesRegex(ValueError, "fpr_at_signal_eff_0p50"):
            LocalCompressionTaggerTrainConfig(
                output_dir="out",
                manifest_path="manifest.json",
                hlt_cache_dir="cache",
                baseline_checkpoint="baseline.pt",
                confirm_split_settings=True,
                selection_metric="accuracy",
            )

    def test_exact_zero_delta_recovers_baseline_and_masks_are_safe(self):
        tokens, mask = make_tokens()
        part_model = DummyReferencePart(num_classes=2)
        model = LocalCompressionFeatureAdapterParT(small_config(), part_model=part_model)
        model.eval()

        with torch.no_grad():
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
            baseline_logits = part_model(
                output.canonical_inputs.points,
                output.canonical_inputs.features,
                output.canonical_inputs.lorentz_vectors,
                output.canonical_inputs.mask,
            )

        self.assertTrue(output.baseline_recoverable_at_zero_delta)
        self.assertTrue(torch.allclose(output.logits, baseline_logits, atol=1.0e-6))
        self.assertEqual(float(output.delta_output.delta_F_rows.abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(output.canonical_inputs.feature_rows(), output.adapted_inputs.feature_rows()))
        self.assertEqual(output.modalities.modality_names, LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(float(output.pool_output.pool_weights[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.gate_output.gates[0, 5].abs().sum().item()), 0.0)

    def test_checkpoint_metadata_rejects_wrong_metric_and_split_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrong_metric = tmp_path / "wrong_metric.pt"
            wrong_split = tmp_path / "wrong_split.pt"
            write_checkpoint(wrong_metric, DummyReferencePart(), selection_metric="accuracy")
            write_checkpoint(wrong_split, DummyReferencePart(), selection_metric=LOCAL_COMPRESSION_PRIMARY_METRIC, split_hash="old")

            with self.assertRaisesRegex(ValueError, "selection_metric mismatch"):
                load_hlt_part_baseline_checkpoint(wrong_metric, DummyReferencePart(), require_metadata=True)
            with self.assertRaisesRegex(ValueError, "split_manifest_hash mismatch"):
                load_hlt_part_baseline_checkpoint(
                    wrong_split,
                    DummyReferencePart(),
                    expected_split_manifest_hash="new",
                    require_metadata=True,
                )

    def test_report_smoke_includes_validation_threshold_final_test_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK / "run_report.json",
                child_report(LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK, final_fpr50=0.30, val_fpr50=0.31),
            )
            write_json(
                root / LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED / "run_report.json",
                child_report(LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED, final_fpr50=0.20, val_fpr50=0.22, arrays=True),
            )

            report = build_local_compression_part_report(
                LocalCompressionPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    variants=(LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK, LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED),
                    confirm_final_test=True,
                    include_parameter_counts=False,
                )
            )

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["comparison_summary"]["primary_metric"], LOCAL_COMPRESSION_PRIMARY_METRIC)
            self.assertEqual(report["comparison_summary"]["primary_metric_direction"], "minimize")
            self.assertEqual(report["comparison_summary"]["best_variant"], LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED)
            row = next(row for row in report["metric_table"] if row["variant"] == LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED)
            self.assertTrue(row["validation_threshold_final_test_available"])
            self.assertEqual(row[LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC], 0.0)
            self.assertTrue((root / "final_report" / "local_compression_part_final_report.json").exists())
            self.assertTrue((root / "final_report" / "metric_table.csv").exists())


if __name__ == "__main__":
    unittest.main()
