from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ArchitectureView10ClassPredictionCacheConfig,
    architecture_view_10class_prediction_paths,
    cache_architecture_view_10class_predictions,
    load_architecture_view_10class_prediction_metadata,
    validate_architecture_view_10class_prediction_manifest,
)


torch = require_torch()


class TinyTenClassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(RAW_TOKEN_DIM, 10)

    def forward(self, tokens, mask, *, max_constits=None):
        del max_constits
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        return self.linear(pooled)


def make_view(split: str, *, n_jets: int = 20) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 10 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 3 + (jet % 4)
        mask[jet, :valid] = True
        for particle in range(valid):
            pt = 10.0 + jet * 0.1 + particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = -0.5 + 0.1 * particle
            tokens[jet, particle, 2] = -2.0 + 0.2 * particle
            tokens[jet, particle, 3] = pt + 0.5
            tokens[jet, particle, 4] = 1.0 if particle % 2 == 0 else -1.0
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 11] = 0.04
            tokens[jet, particle, 13] = 0.06
    jet_ids = [JetIdentity(file=f"{split}.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    metadata = {
        "view": "fixed_hlt",
        "hlt_content_hash": f"hlt-{split}",
        "jet_identity_hash": jet_identity_hash(jet_ids),
        "hlt_params": {"strength": 0.6},
        "seed": 100 + len(split),
    }
    return JetView(tokens=tokens, mask=mask, labels=labels, jet_ids=jet_ids, split=split, metadata=metadata)


def payload(variant: str = ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART) -> dict:
    return {
        "epoch": 3,
        "variant": variant,
        "num_classes": 10,
        "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
        "label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
        "selection_metric": "accuracy",
        "output_contract": "toy_av10_model",
        "metrics": {"selection_metric_value": 0.42},
    }


def test_prediction_cache_runner_writes_shapes_and_metadata(tmp_path: Path):
    variant = ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
    config = ArchitectureView10ClassPredictionCacheConfig(
        output_dir=str(tmp_path / "out"),
        hlt_cache_dir="unused",
        checkpoint_root="unused",
        variants=(variant,),
        splits=("model_val", "stack_val"),
        batch_size=5,
        device="cpu",
        max_model_val_jets=12,
    )

    manifest = cache_architecture_view_10class_predictions(
        config,
        models_by_variant={variant: TinyTenClassModel()},
        payloads_by_variant={variant: payload()},
        views_by_split={
            "model_val": make_view("model_val", n_jets=20),
            "stack_val": make_view("stack_val", n_jets=11),
        },
    )

    assert manifest["contract"] == ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT
    assert len(manifest["prediction_rows"]) == 2
    model_val_npz, model_val_json = architecture_view_10class_prediction_paths(
        tmp_path / "out" / "predictions",
        variant,
        "model_val",
    )
    assert model_val_npz.exists()
    assert model_val_json.exists()
    with np.load(model_val_npz, allow_pickle=False) as data:
        assert data["logits"].shape == (12, 10)
        assert data["labels"].shape == (12,)
        assert data["preds"].shape == (12,)
        assert data["probs"].shape == (12, 10)
    metadata = load_architecture_view_10class_prediction_metadata(
        tmp_path / "out" / "predictions",
        variant,
        "model_val",
    )
    assert metadata["n_jets"] == 12
    assert metadata["label_names"] == list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES)
    assert "binary_projection_metrics" in metadata["metrics"]


def test_prediction_metadata_rejects_mismatched_label_order(tmp_path: Path):
    variant = ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
    config = ArchitectureView10ClassPredictionCacheConfig(
        output_dir=str(tmp_path / "out"),
        hlt_cache_dir="unused",
        checkpoint_root="unused",
        variants=(variant,),
        splits=("model_val",),
        batch_size=8,
        device="cpu",
    )
    cache_architecture_view_10class_predictions(
        config,
        models_by_variant={variant: TinyTenClassModel()},
        payloads_by_variant={variant: payload()},
        views_by_split={"model_val": make_view("model_val", n_jets=10)},
    )
    _npz_path, metadata_path = architecture_view_10class_prediction_paths(
        tmp_path / "out" / "predictions",
        variant,
        "model_val",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["label_names"] = list(reversed(metadata["label_names"]))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="label_names mismatch"):
        load_architecture_view_10class_prediction_metadata(
            tmp_path / "out" / "predictions",
            variant,
            "model_val",
        )


def test_prediction_manifest_rejects_cross_variant_row_order_mismatch():
    row = {
        "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
        "label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
        "split": "stack_val",
        "jet_identity_hash": "hash-a",
    }
    bad = {
        "prediction_rows": [
            row,
            {**row, "variant": "other", "jet_identity_hash": "hash-b"},
        ]
    }
    with pytest.raises(ValueError, match="jet order mismatch"):
        validate_architecture_view_10class_prediction_manifest(bad)


def test_prediction_config_requires_final_test_confirmation(tmp_path: Path):
    with pytest.raises(ValueError, match="final_test"):
        ArchitectureView10ClassPredictionCacheConfig(
            output_dir=str(tmp_path / "out"),
            hlt_cache_dir="hlt",
            checkpoint_root="ckpts",
            variants=(ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,),
            splits=("final_test",),
        )
