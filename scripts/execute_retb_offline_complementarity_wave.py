#!/usr/bin/env python3
"""Train all frozen-bank subset readouts and publish complementarity evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.complementarity import (  # noqa: E402
    build_complementarity_report,
    build_subset_readout,
    build_subset_readout_registry,
    subset_experts,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
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
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


SUBSET_METRICS_CONTRACT = "retb_offline_subset_metrics_v2"
COMPLEMENTARITY_SHAPE = "S8_128"
COMPLEMENTARITY_SEED = 101


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forward(
    model: Any,
    arrays: Mapping[str, Any],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    token_banks = {
        name: torch.from_numpy(
            np.asarray(arrays["token_banks"][name][indices])
        ).float().to(device)
        for name in EXPERT_ORDER
    }
    expert_logits = {
        name: torch.from_numpy(
            np.asarray(arrays["expert_logits"][name][indices])
        ).float().to(device)
        for name in EXPERT_ORDER
    }
    if not subset_experts(getattr(model, "_retb_subset_mask", 0)):
        return model(batch_size=len(indices))
    return model(token_banks=token_banks, expert_logits=expert_logits)


def _evaluate(
    model: Any,
    arrays: Mapping[str, Any],
    *,
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(arrays["labels"]), batch_size):
            indices = np.arange(
                start,
                min(start + batch_size, len(arrays["labels"])),
            )
            rows.append(
                _forward(model, arrays, indices, device=device)
                .float()
                .cpu()
                .numpy()
            )
    return evaluate_classification(
        np.concatenate(rows, axis=0),
        arrays["labels"],
        split=split,
    )


def _train(
    *,
    mask: int,
    kind: str,
    train: Mapping[str, Any],
    val_stop: Mapping[str, Any],
    bank_dimensions: Mapping[str, int],
    class_log_prior: Sequence[float],
    epochs: int,
    batch_size: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[Any, dict[str, Any], str | None]:
    torch.manual_seed(41703 + 1000 * int(mask) + (0 if kind == "SUBSET_LOGIT_LINEAR" else 1))
    model = build_subset_readout(
        mask=mask,
        kind=kind,
        bank_dimensions=bank_dimensions,
        class_log_prior=class_log_prior,
    )
    setattr(model, "_retb_subset_mask", int(mask))
    model.to(device)
    if mask == 0:
        metrics = _evaluate(
            model,
            val_stop,
            split="val_stop",
            device=device,
            batch_size=batch_size,
        )
        return model, metrics, None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-4, weight_decay=1.0e-4
    )
    best = None
    best_metrics = None
    labels = np.asarray(train["labels"], dtype=np.int64)
    for epoch in range(1, epochs + 1):
        generator = np.random.default_rng(
            41703 + 1000 * int(mask) + epoch
        )
        order = generator.permutation(len(labels))
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, train, indices, device=device)
            truth = torch.from_numpy(labels[indices]).long().to(device)
            loss = torch.nn.functional.cross_entropy(logits, truth)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("subset objective is nonfinite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("subset gradient is nonfinite")
            optimizer.step()
        current = _evaluate(
            model,
            val_stop,
            split="val_stop",
            device=device,
            batch_size=batch_size,
        )
        key = (
            -float(current["accuracy"]),
            float(current["cross_entropy"]),
            epoch,
        )
        if best is None or key < best[0]:
            best = (
                key,
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            )
            best_metrics = current
    model.load_state_dict(best[1], strict=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "retb_offline_subset_readout_checkpoint_v1",
            "schema_version": 1,
            "bitmask": int(mask),
            "readout": kind,
            "model_state_dict": best[1],
        },
        checkpoint,
    )
    return model, best_metrics, _sha256(checkpoint)


def _cache(root: Path, split: str) -> Path:
    return (
        root
        / "inputs"
        / "fusion_cache"
        / "offline"
        / COMPLEMENTARITY_SHAPE
        / f"seed_{COMPLEMENTARITY_SEED}"
        / split
        / f"{split}_frozen_tokens.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    for split, role in (
        ("model_train", "training_worker"),
        ("val_stop", "training_worker"),
        ("val_design", "design_worker"),
    ):
        authorize_dataset_access(worker_role=role, requested_resource=split)
    loaded = {
        split: load_frozen_token_cache(_cache(args.campaign_root, split))
        for split in ("model_train", "val_stop", "val_design")
    }
    metadata = {name: value[0] for name, value in loaded.items()}
    arrays = {name: value[1] for name, value in loaded.items()}
    lineage = (
        "shape_id",
        "pipeline_seed",
        "allocation",
        "expert_checkpoint_hashes",
    )
    if any(
        metadata[split][key] != metadata["model_train"][key]
        for split in ("val_stop", "val_design")
        for key in lineage
    ):
        raise ValueError("subset frozen-cache lineage differs")
    counts = np.bincount(
        arrays["model_train"]["labels"], minlength=10
    ).astype(np.float64)
    class_log_prior = np.log(counts / counts.sum()).tolist()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    epochs = 2 if campaign["campaign_profile"] == "miniature_test" else 40
    registry = build_subset_readout_registry(
        shape_id=COMPLEMENTARITY_SHAPE,
        pipeline_seed=COMPLEMENTARITY_SEED,
    )
    output_root = args.campaign_root / "reports" / "stage_c"
    checkpoint_root = (
        args.campaign_root / "runs" / "stage_c" / "subset_readouts"
    )
    selected_rows = []
    for row in registry["rows"]:
        mask = int(row["bitmask"])
        candidates = []
        for kind in row["readouts"]:
            checkpoint = (
                checkpoint_root
                / f"mask_{mask:03d}"
                / kind
                / "best_model_val.pt"
            )
            model, stop_metrics, checkpoint_sha = _train(
                mask=mask,
                kind=kind,
                train=arrays["model_train"],
                val_stop=arrays["val_stop"],
                bank_dimensions={
                    name: shape[1]
                    for name, shape in metadata["model_train"][
                        "allocation"
                    ].items()
                },
                class_log_prior=class_log_prior,
                epochs=epochs,
                batch_size=args.batch_size,
                device=device,
                checkpoint=checkpoint,
            )
            design_metrics = _evaluate(
                model,
                arrays["val_design"],
                split="val_design",
                device=device,
                batch_size=args.batch_size,
            )
            candidates.append(
                {
                    "readout": kind,
                    "val_stop_accuracy": stop_metrics["accuracy"],
                    "val_stop_cross_entropy": stop_metrics[
                        "cross_entropy"
                    ],
                    "val_design_accuracy": design_metrics["accuracy"],
                    "val_design_cross_entropy": design_metrics[
                        "cross_entropy"
                    ],
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
        selected = min(
            candidates,
            key=lambda value: (
                -float(value["val_stop_accuracy"]),
                float(value["val_stop_cross_entropy"]),
                str(value["readout"]),
            ),
        )
        selected_rows.append(
            {
                "subset_id": row["subset_id"],
                "bitmask": mask,
                "experts": row["experts"],
                "accuracy": selected["val_design_accuracy"],
                "cross_entropy": selected["val_design_cross_entropy"],
                "selected_readout": selected["readout"],
                "candidate_readouts": candidates,
                "selection_split": "val_stop",
                "reporting_split": "val_design",
            }
        )
    subset_artifact = bind_source(
        with_content_hash(
            {
                "contract": SUBSET_METRICS_CONTRACT,
                "schema_version": 2,
                "shape_id": COMPLEMENTARITY_SHAPE,
                "pipeline_seed": COMPLEMENTARITY_SEED,
                "subset_registry_sha256": registry["content_hash"],
                "model_train_cache_sha256": metadata["model_train"][
                    "content_hash"
                ],
                "val_stop_cache_sha256": metadata["val_stop"][
                    "content_hash"
                ],
                "val_design_cache_sha256": metadata["val_design"][
                    "content_hash"
                ],
                "subset_count": 128,
                "readout_run_count": 255,
                "rows": selected_rows,
                "all_registered_readouts_trained": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    sensitivity_path = _cache(
        args.campaign_root, "val_design"
    ).parent / "val_design_relation_sensitivity.npz"
    with np.load(sensitivity_path, allow_pickle=False) as payload:
        expected = {
            f"{mode}_{expert}"
            for expert in EXPERT_ORDER
            if expert != "BASE4"
            for mode in ("zero", "within_jet_cyclic")
        }
        if set(payload.files) != expected:
            raise ValueError("relation sensitivity coverage differs")
        sensitivity = {
            name: np.asarray(payload[name]) for name in payload.files
        }
    labels = arrays["val_design"]["labels"]
    active_accuracy = {
        expert: evaluate_classification(
            arrays["val_design"]["expert_logits"][expert],
            labels,
            split="val_design",
        )["accuracy"]
        for expert in EXPERT_ORDER
        if expert != "BASE4"
    }
    bias_zero = {}
    shuffled = {}
    for expert in active_accuracy:
        bias_zero[expert] = active_accuracy[expert] - evaluate_classification(
            sensitivity[f"zero_{expert}"],
            labels,
            split="val_design",
        )["accuracy"]
        shuffled[expert] = active_accuracy[expert] - evaluate_classification(
            sensitivity[f"within_jet_cyclic_{expert}"],
            labels,
            split="val_design",
        )["accuracy"]
    selected_by_mask = {
        int(row["bitmask"]): {
            "accuracy": row["accuracy"],
            "cross_entropy": row["cross_entropy"],
        }
        for row in selected_rows
    }
    full = 127
    leave_one_out = {
        expert: selected_by_mask[full & ~(1 << index)]
        for index, expert in enumerate(EXPERT_ORDER)
    }
    report = bind_source(
        build_complementarity_report(
            shape_id=COMPLEMENTARITY_SHAPE,
            pipeline_seed=COMPLEMENTARITY_SEED,
            cache_manifest_sha256=metadata["val_design"][
                "content_hash"
            ],
            logits_by_expert=arrays["val_design"]["expert_logits"],
            labels=labels,
            tokens_by_expert=arrays["val_design"]["token_banks"],
            subset_metrics=selected_by_mask,
            leave_one_out_metrics=leave_one_out,
            bias_zero_sensitivity=bias_zero,
            relation_shuffle_sensitivity=shuffled,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        output_root / "subset_readout_metrics.json", subset_artifact
    )
    write_immutable_json(
        output_root / "offline_complementarity.json", report
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
