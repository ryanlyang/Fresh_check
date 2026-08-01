#!/usr/bin/env python3
"""Run the complete PT/TRACK design inference wave and lock its optimizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    OfflineExpertDataset,
    collect_expert_diagnostics,
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    task_manifest_path_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (  # noqa: E402
    build_locked_optimization_selection,
    build_optimization_candidate_metrics,
    validate_stage_b_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.profiling import (  # noqa: E402
    profile_model_resources,
)


DESIGN_EVIDENCE_CONTRACT = "retb_offline_expert_val_design_v1"
FOLLOWUP_ROWS_CONTRACT = "retb_stage_b_optimization_followup_rows_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        encoded = temporary.read_bytes()
        if path.exists():
            if path.is_symlink() or path.read_bytes() != encoded:
                raise FileExistsError(
                    "offline expert design prediction differs on reuse"
                )
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


class _ProfileWrapper(torch.nn.Module):
    def __init__(self, model: RetbExpertModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        raw_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            features=features,
            vectors=vectors,
            mask=mask,
            raw_tokens=raw_tokens,
        )


def _particle_encoder_options(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover every checkpoint-bound particle-encoder semantic option."""

    return {
        "measurement_embedding": bool(configuration["measurement_embedding"]),
        "dual_base4_capacity_control": bool(
            configuration.get("dual_base4_capacity_control", False)
        ),
        "activation_checkpointing": False,
        "particle_dropout": float(configuration["particle_dropout"]),
    }


def _model(
    *,
    configuration: Mapping[str, Any],
    relation_normalization: Mapping[str, Any],
    region_normalization: Mapping[str, Any] | None = None,
) -> RetbExpertModel:
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    encoder = RetbParticleEncoder(
        expert_id=configuration["expert_id"],
        topology=configuration["topology"],
        weaver_module=weaver,
        normalization_artifact=relation_normalization,
        region_normalization_artifact=region_normalization,
        **_particle_encoder_options(configuration),
    )
    return RetbExpertModel(
        particle_encoder=encoder,
        shape_id=configuration["shape_id"],
        tokenizer_mode=configuration["tokenizer_mode"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_b_runs.json"
    )
    validate_stage_b_run_registry(registry)
    relation = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "normalization"
        / "offline_500k"
        / "relation.json"
    )
    input_manifest = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "offline"
        / "val_design"
        / "offline_input_manifest.json"
    )
    input_npz = (
        args.campaign_root
        / "inputs"
        / "offline"
        / "val_design"
        / "offline_inputs.npz"
    )
    if input_manifest.get("npz_sha256") != _sha256(input_npz):
        raise ValueError("val_design offline input bytes differ")
    with np.load(input_npz, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    dataset = OfflineExpertDataset(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        labels=arrays["labels"],
        identities=[
            str(value) for value in arrays["identities"].tolist()
        ],
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    rows = []
    for member in registry["full_optimization_grid"]:
        run_id = str(member["run_id"])
        configuration = member["configuration"]
        run_root = (
            args.campaign_root
            / "runs"
            / "stage_b"
            / run_id
            / f"seed_{member['seed']}"
        )
        registration = load_hashed_json(
            run_root / "checkpoint_registration.json"
        )
        checkpoint_path = run_root / "best_model_val.pt"
        if (
            registration.get("run_id") != run_id
            or registration.get("checkpoint_sha256")
            != _sha256(checkpoint_path)
            or registration.get("fixed_epoch_budget_completed") is not True
        ):
            raise ValueError("optimization checkpoint lineage differs")
        model = _model(
            configuration=configuration,
            relation_normalization=relation,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        loader = make_offline_expert_loader(
            dataset,
            seed=0,
            training=False,
            batch_size=args.batch_size,
        )
        _, prediction = collect_expert_diagnostics(
            model, loader, device=device
        )
        metrics = evaluate_classification(
            prediction["logits"],
            prediction["labels"],
            split="val_design",
        )
        evidence_root = args.output_dir / "candidates" / run_id
        prediction_path = evidence_root / "val_design_predictions.npz"
        prediction_sha = _publish_npz(
            prediction_path,
            {
                "identities": prediction["identities"],
                "labels": prediction["labels"],
                "logits": prediction["logits"],
            },
        )
        first_batch = next(iter(loader))
        example = {
            name: value[:1]
            for name, value in first_batch.items()
            if name in {"features", "vectors", "mask", "raw_tokens"}
        }
        profile = profile_model_resources(
            _ProfileWrapper(model),
            example,
            device=device,
            warmup_repetitions=0,
            measured_repetitions=1,
            model_contract_sha256=registration[
                "training_contract_sha256"
            ],
        )
        evidence = bind_source(
            with_content_hash(
                {
                    "contract": DESIGN_EVIDENCE_CONTRACT,
                    "schema_version": 1,
                    "run_id": run_id,
                    "configuration": configuration,
                    "split": "val_design",
                    "checkpoint_sha256": registration[
                        "checkpoint_sha256"
                    ],
                    "prediction_file": prediction_path.name,
                    "prediction_shard_sha256": prediction_sha,
                    "label_manifest_sha256": input_manifest["content_hash"],
                    "metrics": metrics,
                    "parameter_count": int(
                        registration["trainable_parameter_count"]
                    ),
                    "measured_flops": float(
                        profile["forward_flops_per_event"]
                    ),
                    "resource_profile": profile,
                    "checkpoint_selection_affected": False,
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        evidence_path = evidence_root / "val_design_evidence.json"
        write_immutable_json(evidence_path, evidence)
        rows.append(
            {
                "run_id": run_id,
                "configuration": configuration,
                "role": "scientific_candidate",
                "split": "val_design",
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "measured_flops": evidence["measured_flops"],
                "parameter_count": evidence["parameter_count"],
                "checkpoint_sha256": evidence["checkpoint_sha256"],
                "prediction_shard_sha256": prediction_sha,
                "metrics_artifact_sha256": evidence["content_hash"],
                "label_manifest_sha256": input_manifest["content_hash"],
            }
        )
    candidate_metrics = bind_source(
        build_optimization_candidate_metrics(
            run_registry=registry,
            rows=rows,
            val_design_label_manifest_sha256=input_manifest[
                "content_hash"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    selection = build_locked_optimization_selection(
        candidate_metrics=candidate_metrics,
        run_registry=registry,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json"
    )
    manifest = load_hashed_json(
        task_manifest_path_for_graph(
            graph,
            node_id="offline_optimization_selector",
            campaign_root=args.campaign_root,
        )
    )
    followups = bind_source(
        with_content_hash(
            {
                "contract": FOLLOWUP_ROWS_CONTRACT,
                "schema_version": 1,
                "optimization_selection_sha256": selection[
                    "content_hash"
                ],
                "downstream_task_manifest_sha256": manifest["content_hash"],
                "rows": selection["winner_followup_rows"],
                "row_count": len(selection["winner_followup_rows"]),
                "scientific_underperformance_omits_rows": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        args.output_dir / "optimization_candidate_metrics.json",
        candidate_metrics,
    )
    write_immutable_json(
        args.output_dir / "locked_optimization_selection.json",
        selection,
    )
    write_immutable_json(
        args.output_dir / "optimization_followup_rows.json",
        followups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
