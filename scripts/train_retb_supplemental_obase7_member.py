#!/usr/bin/env python3
"""Train one fresh ordinary O_BASE member for the OBASE7 control."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts.train_retb_offline_capacity_controls import _train  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    OfflineClassifierAdapter,
    analytical_particle_transformer_flops,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (  # noqa: E402
    SEVEN_SEEDS,
    validate_supplemental_plan,
)
from teacher_logit_reco.relational_part.capacity import pair_encoder_flops  # noqa: E402
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalParticleTransformer,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.seed not in SEVEN_SEEDS:
        raise ValueError("OBASE7 seed is not registered")
    plan = load_hashed_json(args.plan)
    validate_supplemental_plan(plan)
    parent = Path(plan["parent_campaign_root"])
    campaign = load_hashed_json(parent / "campaign_spec.json")
    source = source_snapshot(REPO_ROOT)
    lineage = {
        "supplemental_plan": plan["content_hash"],
        "parent_campaign_spec": campaign["content_hash"],
        "model_train_input": load_hashed_json(
            parent / "inputs/offline/model_train/offline_input_manifest.json"
        )["content_hash"],
        "val_stop_input": load_hashed_json(
            parent / "inputs/offline/val_stop/offline_input_manifest.json"
        )["content_hash"],
        "val_design_input": load_hashed_json(
            parent / "inputs/offline/val_design/offline_input_manifest.json"
        )["content_hash"],
    }
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model = OfflineClassifierAdapter(
        RelationalParticleTransformer(weaver_module=weaver)
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    training, profile = _train(
        root=parent,
        campaign=campaign,
        control_id="O_BASE",
        model=model,
        flops=(
            analytical_particle_transformer_flops(
                configuration=(128, 4, 8, 8, 2)
            )
            + pair_encoder_flops(4, (64, 64, 64))
        ),
        execution_sha=plan["content_hash"],
        lineage=lineage,
        source=source,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        output_suffix=f"member_seed_{args.seed}",
        output_root=args.output_root,
    )
    # The immutable registration/checkpoint/metrics are sufficient for reuse.
    # Successful optimizer recovery state is intentionally not durable in this
    # storage-constrained supplemental campaign.
    resume = (
        args.output_root
        / "O_BASE"
        / f"member_seed_{args.seed}"
        / "resume_state.pt"
    )
    resume.unlink(missing_ok=True)
    print(json.dumps({"training": training, "profile": profile}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
