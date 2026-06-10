#!/usr/bin/env python3
"""Smoke-test Step 4 teacher-logit loss on one paired batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json  # noqa: E402
from teacher_logit_reco.global_transformer import (  # noqa: E402
    GlobalTransformerReconstructor,
    GlobalTransformerReconstructorConfig,
)
from teacher_logit_reco.losses import (  # noqa: E402
    TeacherLogitRecoLossConfig,
    global_transformer_teacher_training_step,
)
from teacher_logit_reco.teachers import (  # noqa: E402
    TEACHER_ARCHITECTURES,
    assert_teacher_frozen,
    load_frozen_teacher,
)
from teacher_logit_reco.views import load_paired_jet_views, summarize_paired_jet_views, summarize_soft_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="model_train")
    parser.add_argument("--max-jets", type=int, default=8)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-extra-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--correction-budget-weight", type=float, default=0.01)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--output-json", default="checkpoints/teacher_logit_reco_step4_smoke/smoke_report.json")
    return parser.parse_args()


def trainable_grad_norm(module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().cpu())
    return total


def teacher_grad_norm(teacher) -> float:
    total = 0.0
    for param in teacher.model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().cpu())
    return total


def main() -> int:
    torch = require_torch()
    args = parse_args()
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    device = resolve_device(args.device)
    paired = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.split,
        data_dir=args.data_dir,
        max_jets=args.max_jets,
    )
    teacher = load_frozen_teacher(
        args.teacher_checkpoint,
        architecture=args.teacher_architecture,
        device=str(device),
    )
    assert_teacher_frozen(teacher)

    model_config = GlobalTransformerReconstructorConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_extra_candidates=args.num_extra_candidates,
        dropout=0.0,
    )
    loss_config = TeacherLogitRecoLossConfig(
        teacher_kl_weight=args.teacher_kl_weight,
        ce_weight=args.ce_weight,
        correction_budget_weight=args.correction_budget_weight,
        jet_summary_weight=args.jet_summary_weight,
        temperature=args.temperature,
    )
    model = GlobalTransformerReconstructor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    hlt_tokens = torch.as_tensor(paired.hlt.tokens, dtype=torch.float32, device=device)
    hlt_mask = torch.as_tensor(paired.hlt.mask, dtype=torch.bool, device=device)
    offline_tokens = torch.as_tensor(paired.offline.tokens, dtype=torch.float32, device=device)
    offline_mask = torch.as_tensor(paired.offline.mask, dtype=torch.bool, device=device)
    labels = torch.as_tensor(paired.labels, dtype=torch.long, device=device)

    history = []
    last_soft = None
    for step in range(int(args.steps)):
        model.train()
        loss, soft, _, _ = global_transformer_teacher_training_step(
            reconstructor=model,
            teacher=teacher,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            labels=labels,
            optimizer=optimizer,
            config=loss_config,
            jet_ids=paired.jet_ids,
            split=paired.split,
        )
        row = {"step": int(step), **loss.detached_float_dict()}
        row["reconstructor_grad_norm"] = trainable_grad_norm(model)
        row["teacher_grad_norm"] = teacher_grad_norm(teacher)
        history.append(row)
        last_soft = soft

    assert_teacher_frozen(teacher)
    if history[-1]["reconstructor_grad_norm"] <= 0.0:
        raise RuntimeError("Expected nonzero reconstructor gradients in Step 4 smoke test")
    if history[-1]["teacher_grad_norm"] != 0.0:
        raise RuntimeError("Teacher received gradients; frozen-teacher contract is broken")

    report = {
        "ok": True,
        "purpose": "teacher_logit_reco_step4_one_batch_loss_smoke",
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "teacher": teacher.metadata,
        "paired_views": summarize_paired_jet_views(paired),
        "soft_view": summarize_soft_view(last_soft),
        "history": history,
        "leakage_note": (
            "This smoke test uses one model_train/model_val split batch for loss validation only. "
            "No stack or final_test split is loaded."
        ),
    }
    output_path = Path(args.output_json)
    save_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"initial_total_loss={history[0]['total_loss']:.6f}")
    print(f"final_total_loss={history[-1]['total_loss']:.6f}")
    print(f"final_reconstructor_grad_norm={history[-1]['reconstructor_grad_norm']:.6f}")
    print(f"final_teacher_grad_norm={history[-1]['teacher_grad_norm']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
