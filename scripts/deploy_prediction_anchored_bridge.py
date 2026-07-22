#!/usr/bin/env python3
"""Select, confirm, export, and evaluate the repository-owned bridge bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_deployment_execution import (  # noqa: E402
    confirm_deployable_from_execution_spec,
    evaluate_repository_bundle_final_test,
    export_repository_deployable_bundle,
    select_deployable_from_publications,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select")
    select.add_argument("--registry", required=True)
    select.add_argument("--artifact-root", required=True)
    select.add_argument("--r0-checkpoint", required=True)
    select.add_argument("--selected-consumer", required=True)
    select.add_argument("--semantic-evidence-root", required=True)
    select.add_argument("--evidence-output", required=True)
    select.add_argument("--output", required=True)

    confirm = sub.add_parser("confirm")
    confirm.add_argument("--execution-spec", required=True)
    confirm.add_argument("--preconfirmation", required=True)
    confirm.add_argument("--r0-checkpoint", required=True)
    confirm.add_argument("--physical45-scaler", required=True)
    confirm.add_argument("--selected-consumer", required=True)
    confirm.add_argument("--output-dir", required=True)
    confirm.add_argument("--device", default="auto")
    confirm.add_argument("--batch-size", type=int, default=512)

    export = sub.add_parser("export")
    export.add_argument("--execution-spec", required=True)
    export.add_argument("--locked-deployable", required=True)
    export.add_argument("--r0-checkpoint", required=True)
    export.add_argument("--physical45-scaler", required=True)
    export.add_argument("--selected-consumer", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--reservations", required=True)
    export.add_argument("--device", default="cpu")

    final = sub.add_parser("final-test")
    final.add_argument("--bundle", required=True)
    final.add_argument("--locked-deployable", required=True)
    final.add_argument("--clean-reload-audit", required=True)
    final.add_argument("--child-manifest", required=True)
    final.add_argument("--parent-manifest", required=True)
    final.add_argument("--final-hlt-npz", required=True)
    final.add_argument("--final-hlt-metadata", required=True)
    final.add_argument("--output", required=True)
    final.add_argument("--hlt-only", action="store_true")
    final.add_argument("--flags", default="")
    final.add_argument("--device", default="auto")
    final.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        registry = load_hashed_json(args.registry)
        evidence, preconfirmation = select_deployable_from_publications(
            registry,
            artifact_root=args.artifact_root,
            r0_checkpoint_path=args.r0_checkpoint,
            selected_consumer_path=args.selected_consumer,
            semantic_evidence_root=args.semantic_evidence_root,
        )
        evidence_publication = write_immutable_json(args.evidence_output, evidence)
        decision_publication = write_immutable_json(args.output, preconfirmation)
        result = {
            "ok": True,
            "selected_run_id": preconfirmation["selected_run_id"],
            "median_seed_id": preconfirmation["median_seed_id"],
            "evidence_publication": evidence_publication,
            "decision_publication": decision_publication,
        }
    elif args.command == "confirm":
        result = confirm_deployable_from_execution_spec(
            args.execution_spec,
            preconfirmation_path=args.preconfirmation,
            r0_checkpoint_path=args.r0_checkpoint,
            physical45_scaler_path=args.physical45_scaler,
            selected_consumer_path=args.selected_consumer,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=int(args.batch_size),
        )
    elif args.command == "export":
        reservations = load_hashed_json(
            args.reservations,
            expected_contract="prediction_anchored_step9_campaign_reservations_v1",
        )
        result = export_repository_deployable_bundle(
            args.execution_spec,
            locked_deployable_path=args.locked_deployable,
            r0_checkpoint_path=args.r0_checkpoint,
            physical45_scaler_path=args.physical45_scaler,
            selected_consumer_path=args.selected_consumer,
            output_dir=args.output_dir,
            bundle_reservation_bytes=int(
                reservations["final_deployable_bundle_reserved_bytes"]
            ),
            device=args.device,
        )
    else:
        flags = {}
        if args.flags:
            flags = json.loads(Path(args.flags).read_text(encoding="utf-8"))
            if not isinstance(flags, dict):
                raise ValueError("final-test flags must be a JSON object")
        result = evaluate_repository_bundle_final_test(
            bundle_checkpoint_path=args.bundle,
            locked_deployable_path=args.locked_deployable,
            clean_reload_audit_path=args.clean_reload_audit,
            child_manifest_path=args.child_manifest,
            parent_manifest_path=args.parent_manifest,
            final_hlt_npz_path=args.final_hlt_npz,
            final_hlt_metadata_path=args.final_hlt_metadata,
            output_path=args.output,
            hlt_only=bool(args.hlt_only),
            evaluation_flags=flags,
            device=args.device,
            batch_size=int(args.batch_size),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
