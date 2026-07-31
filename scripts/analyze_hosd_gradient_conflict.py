#!/usr/bin/env python3
"""Build authenticated Stage-F redundancy and gradient-conflict diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_gradient_conflict_report,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    with np.load(args.inputs, allow_pickle=False) as payload:
        identities = [str(value) for value in payload["identities"].tolist()]
        families = [str(value) for value in payload["families"].tolist()]
        residuals = {family: payload[f"residuals__{family}"] for family in families}
        errors = {family: payload[f"errors__{family}"] for family in families}
        representations = {
            family: payload[f"representations__{family}"] for family in families
        }
        gradients = {
            key.removeprefix("gradient_cosines__"): payload[key].tolist()
            for key in payload.files
            if key.startswith("gradient_cosines__")
        }
        leave_one_out = {
            family: float(payload[f"leave_one_out__{family}"])
            for family in families
        }
    artifact = build_gradient_conflict_report(
        identities=identities,
        residuals_by_family=residuals,
        target_errors_by_family=errors,
        gradient_cosines=gradients,
        representations_by_family=representations,
        leave_one_out_accuracy_change=leave_one_out,
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root / "combinations" / "gradient_conflict_report.json"
    )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
