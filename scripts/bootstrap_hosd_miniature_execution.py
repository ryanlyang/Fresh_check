#!/usr/bin/env python3
"""Resume all HOSD miniature prerequisites and submit its Slurm DAG.

This controller is intentionally performance blind.  It stops only on a
runtime/integrity failure, is safe to rerun against immutable reusable
artifacts, and never consults validation metrics before submission.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PARENT_STATUS_CONTRACT,
    load_hashed_json,
)


def _run(argv: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        help="Create the miniature first when CAMPAIGN_ROOT has no campaign_spec.json.",
    )
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Complete prerequisites without submitting the scientific DAG.",
    )
    parser.add_argument("--available-storage-bytes", type=int)
    parser.add_argument("--production-batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign_path = root / "campaign_spec.json"
    if not campaign_path.is_file():
        if args.parent_manifest is None:
            raise FileNotFoundError(
                "a new miniature requires --parent-manifest"
            )
        _run(
            [
                sys.executable,
                "-s",
                str(REPO_ROOT / "scripts" / "build_hosd_campaign.py"),
                "--parent-manifest",
                str(args.parent_manifest.resolve()),
                "--output-dir",
                str(root),
                "--campaign-id",
                str(args.campaign_id or root.name),
                "--miniature",
            ]
        )
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    if campaign.get("campaign_profile") != "miniature_test":
        raise ValueError("this controller accepts only miniature_test campaigns")
    python = sys.executable
    common = ["--campaign-root", str(root)]
    for script in (
        "build_hosd_shared_hlt_parents.py",
        "build_hosd_tree_parents.py",
        "fit_hosd_relation_normalizers.py",
    ):
        _run([python, "-s", str(REPO_ROOT / "scripts" / script), *common, "--submit"])
    lock_path = root / "inputs" / "resolved_inherited_parent_lock.json"
    if lock_path.is_file():
        lock = load_hashed_json(
            lock_path, expected_contract=PARENT_STATUS_CONTRACT
        )
        if (
            lock.get("source") != campaign.get("source")
            or lock.get("all_stage_b_parents_reusable") is not True
        ):
            raise ValueError("existing inherited-parent lock is not reusable")
        print(f"Reusing authenticated parent lock: {lock_path}", flush=True)
    else:
        _run(
            [
                python,
                "-s",
                str(REPO_ROOT / "scripts" / "lock_hosd_inherited_parents.py"),
                *common,
            ]
        )
    _run([python, "-s", str(REPO_ROOT / "scripts" / "materialize_hosd_runtime_inputs.py"), *common])
    prepare = [
        python,
        "-s",
        str(REPO_ROOT / "scripts" / "prepare_hosd_execution.py"),
        *common,
        "--profile",
        "miniature_test",
        "--production-batch-size",
        str(args.production_batch_size),
    ]
    if args.available_storage_bytes is not None:
        prepare.extend(
            ["--available-storage-bytes", str(args.available_storage_bytes)]
        )
    _run(prepare)
    if not args.prepare_only:
        environment = dict(os.environ)
        environment["CAMPAIGN_ROOT"] = str(root)
        _run(
            [
                "bash",
                str(REPO_ROOT / "sbatch" / "submit_hosd_tigris_full.sh"),
                "--smoke-submit",
            ],
            environment=environment,
        )
    print(
        "HOSD miniature execution is prepared"
        + ("." if args.prepare_only else " and submitted."),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
