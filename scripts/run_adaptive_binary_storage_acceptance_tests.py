#!/usr/bin/env python3
"""Run the frozen Step-10 component/parity suite and write immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance import (  # noqa: E402
    ABPH_STORAGE_ACCEPTANCE_TEST_FILES,
    ABPH_STORAGE_TEST_EVIDENCE_CONTRACT,
    canonical_hash,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    write_quota_managed_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _git(*args: str) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(REPO_ROOT), *args), stderr=subprocess.DEVNULL
    )


_ISOLATED_ENV_PREFIXES = ("ABPH_", "SLURM_")
_ISOLATED_ENV_NAMES = {
    "DATA_DIR",
    "DEVICE",
    "DIAGNOSTICS_ROOT",
    "DRY_RUN",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "LOG_DIR",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MIRROR_DIAGNOSTICS",
    "OUTPUT_ROOT",
    "OVERWRITE",
    "PRINT_ONLY",
    "RANK",
    "ROLE_NAME",
    "ROLE_RANK",
    "WORLD_SIZE",
}


def isolated_test_environment(
    source: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove live campaign and launcher state from the component tests."""

    environment = dict(os.environ if source is None else source)
    removed = tuple(
        sorted(
            name
            for name in environment
            if name in _ISOLATED_ENV_NAMES
            or name.startswith(_ISOLATED_ENV_PREFIXES)
        )
    )
    for name in removed:
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment, removed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *ABPH_STORAGE_ACCEPTANCE_TEST_FILES,
    )
    test_environment, removed_environment_names = isolated_test_environment()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=test_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    passed_match = re.search(r"(\d+) passed", output)
    skipped_match = re.search(r"(\d+) skipped", output)
    source_commit = _git("rev-parse", "HEAD").decode().strip()
    source_status = _git("status", "--short")
    payload = {
        "contract": ABPH_STORAGE_TEST_EVIDENCE_CONTRACT,
        "ok": completed.returncode == 0 and passed_match is not None,
        "return_code": int(completed.returncode),
        "test_files": list(ABPH_STORAGE_ACCEPTANCE_TEST_FILES),
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "skipped": int(skipped_match.group(1)) if skipped_match else 0,
        "source_git_commit": source_commit,
        "source_status_hash": hashlib.sha256(source_status).hexdigest(),
        "python_executable": sys.executable,
        "pytest_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "pytest_output_tail": output[-4000:],
        "removed_campaign_environment_names": list(removed_environment_names),
    }
    payload["content_hash"] = canonical_hash(payload)
    write_quota_managed_json(
        args.campaign_root,
        args.output,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_acceptance_test_evidence",
        source_provenance_hash=payload["content_hash"],
        run_id="step10_component_parity_tests",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
