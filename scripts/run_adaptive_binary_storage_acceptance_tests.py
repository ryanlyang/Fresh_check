#!/usr/bin/env python3
"""Run the frozen Step-10 component/parity suite and write immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *ABPH_STORAGE_ACCEPTANCE_TEST_FILES,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
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
