#!/usr/bin/env python3
"""Materialize one exact Stage-F/G/H predictor run and its parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (  # noqa: E402
    materialize_predictor_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    run = bind_source(
        materialize_predictor_run(**configuration),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, run)
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "run_sha256": run["content_hash"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
