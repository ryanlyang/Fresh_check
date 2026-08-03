#!/usr/bin/env python3
"""Write the exact Slurm submission ledger for the Stage-D matrix smoke."""

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
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_d_matrix_smoke import (  # noqa: E402
    build_stage_d_matrix_smoke_ledger,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--report-job-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    jobs: dict[str, str] = {}
    for raw in args.job:
        if "=" not in raw:
            raise ValueError("Stage-D matrix job binding must be NODE=JOBID")
        node_id, job_id = raw.split("=", 1)
        if not node_id or node_id in jobs:
            raise ValueError("Stage-D matrix job binding is duplicate")
        jobs[node_id] = job_id
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    ledger = build_stage_d_matrix_smoke_ledger(
        production_graph=graph,
        jobs=jobs,
        report_job_id=args.report_job_id,
    )
    publication = write_immutable_json(args.output, ledger)
    print(
        json.dumps(
            {"ledger": ledger, "publication": publication},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
