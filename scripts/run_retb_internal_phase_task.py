#!/usr/bin/env python3
"""Execute one authenticated internal phased-controller row."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    INTERNAL_PHASE_PLAN_CONTRACT,
    execute_internal_phase_row,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    is_streamed_profile,
    task_local_workspace,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--phase-plan", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    plan = load_hashed_json(
        args.phase_plan, expected_contract=INTERNAL_PHASE_PLAN_CONTRACT
    )
    if (
        Path(plan["campaign_root"]).resolve()
        != args.campaign_root.resolve()
        or plan.get("source") != campaign.get("source")
    ):
        raise ValueError("internal phase campaign lineage differs")
    index = args.task_index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit():
            raise ValueError("internal phase task index is absent")
        index = int(raw)
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    if is_streamed_profile(graph.get("execution_profile")):
        with task_local_workspace(
            campaign_id=str(campaign["campaign_id"]),
            node_id=f"{plan['controller_id']}_{plan['phase_id']}",
            task_index=index,
        ) as workspace:
            keys = ("RETB_TASK_LOCAL_ROOT", "TMPDIR", "TMP", "TEMP", "TORCH_EXTENSIONS_DIR")
            previous = {key: os.environ.get(key) for key in keys}
            for key in keys[:-1]:
                os.environ[key] = str(workspace)
            os.environ["TORCH_EXTENSIONS_DIR"] = str(workspace / "torch_extensions")
            try:
                result = execute_internal_phase_row(
                    plan=plan, task_index=index, repo_root=REPO_ROOT
                )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    else:
        result = execute_internal_phase_row(
            plan=plan, task_index=index, repo_root=REPO_ROOT
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
