#!/usr/bin/env python3
"""Print the fixed compact streamed-smoke physical phase plan."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import SMOKE_PHASES

for row in SMOKE_PHASES:
    print(f"{row['phase_id']}|{row['stage']}|{row['resource']}|{row['kind']}")
