"""Source and campaign validation at every HOSD wrapper entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)

from .access import authorize_access
from .contracts import (
    CAMPAIGN_SPEC_CONTRACT,
    load_hashed_json,
    source_record,
    validate_content_hash,
)


def validate_campaign_source(
    campaign_spec: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    validate_content_hash(campaign_spec, expected_contract=CAMPAIGN_SPEC_CONTRACT)
    current = source_record(source_snapshot(repo_root))
    if campaign_spec.get("source") != current:
        raise ValueError("active source snapshot differs from HOSD campaign")
    return current


def load_and_validate_campaign(
    campaign_root: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    spec = load_hashed_json(
        Path(campaign_root) / "campaign_spec.json",
        expected_contract=CAMPAIGN_SPEC_CONTRACT,
    )
    validate_campaign_source(spec, repo_root=repo_root)
    return spec


__all__ = [
    "authorize_access",
    "load_and_validate_campaign",
    "validate_campaign_source",
]
