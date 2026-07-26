#!/usr/bin/env python3
"""Register an existing offline ParT as selectable or diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_existing_teacher_source_registration,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canonical-train-identity-sha256", required=True)
    parser.add_argument("--observed-train-identity-sha256")
    parser.add_argument("--serialized-recipe")
    parser.add_argument("--recipe-reproduced-exactly", action="store_true")
    parser.add_argument("--provenance-metadata-sha256", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    recipe = None
    if args.serialized_recipe:
        recipe = json.loads(
            Path(args.serialized_recipe).read_text(encoding="utf-8")
        )
        if "content_hash" in recipe:
            recipe = dict(recipe)
            recipe.pop("content_hash")
    registration = build_existing_teacher_source_registration(
        checkpoint_path=args.checkpoint,
        canonical_train_identity_sha256=args.canonical_train_identity_sha256,
        observed_train_identity_sha256=args.observed_train_identity_sha256,
        serialized_recipe=recipe,
        recipe_reproduced_exactly=args.recipe_reproduced_exactly,
        provenance_metadata_sha256=args.provenance_metadata_sha256,
        description=args.description,
    )
    receipt = write_immutable_json(args.output, registration)
    print(
        json.dumps(
            {"registration": registration, "publication": receipt},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
