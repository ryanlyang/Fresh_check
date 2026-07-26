#!/usr/bin/env python3
"""Export a selected particle-view model through a registered factory hook."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view.deployment import (  # noqa: E402
    export_hlt_only_particle_view_bundle,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("registered bundle factory is not callable")
    return factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--predictor-config", type=Path, required=True)
    parser.add_argument("--consumer-config", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--lineage-graph-sha256", required=True)
    parser.add_argument("--fairness-ledger-sha256", required=True)
    parser.add_argument("--split-authorization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    built = _factory(args.factory)(_json(args.factory_config))
    required = {"predictor", "consumer", "exemplar_hlt_inputs"}
    if not isinstance(built, dict) or set(built) != required:
        raise ValueError(f"bundle factory must return exactly {sorted(required)}")
    artifact = export_hlt_only_particle_view_bundle(
        args.output_dir,
        predictor=built["predictor"],
        consumer=built["consumer"],
        exemplar_hlt_inputs=built["exemplar_hlt_inputs"],
        deployment_manifest=_json(args.deployment_manifest),
        source_bundle_sha256=args.source_bundle_sha256,
        predictor_config=_json(args.predictor_config),
        consumer_config=_json(args.consumer_config),
        lineage_graph_sha256=args.lineage_graph_sha256,
        fairness_ledger_sha256=args.fairness_ledger_sha256,
        split_authorization_sha256=args.split_authorization_sha256,
    )
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
