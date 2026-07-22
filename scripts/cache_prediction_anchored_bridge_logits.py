#!/usr/bin/env python3
"""Publish one immutable, binding-locked bridge-teacher logit namespace.

The expensive teacher forward is deliberately separated from publication.  Its
RAM-local NPZ must contain only ``logits``, ``labels``, and ``event_ids``; no
dense bridge field is accepted or persisted by this command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    ALLOWED_NAMESPACES,
    build_live_teacher_config,
    validate_teacher_binding,
    write_teacher_logit_cache,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--input-npz", default="", help="RAM-local detached teacher outputs")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stack-train-distill-sha256", required=True)
    parser.add_argument("--class-order", required=True, help="JSON list or path to one")
    parser.add_argument("--temperature-convention", default="raw_logits__temperature_applied_in_kd_loss")
    parser.add_argument("--namespace", choices=ALLOWED_NAMESPACES, default=None)
    parser.add_argument("--selected-consumer", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _class_order(value: str) -> list[str]:
    source = Path(value)
    raw = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else json.loads(value)
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("--class-order must resolve to a non-empty JSON string list")
    return raw


def _input(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing/unsafe RAM-local teacher output: {source}")
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != {"logits", "labels", "event_ids"}:
            raise ValueError("teacher output NPZ must contain exactly logits, labels, event_ids")
        logits = np.asarray(payload["logits"], dtype=np.float32).copy()
        labels = np.asarray(payload["labels"], dtype=np.int64).copy()
        event_ids = [str(value) for value in np.asarray(payload["event_ids"]).tolist()]
    return logits, labels, event_ids


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = load_hashed_json(args.binding)
    selected = load_hashed_json(args.selected_consumer) if args.selected_consumer else None
    validation = validate_teacher_binding(
        binding,
        expected_kind=str(binding["binding_kind"]),
        primary_selection=selected,
    )
    live = build_live_teacher_config(binding, primary_selection=selected)
    classes = _class_order(args.class_order)
    namespace = str(args.namespace or binding["target_cache_namespace"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "submission_executed": False,
                    "binding_validation": validation,
                    "live_teacher_config": live,
                    "namespace": namespace,
                    "class_order": classes,
                    "input_npz_required_for_execution": True,
                    "persistent_dense_fields_written": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.input_npz:
        raise ValueError("cache publication requires --input-npz unless --dry-run")
    logits, labels, event_ids = _input(args.input_npz)
    manifest = write_teacher_logit_cache(
        binding=binding,
        logits=logits,
        labels=labels,
        event_ids=event_ids,
        stack_train_distill_manifest_sha256=args.stack_train_distill_sha256,
        class_order=classes,
        temperature_convention=args.temperature_convention,
        output_root=args.output_root,
        namespace=namespace,
        live_teacher_config=live,
        primary_selection=selected,
    )
    print(json.dumps({"dry_run": False, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
