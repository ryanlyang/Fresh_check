#!/usr/bin/env python3
"""Build the manifest-bound A-D runtime benchmark matrix for C2F."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


BENCHMARK_PLAN_CONTRACT = "constrained_c2f_runtime_benchmark_plan_v1"
REPRESENTATIVE_VARIANTS = ("C1", "C5-B3", "C6", "C4")


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    variant: str
    matrix_case: str
    runtime_profile: str
    precision_mode: str
    train_batch_size: int
    eval_batch_size: int
    lr_schedule: str
    learning_rate: float
    hlt_encoder_lr_scale: float
    num_workers: int
    prefetch_factor: int | None
    hungarian_executor: str
    hungarian_workers: int


def _pairs(values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for value in values:
        try:
            train, evaluate = (int(piece) for piece in value.split(":", 1))
        except ValueError as error:
            raise ValueError(f"invalid batch pair {value!r}; expected TRAIN:EVAL") from error
        if train <= 0 or evaluate <= 0:
            raise ValueError(f"batch pair must be positive: {value!r}")
        pairs.append((train, evaluate))
    if not pairs:
        raise ValueError("at least one candidate batch pair is required")
    return tuple(pairs)


def _positive(values: Iterable[int], label: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{label} must contain positive integers")
    return result


def _run_id(variant: str, case: str, train_batch: int, evaluate_batch: int, workers: int, hungarian: int) -> str:
    safe_variant = variant.lower().replace("-", "_")
    return f"{safe_variant}_{case.lower()}_b{train_batch}_e{evaluate_batch}_w{workers}_h{hungarian}"


def build_benchmark_runs(
    *,
    single_view_candidates: Iterable[str] = ("32:64", "48:96", "64:128"),
    c6_candidates: Iterable[str] = ("16:32", "24:48", "32:64"),
    c4_candidates: Iterable[str] = ("16:32",),
    input_workers: Iterable[int] = (0, 4, 8, 12),
    c4_hungarian_workers: Iterable[int] = (1, 4, 8, 12),
    peak_learning_rates: Iterable[float] = (2.0e-4, 4.0e-4, 6.0e-4),
) -> tuple[BenchmarkRun, ...]:
    """Return the full A-D matrix, including C4 executor calibration."""

    single_pairs = _pairs(single_view_candidates)
    c6_pairs = _pairs(c6_candidates)
    c4_pairs = _pairs(c4_candidates)
    loader_workers = tuple(int(value) for value in input_workers)
    if not loader_workers or any(value < 0 for value in loader_workers):
        raise ValueError("input_workers must contain nonnegative integers")
    hungarian_workers = _positive(c4_hungarian_workers, "c4_hungarian_workers")
    learning_rates = tuple(float(value) for value in peak_learning_rates)
    if not learning_rates or any(value <= 0.0 for value in learning_rates):
        raise ValueError("peak_learning_rates must contain positive values")
    rows: list[BenchmarkRun] = []

    def add(
        variant: str,
        case: str,
        batch_pair: tuple[int, int],
        *,
        profile: str,
        precision: str,
        schedule: str,
        learning_rate: float = 2.0e-4,
        workers: int = 0,
        executor: str = "serial",
        hungarian: int = 1,
    ) -> None:
        # Preserve the planned conservative HLT-encoder cap of [1e-5, 2e-5].
        hlt_encoder_lr_scale = min(0.05, 2.0e-5 / float(learning_rate))
        lr_tag = f"lr{float(learning_rate):.0e}".replace("-", "m")
        rows.append(
            BenchmarkRun(
                run_id=f"{_run_id(variant, case, batch_pair[0], batch_pair[1], workers, hungarian)}_{lr_tag}",
                variant=variant,
                matrix_case=case,
                runtime_profile=profile,
                precision_mode=precision,
                train_batch_size=batch_pair[0],
                eval_batch_size=batch_pair[1],
                lr_schedule=schedule,
                learning_rate=float(learning_rate),
                hlt_encoder_lr_scale=float(hlt_encoder_lr_scale),
                num_workers=workers,
                prefetch_factor=4 if workers > 0 else None,
                hungarian_executor=executor,
                hungarian_workers=hungarian,
            )
        )

    for variant, reference, candidates in (
        ("C1", (16, 32), single_pairs),
        ("C5-B3", (16, 32), single_pairs),
        ("C6", (8, 16), c6_pairs),
    ):
        add(variant, "A", reference, profile="fp32_reference", precision="fp32", schedule="constant")
        add(
            variant,
            "B",
            reference,
            profile="bf16_calibration",
            precision="bf16_forward_fp32_loss",
            schedule="constant",
        )
        for learning_rate in learning_rates:
            for batch_pair in candidates:
                for workers in loader_workers:
                    add(
                        variant,
                        "C",
                        batch_pair,
                        profile="bf16_calibration",
                        precision="bf16_forward_fp32_loss",
                        schedule="constant",
                        workers=workers,
                        learning_rate=learning_rate,
                    )
                    add(
                        variant,
                        "D",
                        batch_pair,
                        profile="bf16_calibration",
                        precision="bf16_forward_fp32_loss",
                        schedule="warmup_cosine",
                        workers=workers,
                        learning_rate=learning_rate,
                    )

    c4_reference = (16, 32)
    add("C4", "A", c4_reference, profile="fp32_reference", precision="fp32", schedule="constant")
    add(
        "C4",
        "B_serial",
        c4_reference,
        profile="bf16_calibration",
        precision="bf16_forward_fp32_loss",
        schedule="constant",
    )
    for workers in hungarian_workers:
        add(
            "C4",
            "B_thread",
            c4_reference,
            profile="bf16_calibration",
            precision="bf16_forward_fp32_loss",
            schedule="constant",
            executor="thread",
            hungarian=workers,
        )
    for learning_rate in learning_rates:
        for batch_pair in c4_pairs:
            for workers in loader_workers:
                for hungarian in hungarian_workers:
                    for case, schedule in (("C", "constant"), ("D", "warmup_cosine")):
                        add(
                            "C4",
                            case,
                            batch_pair,
                            profile="bf16_calibration",
                            precision="bf16_forward_fp32_loss",
                            schedule=schedule,
                            workers=workers,
                            executor="thread",
                            hungarian=hungarian,
                            learning_rate=learning_rate,
                        )

    identifiers = [row.run_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise AssertionError("benchmark matrix generated duplicate run ids")
    return tuple(rows)


def _payload(args: argparse.Namespace) -> dict[str, object]:
    rows = build_benchmark_runs(
        single_view_candidates=args.single_view_candidates,
        c6_candidates=args.c6_candidates,
        c4_candidates=args.c4_candidates,
        input_workers=args.input_workers,
        c4_hungarian_workers=args.c4_hungarian_workers,
        peak_learning_rates=args.peak_learning_rates,
    )
    payload: dict[str, object] = {
        "contract": BENCHMARK_PLAN_CONTRACT,
        "calibration_root": str(Path(args.calibration_root).resolve()),
        "calibration_manifest": str(Path(args.calibration_manifest).resolve()),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "warmup_fraction": float(args.warmup_fraction),
        "min_lr_ratio": float(args.min_lr_ratio),
        "representative_variants": list(REPRESENTATIVE_VARIANTS),
        "runs": [asdict(row) for row in rows],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["benchmark_plan_hash"] = sha256(canonical).hexdigest()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--single-view-candidates", nargs="+", default=["32:64", "48:96", "64:128"])
    parser.add_argument("--c6-candidates", nargs="+", default=["16:32", "24:48", "32:64"])
    parser.add_argument("--c4-candidates", nargs="+", default=["16:32"])
    parser.add_argument("--input-workers", nargs="+", type=int, default=[0, 4, 8, 12])
    parser.add_argument("--c4-hungarian-workers", nargs="+", type=int, default=[1, 4, 8, 12])
    parser.add_argument("--peak-learning-rates", nargs="+", type=float, default=[2.0e-4, 4.0e-4, 6.0e-4])
    parser.add_argument("--emit-tsv", action="store_true", help="Print one shell-safe tab-separated row per run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    payload = _payload(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.emit_tsv:
        for row in payload["runs"]:
            assert isinstance(row, dict)
            print(
                "\t".join(
                    str(row[key])
                    for key in (
                        "run_id", "variant", "matrix_case", "runtime_profile", "precision_mode",
                        "train_batch_size", "eval_batch_size", "lr_schedule", "learning_rate", "hlt_encoder_lr_scale", "num_workers",
                        "prefetch_factor", "hungarian_executor", "hungarian_workers",
                    )
                )
            )
    else:
        print(json.dumps({"output": str(output), "runs": len(payload["runs"]), "benchmark_plan_hash": payload["benchmark_plan_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
