#!/usr/bin/env python3
"""Calibrate the realistic HLT v2 profile on a small JetClass subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fixed_hlt import (  # noqa: E402
    HLT_PROFILE_V2_REALISTIC,
    build_fixed_hlt_v2_realistic_view,
    scaled_fixed_hlt_v2_realistic_params,
    summarize_hlt_diagnostics,
    wrap_phi_np,
)
from jetclass_fresh.hlt_cache import DEFAULT_HLT_SEEDS, fixed_hlt_params_dict  # noqa: E402
from jetclass_fresh.jetclass_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    JetView,
    SplitManifest,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)


DEFAULT_STRENGTHS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Split manifest path (.json or .json.gz)")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=None,
        help=f"One or more JetClass data directories; defaults to manifest data_dir or {DEFAULT_DATA_DIR}",
    )
    parser.add_argument("--split", default="model_val", choices=list(SPLIT_ORDER))
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required if --split final_test is used. Calibration should normally avoid final_test.",
    )
    parser.add_argument("--output-dir", default="checkpoints/hlt_v2_calibration")
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    parser.add_argument("--seed", type=int, default=19001, help="Subset sampling seed")
    parser.add_argument("--hlt-seed", type=int, default=None, help="HLT RNG seed; defaults to split-specific seed")
    parser.add_argument("--max-jets", type=int, default=20_000)
    parser.add_argument("--max-jets-per-class", type=int, default=None)
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=np.abs(denominator) > 1e-12)
    return out


def _weighted_phi(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.asarray(tokens[:, :, 0], dtype=np.float64) * mask
    phi = np.asarray(tokens[:, :, 2], dtype=np.float64)
    sin_sum = np.sum(pt * np.sin(phi), axis=1)
    cos_sum = np.sum(pt * np.cos(phi), axis=1)
    return np.arctan2(sin_sum, cos_sum)


def _weighted_eta(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pt = np.asarray(tokens[:, :, 0], dtype=np.float64) * mask
    eta = np.asarray(tokens[:, :, 1], dtype=np.float64)
    return _safe_divide(np.sum(pt * eta, axis=1), np.sum(pt, axis=1))


def response_summary(offline_tokens: np.ndarray, offline_mask: np.ndarray, hlt_tokens: np.ndarray, hlt_mask: np.ndarray) -> dict[str, float]:
    """Compute jet-level response shifts for calibration diagnostics."""

    off_pt = np.sum(np.asarray(offline_tokens[:, :, 0], dtype=np.float64) * offline_mask, axis=1)
    hlt_pt = np.sum(np.asarray(hlt_tokens[:, :, 0], dtype=np.float64) * hlt_mask, axis=1)
    frac_shift = _safe_divide(hlt_pt - off_pt, np.maximum(off_pt, 1e-12))

    off_energy = np.sum(np.asarray(offline_tokens[:, :, 3], dtype=np.float64) * offline_mask, axis=1)
    hlt_energy = np.sum(np.asarray(hlt_tokens[:, :, 3], dtype=np.float64) * hlt_mask, axis=1)
    energy_frac_shift = _safe_divide(hlt_energy - off_energy, np.maximum(off_energy, 1e-12))

    off_eta = _weighted_eta(offline_tokens, offline_mask)
    hlt_eta = _weighted_eta(hlt_tokens, hlt_mask)
    off_phi = _weighted_phi(offline_tokens, offline_mask)
    hlt_phi = _weighted_phi(hlt_tokens, hlt_mask)
    eta_shift = hlt_eta - off_eta
    phi_shift = wrap_phi_np(hlt_phi - off_phi)

    return {
        "jet_pt_frac_shift_mean": float(np.mean(frac_shift)) if frac_shift.size else 0.0,
        "jet_pt_abs_frac_shift_p50": float(np.percentile(np.abs(frac_shift), 50)) if frac_shift.size else 0.0,
        "jet_pt_abs_frac_shift_p90": float(np.percentile(np.abs(frac_shift), 90)) if frac_shift.size else 0.0,
        "jet_energy_frac_shift_mean": float(np.mean(energy_frac_shift)) if energy_frac_shift.size else 0.0,
        "jet_energy_abs_frac_shift_p90": float(np.percentile(np.abs(energy_frac_shift), 90)) if energy_frac_shift.size else 0.0,
        "jet_eta_abs_shift_p90": float(np.percentile(np.abs(eta_shift), 90)) if eta_shift.size else 0.0,
        "jet_phi_abs_shift_p90": float(np.percentile(np.abs(phi_shift), 90)) if phi_shift.size else 0.0,
    }


def _diagnostics_for_mask(diagnostics: Mapping[str, np.ndarray], take: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[take] for key, value in diagnostics.items()}


def per_class_diagnostics(diagnostics: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    rows: dict[str, float] = {}
    for label_index, label_name in enumerate(LABEL_NAMES):
        take = labels == int(label_index)
        prefix = f"class_{label_name}"
        rows[f"{prefix}_n_jets"] = int(np.count_nonzero(take))
        if not np.any(take):
            rows[f"{prefix}_drop_total_fraction"] = 0.0
            rows[f"{prefix}_mean_hlt_constits"] = 0.0
            continue
        summary = summarize_hlt_diagnostics(_diagnostics_for_mask(diagnostics, take))
        rows[f"{prefix}_drop_total_fraction"] = summary["drop_total_fraction"]
        rows[f"{prefix}_mean_hlt_constits"] = summary["mean_hlt_constits"]
    return rows


def evaluate_strength(
    offline_view: JetView,
    *,
    strength: float,
    hlt_seed: int,
    show_progress: bool = False,
) -> dict[str, Any]:
    params = scaled_fixed_hlt_v2_realistic_params(float(strength))
    hlt_tokens, hlt_mask, diagnostics = build_fixed_hlt_v2_realistic_view(
        offline_view.tokens,
        offline_view.mask,
        seed=int(hlt_seed),
        params=params,
        show_progress=show_progress,
    )
    row: dict[str, Any] = {
        "hlt_profile": HLT_PROFILE_V2_REALISTIC,
        "hlt_profile_version": params.profile_version,
        "strength": float(strength),
        "n_jets": int(offline_view.tokens.shape[0]),
        "hlt_params": fixed_hlt_params_dict(params),
    }
    row.update(summarize_hlt_diagnostics(diagnostics))
    row.update(response_summary(offline_view.tokens, offline_view.mask, hlt_tokens, hlt_mask))
    row.update(per_class_diagnostics(diagnostics, offline_view.labels))
    return row


def evaluate_strengths(
    offline_view: JetView,
    *,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    hlt_seed: int,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    return [
        evaluate_strength(offline_view, strength=float(strength), hlt_seed=hlt_seed, show_progress=show_progress)
        for strength in strengths
    ]


def select_balanced_identities(
    identities: Sequence[JetIdentity],
    *,
    max_jets: int | None,
    max_jets_per_class: int | None,
    seed: int,
) -> list[JetIdentity]:
    if max_jets is not None and int(max_jets) <= 0:
        raise ValueError("max_jets must be positive when provided")
    if max_jets_per_class is not None and int(max_jets_per_class) <= 0:
        raise ValueError("max_jets_per_class must be positive when provided")

    groups: dict[int, list[int]] = {}
    for index, identity in enumerate(identities):
        groups.setdefault(int(identity.label), []).append(index)
    present_labels = sorted(groups)
    if not present_labels:
        return []

    if max_jets_per_class is None:
        if max_jets is None:
            per_class = max(len(groups[label]) for label in present_labels)
        else:
            per_class = max(1, int(max_jets) // len(present_labels))
    else:
        per_class = int(max_jets_per_class)

    rng = np.random.RandomState(int(seed))
    selected: list[int] = []
    for label in present_labels:
        indices = np.array(groups[label], dtype=np.int64)
        rng.shuffle(indices)
        selected.extend(indices[: min(per_class, len(indices))].tolist())
    if max_jets is not None and len(selected) > int(max_jets):
        selected_array = np.array(selected, dtype=np.int64)
        rng.shuffle(selected_array)
        selected = selected_array[: int(max_jets)].tolist()
    selected = sorted(selected)
    return [identities[index] for index in selected]


def subset_manifest_for_split(
    manifest: SplitManifest,
    *,
    split: str,
    selected_identities: Sequence[JetIdentity],
) -> SplitManifest:
    splits = {name: [] for name in SPLIT_ORDER}
    splits[split] = list(selected_identities)
    split_sizes = {name: len(rows) for name, rows in splits.items()}
    return SplitManifest(
        data_dir=manifest.data_dir,
        max_constits=manifest.max_constits,
        class_names=list(manifest.class_names),
        file_prefix_to_label=dict(manifest.file_prefix_to_label),
        split_sizes=split_sizes,
        split_seeds=dict(manifest.split_seeds),
        file_records=list(manifest.file_records),
        splits=splits,
        metadata={
            **dict(manifest.metadata),
            "calibration_subset_of_split": split,
            "calibration_subset_size": len(selected_identities),
            "source_manifest_hash": manifest_hash(manifest),
        },
    )


def _flatten_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_csv_row(row))


def _format_float(value: Any, precision: int = 6) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    split: str,
    subset_size: int,
    source_manifest_hash: str | None,
    subset_manifest_hash: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for row in rows:
        key_rows.append(
            [
                _format_float(row["strength"], 2),
                int(row["n_jets"]),
                _format_float(row["drop_total_fraction"]),
                _format_float(row["drop_eff_fraction"]),
                _format_float(row["drop_merge_fraction"]),
                _format_float(row["drop_threshold_fraction"]),
                _format_float(row["mean_offline_constits"], 3),
                _format_float(row["mean_hlt_constits"], 3),
                _format_float(row["jet_pt_frac_shift_mean"]),
                _format_float(row["jet_pt_abs_frac_shift_p90"]),
                _format_float(row["jet_eta_abs_shift_p90"]),
                _format_float(row["jet_phi_abs_shift_p90"]),
            ]
        )

    class_lines = []
    for row in rows:
        class_rows = []
        for label_name in LABEL_NAMES:
            class_rows.append(
                [
                    label_name,
                    int(row.get(f"class_{label_name}_n_jets", 0)),
                    _format_float(row.get(f"class_{label_name}_drop_total_fraction", 0.0)),
                    _format_float(row.get(f"class_{label_name}_mean_hlt_constits", 0.0), 3),
                ]
            )
        class_lines.append(f"### Strength {_format_float(row['strength'], 2)}")
        class_lines.append(markdown_table(["class", "n", "drop_total_fraction", "mean_hlt_constits"], class_rows))

    body = [
        "# HLT V2 Calibration Summary",
        "",
        f"- `hlt_profile`: `{HLT_PROFILE_V2_REALISTIC}`",
        f"- `split`: `{split}`",
        f"- `subset_size`: `{subset_size}`",
        f"- `source_manifest_hash`: `{source_manifest_hash}`",
        f"- `subset_manifest_hash`: `{subset_manifest_hash}`",
        "",
        "## Aggregate Sweep",
        "",
        markdown_table(
            [
                "strength",
                "n",
                "drop_total",
                "drop_eff",
                "drop_merge",
                "drop_threshold",
                "mean_offline_n",
                "mean_hlt_n",
                "pt_shift_mean",
                "pt_abs_shift_p90",
                "eta_abs_p90",
                "phi_abs_p90",
            ],
            key_rows,
        ),
        "",
        "## Per-Class Sweep",
        "",
        "\n\n".join(class_lines),
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "final_test" and not bool(args.confirm_final_test):
        raise ValueError("--confirm-final-test is required when calibrating on final_test")
    manifest = load_split_manifest(args.manifest)
    source_manifest_hash = manifest_hash(manifest)
    selected = select_balanced_identities(
        manifest.splits[args.split],
        max_jets=args.max_jets,
        max_jets_per_class=args.max_jets_per_class,
        seed=args.seed,
    )
    subset_manifest = subset_manifest_for_split(manifest, split=args.split, selected_identities=selected)
    subset_manifest_hash = manifest_hash(subset_manifest)
    data_dir = args.data_dir if args.data_dir is not None else (manifest.data_dir or DEFAULT_DATA_DIR)
    offline_view = load_offline_view(
        subset_manifest,
        args.split,
        data_dir=data_dir,
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=int(args.read_chunk_size),
    )
    hlt_seed = DEFAULT_HLT_SEEDS[args.split] if args.hlt_seed is None else int(args.hlt_seed)
    rows = evaluate_strengths(
        offline_view,
        strengths=[float(value) for value in args.strengths],
        hlt_seed=hlt_seed,
        show_progress=bool(args.show_progress),
    )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "hlt_v2_calibration_summary.csv"
    md_path = output_dir / "hlt_v2_calibration_summary.md"
    json_path = output_dir / "hlt_v2_calibration_summary.json"
    write_csv(rows, csv_path)
    write_markdown(
        rows,
        md_path,
        split=args.split,
        subset_size=len(selected),
        source_manifest_hash=source_manifest_hash,
        subset_manifest_hash=subset_manifest_hash,
    )
    report = {
        "ok": True,
        "hlt_profile": HLT_PROFILE_V2_REALISTIC,
        "split": args.split,
        "strengths": [float(value) for value in args.strengths],
        "seed": int(args.seed),
        "hlt_seed": int(hlt_seed),
        "max_jets": None if args.max_jets is None else int(args.max_jets),
        "max_jets_per_class": None if args.max_jets_per_class is None else int(args.max_jets_per_class),
        "source_manifest_path": str(Path(args.manifest)),
        "source_manifest_hash": source_manifest_hash,
        "subset_manifest_hash": subset_manifest_hash,
        "subset_size": len(selected),
        "outputs": {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "json": str(json_path),
        },
        "rows": rows,
    }
    write_json_report(report, json_path)
    return report


def main() -> int:
    args = parse_args()
    report = run_calibration(args)
    print(json.dumps(_jsonable({k: v for k, v in report.items() if k != "rows"}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
