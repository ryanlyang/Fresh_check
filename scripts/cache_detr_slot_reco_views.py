#!/usr/bin/env python3
"""Cache reconstructed views from a trained DETR/free-slot reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.set_matching.detr_slots.cache import (  # noqa: E402
    DEFAULT_DETR_SLOT_CACHE_SPLITS,
    DetrSlotRecoViewCacheConfig,
    cache_detr_slot_reco_views,
)


def label_names_to_indices(values: list[str]) -> tuple[int, ...]:
    if not values:
        return ()
    by_name = {name: index for index, name in enumerate(LABEL_NAMES)}
    output: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            output.append(int(text))
            continue
        if text not in by_name:
            raise ValueError(f"Unknown JetClass label {text!r}; expected one of {list(LABEL_NAMES)}")
        output.append(by_name[text])
    return tuple(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reconstructor-checkpoint", required=True)
    parser.add_argument("--architecture", choices=("gt", "pn", "pfn", "pcnn"), default=None)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_DETR_SLOT_CACHE_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--label-filter-names", nargs="*", default=())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument("--skip-detr-metrics", action="store_true")
    parser.add_argument("--export-max-tokens", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--min-tokens-per-view", type=int, default=8)
    parser.add_argument("--trim-to-valid", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1205)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DetrSlotRecoViewCacheConfig(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        reconstructor_checkpoint=args.reconstructor_checkpoint,
        architecture=args.architecture,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        max_jets_per_split=args.max_jets_per_split,
        label_filter=label_names_to_indices(list(args.label_filter_names)),
        overwrite=args.overwrite,
        skip_existing=not bool(args.no_skip_existing),
        confirm_final_test=args.confirm_final_test,
        strict_checkpoint=not bool(args.non_strict_checkpoint),
        compute_detr_metrics=not bool(args.skip_detr_metrics),
        export_max_tokens=args.export_max_tokens,
        confidence_threshold=args.confidence_threshold,
        min_tokens_per_view=args.min_tokens_per_view,
        trim_to_valid=bool(args.trim_to_valid),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
        seed=args.seed,
    )
    report = cache_detr_slot_reco_views(config)
    print("detr_slot_reconstructed_view_cache_complete:")
    print(f"  architecture: {report['architecture']}")
    print(f"  output_dir: {report['output_dir']}")
    for split, path in report["cache_paths"].items():
        status = "skipped_existing" if report["split_reports"][split].get("skipped_existing") else "wrote"
        print(f"  {split}: {status} {path}")
    print(f"  report: {report['report_path']}")
    print(f"  summary_csv: {report['summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
