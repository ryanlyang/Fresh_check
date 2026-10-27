#!/usr/bin/env python3
"""Preview/remove only completed-run last.pt files in seven named old campaigns.

Python 3.9+, standard library only. No recursive deletion, scheduler calls,
dataset access, source-snapshot changes, or benchmark/prediction cleanup.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import stat


CAMPAIGNS = (
    ("part_ev_screen/context_gate_ampfix_20261016T215608Z", "campaign.json"),
    ("part_ev_message_study/messages_parityfix_20261019T053047Z", "campaign.json"),
    ("part_layerwise_screen/layerwise_20261019T190751Z", "experiment_spec.json"),
    ("part_fullscale/part_ev_100m_20261013T044756Z", "campaign.json"),
    ("part_fullscale/part_plain_ev_100m_20261014T163310Z", "campaign.json"),
    ("part_fullscale/part_ev_adamw_100m_20261014T193546Z", "campaign.json"),
    ("part_fullscale/part_matched_100m_20261014T194945Z", "campaign.json"),
)


@dataclass(frozen=True)
class Candidate:
    campaign: Path
    manifest: str
    path: Path
    identity: tuple
    evidence: tuple


def regular_file(path, root):
    """Reject symlinks (including parent directories), escapes, and special files."""
    path.relative_to(root)
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError(f"Unexpected symlink/path: {path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return info


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root, campaign, manifest, path):
    relative = campaign.relative_to(root).as_posix()
    if (relative, manifest) not in CAMPAIGNS or path.name != "last.pt":
        raise ValueError("Cleanup target is not allowlisted")
    info = regular_file(path, root)
    if info.st_nlink != 1:
        raise ValueError(f"Refusing multiply linked restart file: {path}")
    run = path.parent
    evidence_paths = (campaign / manifest, run / "training_complete.json",
                      run / "best_model_val.pt")
    evidence_stats = [identity(regular_file(p, root)) for p in evidence_paths]
    spec = json.loads(evidence_paths[0].read_text())
    done = json.loads(evidence_paths[1].read_text())
    task = done["task"]
    expected_run = campaign / "runs" / (task.get("run_id") or task["architecture"]) / f"seed_{task['seed']}"
    recipe = spec["recipes"][task["recipe"]] if "recipes" in spec else spec["recipe"]
    checks = done.get("validated_checks", done.get("validated_epochs"))
    if (run != expected_run or task not in spec["tasks"]
            or not spec.get("content_hash")
            or done["campaign_sha256"] != spec["content_hash"]
            or done["step"] != recipe["total_steps"]
            or checks != recipe["epochs"]):
        raise ValueError(f"Incomplete or mismatched training: {run}")
    evidence_hashes = tuple(sha256(p) for p in evidence_paths)
    if evidence_hashes[2] != done["best_checkpoint_sha256"]:
        raise ValueError(f"Selected-model checksum mismatch: {run}")
    if (identity(regular_file(path, root)) != identity(info)
            or [identity(regular_file(p, root)) for p in evidence_paths] != evidence_stats):
        raise ValueError(f"Files changed during verification: {run}")
    return Candidate(campaign, manifest, path, identity(info), evidence_hashes)


def collect(root):
    root = Path(root).absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("Checkpoint root must not be a symlink")
    candidates = []
    for relative, manifest in CAMPAIGNS:
        campaign = root / relative
        for path in sorted(campaign.glob("runs/*/seed_*/last.pt")):
            if not (path.parent / "training_complete.json").exists():
                print(f"KEEP unfinished run: {path.parent}")
                continue
            candidates.append(audit(root, campaign, manifest, path))
    return candidates


def remove_verified(root, candidates):
    """Recheck the ENTIRE plan before deleting any file; never delete directories."""
    for row in candidates:
        if audit(root, row.campaign, row.manifest, row.path) != row:
            raise ValueError(f"Cleanup plan changed: {row.path}")
    removed = 0
    for row in candidates:
        if identity(regular_file(row.path, root)) != row.identity:
            raise ValueError(f"Restart file changed; stopping: {row.path}")
        row.path.unlink()
        removed += row.identity[2]
        print(f"REMOVED: {row.path}")
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Permanently delete the verified restart files; default is read-only")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1] / "checkpoints"
    candidates = collect(root)
    total = sum(row.identity[2] for row in candidates)
    for row in candidates:
        print(f"{row.identity[2] / 1024**2:8.2f} MiB  {row.path}")
    print(f"\nEligible: {len(candidates)} restart files, {total / 1024**3:.3f} GiB logical size")
    print("Keep all selected models, results, benchmarks, sources, datasets, and active campaign files.")
    if args.apply:
        removed = remove_verified(root, candidates)
        print(f"Removed {removed / 1024**3:.3f} GiB of restart files. No backup was made.")
        print("Exact optimizer-state resume of those completed runs is no longer available.")
    else:
        print("PREVIEW ONLY: nothing deleted. Use --apply only after reviewing the list.")
    print(f"Filesystem free: {shutil.disk_usage(root).free / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
