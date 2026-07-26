#!/usr/bin/env python3
"""Compile and authenticate the relational_ca_tree_v1 PyTorch extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_CONTRACT,
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    canonical_json_bytes,
    sha256_file,
    validate_backend_manifest,
    with_content_hash,
)


FLAGS = [
    "-O3",
    "-std=c++17",
    "-fopenmp",
    "-fno-fast-math",
    "-fno-associative-math",
    "-ffp-contract=off",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.contract != ANGULAR_TREE_BACKEND_CONTRACT:
        raise ValueError("unsupported tree backend contract")
    source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "relational_part"
        / "csrc"
        / "relational_ca_tree_v1.cpp"
    )
    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name="relational_ca_tree_v1_ext",
        sources=[str(source)],
        extra_cflags=FLAGS,
        extra_ldflags=["-fopenmp"],
        build_directory=str(args.build_dir),
        verbose=True,
    )
    runtime = dict(module.backend_manifest())
    if runtime.get("openmp_available") is not True:
        raise RuntimeError("compiled tree backend lacks OpenMP")
    binary_source = Path(module.__file__).resolve()
    binary_target = args.output_dir / binary_source.name
    if binary_target.exists() and sha256_file(binary_target) != sha256_file(
        binary_source
    ):
        raise FileExistsError("existing tree backend binary differs")
    if not binary_target.exists():
        shutil.copy2(binary_source, binary_target)
    compiler_line = subprocess.run(
        ["c++", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    self_test = dict(module.self_test())
    manifest = with_content_hash(
        {
            "contract": ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
            "schema_version": 1,
            "contract_id": ANGULAR_TREE_BACKEND_CONTRACT,
            "source_sha256": sha256_file(source),
            "binary_sha256": sha256_file(binary_target),
            "compiler_identity": compiler_line,
            "compiler_major_version": compiler_line,
            "compiler_flags": FLAGS,
            "platform_architecture": platform.machine(),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pytorch_version": torch.__version__,
            "pytorch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "openmp_available": bool(runtime["openmp_available"]),
            "self_test_sha256": hashlib.sha256(
                canonical_json_bytes(self_test)
            ).hexdigest(),
            "binary_filename": binary_target.name,
        }
    )
    validate_backend_manifest(
        manifest, binary_path=binary_target, source_path=source
    )
    manifest_path = args.output_dir / "backend_manifest.json"
    payload = canonical_json_bytes(manifest) + b"\n"
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise FileExistsError("existing backend manifest differs")
    if not manifest_path.exists():
        manifest_path.write_bytes(payload)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
