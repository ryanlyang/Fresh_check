#!/usr/bin/env python3
"""Compile and authenticate the relational_ca_tree_v1 PyTorch extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_CONTRACT,
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    build_compiled_tree,
    build_reference_tree,
    canonical_json_bytes,
    sha256_file,
    tree_content_sha256,
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
    parser.add_argument("--dry-run", action="store_true")
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
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "contract": args.contract,
                    "source": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "build_dir": str(args.build_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "compiler_flags": FLAGS,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
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
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler_command:
        raise ValueError("CXX resolves to an empty compiler command")
    compiler_executable = shutil.which(compiler_command[0])
    if compiler_executable is None:
        raise FileNotFoundError(
            f"selected CXX compiler is absent: {compiler_command[0]}"
        )
    compiler_line = subprocess.run(
        [*compiler_command, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    self_test = dict(module.self_test())
    smoke_tokens = np.zeros((4, 14), dtype=np.float64)
    smoke_tokens[:, 0] = np.asarray((12.0, 7.0, 3.0, 1.0))
    smoke_tokens[:, 1] = np.asarray((-0.4, 0.2, 0.7, -1.1))
    smoke_tokens[:, 2] = np.asarray((0.1, -0.8, 1.4, 2.2))
    smoke_tokens[:, 3] = smoke_tokens[:, 0] * np.cosh(
        smoke_tokens[:, 1]
    )
    smoke_vectors = np.stack(
        (
            smoke_tokens[:, 0] * np.cos(smoke_tokens[:, 2]),
            smoke_tokens[:, 0] * np.sin(smoke_tokens[:, 2]),
            smoke_tokens[:, 0] * np.sinh(smoke_tokens[:, 1]),
            smoke_tokens[:, 3],
        ),
        axis=1,
    )
    smoke_mask = np.asarray((True, True, True, True))
    compiled_smoke = build_compiled_tree(
        module, smoke_vectors, smoke_tokens, smoke_mask
    )
    reference_smoke = build_reference_tree(
        smoke_vectors, smoke_tokens, smoke_mask
    )
    compiled_smoke_sha = tree_content_sha256(compiled_smoke)
    reference_smoke_sha = tree_content_sha256(reference_smoke)
    if compiled_smoke_sha != reference_smoke_sha:
        raise RuntimeError(
            "compiled backend canonical smoke tree differs from Python reference"
        )
    manifest = with_content_hash(
        {
            "contract": ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
            "schema_version": 2,
            "contract_id": ANGULAR_TREE_BACKEND_CONTRACT,
            "backend_schema_version": int(runtime["schema_version"]),
            "source_sha256": sha256_file(source),
            "binary_sha256": sha256_file(binary_target),
            "compiler_identity": str(runtime["compiler_family"]),
            "compiler_major_version": int(
                runtime["compiler_major_version"]
            ),
            "compiler_version": str(runtime["compiler_version"]),
            "compiler_executable": str(Path(compiler_executable).resolve()),
            "compiler_driver_version_line": compiler_line,
            "compiler_flags": FLAGS,
            "platform_architecture": str(
                runtime["platform_architecture"]
            ),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pytorch_version": torch.__version__,
            "pytorch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "openmp_available": bool(runtime["openmp_available"]),
            "self_test_sha256": hashlib.sha256(
                canonical_json_bytes(self_test)
            ).hexdigest(),
            "compiled_reference_smoke_tree_sha256": compiled_smoke_sha,
            "binary_filename": binary_target.name,
        }
    )
    validate_backend_manifest(
        manifest,
        binary_path=binary_target,
        source_path=source,
        check_runtime_environment=True,
        runtime_module=module,
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
