"""Narrow compiler-driver transition; every other original backend check retained."""
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from . import protocol as P


def compiler_identity(manifest):
    if (manifest['compiler_driver_version_line'] != P.OLD_BANNER or
            manifest['compiler_executable'] != '/usr/bin/c++'):
        raise ValueError('Not the explicitly supported original compiler build')
    command = shlex.split(os.environ.get('CXX', 'c++'))
    if command not in (['c++'], ['/usr/bin/c++']):
        raise ValueError('Compiler wrappers/flags/alternate CXX are not admitted')
    executable = shutil.which(command[0])
    if executable is None or Path(executable).resolve() != Path('/usr/bin/c++').resolve():
        raise ValueError('Current compiler executable changed')
    banner = subprocess.run([*command, '--version'], check=True, capture_output=True,
                            text=True, timeout=30).stdout.splitlines()[0]
    if banner != P.NEW_BANNER:
        raise ValueError('Current compiler is not the reviewed -15 build: ' + banner)
    return dict(old_banner=P.OLD_BANNER, new_banner=banner,
                executable=str(Path(executable).resolve()), executable_sha256=P.sha(executable))


def runtime_identity():
    import torch
    aliases = {'AMD64': 'x86_64', 'arm64': 'aarch64'}
    return dict(platform_architecture=aliases.get(platform.machine(), platform.machine()),
        python_major_minor=f'{sys.version_info.major}.{sys.version_info.minor}',
        pytorch_version=torch.__version__, pytorch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))


def validate_runtime(manifest, module, current):
    """Same ABI and embedded build checks as ca_tree; compiler exception is separate."""
    for field, value in current.items():
        if manifest.get(field) != value:
            raise RuntimeError('Tree runtime environment differs at ' + field)
    native = dict(module.backend_manifest())
    values = dict(contract_id=native.get('contract_id'), backend_schema_version=native.get('schema_version'),
        compiler_flags=list(native.get('compiler_flags', ())), compiler_identity=native.get('compiler_family'),
        compiler_major_version=int(native.get('compiler_major_version', -1)),
        compiler_version=native.get('compiler_version'), platform_architecture=native.get('platform_architecture'),
        pytorch_cxx11_abi=native.get('pytorch_cxx11_abi'), openmp_available=native.get('openmp_available'))
    for field, value in values.items():
        expected = list(manifest[field]) if field == 'compiler_flags' else manifest[field]
        if value != expected:
            raise RuntimeError('Loaded tree backend differs at ' + field)
    from teacher_logit_reco.relational_part.contracts import canonical_json_bytes
    digest = hashlib.sha256(canonical_json_bytes(module.self_test())).hexdigest()
    if digest != manifest['self_test_sha256']:
        raise RuntimeError('Loaded tree backend self-test differs')
    return dict(runtime=current, embedded_build=values, self_test_sha256=digest)


def load_reference(plan):
    from teacher_logit_reco.relational_part.ca_tree import validate_backend_manifest
    directory = Path(plan['science']['reference']['path'])/'backend'
    manifest = P.read(directory/'backend_manifest.json')
    binary = directory/manifest['binary_filename']
    source = Path(plan['benchmark'])/'source/teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp'
    # The full original static validator checks contract, hash, flags and frozen
    # smoke tolerances. The runtime validator below preserves every ABI/build
    # check, admitting ONLY the independently recorded driver-banner transition.
    validate_backend_manifest(manifest, binary_path=binary, source_path=source)
    compiler = compiler_identity(manifest)
    spec = importlib.util.spec_from_file_location('relational_ca_tree_v1_ext', binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = validate_runtime(manifest, module, runtime_identity())
    return module, manifest, dict(compiler=compiler, **runtime)


def smoke(module, manifest):
    """Reproduce both historical smoke hashes; never loosen numerical tolerances."""
    import numpy as np
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree, tree_content_sha256
    raw = np.zeros((4, 14), dtype=np.float64)
    raw[:, 0] = (12., 7., 3., 1.)
    raw[:, 1] = (-.4, .2, .7, -1.1)
    raw[:, 2] = (.1, -.8, 1.4, 2.2)
    raw[:, 3] = raw[:, 0]*np.cosh(raw[:, 1])
    vectors = np.stack((raw[:, 0]*np.cos(raw[:, 2]), raw[:, 0]*np.sin(raw[:, 2]),
                        raw[:, 0]*np.sinh(raw[:, 1]), raw[:, 3]), axis=1)
    mask = np.ones(4, dtype=bool)
    hashes = dict(compiled_smoke_tree_sha256=tree_content_sha256(build_compiled_tree(module, vectors, raw, mask)),
                  reference_smoke_tree_sha256=tree_content_sha256(build_reference_tree(vectors, raw, mask)))
    for field, value in hashes.items():
        if value != manifest[field]:
            raise ValueError('Historical smoke result changed: ' + field)
    return hashes


_LIBRARY_ROLES = {
    'C++ runtime': ('libstdc++',),
    'GCC support runtime': ('libgcc_s',),
    'OpenMP runtime': ('libgomp', 'libomp'),
    'C runtime': ('libc',),
    'math runtime': ('libm',),
}


def _library_role(name):
    # Accept SONAME versions and bundled/build suffixes, not e.g. libc10.so,
    # libomptarget.so or libomp.so.backup. Discovery is not equivalence: the
    # resolved path and file hash of EVERY matching provider remain pinned.
    for role, stems in _LIBRARY_ROLES.items():
        if any(re.fullmatch(re.escape(stem)+r'(?:-[A-Za-z0-9_.-]+)?\.so(?:\.[0-9]+)*', name)
               for stem in stems):
            return role
    return None


def libraries():
    """Pin actually mapped C/C++/math/OpenMP libraries, not hypothetical ldd paths."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = Path(parts[5])
            if path.name.endswith(' (deleted)') and _library_role(path.name[:-10]):
                raise ValueError('Mapped runtime library was deleted: ' + str(path))
            if _library_role(path.name):
                paths.add(path.resolve(strict=True))
    found = {_library_role(path.name) for path in paths}
    missing = [role for role in _LIBRARY_ROLES if role not in found]
    if missing:
        raise ValueError('Cannot identify all loaded C/C++/math/OpenMP libraries; missing: '
                         + ', '.join(missing) + '; recognized paths: '
                         + ', '.join(str(path) for path in sorted(paths)))
    return {str(p): P.sha(p) for p in sorted(paths)}


def verify(plan, *, expected=None):
    """Run before admission AND on every recovery worker; CPU fixtures only."""
    from part_selected_fast.trees import load_backend, backend_parity
    reference, manifest, identity = load_reference(plan)
    fast = load_backend(Path(plan['benchmark'])/'backend')  # Unmodified strict loader.
    parity = backend_parity(fast, reference, threads=4)
    if any(parity.get(key) is not True for key in ('passed', 'exact', 'invalid_input_rejected')):
        raise ValueError('Backend parity did not pass every admission requirement')
    result = dict(**identity, libraries=libraries(), smoke=smoke(reference, manifest),
                  batch_parity=parity)
    if expected is not None and result != expected:
        raise ValueError('Compatibility evidence changed since admission; no training started')
    return reference, fast, result
