"""Exercise Linux maps discovery on any local test platform; no native build."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from part_selected_compiler_recovery import backend as B, protocol as P


@pytest.fixture
def mapped(tmp_path, monkeypatch):
    names = ['libstdc++.so.6.0.34', 'libgcc_s.so.1', 'libomp.so', 'libc.so.6', 'libm.so.6']
    files = {}
    state = SimpleNamespace(names=names, aliases={})
    real_read, real_resolve = Path.read_text, Path.resolve

    def mapped_file(name):
        if name not in files:
            path = tmp_path/name
            path.write_bytes(('original '+name).encode())
            files[name] = path
        return files[name]

    def read(path, *args, **kwargs):
        if path == Path('/proc/self/maps'):
            lines = ['1000-2000 rw-p 00000000 00:00 0 [heap]']
            for name in state.names:
                # Multiple mmap segments of each library must be deduplicated.
                for segment in range(2):
                    lines.append(f'2000-3000 r-xp 00000000 08:01 {segment+1} /native/{name}')
            return '\n'.join(lines)
        return real_read(path, *args, **kwargs)

    def resolve(path, *args, **kwargs):
        if path.parent == Path('/native'):
            assert kwargs.get('strict') is True
            name = state.aliases.get(path.name, path.name)
            return real_resolve(mapped_file(name), strict=True)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(Path, 'resolve', resolve)
    state.file = mapped_file
    return state


@pytest.mark.parametrize('openmp', ['libomp.so', 'libomp.so.5', 'libgomp.so.1.0.0', 'libgomp-947d5fa1.so.1'])
def test_supported_runtime_library_is_found_and_hashed(mapped, openmp):
    mapped.names[2] = openmp
    expected = {str(mapped.file(name).resolve()): P.sha(mapped.file(name)) for name in mapped.names}
    assert B.libraries() == expected


def test_full_tigris_inventory_excludes_unrelated_libraries(mapped):
    expected = {str(mapped.file(name).resolve()): P.sha(mapped.file(name)) for name in mapped.names}
    mapped.names += ['libc10.so', 'libcrypto.so.3', 'libcublas.so.12', 'libmvec.so.1',
                     'libnvpl_blas_lp64_gomp.so.0', 'selected_tree_batch_v1_ext.so', 'libomptarget.so']
    assert B.libraries() == expected


def test_all_loaded_openmp_providers_are_pinned_not_treated_as_equivalent(mapped):
    mapped.names += ['libgomp.so.1']
    result = B.libraries()
    assert str(mapped.file('libomp.so').resolve()) in result
    assert str(mapped.file('libgomp.so.1').resolve()) in result
    assert len(result) == 6


def test_resolved_openmp_alias_pins_real_file(mapped):
    mapped.names[2] = 'libgomp.so.1'
    mapped.aliases['libgomp.so.1'] = 'libomp.so'
    result = B.libraries()
    assert str(mapped.file('libomp.so').resolve()) in result
    assert not any(Path(name).name == 'libgomp.so.1' for name in result)


@pytest.mark.parametrize('missing,role', [('libstdc++.so.6.0.34', 'C++'), ('libgcc_s.so.1', 'GCC'),
    ('libomp.so', 'OpenMP'), ('libc.so.6', 'C runtime'), ('libm.so.6', 'math')])
def test_missing_required_family_still_rejected_with_specific_error(mapped, missing, role):
    mapped.names.remove(missing)
    with pytest.raises(ValueError) as error:
        B.libraries()
    assert 'missing: '+role in str(error.value)


def test_worker_inventory_changes_when_provider_or_bytes_change(mapped):
    before = B.libraries()
    mapped.file('libomp.so').write_bytes(b'changed runtime')
    assert B.libraries() != before
    mapped.names[2] = 'libgomp.so.1'
    assert B.libraries().keys() != before.keys()


def test_deleted_library_is_not_accepted_as_current_file(mapped):
    mapped.names[2] = 'libomp.so (deleted)'
    with pytest.raises(ValueError, match='deleted'):
        B.libraries()


@pytest.mark.parametrize('lookalike', ['libomptarget.so', 'libomp.so.backup', 'libomp-plugin', 'libomp_extra.so'])
def test_similarly_named_nonruntime_cannot_supply_openmp_role(mapped, lookalike):
    mapped.names[2] = lookalike
    with pytest.raises(ValueError, match='OpenMP'):
        B.libraries()
