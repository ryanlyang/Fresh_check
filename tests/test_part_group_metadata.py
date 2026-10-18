"""Hard-group metadata must work on CUDA builds without Boolean sorting."""
import pytest
import torch

from part_group_screen.grouping import fixed_assignment, random_assignment
from part_group_screen.metadata import canonical_order, random_order


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def metadata_inputs(length):
    generator = torch.Generator().manual_seed(412)
    features = torch.randn(3, length, 17, generator=generator)
    momentum = torch.randn(3, length, 3, generator=generator)
    energy = (momentum.square().sum(-1, keepdim=True) + .25).sqrt()
    vectors = torch.cat((momentum, energy), -1)
    if length > 1:
        # Exact ties, including one where a valid and padded input are identical.
        features[:, 1] = features[:, 0]
        vectors[:, 1] = vectors[:, 0]
    counts = torch.tensor([length, max(1, length // 2), 1])
    valid = torch.arange(length)[None] < counts[:, None]
    permutation = torch.randperm(length, generator=generator)
    # Padding is interspersed and has nonzero values: sorting must use the mask.
    return features[:, permutation], vectors[:, permutation], valid[:, permutation]


def reject_bool_sort(monkeypatch):
    original = torch.argsort

    def guarded(values, *args, **kwargs):
        if values.dtype == torch.bool:
            raise RuntimeError('Sort currently does not support bool dtype on CUDA.')
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch, 'argsort', guarded)


@pytest.mark.parametrize('length', [1, 5, 13, 128])
def test_canonical_mask_order_is_stable_without_bool_sort(monkeypatch, length):
    features, vectors, valid = metadata_inputs(length)
    values = torch.cat((vectors, features), -1).tolist()
    expected = torch.tensor([
        sorted(range(length), key=lambda i: (not bool(valid[b, i]), *values[b][i]))
        for b in range(3)
    ])
    reject_bool_sort(monkeypatch)
    before = torch.get_rng_state().clone()
    assert torch.equal(canonical_order(features, vectors, valid), expected)
    order = random_order(features, vectors, valid, 731019)
    assert torch.equal(torch.get_rng_state(), before)
    ordered_mask = valid.gather(1, order)
    assert torch.equal(ordered_mask, torch.arange(length)[None] < valid.sum(1, keepdim=True))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('assignment', [fixed_assignment, random_assignment])
@pytest.mark.parametrize('length', [13, 128])
def test_hard_assignments_without_bool_sort(monkeypatch, device, assignment, length):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Requires CUDA')
    f_cpu, v_cpu, m_cpu = metadata_inputs(length)
    # Ordering is based only on unchanged input bits, so compare exactly across devices.
    expected_order = canonical_order(f_cpu, v_cpu, m_cpu)
    expected_random = random_order(f_cpu, v_cpu, m_cpu, 731019)
    f, v, valid = (x.to(device) for x in (f_cpu, v_cpu, m_cpu))
    permutation = torch.arange(length - 1, -1, -1, device=device)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    reject_bool_sort(monkeypatch)
    assert torch.equal(canonical_order(f, v, valid).cpu(), expected_order)
    assert torch.equal(random_order(f, v, valid, 731019).cpu(), expected_random)
    a, active = assignment(v, f, valid)
    torch.testing.assert_close(a.sum(-1), valid.float(), rtol=0, atol=0)
    assert (a * ~active[:, None]).sum() == 0
    fixed, _ = fixed_assignment(v, f, valid)
    assert torch.equal(a.sum(1).sort().values, fixed.sum(1).sort().values)
    # Duplicate particles are interchangeable; compare physical group contents.
    shuffled, active2 = assignment(v[:, permutation], f[:, permutation], valid[:, permutation])
    assert torch.equal(active, active2)
    for b in range(3):
        for k in range(a.shape[-1]):
            left = v[b, a[b, :, k].bool()].cpu().tolist()
            right = v[b, permutation][shuffled[b, :, k].bool()].cpu().tolist()
            assert sorted(left) == sorted(right)
    assert torch.equal(torch.get_rng_state(), before_cpu)
    if device == 'cuda':
        assert torch.equal(torch.cuda.get_rng_state(), before_cuda)
