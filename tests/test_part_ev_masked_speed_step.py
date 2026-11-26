import copy
import pytest
import torch
from part_ev_masked_predictive import runtime as R, restart as K, verification as V, modes
from part_ev_masked_predictive.step import attempt as reference
from part_ev_masked_speed.step import attempt, teacher_commit, assert_buffers_equal

torch.set_num_threads(1)
CPU = torch.device('cpu')


@pytest.mark.parametrize('microbatch', [2, 4, 8, 32, 64, 128, 256])
@pytest.mark.parametrize('label', [20000, 40000, 900000])
def test_full_attempt_matches_reference_gradients_rng_buffers_and_ema(microbatch, label):
    s = R.fresh('masked', CPU, 1, trim=False)
    V.activate(s.base.student)
    initial = K.capture(s); x, y = R.tensors(R.fixed_batch(16, 17), CPU)
    gradients = []
    hook = s.opt.register_step_pre_hook(lambda opt, args, kwargs: gradients.append(
        {n: None if p.grad is None else p.grad.detach().clone() for n, p in s.base.named_parameters()}))
    try:
        old = reference(s, x, y, schedule_label=label, diagnostic=True, amp=False, audit=True)
        saved = K.capture(s)
        K.restore(s, initial)
        new = attempt(s, x, y, microbatch=microbatch, schedule_label=label, diagnostic=True, amp=False, audit=True)
        actual = K.capture(s)
    finally:
        hook.remove()
    assert old.keys() == new.keys()
    for name in ('mask_sha256', 'modes', 'eligible', 'targets', 'skipped', 'scale', 'ema_successes',
                 'attempt', 'schedule_label', 'coefficient', 'momentum', 'bn_isolated', 'rng_isolated'):
        assert old[name] == new[name], name
    for name in ('loss', 'ce', 'auxiliary'):
        assert new[name] == pytest.approx(old[name], abs=1e-6, rel=1e-5)
    for name, gradient in gradients[0].items():
        other = gradients[1][name]
        if gradient is None: assert other is None
        else: V.close(other, gradient, gradient=True)
    for name in ('rng', 'generators', 'scaler'):
        K.compare(actual[name], saved[name], exact=True)
    K.compare(actual['model'], saved['model'])
    K.compare(actual['optimizer'], saved['optimizer'])
    K.compare(actual['teacher'], saved['teacher'])
    for name, value in s.base.student.named_buffers():
        assert torch.equal(value, saved['model']['student.'+name])


@pytest.mark.parametrize('success', [True, False])
def test_foreach_teacher_matches_original_transaction(success):
    s = R.fresh('masked', CPU, 1, trim=False)
    with torch.no_grad():
        for p in s.base.student.parameters(): p.add_(.001)
        for b in s.base.student.buffers(): b.add_(1)
    other = copy.deepcopy(s.teacher)
    s.teacher.commit(s.base.student, 500000, success)
    teacher_commit(other, s.base.student, 500000, success)
    K.compare(other.state_dict(), s.teacher.state_dict(), exact=True)
    assert other.successes == int(success) and other.buffer_copies == 1


def test_single_sync_buffer_audit_is_still_exact_and_nan_rejecting():
    a = {'running': torch.ones(3), 'count': torch.tensor(1)}
    b = {k: v.clone() for k, v in a.items()}
    assert_buffers_equal(a, b)
    b['running'][0] += 1e-6
    with pytest.raises(ValueError, match='mutated'): assert_buffers_equal(a, b)
    b['running'][0] = float('nan'); a['running'][0] = float('nan')
    with pytest.raises(ValueError): assert_buffers_equal(a, b)


def test_refuses_ddp_bad_microbatch_or_non_diagnostic_schedule():
    s = R.fresh('masked', CPU, 1, trim=False); x, y = R.tensors(R.fixed_batch(8, 9), CPU)
    for mb in (0, -1, 257, True):
        with pytest.raises(ValueError): attempt(s, x, y, microbatch=mb)
    with pytest.raises(ValueError, match='diagnostic'): attempt(s, x, y, schedule_label=100000)
    s.world = 2
    with pytest.raises(ValueError, match='single-GPU'): attempt(s, x, y)


def test_empty_auxiliary_population_and_clean_only_do_not_change_predictor():
    s = R.fresh('masked', CPU, 1, trim=False); x, y = R.tensors(R.fixed_batch(8, 3), CPU)
    before = {n: p.detach().clone() for n, p in s.base.predictor.named_parameters()}
    row = attempt(s, x, y, microbatch=2, schedule_label=40000, diagnostic=True, amp=False)
    assert row['eligible'] == row['targets'] == 0 and row['auxiliary'] == 0
    assert s.generators.draws == 1
    for name, p in s.base.predictor.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is None
