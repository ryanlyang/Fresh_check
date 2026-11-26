"""Same masked objective, fewer serial slots and host synchronizations.

The historical step is intentionally not edited. This path is currently a
single-GPU diagnostic implementation; it cannot silently change SPORC DDP.
"""
import torch
from part_ev_masked_predictive import runtime as R, modes, views
from part_ev_masked_predictive.config import coefficient, momentum, recipe
from part_ev_fullscale.protocol import learning_rate
from part_ev_fullscale.checkpoint import capture_rank_state
from part_ev_screen.precision import model_precision, audit_precision
from contextlib import nullcontext


def assert_buffers_equal(actual, expected):
    """Same exact value check, one device-to-host decision instead of one per BN buffer."""
    if actual.keys() != expected.keys():
        raise ValueError('Auxiliary buffer keys changed')
    checks = []
    for name, value in actual.items():
        other = expected[name]
        if value.shape != other.shape or value.dtype != other.dtype or value.device != other.device:
            raise ValueError('Auxiliary buffer identity changed: ' + name)
        checks.append(torch.eq(value, other).all())
    if checks and not bool(torch.stack(checks).all()):
        raise ValueError('Auxiliary phase mutated clean BN buffers')


@torch.no_grad()
def teacher_commit(teacher, student, u, success):
    """Keep multiply-then-add EMA arithmetic and the skipped-step BN-copy rule."""
    teacher.validate_subset(student)
    source = dict(student.named_parameters())
    if success:
        groups = {}
        for name, target in teacher.model.named_parameters():
            value = source[name]
            if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                raise ValueError('EMA parameter identity differs')
            pair = groups.setdefault((target.device, target.dtype), ([], []))
            pair[0].append(target); pair[1].append(value)
        m = momentum(u)
        for targets, values in groups.values():
            # Do not substitute lerp: preserve the original operation order.
            torch._foreach_mul_(targets, m)
            torch._foreach_add_(targets, values, alpha=1-m)
        teacher.successes += 1
    buffers = dict(student.named_buffers())
    for name, value in teacher.model.named_buffers():
        value.copy_(buffers[name])
    teacher.buffer_copies += 1


def attempt(s, values, truth, *, microbatch=32, schedule_label=None, amp=True,
            audit=False, diagnostic=False):
    if s.world != 1 or s.arm != 'masked':
        raise ValueError('Masked speed path is single-GPU masked-only')
    if type(microbatch) is not int or microbatch < 1 or microbatch > 256:
        raise ValueError('Auxiliary microbatch must be between 1 and 256')
    u = s.attempts+1 if schedule_label is None else schedule_label
    if schedule_label is not None and not diagnostic:
        raise ValueError('Schedule override is diagnostic only')
    device = s.device
    R.input_health(values, device, 1)
    if not (truth.ndim == 1 and len(truth) == len(values[0]) and bool(((truth>=0)&(truth<10)).all())):
        raise ValueError('Malformed clean labels')
    lam = coefficient(u); active = lam > 0
    s.base.train(); s.opt.zero_grad(set_to_none=True)
    for group in s.opt.param_groups: group['lr'] = learning_rate(u-1, recipe())
    with torch.profiler.record_function('masked.bn_snapshot'):
        b0 = modes.snapshot(s.base.student) if active else None
    expected = torch.float16 if amp and device.type == 'cuda' else torch.float32
    precision = None
    with torch.profiler.record_function('masked.clean'):
        with model_precision(s.base.student, device, training=amp), (
                audit_precision(s.base.student, expected_dtype=expected) if audit else nullcontext()) as precision:
            logits, retained, anchor = s.model('clean', values, retain=active)
            ce = torch.nn.functional.cross_entropy(logits, truth)+anchor
        if not bool(torch.isfinite(ce)): raise FloatingPointError('Nonfinite clean CE')
        with torch.profiler.record_function('masked.clean_backward'): s.scaler.scale(ce).backward()
    ce_value = float(ce.detach()); del logits, ce, anchor
    b1 = modes.snapshot(s.base.student) if active else dict(s.base.student.named_buffers())
    if not bool(torch.stack([torch.isfinite(x).all() for x in b1.values()]).all()):
        raise FloatingPointError('Nonfinite clean BN buffers')
    local_sum = 0.; eligible = 0; target_count = 0; mask_hash = None; mode_counts = [0, 0]
    aux_precision = visible_precision = teacher_precision = None
    native_rng = capture_rank_state(s.base) if audit else None
    if active:
        v = views.create(retained, s.generators)
        if not bool(v.healthy): raise ValueError('Malformed auxiliary geometry/features')
        eligible = int(v.eligible.sum())
        if audit: mask_hash = v.fingerprint()
        if not eligible:
            with torch.profiler.record_function('masked.auxiliary_backward'):
                s.scaler.scale(s.model('flush')).backward()
        else:
            with modes.auxiliary(s.base.student), model_precision(s.base.student, device, training=amp):
                for start in range(0, len(v.selected), microbatch):
                    with torch.profiler.record_function('masked.pack'):
                        item = views.slot(v, start, min(start+microbatch, len(v.selected)))
                    if item is None:
                        loss = s.model('flush')
                    else:
                        with torch.profiler.record_function('masked.teacher'), (
                                audit_precision(s.teacher.model, expected_dtype=expected) if audit else nullcontext()) as teacher_precision:
                            targets = s.teacher.targets(item, amp=amp)
                        with torch.profiler.record_function('masked.visible_predictor'), (
                                audit_precision(s.base.predictor, expected_dtype=expected) if audit else nullcontext()) as aux_precision, (
                                audit_precision(s.base.student, expected_dtype=expected) if audit else nullcontext()) as visible_precision:
                            losses = s.model('auxiliary', item['visible'], locations=item['locations'],
                                query_valid=item['query_valid'], targets=targets, buffers=b0)
                            loss = lam/eligible*losses.sum()
                            local_sum += float(losses.detach().sum())
                            target_count += int(item['query_valid'].sum())
                            mode_counts[0] += int((~item['modes']).sum()); mode_counts[1] += int(item['modes'].sum())
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite auxiliary loss')
                    with torch.profiler.record_function('masked.auxiliary_backward'): s.scaler.scale(loss).backward()
                    if item is not None: del targets, losses
                    del loss, item
        with torch.profiler.record_function('masked.buffer_audit'):
            assert_buffers_equal(dict(s.base.student.named_buffers()), b1)
        del v, retained, b0
    if audit:
        from part_ev_masked_predictive.restart import compare
        compare(capture_rank_state(s.base), native_rng, exact=True)
    if not active or not eligible:
        for parameter in s.base.predictor.parameters(): parameter.grad = None
    old = s.scaler.get_scale(); s.scaler.unscale_(s.opt)
    gradients = [p.grad for p in s.base.parameters() if p.grad is not None]
    healthy = bool(torch.stack([p.detach().float().norm() for p in gradients]).norm().isfinite())
    if not healthy and not s.scaler.is_enabled(): raise FloatingPointError('Nonfinite unscaled gradients')
    with torch.profiler.record_function('masked.optimizer'):
        s.scaler.step(s.opt); s.scaler.update()
    skipped = int(s.scaler.get_scale() < old)
    if not (healthy or skipped): raise FloatingPointError('Overflow without scaler recovery')
    with torch.profiler.record_function('masked.ema'):
        teacher_commit(s.teacher, s.base.student, u, not skipped)
    s.attempts += 1; s.successes += 1-skipped; s.auxiliary_attempts += int(active); s.last_schedule_label = u
    aux = local_sum/max(1, eligible)
    return dict(loss=ce_value+lam*aux, ce=ce_value, auxiliary=aux, coefficient=lam, momentum=momentum(u),
        eligible=eligible, targets=target_count, modes=mode_counts, skipped=skipped, scale=s.scaler.get_scale(),
        ema_successes=s.teacher.successes, attempt=s.attempts, schedule_label=u, diagnostic=diagnostic,
        precision=precision, auxiliary_precision=aux_precision, visible_precision=visible_precision,
        teacher_precision=teacher_precision, mask_sha256=mask_hash, bn_isolated=True,
        rng_isolated=True if audit else None, latents=None)
