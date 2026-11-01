"""Deadline and handoff wrapper around a PRIVATE copy of the unchanged trainer."""
from contextlib import nullcontext
import importlib.util
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from . import protocol as S
from part_ev_layerwise_ddp import engine as original


def create_engine():
    module_spec = importlib.util.spec_from_file_location(
        'part_ev_layerwise_ddp._segment_engine', original.__file__)
    engine = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(engine)
    return engine


def allocation_deadline(environ=None, *, wall_clock=time.time, monotonic=time.monotonic):
    """Include startup and replay in the allocation, not just gradient updates."""
    env = os.environ if environ is None else environ
    try:
        start, end = float(env['SLURM_JOB_START_TIME']), float(env['SLURM_JOB_END_TIME'])
    except (KeyError, ValueError) as exc:
        raise ValueError('Need Slurm allocation start/end timestamps; refusing an unbounded segment') from exc
    if (not math.isfinite(start) or not math.isfinite(end)
            or end-start != S.POLICY['walltime_seconds']):
        raise ValueError('Segment allocation must be exactly 72 hours')
    remaining = end-wall_clock()-S.POLICY['shutdown_margin_seconds']
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError('No safe segment runtime remains')
    return monotonic()+remaining


class Deadline:
    def __init__(self, engine, deadline, *, clock=time.monotonic):
        self.engine, self.deadline, self.clock = engine, deadline, clock
        self.expired = False
        self.external_stop = False

    def stopping(self, device, world):
        # Every rank makes the same two collectives, including validation/replay.
        self.external_stop |= not self.engine.B.agree(not self.engine.B.STOP, device, world)
        self.expired |= not self.engine.B.agree(self.clock() < self.deadline, device, world)
        return self.external_stop or self.expired


def train_segment(chain, plan, spec, index, rank, world, device, runtime, deadline, *,
                  engine=None, clock=time.monotonic):
    engine = create_engine() if engine is None else engine
    campaign = Path(plan['campaign'])
    run = S.P.run_directory(campaign)
    if world != 2 or runtime != S.P.expected_runtime(spec):
        raise ValueError('Segments must retain the approved two-GPU runtime')
    # Own the existing training lock through receipt commit, not just model updates.
    guard = S.P.file_lock(run/'.training.lock', blocking=False) if rank == 0 else nullcontext()
    engine.root_call(lambda:guard.__enter__(), rank, world)
    try:
        # Replace the inner lock only in this private module, never in historical P/E.
        engine.P = SimpleNamespace(**vars(S.P))
        def inner_lock(path, **kwargs):
            if Path(path) == run/'.training.lock': return nullcontext()
            return S.P.file_lock(path, **kwargs)
        engine.P.file_lock = inner_lock
        expected = engine.root_call(lambda:S.previous_snapshot(chain, plan, index), rank, world)
        before = engine.root_call(lambda:S.snapshot(campaign, spec), rank, world)
        if before != expected:
            raise ValueError('Checkpoint changed since the last handoff/chain preparation')
        started = time.monotonic()
        controller = Deadline(engine, deadline, clock=clock)
        engine.stopping = controller.stopping
        replay_seconds = 0.
        replay_impl = engine.replay
        def timed_replay(*args, **kwargs):
            nonlocal replay_seconds
            start = time.monotonic()
            try: return replay_impl(*args, **kwargs)
            finally: replay_seconds += time.monotonic()-start
        engine.replay = timed_replay
        planned = False
        if not before['complete']:
            try:
                engine.train(campaign, spec, rank, world, device, runtime)
            except SystemExit as exc:
                # Never relabel an external signal, crash or failed replay as success.
                if exc.code != 75 or not controller.expired or controller.external_stop:
                    raise
                planned = True
        if not engine.B.agree(not engine.B.STOP, device, world):
            raise RuntimeError('External termination is not a planned successful segment')
        after = engine.root_call(lambda:S.snapshot(campaign, spec), rank, world)
        if after['complete']:
            status = 'already_complete' if before['complete'] else 'complete'
        elif planned and (after['step'],after['validated_checks']) > (before['step'],before['validated_checks']):
            status = 'paused'
        else:
            raise RuntimeError('No committed training progress in this segment; chain stopped. '
                               'Inspect restart replay cost and logs before retrying.')
        timings = engine.B.gather(dict(elapsed_seconds=time.monotonic()-started,
                                      replay_seconds=replay_seconds), world)
        def commit():
            row = dict(chain_sha256=plan['content_hash'],index=index,status=status,
                       before=before,after=after,timings=timings,job_id=os.environ.get('SLURM_JOB_ID'))
            row['content_hash'] = S.P.digest(row)
            S.P.write_json(S.receipt_path(chain,index),row)
            return row
        result = engine.root_call(commit, rank, world)
        if rank == 0: engine.progress('segment_handoff',**result)
        return result
    finally:
        engine.root_call(lambda:guard.__exit__(None,None,None), rank, world)
