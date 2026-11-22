#!/usr/bin/env python3
"""Collect two-rank restart evidence without modifying the donor or its gates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_triangle_restart import protocol as P


def diagnostic(output):
    started = time.monotonic()
    plan = P.load_plan(output)
    campaign = Path(plan['campaign'])
    # Only the admitted donor provides model, optimizer, precision and step code.
    sys.path.insert(0, str(campaign/'source'))
    from part_ev_triangles import protocol as science, benchmark as C
    spec = science.load(campaign)  # Environment check before lazy optimizer imports.
    from part_ev_triangle_restart.engine import cpu_copy, trials, compare_tree
    from part_ev_fullscale.checkpoint import trusted_load, capture_rank_state, restore_rank_state
    from part_ev_pair_readout.benchmark import compare as original_loss_compare
    import torch
    import torch.distributed as dist

    rank, world, device = C.B.setup(2)
    try:
        properties = torch.cuda.get_device_properties(device)
        hardware = C.B.gather(dict(name=properties.name, memory=properties.total_memory), world)
        if any('A100' not in row['name'] for row in hardware) or hardware[0] != hardware[1]:
            raise ValueError('Need two matching same-node A100s, not substituted GPU hardware')
        deadline = started+P.POLICY['walltime_seconds']-P.POLICY['shutdown_margin_seconds']

        def stop():
            if not C.B.agree(not C.B.STOP and time.monotonic() < deadline, device, world):
                raise TimeoutError('Bounded diagnostic interrupted; no admission')

        base, model = C.fresh(device, world)
        optimizer = C.build_optimizer(base, spec['recipe'])
        scaler = torch.amp.GradScaler('cuda')
        values, truth = C.B.tensors(C.B.split_batch(C.B.fixed_batch(512, 128), rank, world), device)
        probe = campaign/'probes/restart.pt'

        def disk_initial():
            if P.sha(probe) != plan['artifacts']['probes/restart.pt']:
                raise ValueError('Original probe checkpoint changed')
            saved = trusted_load(probe)
            P.validate_checkpoint(saved, spec, world)
            return cpu_copy(dict(model=saved['model'], optimizer=saved['optimizer'], scaler=saved['scaler'],
                                 rng=saved['rank_states'][rank], progress=saved['progress']))

        def restore(saved):
            base.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            base.zero_grad(set_to_none=True)
            base.train()
            base._state_pair_cache.clear()
            for module in base.modules():
                if type(module).__name__ == 'SequenceTrimmer':
                    module.enabled = True
            restore_rank_state(base, saved['rng'])

        def snapshot(progress):
            return dict(model=base.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), rng=capture_rank_state(base), progress=progress)

        def step(progress):
            inputs = [v.roll(progress['fixture_cursor'], dims=-1) for v in values]
            before_scale, before_lookahead = scaler.get_scale(), optimizer.step_counter
            row = C.B.step(base, model, optimizer, scaler, inputs, truth,
                           spec['recipe'], progress['attempt'], world)
            return dict(attempt=progress['attempt'], fixture_cursor=progress['fixture_cursor'],
                        loss=row['loss'], skipped=row['skipped'], scale_before=before_scale,
                        scale_after=scaler.get_scale(), lookahead_before=before_lookahead,
                        lookahead_after=optimizer.step_counter, lr=optimizer.param_groups[0]['lr'])

        def boundary(progress):
            # Exactly the original fixture's rank-zero validation at step 20,000.
            inputs = [v.roll(progress['fixture_cursor']-1, dims=-1) for v in values]
            def validate():
                base.eval()
                with torch.no_grad(), C.model_precision(base, device, training=False):
                    logits = base(*(v[:8] for v in inputs))
                if not logits.isfinite().all():
                    raise ValueError('Restart validation not finite')
                return logits.cpu()
            C.root_call(validate, rank, world)

        def roundtrip(state):
            # Rank-local diagnostic checkpoint; never overwrite any old artifact.
            path = output/f'roundtrip_rank_{rank}.pt'
            with path.open('xb') as handle:
                torch.save(state, handle)
            return trusted_load(path)

        def emit(name, row):
            with (output/f'events_rank_{rank}.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(stage=name, **row), allow_nan=False)+'\n')
            if '_update_' not in name:
                print(json.dumps(dict(stage=name, rank=rank)), flush=True)

        initial = disk_initial()
        result = trials(initial, restore=restore, snapshot=snapshot, step=step, boundary=boundary,
                        disk_initial=disk_initial, roundtrip=roundtrip, stop=stop, emit=emit)
        low = result['low_instrumentation']
        pairs = lambda rows: torch.tensor([(r['loss'], r['skipped']) for r in rows], dtype=torch.float32)
        try:
            original_loss_compare(pairs(low['disk']['updates']), pairs(low['memory']['updates']))
            original_gate = dict(passed=True)
        except AssertionError as exc:
            original_gate = dict(passed=False, error=str(exc))
        differences = [abs(a['loss']-b['loss']) for a, b in zip(low['memory']['updates'], low['disk']['updates'])]
        result.update(rank=rank, hardware=hardware, original_loss_gate=original_gate,
                      maximum_low_instrumentation_loss_difference=max(differences),
                      original_probe_sha256=plan['artifacts']['probes/restart.pt'],
                      campaign_sha256=spec['content_hash'], policy=P.POLICY,
                      runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                          cudnn_deterministic=torch.backends.cudnn.deterministic,
                          cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
        # Explicit rank agreement evidence, rather than confusing a local repeat
        # with agreement of the two distributed workers.
        schedules = {name: [{k: v for k, v in u.items() if k != 'loss'} for u in row['updates']]
                     for name, row in result['trials'].items()}
        all_schedules = C.B.gather(schedules, world)
        result['ranks_agree_on_update_schedule'] = compare_tree(all_schedules[0], all_schedules[1])['exact']
        P.write_new(output/f'rank_{rank}.json', result)
        summaries = C.B.gather(dict(rank=rank, complete=result['complete'],
            initial_restores_exact=result['all_initial_restores_exact'],
            mid_roundtrip_exact=result['mid_roundtrip_exact'],
            original_loss_gate_passed=original_gate['passed'],
            ranks_agree_on_update_schedule=result['ranks_agree_on_update_schedule'],
            trial_update_schedules_equal=result['trial_update_schedules_equal'],
            total_skipped_updates=result['total_skipped_updates'],
            maximum_loss_difference=max(differences),
            first_repeat_divergence=result['trials']['memory_b']['first_divergence'],
            first_disk_divergence=result['trials']['disk_start']['first_divergence'],
            first_resume_divergence=result['trials']['resume_after_3']['first_divergence']), world)
        # Detect any accidental mutation of donor artifacts after the inspection.
        if any(P.sha(campaign/name) != expected for name, expected in plan['artifacts'].items()):
            raise ValueError('Donor artifacts changed during diagnostic')
        if rank == 0:
            report = dict(complete=True, ranks=summaries, campaign=str(campaign),
                authorizes_training=False, authorizes_benchmark_retry=False, changes_tolerances=False,
                note='Completion means evidence collected, not restart approval. Synthetic fixed batch only. '
                     'The old optimizer/DDP kernel history before the saved probe cannot be reconstructed; '
                     'absence of divergence does not exonerate the original benchmark. '
                     'Instrumented comparisons add synchronization. Inspect both rank JSONs.')
            P.write_new(output/'report.json', report)
            print(json.dumps(report, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    output = P.canonical(args.output)
    try:
        diagnostic(output)
    except Exception as exc:
        rank = int(os.environ.get('RANK', '0'))
        error = dict(complete=False, rank=rank, authorizes_training=False,
                     error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        path = output/f'error_rank_{rank}.json'
        if output.is_dir() and not path.exists():
            P.write_new(path, error)
        raise


if __name__ == '__main__':
    main()
