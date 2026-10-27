from types import SimpleNamespace

import pytest
import torch

from part_ev_state_compare import protocol as P, engine as E


@pytest.mark.parametrize('index', [0, 3, 4])
def test_evaluation_preserves_training_origin_and_reuses_committed_files(tmp_path, monkeypatch, index):
    engine = E.create_engine()
    task = P.TASKS[index]
    spec = dict(tasks=P.TASKS, content_hash='comparison', inventory=dict(splits=dict(
        val=dict(files=[dict(name=f'file_{i}.root', num_entries=10) for i in range(2)]))))
    runtime = dict(gpu='fixture', campaign_sha256='comparison')
    training_hash = 'comparison' if index < 3 else 'original'
    training_runtime = dict(runtime, campaign_sha256=training_hash)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self, *values):
            return self.weight.expand(len(values[0]), -1)

    checkpoint = tmp_path / 'selected.pt'
    torch.save(dict(task=task, campaign_sha256=training_hash, runtime=training_runtime,
                    model=Tiny().state_dict()), checkpoint)
    checksum = P.file_sha256(checkpoint)
    selected = dict(checkpoint=str(checkpoint), sha256=checksum,
                    training_campaign_sha256=training_hash, training_runtime=training_runtime)
    monkeypatch.setattr(P, 'load_campaign', lambda _: spec)
    monkeypatch.setattr(P, 'benchmark_summary', lambda *args: {})
    monkeypatch.setattr(E, 'lock_finalists', lambda *args: dict(checkpoints=[selected]*5))
    monkeypatch.setattr(engine, 'gpu_setup', lambda _: (torch.device('cpu'), runtime))
    monkeypatch.setattr(engine, 'make_model', lambda *args: Tiny())
    calls = []
    def loader(*args):
        calls.append(True)
        return [(dict((k, torch.zeros(10, 1, 1)) for k in
                      ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')),
                 dict(_label_=torch.arange(10)))]
    monkeypatch.setattr(engine, 'finite_validation_loader', loader)
    args = SimpleNamespace(output=tmp_path, task_index=index)
    # Deliberately tiny fixtures must never be labelled a complete 5M evaluation.
    with pytest.raises(ValueError, match='all 5M'):
        E.evaluate(args, engine)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='all 5M'):
        E.evaluate(args, engine)
    assert len(calls) == 2  # Retried files reuse logits without repeated inference.
    assert P.file_sha256(checkpoint) == checksum
    directory = P.task_directory(tmp_path, index) / 'full_validation'
    assert not (directory / 'metrics.json').exists()
    sidecar = directory / 'file_000.json'
    row = P.read_json(sidecar)
    assert row['contract']['checkpoint_sha256'] == checksum
    row['contract']['checkpoint_sha256'] = 'different'
    P.write_json(sidecar, row)
    with pytest.raises(ValueError, match='prediction changed'):
        E.evaluate(args, engine)
    assert len(calls) == 2


def test_evaluation_rejects_rewritten_control_provenance(tmp_path, monkeypatch):
    engine = E.create_engine()
    runtime = dict(gpu='fixture', campaign_sha256='comparison')
    spec = dict(tasks=P.TASKS, content_hash='comparison')
    monkeypatch.setattr(P, 'load_campaign', lambda _: spec)
    monkeypatch.setattr(P, 'benchmark_summary', lambda *args: {})
    monkeypatch.setattr(engine, 'gpu_setup', lambda _: (torch.device('cpu'), runtime))
    monkeypatch.setattr(engine, 'make_model', lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(engine, 'trusted_load', lambda _: dict(task=P.TASKS[3],
        runtime=runtime, campaign_sha256='comparison'))
    selected = dict(checkpoint='unused', training_runtime=dict(runtime, campaign_sha256='original'),
                    training_campaign_sha256='original')
    monkeypatch.setattr(E, 'lock_finalists', lambda *args: dict(checkpoints=[selected]*5))
    with pytest.raises(ValueError, match='origin/runtime mismatch'):
        E.evaluate(SimpleNamespace(output=tmp_path, task_index=3), engine)
