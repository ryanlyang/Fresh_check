import hashlib
import json

import pytest

from scripts import cleanup_part_completed_resumes as C


def fixture_run(root, index=0):
    relative, manifest = C.CAMPAIGNS[index]
    campaign = root / relative
    task = dict(architecture="part", seed=101)
    recipe = dict(total_steps=200000, epochs=10)
    spec = dict(content_hash="frozen", tasks=[task], recipe=recipe)
    if index < 3:
        task.update(run_id="PART_RANGER", recipe="short")
        spec.pop("recipe")
        spec["recipes"] = dict(short=recipe)
    run = campaign / "runs" / task.get("run_id", "part") / "seed_101"
    run.mkdir(parents=True)
    (campaign / manifest).write_text(json.dumps(spec))
    (run / "best_model_val.pt").write_bytes(b"best model")
    (run / "last.pt").write_bytes(b"optimizer restart")
    done = dict(task=task, campaign_sha256="frozen", step=200000,
                best_checkpoint_sha256=hashlib.sha256(b"best model").hexdigest())
    done["validated_checks" if index < 3 else "validated_epochs"] = 10
    (run / "training_complete.json").write_text(json.dumps(done))
    return campaign, run


@pytest.mark.parametrize("index", range(len(C.CAMPAIGNS)))
def test_formats_preview_and_narrow_removal(tmp_path, index):
    campaign, run = fixture_run(tmp_path, index)
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    rows = C.collect(tmp_path)
    assert len(rows) == 1
    assert before == {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    assert C.remove_verified(tmp_path, rows) == len(b"optimizer restart")
    for path, payload in before.items():
        if path.name == "last.pt":
            assert not path.exists()
        else:
            assert path.read_bytes() == payload
    assert C.collect(tmp_path) == []


def test_excludes_active_campaigns_and_unfinished_runs(tmp_path):
    _, run = fixture_run(tmp_path)
    (run / "training_complete.json").unlink()
    for family in ("part_ev_state_fullscale", "part_ev_state_speed", "part_ev_state_capacity"):
        protected = tmp_path / family / "active" / "runs" / "part" / "seed_101"
        protected.mkdir(parents=True)
        (protected / "last.pt").write_bytes(b"preserve")
        (protected / "training_complete.json").write_text("{}")
    assert C.collect(tmp_path) == []
    assert len(list(tmp_path.rglob("last.pt"))) == 4


@pytest.mark.parametrize("change", ("step", "checks", "hash", "task", "best", "manifest"))
def test_bad_evidence_blocks_entire_plan(tmp_path, change):
    _, first = fixture_run(tmp_path, 0)
    campaign, run = fixture_run(tmp_path, 2)
    done_path = run / "training_complete.json"
    done = json.loads(done_path.read_text())
    if change == "step":
        done["step"] -= 1
    elif change == "checks":
        done["validated_checks"] -= 1
    elif change == "hash":
        done["campaign_sha256"] = "wrong"
    elif change == "task":
        done["task"]["seed"] = 202
    elif change == "best":
        (run / "best_model_val.pt").write_bytes(b"changed")
    else:
        (campaign / "experiment_spec.json").unlink()
    done_path.write_text(json.dumps(done))
    with pytest.raises((ValueError, FileNotFoundError)):
        C.collect(tmp_path)
    assert (first / "last.pt").exists()
    assert (run / "last.pt").exists()


@pytest.mark.parametrize("change", ("last", "best", "done"))
def test_apply_revalidates_everything_before_deletion(tmp_path, change):
    _, first = fixture_run(tmp_path, 0)
    _, run = fixture_run(tmp_path, 1)
    rows = C.collect(tmp_path)
    if change == "last":
        (run / "last.pt").write_bytes(b"new restart")
    elif change == "best":
        (run / "best_model_val.pt").write_bytes(b"changed selected model")
    else:
        done_path = run / "training_complete.json"
        done = json.loads(done_path.read_text())
        done["step"] -= 1
        done_path.write_text(json.dumps(done))
    with pytest.raises(ValueError):
        C.remove_verified(tmp_path, rows)
    assert (first / "last.pt").exists()
    assert (run / "last.pt").exists()


def test_reject_symlinks(tmp_path):
    _, run = fixture_run(tmp_path)
    (run / "last.pt").unlink()
    target = tmp_path / "outside.pt"
    target.write_bytes(b"do not delete")
    try:
        (run / "last.pt").symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is unavailable")
    with pytest.raises(ValueError, match="symlink/path"):
        C.collect(tmp_path)
    assert target.read_bytes() == b"do not delete"


def test_disallow_non_resume_target(tmp_path):
    campaign, run = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="not allowlisted"):
        C.audit(tmp_path, campaign, C.CAMPAIGNS[0][1], run / "best_model_val.pt")


def test_reject_hardlink(tmp_path):
    import os
    _, run = fixture_run(tmp_path)
    os.link(run / "last.pt", tmp_path / "another_link.pt")
    with pytest.raises(ValueError, match="multiply linked"):
        C.collect(tmp_path)


def test_cli_requires_explicit_apply(tmp_path, monkeypatch, capsys):
    root = tmp_path / "checkpoints"
    _, run = fixture_run(root)
    monkeypatch.setattr(C, "__file__", str(tmp_path / "scripts" / "cleanup.py"))
    assert C.main([]) == 0
    assert "PREVIEW ONLY" in capsys.readouterr().out
    assert (run / "last.pt").exists()
    assert C.main(["--apply"]) == 0
    assert "No backup was made" in capsys.readouterr().out
    assert not (run / "last.pt").exists()
    assert (run / "best_model_val.pt").exists()
