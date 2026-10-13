"""One-archive JetClass staging: integrity, resume and narrowly scoped cleanup."""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from scripts import stage_jetclass_archive as S


def fixture_archive(tmp_path, name="JetClass_Pythia_train_100M_part0.tar", mutate=None):
    root = tmp_path / "datasets" / "JetClass"
    (root / "incoming").mkdir(parents=True)
    probe = S.Stage(root, name, "a" * 64, 1, headroom_gib=0)
    path = root / "incoming" / name
    files = [(probe.member_prefix + basename, (basename + "\n").encode())
             for basename in sorted(probe.names)]
    if mutate:
        files = mutate(files)
    with tarfile.open(path, "w") as archive:
        if name == S.VAL_NAME:
            directory = tarfile.TarInfo("val_5M")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        for member_name, payload in files:
            if isinstance(payload, tarfile.TarInfo):
                archive.addfile(payload)
            else:
                member = tarfile.TarInfo(member_name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return S.Stage(root, name, digest, path.stat().st_size, headroom_gib=0)


def symlink_or_skip(target, link, is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable to this Windows account")


def test_finalize_exact_train_members_and_idempotent_status(tmp_path):
    stage = fixture_archive(tmp_path)
    assert stage.status()["complete"] is False
    assert stage.prepare()["complete"] is False
    result = stage.finalize()
    assert result["complete"] is True
    assert result["archive_removed"] is True
    assert not stage.archive_path.exists()
    assert len(list((stage.root / stage.destination).glob("*.root"))) == 100
    receipt = json.loads(stage.receipt_path.read_text())
    assert receipt["archive_sha256"] == stage.sha256
    assert {row["path"] for row in receipt["members"]} == stage.expected
    assert all((stage.root / row["path"]).stat().st_nlink == 1 for row in receipt["members"])
    assert stage.status()["complete"] is True
    assert stage.finalize()["complete"] is True


def test_validation_directory_and_file_indices(tmp_path):
    stage = fixture_archive(tmp_path, S.VAL_NAME)
    stage.finalize()
    assert len(list((stage.root / "Pythia/val_5M").glob("*.root"))) == 50
    assert not (stage.root / "Pythia/val_5M/val_5M").exists()
    assert (stage.root / "Pythia/val_5M/HToBB_124.root").exists()


@pytest.mark.parametrize("name", ["JetClass_Pythia_test_20M.tar", "../evil.tar",
                                  "JetClass_Pythia_train_100M_part10.tar", "part0.tar"])
def test_archive_name_allowlist(tmp_path, name):
    root = tmp_path / "JetClass"
    root.mkdir()
    with pytest.raises(ValueError, match="Only JetClass"):
        S.Stage(root, name, "a" * 64, 1)


def test_bad_sha_and_size_never_extract_or_delete(tmp_path):
    stage = fixture_archive(tmp_path)
    stage.sha256 = "f" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage.finalize()
    assert stage.archive_path.exists()
    assert not (stage.root / "Pythia").exists()
    stage.archive_bytes += 1
    with pytest.raises(ValueError, match="size mismatch"):
        stage.finalize()
    assert stage.archive_path.exists()


@pytest.mark.parametrize("bad_name", ["../escaped.root", "/tmp/escaped.root", "x/../../escaped.root",
                                      "HToBB_010.root", "HToBB_0.root", "alien_000.root",
                                      "nested/HToBB_000.root", "..\\escaped.root"])
def test_tar_traversal_layout_and_index_rejected_before_extraction(tmp_path, bad_name):
    stage = fixture_archive(tmp_path, mutate=lambda rows: rows + [(bad_name, b"evil")])
    with pytest.raises(ValueError):
        stage.finalize()
    assert stage.archive_path.exists()
    assert not (stage.root / "Pythia").exists()
    assert not (tmp_path / "escaped.root").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_tar_links_and_special_files_rejected(tmp_path, kind):
    link = tarfile.TarInfo("HToBB_000.root")
    link.type = kind
    link.linkname = "../../outside"
    stage = fixture_archive(tmp_path, mutate=lambda rows: [(link.name, link)] + rows[1:])
    with pytest.raises(ValueError, match="regular ROOT"):
        stage.finalize()
    assert stage.archive_path.exists()


@pytest.mark.parametrize("mutation", [lambda rows: rows[:-1], lambda rows: rows + [rows[0]]])
def test_missing_and_duplicate_tar_members_rejected(tmp_path, mutation):
    stage = fixture_archive(tmp_path, mutate=mutation)
    with pytest.raises(ValueError):
        stage.finalize()
    assert stage.archive_path.exists()


def test_existing_conflict_untouched_and_original_archive_retained(tmp_path):
    stage = fixture_archive(tmp_path)
    target = stage.root / sorted(stage.expected)[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unrelated original data")
    with pytest.raises(ValueError, match="conflicts"):
        stage.finalize()
    assert target.read_bytes() == b"unrelated original data"
    assert stage.archive_path.exists()
    assert not stage.receipt_path.exists()


def test_interruption_after_member_then_resume_without_rewriting(tmp_path, monkeypatch):
    stage = fixture_archive(tmp_path)
    original = stage._extract_member
    calls = []

    def interrupted(*args):
        if calls:
            raise KeyboardInterrupt()
        result = original(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(stage, "_extract_member", interrupted)
    with pytest.raises(KeyboardInterrupt):
        stage.finalize()
    target = stage.root / calls[0]["path"]
    before = target.stat().st_mtime_ns
    assert stage.archive_path.exists()
    assert not stage.receipt_path.exists()
    monkeypatch.setattr(stage, "_extract_member", original)
    stage.finalize()
    assert target.stat().st_mtime_ns == before
    assert stage.status()["complete"]


def test_owned_incomplete_member_is_redone(tmp_path):
    stage = fixture_archive(tmp_path)
    target = stage.root / sorted(stage.expected)[0]
    target.parent.mkdir(parents=True)
    partial = stage._partial(target)
    partial.write_bytes(b"interrupted copy")
    stage.finalize()
    assert not partial.exists()
    assert target.read_bytes() == (target.name + "\n").encode()


def test_crash_after_atomic_publication_recovers_owned_temporary_hardlink(tmp_path):
    stage = fixture_archive(tmp_path)
    target = stage.root / sorted(stage.expected)[0]
    target.parent.mkdir(parents=True)
    partial = stage._partial(target)
    partial.write_bytes((target.name + "\n").encode())
    os.link(partial, target)
    assert target.stat().st_nlink == 2
    stage.finalize()
    assert target.stat().st_nlink == 1
    assert not partial.exists()


def test_completed_receipt_retains_tar_until_explicit_finalize_cleanup(tmp_path, monkeypatch):
    stage = fixture_archive(tmp_path)
    original = stage._delete_verified_archive
    monkeypatch.setattr(stage, "_delete_verified_archive", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        stage.finalize()
    assert stage.status()["complete"] is True
    assert stage.status()["archive_present"] is True
    assert stage.prepare()["complete"] is True
    assert stage.archive_path.exists()  # status and prepare never delete
    monkeypatch.setattr(stage, "_delete_verified_archive", original)
    assert stage.finalize()["archive_removed"] is True


def test_complete_receipt_never_authorizes_deletion_of_corrupt_tar(tmp_path, monkeypatch):
    stage = fixture_archive(tmp_path)
    original = stage._delete_verified_archive
    monkeypatch.setattr(stage, "_delete_verified_archive", lambda *_: None)
    stage.finalize()
    with stage.archive_path.open("r+b") as stream:
        stream.write(b"corrupt")
    monkeypatch.setattr(stage, "_delete_verified_archive", original)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage.finalize()
    assert stage.archive_path.exists()


def test_receipt_missing_member_marks_incomplete(tmp_path):
    stage = fixture_archive(tmp_path)
    stage.finalize()
    target = stage.root / sorted(stage.expected)[0]
    target.unlink()
    assert not stage.status()["complete"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(archive_sha256="f" * 64),
    lambda r: r.update(archive_bytes=1),
    lambda r: r["members"].pop(),
    lambda r: r["members"][0].update(path="../../outside"),
    lambda r: r["members"][0].update(path=[]),
    lambda r: r["members"][0].update(size=-1),
])
def test_receipt_binding_and_members_cannot_be_forged(tmp_path, mutation):
    stage = fixture_archive(tmp_path)
    stage.finalize()
    receipt = json.loads(stage.receipt_path.read_text())
    mutation(receipt)
    stage.receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        stage.status()


def test_prepare_space_projection_and_other_archive_refusal(tmp_path, monkeypatch):
    stage = fixture_archive(tmp_path)
    archive_size = stage.archive_bytes
    stage.archive_path.write_bytes(b"partial")
    stage.headroom_bytes = 100
    required = (archive_size - 7) + archive_size + 100
    monkeypatch.setattr(shutil, "disk_usage", lambda *_: SimpleNamespace(free=required - 1))
    with pytest.raises(OSError, match="Nothing deleted"):
        stage.prepare()
    assert stage.archive_path.read_bytes() == b"partial"
    monkeypatch.setattr(shutil, "disk_usage", lambda *_: SimpleNamespace(free=required))
    assert stage.prepare()["required_free_bytes"] == required
    other = stage.incoming / "JetClass_Pythia_train_100M_part1.tar"
    other.write_bytes(b"do not remove")
    with pytest.raises(ValueError, match="Another archive"):
        stage.prepare()
    assert other.read_bytes() == b"do not remove"


def test_unrelated_outside_and_incoming_files_never_deleted(tmp_path):
    stage = fixture_archive(tmp_path)
    outside = tmp_path / "keep.tar"
    outside.write_bytes(b"outside")
    helper = stage.incoming / ".stage_jetclass_archive_helper.py"
    helper.write_bytes(b"helper")
    other_archive = stage.incoming / "JetClass_Pythia_train_100M_part1.tar"
    other_archive.write_bytes(b"another staging archive")
    stage.finalize()
    assert outside.read_bytes() == b"outside"
    assert helper.read_bytes() == b"helper"
    assert other_archive.read_bytes() == b"another staging archive"


def test_incoming_hardlink_refused_without_deleting_either_name(tmp_path):
    stage = fixture_archive(tmp_path)
    outside = tmp_path / "original.tar"
    os.link(stage.archive_path, outside)
    with pytest.raises(ValueError, match="hard links"):
        stage.finalize()
    assert stage.archive_path.exists()
    assert outside.exists()


def test_final_hardlink_refused(tmp_path):
    stage = fixture_archive(tmp_path)
    target = stage.root / sorted(stage.expected)[0]
    target.parent.mkdir(parents=True)
    target.write_bytes((target.name + "\n").encode())
    outside = tmp_path / "original.root"
    os.link(target, outside)
    with pytest.raises(ValueError, match="hard links"):
        stage.finalize()
    assert outside.exists()
    assert stage.archive_path.exists()


def test_root_symlink_component_refused(tmp_path):
    actual = tmp_path / "actual"
    (actual / "JetClass").mkdir(parents=True)
    linked = tmp_path / "linked"
    symlink_or_skip(actual, linked, True)
    with pytest.raises(ValueError, match="Links"):
        S.Stage(linked / "JetClass", "JetClass_Pythia_train_100M_part0.tar", "a" * 64, 1)


def test_data_directory_symlink_refused(tmp_path):
    stage = fixture_archive(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(outside, stage.root / "Pythia", True)
    with pytest.raises(ValueError, match="Links"):
        stage.finalize()
    assert not list(outside.iterdir())
    assert stage.archive_path.exists()


def test_status_json_only_stdout_and_no_mutation(tmp_path, capsys):
    stage = fixture_archive(tmp_path)
    before = set(stage.root.rglob("*"))
    assert S.main(["status", "--root", str(stage.root), "--archive", stage.archive,
                   "--sha256", stage.sha256, "--archive-bytes", str(stage.archive_bytes)]) == 0
    streams = capsys.readouterr()
    assert json.loads(streams.out)["complete"] is False
    assert streams.err == ""
    assert set(stage.root.rglob("*")) == before


def test_concurrent_finalization_lock_refused(tmp_path):
    stage = fixture_archive(tmp_path)
    with stage.lock():
        with pytest.raises((RuntimeError, PermissionError)):
            stage.prepare()

