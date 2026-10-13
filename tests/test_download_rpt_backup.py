"""Offline tests for the download-only RC archive helper."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "download_rpt_backup", Path(__file__).resolve().parents[1] / "scripts/download_rpt_backup.py")
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


def make_archive(path, omitted=None, extra=None):
    with tarfile.open(path, "w") as archive:
        for root in backup.ROOTS:
            if root != omitted:
                info = tarfile.TarInfo(root)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
        for name in ("Fresh_check/checkpoints/model.pt", "Fresh_check/checkpoints/logits.npz"):
            info = tarfile.TarInfo(name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"data"))
        info = tarfile.TarInfo("Fresh_check/source_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/home/ryreu/atlas/.rpt_worktrees/example"
        archive.addfile(info)
        if extra:
            archive.addfile(tarfile.TarInfo(extra))


class FakeProcess:
    def __init__(self, payload, messages, code=0):
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO(messages)
        self.returncode = code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "source.tar"
        make_archive(self.archive)

    def messages(self, *, digest=None, complete=True):
        records = [{"stage": "inventory", "estimated_bytes": self.archive.stat().st_size}]
        if complete:
            records.append({"stage": "complete", "archive_bytes": self.archive.stat().st_size,
                            "sha256": digest or backup.sha256_file(self.archive)})
        return ("Banner\n" + "\n".join(backup.MARKER + json.dumps(row) for row in records)
                + "\n").encode()

    def run_download(self, *, messages=None, code=0):
        target = self.root / "download.tar"
        process = FakeProcess(self.archive.read_bytes(), messages or self.messages(), code)
        with patch.object(backup.shutil, "which", return_value="ssh"), \
                patch.object(backup.subprocess, "Popen", return_value=process) as popen, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            receipt = backup.download(target, "test@host")
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "ssh")
        self.assertNotIn("shell", popen.call_args.kwargs)
        return target, receipt

    def test_inventory_includes_models_predictions_and_symlinks(self):
        result = backup.archive_inventory(self.archive)
        self.assertEqual(result["checkpoint_pt_files"], 1)
        self.assertEqual(result["npz_files"], 1)
        self.assertEqual(result["symlinks"], 1)
        self.assertEqual(result["roots"], sorted(backup.ROOTS))

    def test_missing_root_rejected(self):
        make_archive(self.archive, omitted=".rpt_worktrees")
        with self.assertRaisesRegex(ValueError, "missing roots"):
            backup.archive_inventory(self.archive)

    def test_unsafe_or_unexpected_members_rejected(self):
        for name in ("../escaped", "/absolute", "Fresh_check/../../escaped", "other/data"):
            with self.subTest(name=name):
                make_archive(self.archive, extra=name)
                with self.assertRaises(ValueError):
                    backup.archive_inventory(self.archive)

    def test_success_rechecks_disk_and_records_receipt(self):
        target, receipt = self.run_download()
        self.assertEqual(target.read_bytes(), self.archive.read_bytes())
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["remote_sha256"], backup.sha256_file(target))
        self.assertTrue(backup.receipt_path(target).is_file())
        self.assertFalse(target.with_name(target.name + ".partial").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(backup.verify(target)["checkpoint_pt_files"], 1)

    def test_remote_digest_mismatch_not_verified(self):
        with self.assertRaisesRegex(RuntimeError, "differs from remote"):
            self.run_download(messages=self.messages(digest="0" * 64))
        self.assertFalse((self.root / "download.tar").exists())
        self.assertFalse(backup.receipt_path(self.root / "download.tar").exists())
        self.assertTrue((self.root / "download.tar.partial").exists())

    def test_failed_tar_or_ssh_not_verified(self):
        with self.assertRaisesRegex(RuntimeError, "SSH/tar failed"):
            self.run_download(code=1)
        self.assertFalse(backup.receipt_path(self.root / "download.tar").exists())

    def test_missing_remote_confirmation_not_verified(self):
        with self.assertRaisesRegex(RuntimeError, "messages missing"):
            self.run_download(messages=self.messages(complete=False))

    def test_refuses_to_overwrite_any_existing_download(self):
        for suffix in ("", ".partial", ".verified.json"):
            with self.subTest(suffix=suffix):
                output = self.root / f"existing{suffix}"
                collision = output.with_name(output.name + suffix)
                collision.write_bytes(b"keep")
                with patch.object(backup.shutil, "which", return_value="ssh"), \
                        self.assertRaises(FileExistsError):
                    backup.download(output, "test@host")
                self.assertEqual(collision.read_bytes(), b"keep")

    def test_disk_corruption_fails_reverification(self):
        target, _ = self.run_download()
        content = bytearray(target.read_bytes())
        content[-1] ^= 1
        target.write_bytes(content)
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            backup.verify(target)

    def test_remote_program_parses(self):
        compile(backup.REMOTE_PROGRAM, "remote_backup", "exec")


if __name__ == "__main__":
    unittest.main()
