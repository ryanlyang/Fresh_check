"""Read-only uploader CLI tests: all external operations are forbidden in dry runs.

Uses only unittest and a local Bash interpreter. No SSH connection is opened and
no real archive is read. Windows uses Git Bash for the portable dry-run checks;
the actual uploader is intended to be launched under WSL/Linux.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADER = REPO_ROOT / "scripts/upload_jetclass_train_val.sh"


def find_bash():
    if os.name == "nt":
        # WindowsApps/bash.exe is a WSL dispatcher, not a portable local shell.
        for path in (Path("C:/Program Files/Git/bin/bash.exe"),
                     Path("C:/Program Files/Git/usr/bin/bash.exe")):
            if path.is_file():
                return str(path)
        return None
    return shutil.which("bash")


BASH = find_bash()


def shell_path(path):
    path = Path(path).resolve()
    if os.name == "nt":
        return "/" + path.drive[0].lower() + path.as_posix()[2:]
    return str(path)


@unittest.skipUnless(BASH, "Local Bash interpreter unavailable")
class UploaderCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jetclass-uploader-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scripts = self.root / "scripts"
        self.source = self.root / "Downloads with spaces"
        self.bin = self.root / "fakebin"
        for directory in (self.scripts, self.source, self.bin):
            directory.mkdir()
        self.script = self.scripts / UPLOADER.name
        shutil.copyfile(UPLOADER, self.script)
        # Dry runs should check presence but never execute the companion helper.
        (self.scripts / "stage_jetclass_archive.py").write_text(
            "raise RuntimeError('The dry run must not execute a helper')\n", encoding="utf-8")
        self.names = [f"JetClass_Pythia_train_100M_part{i}.tar" for i in range(10)]
        self.names.append("JetClass_Pythia_val_5M.tar")
        for index, name in enumerate(self.names):
            (self.source / name).write_bytes(b"fixture" * (index + 1))
        self.marker = self.root / "unexpected-external-operation.txt"
        for command in ("ssh", "rsync", "scp", "sha256sum", "python3", "flock", "mktemp"):
            stub = self.bin / command
            stub.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$0" >> "$UPLOAD_TEST_MARKER"\nexit 99\n',
                encoding="utf-8")
            stub.chmod(0o755)
        self.environment = dict(os.environ)
        self.environment["UPLOAD_TEST_MARKER"] = shell_path(self.marker)

    def run_cli(self, *args):
        # Set PATH inside Bash so this harness works with native Unix as well as
        # Git Bash's Windows path translation. Remaining arguments stay quoted.
        command = [BASH, "--noprofile", "--norc", "-c",
                   'export PATH="$1:$PATH"; shift; exec bash "$@"',
                   "uploader-test", shell_path(self.bin), shell_path(self.script),
                   "--source-dir", shell_path(self.source), *args]
        return subprocess.run(command, env=self.environment, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30)

    def archive_snapshot(self):
        return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.source.iterdir()}

    def assert_no_external_operation(self):
        self.assertFalse(self.marker.exists(), "Dry-run/argument checks invoked an external operation")

    def test_all_inputs_dry_run_does_not_hash_connect_write_or_mutate_archives(self):
        before = self.archive_snapshot()
        entries_before = {str(path.relative_to(self.root)) for path in self.root.rglob("*")}
        result = self.run_cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Dry run only: no connection or changes made.", result.stdout)
        for name in self.names:
            self.assertEqual(result.stdout.count(name), 1)
        expected_size = sum(len(value[0]) for value in before.values())
        self.assertIn(f"Selected archive bytes: {expected_size}", result.stdout)
        self.assertEqual(before, self.archive_snapshot())
        self.assertEqual(entries_before,
                         {str(path.relative_to(self.root)) for path in self.root.rglob("*")})
        self.assert_no_external_operation()

    def test_only_part_and_validation_require_only_the_selected_input(self):
        for selector, name in (("part3", self.names[3]), ("val", self.names[-1])):
            with self.subTest(selector=selector):
                selected = self.root / selector
                selected.mkdir()
                shutil.copyfile(self.source / name, selected / name)
                result = self.run_cli("--source-dir", shell_path(selected),
                                      "--only", selector, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(name, result.stdout)
                for other in set(self.names) - {name}:
                    self.assertNotIn(other, result.stdout)
        self.assert_no_external_operation()

    def test_missing_input_fails_before_connection(self):
        # Fixture-only deletion, never any user's archive.
        (self.source / self.names[-1]).unlink()
        result = self.run_cli("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or linked local archive", result.stderr)
        self.assertIn(self.names[-1], result.stderr)
        self.assert_no_external_operation()

    def test_empty_input_fails_before_connection(self):
        (self.source / self.names[0]).write_bytes(b"")
        result = self.run_cli("--only", "part0", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Empty archive", result.stderr)
        self.assert_no_external_operation()

    def test_invalid_arguments_fail_before_connection(self):
        cases = (
            (("--remote", "ryreu@host;touch marker"), "Invalid remote host"),
            (("--remote", "-oProxyCommand=bad"), "Invalid remote host"),
            (("--remote-root", "/"), "absolute remote path"),
            (("--remote-root", "/home/ryreu/../JetClass"), "absolute remote path"),
            (("--remote-root", "/home/ryreu/./JetClass"), "absolute remote path"),
            (("--only", "part10"), "Invalid --only"),
            (("--headroom-gib", "0"), "positive integer"),
            (("--headroom-gib", "not-a-number"), "positive integer"),
            (("--bogus",), "Unknown option"),
            (("--remote",), "Missing value"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assert_no_external_operation()

    def test_help_and_shell_syntax(self):
        help_result = self.run_cli("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Rerun the SAME command", help_result.stdout)
        result = subprocess.run([BASH, "-n", shell_path(self.script)],
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_no_external_operation()

    def test_transfer_contract_reuses_connection_and_repairs_same_size_partial(self):
        source = self.script.read_text(encoding="utf-8")
        self.assertIn("--partial --inplace --checksum", source)
        self.assertIn("ControlPath=${control_socket}", source)
        self.assertIn("ControlMaster=yes", source)
        self.assertIn("BatchMode=yes", source)
        self.assertNotIn("StrictHostKeyChecking=no", source)
        self.assertNotIn("--remove-source-files", source)
        self.assertNotIn("--delete", source)
        self.assertIn("remote_step finalize", source)
        self.assertIn("trap cleanup EXIT", source)


if __name__ == "__main__":
    unittest.main()
