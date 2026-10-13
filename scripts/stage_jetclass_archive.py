#!/usr/bin/env python3
"""Verify and resumably unpack one JetClass training/validation archive.

The caller uploads to ROOT/incoming/NAME. ``finalize`` verifies the SHA-256,
publishes complete ROOT files without overwriting conflicting data, commits a
receipt, and THEN removes only that verified incoming archive. Local archives
are never touched. All commands print one JSON result; progress goes to stderr.
Requires Python 3.8+ and only the standard library.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile


CONTRACT = "jetclass_archive_stage_v1"
CLASSES = (
    "HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q", "TTBar",
    "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu",
)
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
TRAIN_RE = re.compile(r"JetClass_Pythia_train_100M_part([0-9])\.tar\Z")
VAL_NAME = "JetClass_Pythia_val_5M.tar"
BLOCK = 4 * 1024 * 1024


def _lstat(path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _reject_link(path, info):
    # Windows junctions are reparse points but are not always is_symlink().
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
        raise ValueError("Links/reparse points are not allowed: {}".format(path))


def _safe_components(path):
    for candidate in reversed((path,) + tuple(path.parents)):
        info = _lstat(candidate)
        if info is not None:
            _reject_link(candidate, info)
            if candidate != path and not stat.S_ISDIR(info.st_mode):
                raise ValueError("Path parent is not a directory: {}".format(candidate))


def _regular(path, missing_ok=False):
    _safe_components(path)
    info = _lstat(path)
    if info is None:
        if missing_ok:
            return None
        raise FileNotFoundError("Missing file: {}".format(path))
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Expected a regular file with no hard links: {}".format(path))
    return info


def _directory(path, create=False):
    _safe_components(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    info = _lstat(path)
    if info is not None and not stat.S_ISDIR(info.st_mode):
        raise ValueError("Expected directory: {}".format(path))
    return info is not None


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _hash_file(path):
    before = _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            digest.update(chunk)
    if _signature(_regular(path)) != _signature(before):
        raise ValueError("File changed while being verified: {}".format(path))
    return digest.hexdigest()


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Stage:
    def __init__(self, root, archive, sha256, archive_bytes, headroom_gib=10):
        self.root = Path(root)
        if (not self.root.is_absolute() or self.root.name != "JetClass"
                or ".." in self.root.parts or len(self.root.parts) < 3):
            raise ValueError("Root must be an absolute, non-broad path ending in JetClass")
        _safe_components(self.root)
        if self.root.resolve(strict=True) != self.root or not self.root.is_dir():
            raise ValueError("Root must be an existing canonical directory without links")
        train = TRAIN_RE.fullmatch(archive)
        if train:
            indices = range(int(train.group(1)) * 10, int(train.group(1)) * 10 + 10)
            self.destination = "Pythia/train_100M"
            self.member_prefix = ""
        elif archive == VAL_NAME:
            indices = range(120, 125)
            self.destination = "Pythia/val_5M"
            self.member_prefix = "val_5M/"
        else:
            raise ValueError("Only JetClass training parts 0-9 and validation 5M are allowed")
        if not isinstance(sha256, str) or not HASH_RE.fullmatch(sha256):
            raise ValueError("Expected a lowercase 64-character SHA-256")
        if type(archive_bytes) is not int or archive_bytes <= 0:
            raise ValueError("archive-bytes must be a positive integer")
        if not math.isfinite(headroom_gib) or headroom_gib < 0:
            raise ValueError("headroom-gib must be finite and nonnegative")
        self.archive = archive
        self.sha256 = sha256
        self.archive_bytes = archive_bytes
        self.headroom_bytes = math.ceil(headroom_gib * 1024 ** 3)
        self.names = {"{}_{:03d}.root".format(label, index)
                      for label in CLASSES for index in indices}
        self.expected = {self.destination + "/" + name for name in self.names}
        self.incoming = self.root / "incoming"
        self.archive_path = self.incoming / archive
        self.receipts = self.root / "receipts"
        self.receipt_path = self.receipts / (archive + ".json")
        for directory in (self.incoming, self.receipts, self.root / self.destination):
            _directory(directory)

    def _partial(self, target):
        return target.with_name(".{}.{}.stage-partial".format(target.name, self.sha256))

    def _receipt(self):
        if _regular(self.receipt_path, missing_ok=True) is None:
            return None
        with self.receipt_path.open(encoding="utf-8") as stream:
            receipt = json.load(stream)
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in (
            ("contract", CONTRACT), ("archive", self.archive),
            ("archive_sha256", self.sha256), ("archive_bytes", self.archive_bytes),
        )):
            raise ValueError("Completion receipt does not match the requested archive")
        members = receipt.get("members")
        if not isinstance(members, list) or len(members) != len(self.expected):
            raise ValueError("Invalid completion receipt member count")
        paths = set()
        for member in members:
            if (not isinstance(member, dict) or not isinstance(member.get("path"), str)
                    or member["path"] not in self.expected
                    or member["path"] in paths or type(member.get("size")) is not int
                    or member["size"] <= 0 or not isinstance(member.get("sha256"), str)
                    or not HASH_RE.fullmatch(member["sha256"])):
                raise ValueError("Invalid completion receipt member")
            paths.add(member["path"])
        if paths != self.expected:
            raise ValueError("Invalid completion receipt member set")
        return receipt

    def status(self):
        receipt = self._receipt()
        archive_info = _regular(self.archive_path, missing_ok=True)
        complete = receipt is not None
        if receipt is not None:
            for member in receipt["members"]:
                info = _regular(self.root / member["path"], missing_ok=True)
                if info is None or info.st_size != member["size"]:
                    complete = False
        return {"complete": complete, "archive": self.archive,
                "archive_present": archive_info is not None,
                "member_count": len(self.expected)}

    @contextlib.contextmanager
    def lock(self):
        path = self.root / ".stage_jetclass_archive.lock"
        _regular(path, missing_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags, 0o600)
        locked = False
        try:
            _regular(path)
            if os.name == "nt":
                import msvcrt
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            yield
        except BlockingIOError as exc:
            raise RuntimeError("Another archive preparation/finalization is active; retry later") from exc
        finally:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _check_other_archives(self):
        if self.incoming.exists():
            for path in self.incoming.iterdir():
                if path.name != self.archive and (path.name.endswith(".tar") or ".tar." in path.name):
                    raise ValueError("Another archive is staged; finish it first: {}".format(path))

    def prepare(self):
        with self.lock():
            result = self.status()
            if result["complete"]:
                return result
            self._check_other_archives()
            current = _regular(self.archive_path, missing_ok=True)
            remaining = max(0, self.archive_bytes - (current.st_size if current else 0))
            required = remaining + self.archive_bytes + self.headroom_bytes
            available = shutil.disk_usage(str(self.root)).free
            if available < required:
                raise OSError("Need {:.2f} GiB free for remaining upload, extraction and headroom; "
                              "have {:.2f} GiB. Nothing deleted.".format(
                                  required / 1024 ** 3, available / 1024 ** 3))
            _directory(self.incoming, create=True)
            _directory(self.root / self.destination, create=True)
            _directory(self.receipts, create=True)
            result.update(required_free_bytes=required, available_free_bytes=available)
            return result

    def _members(self, archive):
        result = {}
        for member in archive.getmembers():
            name = member.name
            if "\\" in name or "\x00" in name or name.startswith("/"):
                raise ValueError("Unsafe archive member: {!r}".format(name))
            while name.startswith("./"):
                name = name[2:]
            if ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe archive member: {!r}".format(name))
            if member.isdir() and name.rstrip("/") in ("", ".", "val_5M"):
                if name.rstrip("/") == "val_5M" and not self.member_prefix:
                    raise ValueError("Unexpected archive directory: {!r}".format(name))
                continue
            if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or member.issparse() or member.size <= 0):
                raise ValueError("Only nonempty regular ROOT members are allowed: {!r}".format(name))
            if not name.startswith(self.member_prefix):
                raise ValueError("Unexpected archive layout: {!r}".format(name))
            basename = name[len(self.member_prefix):]
            if basename not in self.names:
                raise ValueError("Unexpected class, file index or path: {!r}".format(name))
            target = self.destination + "/" + basename
            if target in result:
                raise ValueError("Duplicate archive member: {!r}".format(name))
            result[target] = member
        if set(result) != self.expected:
            raise ValueError("Archive must contain exactly {} expected ROOT files; found {}".format(
                len(self.expected), len(result)))
        if sum(member.size for member in result.values()) > self.archive_bytes:
            raise ValueError("Uncompressed member size exceeds archive size")
        return result

    def _discard_own_partial(self, target):
        partial = self._partial(target)
        _safe_components(partial)
        info = _lstat(partial)
        if info is None:
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Unsafe partial file: {}".format(partial))
        # A crash after no-overwrite hard-link publication can leave our exact
        # partial name and final name referring to the same complete inode.
        target_info = _lstat(target)
        if info.st_nlink == 2 and target_info is not None:
            _safe_components(target)
            if (stat.S_ISREG(target_info.st_mode) and target_info.st_nlink == 2
                    and os.path.samefile(partial, target)):
                partial.unlink()
                _regular(target)
                return
        if info.st_nlink != 1:
            raise ValueError("Partial file has unexpected hard links: {}".format(partial))
        partial.unlink()

    def _extract_member(self, archive, member, relative):
        target = self.root / relative
        _directory(target.parent, create=True)
        self._discard_own_partial(target)
        existing = _regular(target, missing_ok=True)
        partial = self._partial(target)
        digest = hashlib.sha256()
        total = 0
        with archive.extractfile(member) as source:
            if existing is not None:
                for chunk in iter(lambda: source.read(BLOCK), b""):
                    total += len(chunk)
                    digest.update(chunk)
            else:
                with partial.open("xb") as destination:
                    for chunk in iter(lambda: source.read(BLOCK), b""):
                        destination.write(chunk)
                        total += len(chunk)
                        digest.update(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
        if total != member.size:
            raise ValueError("Incomplete archive payload: {}".format(relative))
        checksum = digest.hexdigest()
        if existing is not None:
            if existing.st_size != total or _hash_file(target) != checksum:
                raise ValueError("Existing ROOT file conflicts with archive; nothing overwritten: {}".format(target))
        else:
            # Unlike POSIX rename(), link() atomically fails if target exists.
            # Both names are on the same filesystem; the temporary hard link
            # is immediately removed, leaving an ordinary one-link ROOT file.
            try:
                os.link(str(partial), str(target))
            except FileExistsError:
                if _regular(target).st_size != total or _hash_file(target) != checksum:
                    raise ValueError("ROOT target appeared during extraction; nothing overwritten: {}".format(target))
            partial.unlink()
            _regular(target)
            _fsync_directory(target.parent)
        return {"path": relative, "size": total, "sha256": checksum}

    def _write_receipt(self, members):
        _directory(self.receipts, create=True)
        # Validate any old receipt before replacing it during a missing-file
        # repair; an unrelated/malformed receipt is never overwritten.
        self._receipt()
        receipt = {"contract": CONTRACT, "archive": self.archive,
                   "archive_sha256": self.sha256, "archive_bytes": self.archive_bytes,
                   "members": members}
        temporary = self.receipt_path.with_name(".{}.{}.partial".format(
            self.receipt_path.name, self.sha256))
        if _regular(temporary, missing_ok=True) is not None:
            temporary.unlink()
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _regular(self.receipt_path, missing_ok=True)
        os.replace(str(temporary), str(self.receipt_path))
        _fsync_directory(self.receipts)

    def _delete_verified_archive(self, verified):
        current = _regular(self.archive_path)
        if _signature(current) != _signature(verified):
            raise ValueError("Archive changed after verification; leaving it untouched")
        self.archive_path.unlink()
        _fsync_directory(self.incoming)

    def finalize(self):
        with self.lock():
            result = self.status()
            if not result["archive_present"]:
                if result["complete"]:
                    return result
                raise FileNotFoundError("Upload the archive before finalizing: {}".format(self.archive_path))
            verified = _regular(self.archive_path)
            if verified.st_size != self.archive_bytes:
                raise ValueError("Archive size mismatch; resume the upload before finalizing")
            print("Verifying SHA-256: {}".format(self.archive), file=sys.stderr, flush=True)
            if _hash_file(self.archive_path) != self.sha256:
                raise ValueError("Archive SHA-256 mismatch; archive retained, no extraction or deletion")
            if not result["complete"]:
                with tarfile.open(str(self.archive_path), mode="r:") as archive:
                    members = self._members(archive)
                    records = []
                    for index, relative in enumerate(sorted(members), 1):
                        records.append(self._extract_member(archive, members[relative], relative))
                        print("Verified/extracted {}/{}: {}".format(index, len(members), relative),
                              file=sys.stderr, flush=True)
                self._write_receipt(records)
                if not self.status()["complete"]:
                    raise ValueError("Completion check failed; archive retained")
            self._delete_verified_archive(verified)
            result = self.status()
            result["archive_removed"] = True
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "prepare", "finalize"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--archive-bytes", required=True, type=int)
    parser.add_argument("--headroom-gib", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        stage = Stage(args.root, args.archive, args.sha256, args.archive_bytes, args.headroom_gib)
        result = getattr(stage, args.command)()
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print("JetClass staging error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Rerun the uploader to resume; archive retained unless already finalized.",
              file=sys.stderr)
        return 130
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
