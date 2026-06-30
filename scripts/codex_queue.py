#!/usr/bin/env python3
"""Run queued Codex CLI prompts one at a time.

Edit the configuration block below for the lightweight workflow:

    PROMPTS = [
        "Read PLAN.md and implement step 3 only. Stop after verification.",
        "Read PLAN.md and implement step 4 only. Stop after verification.",
    ]

Then run:

    python scripts/codex_queue.py --auto-approve

You can also pass prompts on the command line with repeated --prompt flags, or
load a multi-prompt text file where prompts are separated by a line containing
only "---".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# User-editable defaults
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
    # "Read PLAN.md and implement step 3 only. Stop after verification.",
    # "Read PLAN.md and implement step 4 only. Stop after verification.",
]

# same-session: all prompts are sent to SESSION_ID, or to a new captured session
# if SESSION_ID is None.
# independent: every prompt starts a fresh `codex exec` run.
MODE = "same-session"

# The session id you provided. Set to None to create a new first session and
# capture its id from `codex exec --json`.
SESSION_ID: str | None = "019f14fb-58c4-7c82-b8ad-7beab91a1745"

REPO_DIR = REPO_ROOT
OUTPUT_DIR = REPO_ROOT / ".codex_queue_outputs"
STOP_ON_FAILURE = True
WRITE_LAST_MESSAGES = True
RESUME_ALL = True

# Keep this false by default. Use --auto-approve or set it true only when you
# are comfortable with Codex's --dangerously-bypass-approvals-and-sandbox flag.
AUTO_APPROVE = False


UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
SESSION_KEYWORDS = ("session", "conversation", "thread")


@dataclass(frozen=True)
class RunResult:
    index: int
    prompt: str
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str
    last_message_path: str | None
    stdout_log_path: str
    stderr_log_path: str
    session_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queued Codex CLI prompts sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("same-session", "independent"),
        default=MODE,
        help="Whether prompts share one Codex session or each start fresh.",
    )
    parser.add_argument(
        "--session-id",
        default=SESSION_ID,
        help=(
            "Existing Codex session id or thread name to resume in same-session "
            "mode. Use 'none' to start a new session from the first prompt."
        ),
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_DIR,
        help="Working directory passed to fresh `codex exec` runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for state, JSONL event logs, stderr logs, and final messages.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to queue. Repeat this flag to queue multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help=(
            "Text file containing one prompt, prompts separated by a line with "
            "'---', or a JSON list of prompt strings."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=not STOP_ON_FAILURE,
        help="Continue to later prompts even if a Codex subprocess fails.",
    )
    parser.add_argument(
        "--no-output-last-message",
        action="store_true",
        default=not WRITE_LAST_MESSAGES,
        help="Do not pass --output-last-message to Codex.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=AUTO_APPROVE,
        help=(
            "Pass Codex's --dangerously-bypass-approvals-and-sandbox flag so "
            "runs do not stop for command approval prompts."
        ),
    )
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help="Alias for --auto-approve, matching the underlying Codex flag name.",
    )
    parser.add_argument(
        "--bypass-hook-trust",
        action="store_true",
        help="Pass Codex's --dangerously-bypass-hook-trust flag.",
    )
    parser.add_argument(
        "--resume-all",
        action=argparse.BooleanOptionalAction,
        default=RESUME_ALL,
        help="Pass --all to `codex exec resume` to disable cwd filtering.",
    )
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable.")
    parser.add_argument("--model", help="Model passed through to Codex.")
    parser.add_argument(
        "--profile",
        help="Config profile for fresh `codex exec` runs. Not used for resume.",
    )
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        help="Sandbox mode for fresh `codex exec` runs. Not used for resume.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Codex config override. Repeatable.",
    )
    parser.add_argument(
        "--enable",
        action="append",
        default=[],
        metavar="FEATURE",
        help="Codex feature to enable. Repeatable.",
    )
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        metavar="FEATURE",
        help="Codex feature to disable. Repeatable.",
    )
    parser.add_argument(
        "--skip-git-repo-check",
        action="store_true",
        help="Pass --skip-git-repo-check to Codex.",
    )
    parser.add_argument(
        "--ignore-user-config",
        action="store_true",
        help="Pass --ignore-user-config to Codex.",
    )
    parser.add_argument(
        "--ignore-rules",
        action="store_true",
        help="Pass --ignore-rules to Codex.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Codex commands that would run without executing them.",
    )
    return parser.parse_args()


def normalize_session_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() in {"none", "new"}:
        return None
    return value


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    prompts.extend(prompt.strip() for prompt in PROMPTS if prompt.strip())

    if args.prompts_file:
        prompts.extend(read_prompts_file(args.prompts_file))

    prompts.extend(prompt.strip() for prompt in args.prompt if prompt.strip())
    return prompts


def read_prompts_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("prompts")
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{path} must be a JSON list of strings or an object with a prompts list")
        return [item.strip() for item in value if item.strip()]

    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = []
        else:
            current.append(line)

    tail = "\n".join(current).strip()
    if tail:
        blocks.append(tail)

    return blocks


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_stem(index: int) -> str:
    return f"{index:03d}-{timestamp()}"


def build_common_options(
    args: argparse.Namespace,
    last_message_path: Path | None,
) -> list[str]:
    options: list[str] = []
    for item in args.config:
        options.extend(["--config", item])
    for feature in args.enable:
        options.extend(["--enable", feature])
    for feature in args.disable:
        options.extend(["--disable", feature])
    if args.model:
        options.extend(["--model", args.model])
    if args.auto_approve or args.dangerously_bypass_approvals_and_sandbox:
        options.append("--dangerously-bypass-approvals-and-sandbox")
    if args.bypass_hook_trust:
        options.append("--dangerously-bypass-hook-trust")
    if args.skip_git_repo_check:
        options.append("--skip-git-repo-check")
    if args.ignore_user_config:
        options.append("--ignore-user-config")
    if args.ignore_rules:
        options.append("--ignore-rules")
    options.append("--json")
    if last_message_path is not None:
        options.extend(["--output-last-message", str(last_message_path)])
    return options


def build_fresh_command(
    args: argparse.Namespace,
    prompt: str,
    last_message_path: Path | None,
) -> list[str]:
    command = [args.codex_bin, "exec"]
    command.extend(build_common_options(args, last_message_path))
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.sandbox:
        command.extend(["--sandbox", args.sandbox])
    command.extend(["--cd", str(args.repo_dir)])
    command.append(prompt)
    return command


def build_resume_command(
    args: argparse.Namespace,
    session_id: str,
    prompt: str,
    last_message_path: Path | None,
) -> list[str]:
    command = [args.codex_bin, "exec", "resume"]
    command.extend(build_common_options(args, last_message_path))
    if args.resume_all:
        command.append("--all")
    command.extend([session_id, prompt])
    return command


def command_for_display(command: list[str]) -> str:
    display = list(command)
    if display:
        display[-1] = preview(display[-1], 80)
    if os.name == "nt":
        return subprocess.list2cmdline(display)
    return shlex.join(display)


def preview(text: str, max_chars: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3] + "..."


def run_command(
    command: list[str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    cwd: Path,
) -> tuple[int, list[str]]:
    started = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_thread = threading.Thread(
        target=stream_lines,
        args=(started.stdout, sys.stdout, stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_lines,
        args=(started.stderr, sys.stderr, stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    returncode = started.wait()
    stdout_thread.join()
    stderr_thread.join()

    stdout_log_path.write_text("".join(stdout_lines), encoding="utf-8")
    stderr_log_path.write_text("".join(stderr_lines), encoding="utf-8")
    return returncode, stdout_lines


def stream_lines(pipe: Any, target: Any, sink: list[str]) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        sink.append(line)
        target.write(line)
        target.flush()
    pipe.close()


def extract_session_id(lines: list[str]) -> str | None:
    candidates: dict[str, float] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            for uuid in UUID_RE.findall(stripped):
                candidates[uuid] = candidates.get(uuid, 0.0) + 0.1
            continue
        collect_uuid_candidates(value, (), candidates)

    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: item[1])[0]


def collect_uuid_candidates(
    value: Any,
    key_path: tuple[str, ...],
    candidates: dict[str, float],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            collect_uuid_candidates(child, (*key_path, str(key)), candidates)
        return

    if isinstance(value, list):
        for child in value:
            collect_uuid_candidates(child, key_path, candidates)
        return

    if not isinstance(value, str):
        return

    matches = UUID_RE.findall(value)
    if not matches:
        return

    joined_keys = ".".join(key_path).lower()
    score = 1.0
    if any(keyword in joined_keys for keyword in SESSION_KEYWORDS):
        score = 10.0
    elif joined_keys.endswith(".id") or joined_keys == "id":
        score = 3.0

    for uuid in matches:
        candidates[uuid] = candidates.get(uuid, 0.0) + score


def write_state(
    state_path: Path,
    args: argparse.Namespace,
    session_id: str | None,
    results: list[RunResult],
) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "repo_dir": str(args.repo_dir),
        "session_id": session_id,
        "continue_on_error": args.continue_on_error,
        "auto_approve": bool(args.auto_approve or args.dangerously_bypass_approvals_and_sandbox),
        "runs": [
            {
                "index": result.index,
                "returncode": result.returncode,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "last_message_path": result.last_message_path,
                "stdout_log_path": result.stdout_log_path,
                "stderr_log_path": result.stderr_log_path,
                "session_id": result.session_id,
                "prompt_preview": preview(result.prompt, 160),
                "command": command_for_display(result.command),
            }
            for result in results
        ],
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.session_id = normalize_session_id(args.session_id)
    args.repo_dir = args.repo_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    prompts = load_prompts(args)
    if not prompts:
        print(
            "No prompts queued. Edit PROMPTS in this file, pass --prompt, or use --prompts-file.",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "queue_state.json"

    session_id = args.session_id
    results: list[RunResult] = []

    print(f"Queue mode: {args.mode}")
    print(f"Repo dir:   {args.repo_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.mode == "same-session":
        print(f"Session:    {session_id or '(new session from first prompt)'}")
    if args.auto_approve or args.dangerously_bypass_approvals_and_sandbox:
        print("Auto approve: Codex approval prompts and sandboxing will be bypassed.")

    for index, prompt in enumerate(prompts, start=1):
        run_stem = make_run_stem(index)
        last_message_path = None
        if not args.no_output_last_message:
            last_message_path = args.output_dir / f"{run_stem}-last-message.md"

        stdout_log_path = args.output_dir / f"{run_stem}-events.jsonl"
        stderr_log_path = args.output_dir / f"{run_stem}-stderr.log"

        if args.mode == "independent":
            command = build_fresh_command(args, prompt, last_message_path)
        elif session_id:
            command = build_resume_command(args, session_id, prompt, last_message_path)
        else:
            command = build_fresh_command(args, prompt, last_message_path)

        print()
        print(f"[{index}/{len(prompts)}] {preview(prompt, 120)}")
        print(f"$ {command_for_display(command)}")

        started_at = datetime.now().isoformat(timespec="seconds")
        if args.dry_run:
            returncode = 0
            stdout_lines: list[str] = []
            stdout_log_path.write_text("", encoding="utf-8")
            stderr_log_path.write_text("", encoding="utf-8")
        else:
            returncode, stdout_lines = run_command(
                command,
                stdout_log_path,
                stderr_log_path,
                args.repo_dir,
            )
        finished_at = datetime.now().isoformat(timespec="seconds")

        if args.mode == "same-session" and not session_id and returncode == 0:
            session_id = "<captured-session-id>" if args.dry_run else extract_session_id(stdout_lines)
            if session_id:
                if args.dry_run:
                    print(f"Dry run: would capture session id as {session_id}")
                else:
                    print(f"Captured session id: {session_id}")
            else:
                print(
                    "Could not find a session id in JSON output; stopping before the next prompt.",
                    file=sys.stderr,
                )
                returncode = 3

        if args.mode == "independent" and returncode == 0:
            independent_session_id = extract_session_id(stdout_lines)
        else:
            independent_session_id = session_id

        result = RunResult(
            index=index,
            prompt=prompt,
            command=command,
            returncode=returncode,
            started_at=started_at,
            finished_at=finished_at,
            last_message_path=str(last_message_path) if last_message_path is not None else None,
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            session_id=independent_session_id,
        )
        results.append(result)
        write_state(state_path, args, session_id, results)

        if returncode != 0:
            print(f"Codex exited with status {returncode}.")
            if not args.continue_on_error:
                print(f"Stopping queue. State written to {state_path}")
                return returncode

    print()
    print(f"Queue complete. State written to {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
