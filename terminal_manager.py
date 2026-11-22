"""
Mobile Web Terminal & Server Log Buffer for Qwen-Image-2.1 INT8 Colab Studio.
Supports persistent working directory (`cd`), incremental UTF-8 streaming output polling,
process-group kill (`Ctrl+C`), and live server log inspection.
"""

from __future__ import annotations

import codecs
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

DANGEROUS_CMD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|[;&|\s])rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/($|\s)|/\*|\~|\$HOME)(\s|$)", re.IGNORECASE), "禁止自动递归强制删除根目录或家目录"),
    (re.compile(r"(^|[;&|\s])rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/($|\s)|/\*|\~|\$HOME)(\s|$)", re.IGNORECASE), "禁止自动递归强制删除根目录或家目录"),
    (re.compile(r"(^|[;&|\s])(mkfs(\.\w+)?|fdisk|parted|wipefs)\b", re.IGNORECASE), "禁止自动执行磁盘格式化/分区命令"),
    (re.compile(r"(^|[;&|\s])dd\s+.*\bof=/dev/(sd|nvme|vd|hd|loop|mmc)", re.IGNORECASE), "禁止自动向块设备写入数据"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "检测到 Fork Bomb 恶意命令"),
    (re.compile(r"(^|[;&|\s])(shutdown|reboot|poweroff|halt|init\s+[06])\b", re.IGNORECASE), "禁止自动关机或重启系统"),
    (re.compile(r"(^|[;&|\s])chmod\s+-R\s+777\s+/($|\s)", re.IGNORECASE), "禁止自动递归修改根目录权限"),
    (re.compile(r">\s*/dev/(sd|nvme|vd|hd)", re.IGNORECASE), "禁止重定向覆盖磁盘设备"),
    (re.compile(r"(^|[;&|\s])(del|erase|rmdir|rd)\s+/[sqf]+\s+[a-zA-Z]:\\", re.IGNORECASE), "禁止自动批量强制删除系统盘目录"),
    (re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b|\bwget\b.*\|\s*(sh|bash)\b", re.IGNORECASE), "从网络直接管道执行未知脚本需手动确认"),
]

_CD_FORBIDDEN_META = ("&&", "||", ";", "|", ">", "<", "`", "$(", "(", ")", "&", "\n", "\r")


def is_dangerous_command(command: str) -> tuple[bool, Optional[str]]:
    """Return `(True, reason)` if `command` matches high-risk destructive patterns."""
    cmd = (command or "").strip()
    if not cmd:
        return False, None
    for pat, reason in DANGEROUS_CMD_PATTERNS:
        if pat.search(cmd):
            return True, reason
    return False, None


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _terminate_popen(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a subprocess (and its process group on POSIX), escalating to SIGKILL if needed."""
    if os.name != "nt":
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=1.0)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)


class LogRingBuffer(logging.Handler):
    """Captures recent server logs in memory for the mobile Terminal log viewer."""

    def __init__(self, max_lines: int = 300) -> None:
        super().__init__()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._lines.append(msg)
        except Exception:
            pass

    def get_lines(self, limit: int = 120) -> list[str]:
        with self._lock:
            items = list(self._lines)
        return items[-limit:]


@dataclass
class TerminalTask:
    cmd_id: str
    command: str
    cwd: str
    running: bool = True
    killed: bool = False
    exit_code: Optional[int] = None
    output: str = ""
    base_offset: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    proc: Optional[subprocess.Popen[bytes]] = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class TerminalManager:
    """Thread-safe shell command runner with persistent `cwd` and streaming poll."""

    MAX_OUTPUT_CHARS = 200_000

    def __init__(self, initial_cwd: Path) -> None:
        self._cwd = str(initial_cwd.resolve())
        self._lock = threading.Lock()
        self._tasks: dict[str, TerminalTask] = {}

    @property
    def cwd(self) -> str:
        with self._lock:
            return self._cwd

    @cwd.setter
    def cwd(self, value: str) -> None:
        with self._lock:
            self._cwd = value

    def _resolve_cd(self, target_arg: str) -> tuple[bool, str, str]:
        """Resolve and update persistent working directory atomically under `self._lock`."""
        target_arg = target_arg.strip()
        if (target_arg.startswith('"') and target_arg.endswith('"')) or (
            target_arg.startswith("'") and target_arg.endswith("'")
        ):
            target_arg = target_arg[1:-1]

        with self._lock:
            if not target_arg or target_arg == "~":
                dest = Path.home()
            else:
                expanded = os.path.expanduser(target_arg)
                candidate = Path(expanded)
                if not candidate.is_absolute():
                    candidate = Path(self._cwd) / candidate
                dest = candidate.resolve()

            if dest.exists() and dest.is_dir():
                self._cwd = str(dest)
                return True, f"已切换目录: {self._cwd}\n", self._cwd
            return False, f"cd: 目录不存在: {dest}\n", self._cwd

    def _prune_tasks_locked(self) -> None:
        if len(self._tasks) <= 40:
            return
        oldest_keys = sorted(self._tasks.keys(), key=lambda k: self._tasks[k].started_at)[:15]
        for k in oldest_keys:
            t = self._tasks.get(k)
            if t is None:
                continue
            with t.lock:
                is_running = t.running
            if not is_running:
                self._tasks.pop(k, None)

    def _append_task_output(self, task: TerminalTask, text: str) -> None:
        if not text:
            return
        with task.lock:
            combined = task.output + text
            if len(combined) > self.MAX_OUTPUT_CHARS:
                trim = len(combined) - self.MAX_OUTPUT_CHARS
                task.output = combined[trim:]
                task.base_offset += trim
            else:
                task.output = combined

    def run_command(self, command: str) -> TerminalTask:
        cmd_str = command.strip()
        cmd_id = uuid.uuid4().hex[:10]

        # Handle standalone `cd` / `cd <dir>` directly without spawning shell, rejecting metacharacters
        is_pure_cd = (cmd_str == "cd" or cmd_str.startswith("cd ")) and not any(
            op in cmd_str for op in _CD_FORBIDDEN_META
        )
        if is_pure_cd:
            arg = cmd_str[2:].strip()
            ok, msg, cur_cwd = self._resolve_cd(arg)
            task = TerminalTask(
                cmd_id=cmd_id,
                command=cmd_str,
                cwd=cur_cwd,
                running=False,
                exit_code=0 if ok else 1,
                output=msg,
                finished_at=time.time(),
            )
            with self._lock:
                self._tasks[cmd_id] = task
                self._prune_tasks_locked()
            return task

        cur_cwd = self.cwd
        task = TerminalTask(cmd_id=cmd_id, command=cmd_str, cwd=cur_cwd)
        with self._lock:
            self._tasks[cmd_id] = task
            self._prune_tasks_locked()

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("TERM", "dumb")

        def _decode_chunk(raw: bytes, utf8_dec: Any, gbk_dec: Any) -> str:
            if os.name != "nt":
                return utf8_dec.decode(raw, final=False)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return gbk_dec.decode(raw, final=False)

        def _worker() -> None:
            utf8_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
            gbk_dec = codecs.getincrementaldecoder("gbk")(errors="replace")
            try:
                popen_kwargs: dict[str, Any] = {
                    "shell": True,
                    "cwd": task.cwd,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "env": env,
                }
                if os.name != "nt":
                    popen_kwargs["start_new_session"] = True

                proc = subprocess.Popen(cmd_str, **popen_kwargs)
                with task.lock:
                    task.proc = proc
                    was_killed = task.killed

                if was_killed:
                    try:
                        _terminate_popen(proc)
                    except Exception:
                        pass
                    with task.lock:
                        task.exit_code = -1
                    return

                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(1024)
                    if not chunk:
                        tail = utf8_dec.decode(b"", final=True) + (gbk_dec.decode(b"", final=True) if os.name == "nt" else "")
                        if tail:
                            self._append_task_output(task, strip_ansi(tail))
                        break
                    text = strip_ansi(_decode_chunk(chunk, utf8_dec, gbk_dec))
                    self._append_task_output(task, text)
                proc.wait()
                with task.lock:
                    if not task.killed:
                        task.exit_code = proc.returncode
            except Exception as exc:
                self._append_task_output(task, f"\n[终端执行错误] {exc}\n")
                with task.lock:
                    if task.exit_code is None:
                        task.exit_code = 1
            finally:
                with task.lock:
                    task.running = False
                    if task.finished_at is None:
                        task.finished_at = time.time()

        threading.Thread(target=_worker, daemon=True).start()
        return task

    def poll(self, cmd_id: str, offset: int = 0) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(cmd_id)
            cur_cwd = self._cwd
        if not task:
            return None
        with task.lock:
            rel_offset = max(0, offset - task.base_offset)
            new_text = task.output[rel_offset:]
            next_offset = task.base_offset + len(task.output)
            running = task.running
            exit_code = task.exit_code
            elapsed = round((task.finished_at or time.time()) - task.started_at, 2)
        return {
            "cmd_id": task.cmd_id,
            "command": task.command,
            "cwd": cur_cwd,
            "running": running,
            "exit_code": exit_code,
            "chunk": new_text,
            "offset": next_offset,
            "elapsed_sec": elapsed,
        }

    def kill(self, cmd_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(cmd_id)
        if not task:
            return False

        with task.lock:
            if not task.running:
                return False
            task.killed = True
            proc = task.proc

        if proc is not None:
            try:
                _terminate_popen(proc)
            except Exception:
                return False

        self._append_task_output(task, "\n^C (已手动中断)\n")
        with task.lock:
            task.running = False
            task.exit_code = -1
            task.finished_at = time.time()
        return True
