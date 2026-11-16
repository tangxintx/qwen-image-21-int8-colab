"""
On-demand ComfyUI Launcher for Google Colab.
Never clones, downloads, or starts ComfyUI automatically on startup.
Only runs when explicitly triggered by the user from Settings.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("qwen_easycolab.comfy")

COMFY_DIR = Path("/content/ComfyUI") if (os.name != "nt" and Path("/content").exists()) else Path(__file__).resolve().parent / "ComfyUI"
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
CLOUDFLARED_BIN = Path("/tmp/cloudflared")


class ComfyUILauncher:
    """Manages on-demand ComfyUI installation, execution, LoRA symlinking, and Cloudflared tunnel."""

    def __init__(self, loras_dir: Path) -> None:
        self.loras_dir = loras_dir
        self._lock = threading.Lock()
        self._status: str = "idle"  # idle | installing | starting | running | error
        self._progress: float = 0.0
        self._message: str = "未启动（按需手动启动）"
        self._public_url: Optional[str] = None
        self._local_url: str = f"http://127.0.0.1:{COMFY_PORT}"
        self._error: Optional[str] = None
        self._proc: Optional[subprocess.Popen[Any]] = None
        self._tunnel_proc: Optional[subprocess.Popen[str]] = None
        self._log_file: Optional[Any] = None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            if self._status == "running" and self._proc is not None and self._proc.poll() is not None:
                self._status = "idle"
                self._message = "ComfyUI 进程已退出"
                self._public_url = None
            return {
                "status": self._status,
                "progress": round(self._progress, 3),
                "message": self._message,
                "public_url": self._public_url,
                "local_url": self._local_url,
                "error": self._error,
                "installed": COMFY_DIR.exists(),
            }

    def _set_state(
        self,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        public_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if status is not None:
                self._status = status
            if progress is not None:
                self._progress = progress
            if message is not None:
                self._message = message
            if public_url is not None:
                self._public_url = public_url
            self._error = error

    def _link_loras_dir(self) -> None:
        comfy_loras = COMFY_DIR / "models" / "loras"
        comfy_loras.mkdir(parents=True, exist_ok=True)
        link_path = comfy_loras / "colab_loras"
        if not link_path.exists() and self.loras_dir.exists():
            try:
                os.symlink(str(self.loras_dir), str(link_path), target_is_directory=True)
            except OSError:
                pass

    def _ensure_cloudflared(self) -> Optional[Path]:
        if CLOUDFLARED_BIN.exists() and CLOUDFLARED_BIN.stat().st_size > 1_000_000:
            return CLOUDFLARED_BIN
        if os.name == "nt":
            return None
        try:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            urllib.request.urlretrieve(url, str(CLOUDFLARED_BIN))
            CLOUDFLARED_BIN.chmod(CLOUDFLARED_BIN.stat().st_mode | 0o111)
            return CLOUDFLARED_BIN
        except Exception as exc:
            logger.warning("Failed to download cloudflared for ComfyUI: %s", exc)
            return None

    def launch(self, demo_mode: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._status in {"installing", "starting", "running"}:
                return self.get_status()
            self._status = "installing"
            self._progress = 0.05
            self._message = "准备启动 ComfyUI..."
            self._public_url = None
            self._error = None

        def _worker() -> None:
            try:
                if demo_mode:
                    for pct, msg in [
                        (0.25, "[演示模式] 模拟检查 ComfyUI 仓库..."),
                        (0.55, "[演示模式] 模拟链接 /content/loras..."),
                        (0.85, "[演示模式] 模拟启动 ComfyUI (8188) 与隧道..."),
                    ]:
                        time.sleep(0.2)
                        self._set_state("installing", pct, msg)
                    self._set_state(
                        status="running",
                        progress=1.0,
                        message="ComfyUI 演示服务已就绪",
                        public_url=self._local_url,
                    )
                    return

                # 1) Clone ComfyUI if not present
                if not (COMFY_DIR / "main.py").exists():
                    self._set_state("installing", 0.15, "正在克隆 ComfyUI 仓库...")
                    res = subprocess.run(
                        ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if res.returncode != 0:
                        raise RuntimeError(f"git clone ComfyUI 失败: {(res.stdout or '')[-300:]}")

                # 2) Install ComfyUI requirements
                req_file = COMFY_DIR / "requirements.txt"
                if req_file.exists():
                    self._set_state("installing", 0.42, "正在安装 ComfyUI 依赖...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                self._link_loras_dir()
                self._set_state("starting", 0.65, f"正在启动 ComfyUI (端口 {COMFY_PORT})...")

                log_path = Path("/tmp/comfyui.log") if os.name != "nt" else (COMFY_DIR / "comfyui.log")
                self._log_file = open(log_path, "ab")
                self._proc = subprocess.Popen(
                    [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(COMFY_PORT)],
                    cwd=str(COMFY_DIR),
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                )

                ready = False
                for i in range(60):
                    if self._proc.poll() is not None:
                        raise RuntimeError("ComfyUI 进程意外退出，请检查 /tmp/comfyui.log")
                    try:
                        with urllib.request.urlopen(f"{self._local_url}/system_stats", timeout=1) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        pass
                    self._set_state("starting", 0.65 + min(0.20, i * 0.004), f"等待 ComfyUI 监听端口 {COMFY_PORT}...")
                    time.sleep(1)

                if not ready:
                    raise RuntimeError(f"ComfyUI 在 60 秒内未能监听端口 {COMFY_PORT}")

                # 3) Establish Cloudflared public tunnel
                public_url = self._local_url
                cf_bin = self._ensure_cloudflared()
                if cf_bin is not None:
                    self._set_state("starting", 0.88, "正在建立 ComfyUI Cloudflare 公网隧道...")
                    self._tunnel_proc = subprocess.Popen(
                        [str(cf_bin), "tunnel", "--no-autoupdate", "--url", self._local_url],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    pat = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
                    found: list[str] = []
                    done_evt = threading.Event()

                    def _read_tunnel() -> None:
                        assert self._tunnel_proc is not None and self._tunnel_proc.stdout is not None
                        for line in self._tunnel_proc.stdout:
                            m = pat.search(line)
                            if m:
                                found.append(m.group(0))
                                done_evt.set()
                                break
                        done_evt.set()

                    threading.Thread(target=_read_tunnel, daemon=True).start()
                    done_evt.wait(timeout=35)
                    if found:
                        public_url = found[0]

                self._set_state(
                    status="running",
                    progress=1.0,
                    message="ComfyUI 已启动就绪",
                    public_url=public_url,
                )
            except Exception as exc:
                logger.exception("ComfyUI launch failed")
                self.stop()
                self._set_state(
                    status="error",
                    progress=0.0,
                    message=f"ComfyUI 启动失败: {exc}",
                    error=str(exc),
                )

        threading.Thread(target=_worker, daemon=True).start()
        return self.get_status()

    def stop(self) -> dict[str, Any]:
        for proc in (self._tunnel_proc, self._proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._proc = None
        self._tunnel_proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self._set_state(status="idle", progress=0.0, message="ComfyUI 已停止", public_url=None)
        return self.get_status()
