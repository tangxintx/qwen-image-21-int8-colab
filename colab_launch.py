"""
Optimized Launcher for Qwen-Image-2.1 INT8 Mobile Web Studio.
Reference: https://github.com/wy51ai/easycolab (AI-generated codebase).

Features:
- Stale port / process cleanup before startup (safe Colab cell re-runs)
- Atomic, retried `cloudflared` binary download with progress bar
- Continuous non-blocking tunnel stdout drain (prevents pipe buffer deadlock)
- Native Colab tqdm progress bars + QR code rendering
- Supports both Colab tunnel mode and `--local` / `--demo` local mode
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED_BIN = Path("/tmp/cloudflared")
SERVER_LOG_PATH = Path("/tmp/qwen_server.log") if os.name != "nt" else (BASE_DIR / "qwen_server.log")
DEFAULT_PORT = int(os.environ.get("PORT", "7860"))


def _make_pbar(total: int, desc: str, unit: str = "it", unit_scale: bool = False):
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit=unit, unit_scale=unit_scale, leave=True)
    except Exception:
        return None


def cleanup_stale_processes(port: int) -> None:
    """Terminate leftover `server.py` or `cloudflared` instances occupying `port` from prior runs."""
    try:
        import psutil
    except ImportError:
        return

    my_pid = os.getpid()
    killed: list[int] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == my_pid:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = (proc.info.get("name") or "").lower()
            is_stale_server = "server.py" in cmdline and str(BASE_DIR.name) in cmdline
            is_stale_tunnel = "cloudflared" in name and f"127.0.0.1:{port}" in cmdline
            if is_stale_server or is_stale_tunnel:
                proc.terminate()
                killed.append(pid)
        except Exception:
            continue

    if killed:
        print(f"[清理] 已停止残留进程 (PID: {', '.join(map(str, killed))})...", flush=True)
        time.sleep(0.8)


def _cloudflared_download_url() -> str:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
    return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def ensure_cloudflared(retries: int = 3) -> Path:
    if CLOUDFLARED_BIN.exists() and CLOUDFLARED_BIN.stat().st_size > 5_000_000:
        pbar = _make_pbar(1, "cloudflared 已就绪")
        if pbar:
            pbar.update(1)
            pbar.close()
        return CLOUDFLARED_BIN

    url = _cloudflared_download_url()
    part_path = CLOUDFLARED_BIN.with_suffix(".part")
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        pbar = None
        try:
            def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
                nonlocal pbar
                if pbar is None:
                    desc = "下载 cloudflared" if attempt == 1 else f"重试下载 cloudflared ({attempt}/{retries})"
                    pbar = _make_pbar(max(total_size, 1), desc, unit="B", unit_scale=True)
                if pbar is not None:
                    downloaded = min(block_num * block_size, total_size) if total_size > 0 else block_num * block_size
                    delta = max(0, downloaded - pbar.n)
                    if delta:
                        pbar.update(delta)

            urllib.request.urlretrieve(url, str(part_path), reporthook=_reporthook)
            if part_path.exists() and part_path.stat().st_size > 5_000_000:
                part_path.replace(CLOUDFLARED_BIN)
                CLOUDFLARED_BIN.chmod(CLOUDFLARED_BIN.stat().st_mode | 0o111)
                return CLOUDFLARED_BIN
            raise RuntimeError("下载的 cloudflared 文件体积异常，可能不完整")
        except Exception as exc:
            last_exc = exc
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            time.sleep(1.5 * attempt)
        finally:
            if pbar is not None:
                if pbar.total and pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.close()

    raise RuntimeError(f"cloudflared 下载失败: {last_exc}")


def setup_swap_with_progress() -> None:
    if os.name == "nt":
        return
    if os.environ.get("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}:
        return
    raw_gb = os.environ.get("QWEN_ENABLE_SWAP_GB", "0").strip()
    try:
        size_gb = int(raw_gb)
    except ValueError:
        size_gb = 0
    if size_gb <= 0:
        print("[Swap] 默认跳过 Linux Swap 配置（可在 Colab 参数中选择开启）。", flush=True)
        return

    from swap_manager import enable_linux_swap

    print(f"正在检测并配置 {size_gb}GB Linux Swap 虚拟内存...", flush=True)
    res = enable_linux_swap(size_gb=size_gb, use_tqdm=True)
    print(f"[Swap] {res.get('message')}", flush=True)


def preload_model_with_progress() -> None:
    if os.environ.get("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}:
        return
    if os.environ.get("QWEN_PRELOAD_MODEL", "1").strip().lower() not in {"1", "true", "yes"}:
        return

    model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo")
    quant_mode = os.environ.get("QWEN_QUANT_MODE", "int8-torchao")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from hf_downloader import (
        GGUF_BASE_REPO,
        GGUF_REDIRECTED_REPO,
        download_gguf_bundle_fast,
        download_model_fast,
        strip_model_variant_suffix,
    )

    base_repo, suffix = strip_model_variant_suffix(model_id)
    qm_low = quant_mode.strip().lower()
    is_gguf = (
        base_repo.lower() in {GGUF_BASE_REPO.lower(), GGUF_REDIRECTED_REPO.lower()}
        or "gguf" in base_repo.lower()
        or qm_low in {"gguf-q8-full", "gguf-q4"}
    )
    if is_gguf:
        variant = suffix if suffix in {"q8", "q4"} else ("q4" if "q4" in qm_low else "q8")
        print(f"正在通过 Hugging Face API 极速按需预下载 GGUF {variant.upper()} 模型组件: {base_repo}", flush=True)
        bundle = download_gguf_bundle_fast(
            variant=variant,
            token=hf_token,
            use_tqdm=True,
            include_diffusers_configs=True,
            include_mmproj=True,
        )
        print(f"GGUF {variant.upper()} 模型组件已高速缓存至: {bundle.get('dit_path')}", flush=True)
        return

    print(f"正在通过 Hugging Face API 极速预下载模型: {model_id}", flush=True)
    local_dir = download_model_fast(repo_id=model_id, token=hf_token, use_tqdm=True)
    print(f"模型已高速缓存至: {local_dir}", flush=True)


def wait_server_ready(server_proc: subprocess.Popen[bytes], port: int, timeout_sec: int = 45) -> bool:
    pbar = _make_pbar(timeout_sec, "启动 FastAPI 服务", unit="s")
    ready = False
    for _ in range(timeout_sec):
        if server_proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(1)
        if pbar:
            pbar.update(1)
    if pbar:
        if ready and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    if not ready and SERVER_LOG_PATH.exists():
        print("\n[服务启动日志尾部]", flush=True)
        print(SERVER_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-2000:], flush=True)
    return ready


def wait_tunnel_url(tunnel_proc: subprocess.Popen[str], timeout_sec: int = 45) -> Optional[str]:
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    found: list[str] = []
    recent_lines: deque[str] = deque(maxlen=25)
    done_evt = threading.Event()

    def _reader() -> None:
        assert tunnel_proc.stdout is not None
        # Keep draining stdout for the entire lifetime of cloudflared to avoid pipe buffer deadlock
        for line in tunnel_proc.stdout:
            recent_lines.append(line.rstrip())
            if not found:
                match = url_pattern.search(line)
                if match:
                    found.append(match.group(0))
                    done_evt.set()
        done_evt.set()

    threading.Thread(target=_reader, daemon=True).start()

    pbar = _make_pbar(timeout_sec, "建立 Cloudflare 公网隧道", unit="s")
    for _ in range(timeout_sec):
        if tunnel_proc.poll() is not None:
            break
        if done_evt.wait(timeout=1.0):
            break
        if pbar:
            pbar.update(1)

    if pbar:
        if found and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    if not found and recent_lines:
        print("\n[Cloudflared 日志尾部]", flush=True)
        print("\n".join(recent_lines), flush=True)
    return found[0] if found else None


def display_qr_and_url(public_url: str, is_local: bool = False) -> None:
    print("\n==========================================", flush=True)
    print("Qwen-Image-2.1 INT8 WebUI (代码由 AI 生成 · 参考 wy51ai/easycolab)", flush=True)
    print(f"访问地址: {public_url}", flush=True)
    print("==========================================\n", flush=True)
    if is_local:
        return
    try:
        import io
        import qrcode
        from IPython.display import HTML, Image as IPyImage, display

        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(public_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        display(
            HTML(
                f'<div style="padding:14px;background:#121826;color:#eef2ff;border-radius:12px;font-family:sans-serif">'
                f'<h3 style="margin:0 0 8px">Qwen-Image-2.1 INT8 手机端已就绪</h3>'
                f'<p style="margin:0 0 10px">点击或手机扫码打开：<a href="{public_url}" target="_blank" style="color:#22d3ee;font-weight:bold">{public_url}</a></p>'
                f'<p style="margin:0;font-size:12px;color:#94a3b8">说明：代码由 AI 生成，参考 wy51ai/easycolab。在手机网页右上角点击「关闭」可自动释放服务。</p>'
                f"</div>"
            )
        )
        display(IPyImage(data=buf.getvalue()))
    except Exception:
        pass


def launch_in_colab(
    port: int = DEFAULT_PORT,
    local_only: bool = False,
    demo_mode: Optional[bool] = None,
    no_preload: bool = False,
) -> None:
    env = os.environ.copy()
    env["PORT"] = str(port)
    if demo_mode is True:
        env["DEMO_MODE"] = "1"
        os.environ["DEMO_MODE"] = "1"
    if no_preload:
        env["QWEN_PRELOAD_MODEL"] = "0"
        os.environ["QWEN_PRELOAD_MODEL"] = "0"

    use_tunnel = not local_only and os.name != "nt" and os.environ.get("NO_TUNNEL", "0") != "1"

    cleanup_stale_processes(port)
    setup_swap_with_progress()
    preload_model_with_progress()

    tunnel_proc: Optional[subprocess.Popen[str]] = None
    if use_tunnel:
        cf_bin = ensure_cloudflared()
        tunnel_proc = subprocess.Popen(
            [str(cf_bin), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        env["CLOUDFLARED_PID"] = str(tunnel_proc.pid)

    log_file = open(SERVER_LOG_PATH, "wb")
    server_proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "server.py")],
        cwd=str(BASE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        if not wait_server_ready(server_proc, port=port, timeout_sec=45):
            print("FastAPI 服务未能正常启动，请检查上方日志。", flush=True)
            return

        if tunnel_proc is not None:
            public_url = wait_tunnel_url(tunnel_proc, timeout_sec=45)
            if public_url:
                display_qr_and_url(public_url, is_local=False)
            else:
                print(f"未能在 45 秒内获取 Cloudflare 公网地址，本地服务运行于 http://127.0.0.1:{port}", flush=True)
        else:
            display_qr_and_url(f"http://127.0.0.1:{port}", is_local=True)

        print("服务运行中（按 Ctrl+C 或在网页端点击「关闭」可退出）...", flush=True)
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务...", flush=True)
    finally:
        for proc in (server_proc, tunnel_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        log_file.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Qwen-Image-2.1 INT8 Mobile WebUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web server port (default: 7860)")
    parser.add_argument("--local", action="store_true", help="Run locally without Cloudflare tunnel")
    parser.add_argument("--demo", action="store_true", help="Enable DEMO_MODE=1 without GPU")
    parser.add_argument("--no-preload", action="store_true", help="Skip Hugging Face model pre-download")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    launch_in_colab(
        port=args.port,
        local_only=args.local,
        demo_mode=True if args.demo else None,
        no_preload=args.no_preload,
    )
