"""
Optimized Launcher for Qwen-Image-2.1 Studio (Google Colab & Local).
Reference: https://github.com/wy51ai/easycolab (AI-generated codebase).

Features:
- Deep stale-process & port-holder cleanup before startup (safe Colab cell re-runs)
- Timeout-protected, atomic, retried `cloudflared` binary download with progress bar
- Continuous non-blocking tunnel stdout drain + automatic tunnel retry
- Native Colab tqdm progress bars + clean QR code card
- Supports Colab tunnel mode and `--local` / `--demo` / `--ai-auto-run` flags
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
    """Terminate leftover `server.py`, port-holding processes, or `cloudflared` instances on `port`."""
    try:
        import psutil
    except ImportError:
        return

    my_pid = os.getpid()
    to_kill: dict[int, psutil.Process] = {}

    # 1) Match processes currently listening on target port
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid and conn.pid != my_pid:
                try:
                    to_kill[conn.pid] = psutil.Process(conn.pid)
                except Exception:
                    pass
    except Exception:
        pass

    # 2) Match stale server.py or cloudflared processes by cmdline / cwd
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == my_pid or pid in to_kill:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = (proc.info.get("name") or "").lower()
            is_stale_server = "server.py" in cmdline and (
                str(BASE_DIR.name) in cmdline or str(BASE_DIR) in cmdline
            )
            if not is_stale_server and "server.py" in cmdline:
                try:
                    if Path(proc.cwd()).resolve() == BASE_DIR:
                        is_stale_server = True
                except Exception:
                    pass
            is_stale_tunnel = "cloudflared" in name and f"127.0.0.1:{port}" in cmdline
            if is_stale_server or is_stale_tunnel:
                to_kill[pid] = proc
        except Exception:
            continue

    if not to_kill:
        return

    killed_pids: list[int] = []
    for pid, proc in to_kill.items():
        try:
            proc.terminate()
            killed_pids.append(pid)
        except Exception:
            continue

    if killed_pids:
        print(f"[清理] 已停止残留进程 (PID: {', '.join(map(str, killed_pids))})...", flush=True)
        _, alive = psutil.wait_procs(list(to_kill.values()), timeout=2.5)
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass


def _cloudflared_download_url() -> str:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
    return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def ensure_cloudflared(retries: int = 3, timeout_sec: int = 30) -> Path:
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
            req = urllib.request.Request(url, headers={"User-Agent": "Qwen-Image-2.1-Studio/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                total_size = int(resp.headers.get("Content-Length") or 0)
                desc = "下载 cloudflared" if attempt == 1 else f"重试下载 cloudflared ({attempt}/{retries})"
                pbar = _make_pbar(max(total_size, 1), desc, unit="B", unit_scale=True)
                with open(part_path, "wb") as f:
                    while True:
                        chunk = resp.read(262_144)
                        if not chunk:
                            break
                        f.write(chunk)
                        if pbar is not None:
                            if total_size > 0 and pbar.n + len(chunk) > pbar.total:
                                pbar.total = pbar.n + len(chunk)
                            pbar.update(len(chunk))

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

    model_id = os.environ.get("QWEN_MODEL_ID", "abenzerps/Qwen-Image-2.1-Turbo-Quantized:q8")
    quant_mode = os.environ.get("QWEN_QUANT_MODE", "gguf-q8-full")
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


def wait_tunnel_url(tunnel_proc: subprocess.Popen[str], timeout_sec: int = 35) -> Optional[str]:
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


def start_tunnel_with_retry(
    cf_bin: Path,
    port: int,
    retries: int = 2,
    timeout_sec: int = 35,
) -> tuple[Optional[subprocess.Popen[str]], Optional[str]]:
    """Start `cloudflared` quick tunnel with automatic retry if trycloudflare stalls."""
    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"[隧道] 第 {attempt}/{retries} 次重试建立 Cloudflare 公网隧道...", flush=True)
        proc = subprocess.Popen(
            [str(cf_bin), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        url = wait_tunnel_url(proc, timeout_sec=timeout_sec)
        if url:
            return proc, url
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(1.2)
    return None, None


def _print_startup_summary(port: int, use_tunnel: bool) -> None:
    model_id = os.environ.get("QWEN_MODEL_ID", "abenzerps/Qwen-Image-2.1-Turbo-Quantized:q8")
    quant_mode = os.environ.get("QWEN_QUANT_MODE", "gguf-q8-full")
    offload = os.environ.get("QWEN_OFFLOAD", "auto")
    gpu_prof = os.environ.get("QWEN_GPU_PROFILE", "auto")
    demo = os.environ.get("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}
    ai_auto = os.environ.get("QWEN_AI_AUTO_RUN", "0").strip().lower() in {"1", "true", "yes"}
    print("\n--- Qwen-Image-2.1 Studio 启动配置 ---", flush=True)
    print(f"  模型仓库   : {model_id}", flush=True)
    print(f"  量化策略   : {quant_mode} | 显存调度: {offload} | GPU预设: {gpu_prof}", flush=True)
    print(f"  AI自动运行 : {'开启 (含安全护栏)' if ai_auto else '关闭 (可在界面按需开启)'}", flush=True)
    print(f"  运行模式   : {'DEMO 演示' if demo else 'GPU 推理'} | 端口: {port} | 公网隧道: {'开启' if use_tunnel else '关闭'}", flush=True)
    print("--------------------------------------\n", flush=True)


def display_qr_and_url(public_url: str, is_local: bool = False) -> None:
    print("\n==========================================", flush=True)
    print("Qwen-Image-2.1 Studio 已就绪 (参考 wy51ai/easycolab)", flush=True)
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
        img = qr.make_image(fill_color="#18181b", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        display(
            HTML(
                f'<div style="padding:16px 18px;background:#ffffff;color:#18181b;border:1px solid #e4e4e7;'
                f'border-radius:14px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:560px;'
                f'box-shadow:0 2px 10px rgba(15,23,42,0.06)">'
                f'<div style="font-size:15px;font-weight:700;margin-bottom:6px">Qwen-Image-2.1 Studio 已就绪</div>'
                f'<div style="margin-bottom:10px;font-size:13px;color:#52525b">点击或手机扫码打开：'
                f'<a href="{public_url}" target="_blank" style="color:#2563eb;font-weight:600;text-decoration:none">{public_url}</a></div>'
                f'<div style="font-size:12px;color:#71717a">提示：网页右上角点击「关闭」可安全卸载模型并释放 Colab 资源。</div>'
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
    ai_auto_run: Optional[bool] = None,
) -> None:
    env = os.environ.copy()
    env["PORT"] = str(port)
    if demo_mode is True:
        env["DEMO_MODE"] = "1"
        os.environ["DEMO_MODE"] = "1"
    if no_preload:
        env["QWEN_PRELOAD_MODEL"] = "0"
        os.environ["QWEN_PRELOAD_MODEL"] = "0"
    if ai_auto_run is not None:
        val = "1" if ai_auto_run else "0"
        env["QWEN_AI_AUTO_RUN"] = val
        os.environ["QWEN_AI_AUTO_RUN"] = val

    use_tunnel = not local_only and os.name != "nt" and os.environ.get("NO_TUNNEL", "0") != "1"

    _print_startup_summary(port=port, use_tunnel=use_tunnel)
    cleanup_stale_processes(port)
    setup_swap_with_progress()
    preload_model_with_progress()

    cf_bin: Optional[Path] = None
    if use_tunnel:
        cf_bin = ensure_cloudflared()

    log_file = open(SERVER_LOG_PATH, "wb")
    server_proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "server.py")],
        cwd=str(BASE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    tunnel_proc: Optional[subprocess.Popen[str]] = None
    try:
        if not wait_server_ready(server_proc, port=port, timeout_sec=45):
            print("FastAPI 服务未能正常启动，请检查上方日志。", flush=True)
            return

        if use_tunnel and cf_bin is not None:
            tunnel_proc, public_url = start_tunnel_with_retry(cf_bin, port=port, retries=2, timeout_sec=35)
            if public_url:
                if tunnel_proc is not None:
                    os.environ["CLOUDFLARED_PID"] = str(tunnel_proc.pid)
                display_qr_and_url(public_url, is_local=False)
            else:
                print(f"未能获取 Cloudflare 公网地址，本地服务运行于 http://127.0.0.1:{port}", flush=True)
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
    parser = argparse.ArgumentParser(description="Launch Qwen-Image-2.1 Studio WebUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web server port (default: 7860)")
    parser.add_argument("--local", action="store_true", help="Run locally without Cloudflare tunnel")
    parser.add_argument("--demo", action="store_true", help="Enable DEMO_MODE=1 without GPU")
    parser.add_argument("--no-preload", action="store_true", help="Skip Hugging Face model pre-download")
    parser.add_argument("--ai-auto-run", action="store_true", help="Enable opt-in AI autonomous terminal execution")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    launch_in_colab(
        port=args.port,
        local_only=args.local,
        demo_mode=True if args.demo else None,
        no_preload=args.no_preload,
        ai_auto_run=True if args.ai_auto_run else None,
    )
