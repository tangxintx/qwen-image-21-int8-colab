"""
Linux Swap (virtual memory) helper for Google Colab.
Uses a fast 4 MB preflight probe (< 0.5s) to verify kernel/filesystem support before
allocating 8 GB, supports OverlayFS via loopback (`losetup`), and shows real-time
byte progress (`tqdm` / callback) when `dd` zero-fill is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

DEFAULT_SWAP_PATH = "/swapfile"
MANUAL_SWAP_COMMAND = (
    "PYTHONPATH=/content/qwen_studio:. python -c "
    "\"from swap_manager import enable_linux_swap; print(enable_linux_swap(8, use_tqdm=False)['message'])\" && free -h"
)

SwapProgressCallback = Callable[[int, int, float, str], None]


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _get_fs_type(path: str) -> str:
    if not _is_linux():
        return sys.platform
    try:
        res = subprocess.run(
            ["stat", "-f", "-c", "%T", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().lower()
    except Exception:
        pass
    return ""


def _read_swap_stats() -> tuple[int, int, int, float]:
    """Return `(total_bytes, used_bytes, free_bytes, percent)` safely across OS/sandboxes."""
    try:
        sm = psutil.swap_memory()
        return int(sm.total), int(sm.used), int(sm.free), float(sm.percent)
    except Exception:
        if _is_linux():
            try:
                meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
                kv: dict[str, int] = {}
                for line in meminfo.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parts = v.strip().split()
                        if parts and parts[0].isdigit():
                            kv[k] = int(parts[0]) * 1024
                tot = kv.get("SwapTotal", 0)
                free = kv.get("SwapFree", 0)
                used = max(0, tot - free)
                pct = (used / tot * 100.0) if tot > 0 else 0.0
                return tot, used, free, pct
            except Exception:
                pass
        return 0, 0, 0, 0.0


def get_swap_status(swap_path: str = DEFAULT_SWAP_PATH) -> dict[str, Any]:
    """Return current swap memory metrics and swapfile state."""
    tot_b, used_b, free_b, pct = _read_swap_stats()
    p = Path(swap_path)
    swapfile_exists = False
    swapfile_gb = 0.0
    try:
        if p.exists():
            swapfile_exists = True
            swapfile_gb = round(p.stat().st_size / (1024**3), 2)
    except OSError:
        pass

    return {
        "supported": _is_linux(),
        "enabled": tot_b > 0,
        "used_gb": round(used_b / (1024**3), 2),
        "total_gb": round(tot_b / (1024**3), 2),
        "free_gb": round(free_b / (1024**3), 2),
        "percent": round(pct, 1),
        "swapfile_exists": swapfile_exists,
        "swapfile_gb": swapfile_gb,
        "fs_type": _get_fs_type(str(Path(swap_path).parent or "/")),
        "manual_command": MANUAL_SWAP_COMMAND,
    }


def _run_cmd(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _cleanup_swapfile(swap_path: str, loop_dev: str = "") -> None:
    if loop_dev:
        _run_cmd(["swapoff", loop_dev])
        _run_cmd(["losetup", "-d", loop_dev])
    _run_cmd(["swapoff", swap_path])
    p = Path(swap_path)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def _probe_swap_method(target_dir: str) -> tuple[str, str]:
    """
    Fast (<0.5s) 4 MB probe to determine whether swap works in this container
    and which allocation/mount strategy succeeds (`direct_fallocate`, `loop_fallocate`,
    `direct_dd`, or `loop_dd`). Returns `("", error_text)` if unsupported.
    """
    probe_path = str(Path(target_dir) / ".qwen_swap_probe_4m")
    _cleanup_swapfile(probe_path)

    last_err = ""
    try:
        # Try 1 & 2: 4 MB via fallocate
        r_fa = _run_cmd(["fallocate", "-l", "4M", probe_path], timeout=10)
        if r_fa.returncode == 0 and Path(probe_path).exists() and Path(probe_path).stat().st_size > 0:
            os.chmod(probe_path, 0o600)
            r_mk = _run_cmd(["mkswap", probe_path], timeout=10)
            if r_mk.returncode == 0:
                r_on = _run_cmd(["swapon", probe_path], timeout=10)
                if r_on.returncode == 0:
                    return "direct_fallocate", ""
                last_err = (r_on.stdout or "").strip()

                if "invalid argument" in last_err.lower():
                    r_lp = _run_cmd(["losetup", "-f", "--show", probe_path], timeout=10)
                    lp_out = (r_lp.stdout or "").strip()
                    if r_lp.returncode == 0 and lp_out.startswith("/dev/loop"):
                        loop_dev = lp_out.splitlines()[-1].strip()
                        try:
                            r_lon = _run_cmd(["swapon", loop_dev], timeout=10)
                            if r_lon.returncode == 0:
                                return "loop_fallocate", ""
                            last_err = (r_lon.stdout or "").strip()
                        finally:
                            _cleanup_swapfile(probe_path, loop_dev=loop_dev)
                    elif lp_out:
                        last_err = f"{last_err}; {lp_out}"

        # Try 3 & 4: 4 MB via contiguous dd zero-fill
        _cleanup_swapfile(probe_path)
        r_dd = _run_cmd(["dd", "if=/dev/zero", f"of={probe_path}", "bs=1M", "count=4"], timeout=15)
        if r_dd.returncode == 0 and Path(probe_path).exists() and Path(probe_path).stat().st_size > 0:
            os.chmod(probe_path, 0o600)
            r_mk = _run_cmd(["mkswap", probe_path], timeout=10)
            if r_mk.returncode == 0:
                r_on = _run_cmd(["swapon", probe_path], timeout=10)
                if r_on.returncode == 0:
                    return "direct_dd", ""
                last_err = (r_on.stdout or "").strip()

                if "invalid argument" in last_err.lower():
                    r_lp = _run_cmd(["losetup", "-f", "--show", probe_path], timeout=10)
                    lp_out = (r_lp.stdout or "").strip()
                    if r_lp.returncode == 0 and lp_out.startswith("/dev/loop"):
                        loop_dev = lp_out.splitlines()[-1].strip()
                        try:
                            r_lon = _run_cmd(["swapon", loop_dev], timeout=10)
                            if r_lon.returncode == 0:
                                return "loop_dd", ""
                            last_err = (r_lon.stdout or "").strip()
                        finally:
                            _cleanup_swapfile(probe_path, loop_dev=loop_dev)
                    elif lp_out:
                        last_err = f"{last_err}; {lp_out}"
            else:
                last_err = (r_mk.stdout or "").strip()
        else:
            last_err = (r_dd.stdout or r_fa.stdout or "").strip()
    except Exception as exc:
        last_err = str(exc)
    finally:
        _cleanup_swapfile(probe_path)

    return "", last_err


def _allocate_dd_with_progress(
    swap_path: str,
    size_gb: int,
    use_tqdm: bool = False,
    progress_cb: Optional[SwapProgressCallback] = None,
) -> tuple[bool, str]:
    target_bytes = size_gb * (1024**3)
    p = Path(swap_path)
    _cleanup_swapfile(swap_path)

    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=target_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"写入 {size_gb}GB Swap 文件",
            )
        except Exception:
            pbar = None

    proc = subprocess.Popen(
        ["dd", "if=/dev/zero", f"of={swap_path}", "bs=16M", f"count={size_gb * 64}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stop_evt = threading.Event()
    t_last = time.time()
    b_last = 0

    def _watch() -> None:
        nonlocal t_last, b_last
        while not stop_evt.wait(0.35):
            if proc.poll() is not None:
                break
            cur_b = p.stat().st_size if p.exists() else 0
            now = time.time()
            dt = max(0.1, now - t_last)
            spd = max(0.0, (cur_b - b_last) / (1024 * 1024) / dt)
            b_last = cur_b
            t_last = now
            if pbar is not None:
                delta = max(0, cur_b - pbar.n)
                if delta > 0:
                    pbar.update(min(delta, max(0, target_bytes - pbar.n)))
            if progress_cb is not None:
                pct = min(99, int(cur_b / max(1, target_bytes) * 100))
                progress_cb(
                    cur_b,
                    target_bytes,
                    spd,
                    f"写入 Swap 文件: {cur_b / (1024**3):.2f}/{size_gb:.2f} GB ({pct}% · {spd:.1f} MB/s)",
                )

    wt = threading.Thread(target=_watch, daemon=True)
    wt.start()
    try:
        out, _ = proc.communicate()
    finally:
        stop_evt.set()
        wt.join(timeout=1.0)
        if pbar is not None:
            if proc.returncode == 0 and pbar.n < target_bytes:
                pbar.update(target_bytes - pbar.n)
            pbar.close()

    ok = proc.returncode == 0 and p.exists() and p.stat().st_size > 0
    return ok, (out or "").strip()


def enable_linux_swap(
    size_gb: int = 8,
    swap_path: str = DEFAULT_SWAP_PATH,
    swappiness: int = 10,
    use_tqdm: bool = False,
    progress_cb: Optional[SwapProgressCallback] = None,
) -> dict[str, Any]:
    """
    Create and activate a Linux swapfile on disk.
    Preflight-probes container permissions in <0.5s before allocating 8 GB,
    and streams real-time byte progress when `dd` zero-fill is needed.
    """
    size_gb = max(1, min(32, int(size_gb)))
    if not _is_linux():
        return {
            "ok": False,
            "already_enabled": False,
            "message": "当前非 Linux 环境，仅在 Google Colab / Linux 容器中支持开启 Swap",
            "swap": get_swap_status(swap_path),
        }

    tot_b, _, _, _ = _read_swap_stats()
    if tot_b >= int(size_gb * 0.9 * (1024**3)):
        return {
            "ok": True,
            "already_enabled": True,
            "message": f"Linux Swap 已开启 ({tot_b / (1024**3):.1f} GB)，无需重复创建",
            "swap": get_swap_status(swap_path),
        }

    target_dir = str(Path(swap_path).parent or "/")
    disk = shutil.disk_usage(target_dir)
    free_gb = disk.free / (1024**3)
    reserve_gb = 3.0
    if free_gb < 4.0:
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"磁盘剩余空间仅 {free_gb:.1f} GB，无法安全划出 Swap 虚拟内存",
            "swap": get_swap_status(swap_path),
        }

    actual_gb = size_gb
    if free_gb < size_gb + reserve_gb:
        actual_gb = max(1, int(free_gb - reserve_gb))

    # Clean up any leftover inactive swapfile before probing/creating
    _cleanup_swapfile(swap_path)

    # Fast 4 MB preflight probe (< 0.5s) avoids waiting minutes on `dd` if container blocks swapon
    method, probe_err = _probe_swap_method(target_dir)
    if not method:
        lower_err = probe_err.lower()
        hint = (
            "（当前 Colab 容器内核隔离策略禁止 swapon/loop 挂载，已自动跳过，继续启动模型；建议在设置中保持 Sequential Offload + 2.1-Turbo 降低峰值内存）"
            if any(k in lower_err for k in ("operation not permitted", "permission", "invalid argument", "loop"))
            else ""
        )
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"当前运行环境不支持挂载 Swap: {probe_err[-200:]} {hint}".strip(),
            "swap": get_swap_status(swap_path),
        }

    # Allocate full size using the strategy proven by the 4 MB probe
    if method in {"direct_fallocate", "loop_fallocate"}:
        res_fa = _run_cmd(["fallocate", "-l", f"{actual_gb}G", swap_path])
        if res_fa.returncode != 0 or not Path(swap_path).exists() or Path(swap_path).stat().st_size == 0:
            ok_alloc, alloc_msg = _allocate_dd_with_progress(
                swap_path, actual_gb, use_tqdm=use_tqdm, progress_cb=progress_cb
            )
        else:
            ok_alloc, alloc_msg = True, ""
    else:
        ok_alloc, alloc_msg = _allocate_dd_with_progress(
            swap_path, actual_gb, use_tqdm=use_tqdm, progress_cb=progress_cb
        )

    if not ok_alloc:
        _cleanup_swapfile(swap_path)
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"分配 Swap 文件失败: {alloc_msg[-240:]}",
            "swap": get_swap_status(swap_path),
        }

    try:
        os.chmod(swap_path, 0o600)
    except OSError as exc:
        _cleanup_swapfile(swap_path)
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"设置 Swap 文件权限失败: {exc}",
            "swap": get_swap_status(swap_path),
        }

    res_mk = _run_cmd(["mkswap", swap_path])
    if res_mk.returncode != 0:
        _cleanup_swapfile(swap_path)
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"mkswap 格式化失败: {(res_mk.stdout or '').strip()[-200:]}",
            "swap": get_swap_status(swap_path),
        }

    loop_dev = ""
    if method.startswith("loop_"):
        r_lp = _run_cmd(["losetup", "-f", "--show", swap_path])
        lp_out = (r_lp.stdout or "").strip()
        if r_lp.returncode == 0 and lp_out.startswith("/dev/loop"):
            loop_dev = lp_out.splitlines()[-1].strip()
            res_on = _run_cmd(["swapon", loop_dev])
        else:
            res_on = r_lp
    else:
        res_on = _run_cmd(["swapon", swap_path])

    if res_on.returncode != 0:
        err_out = (res_on.stdout or "").strip()
        _cleanup_swapfile(swap_path, loop_dev=loop_dev)
        return {
            "ok": False,
            "already_enabled": False,
            "message": f"swapon 挂载失败: {err_out[-240:]}",
            "swap": get_swap_status(swap_path),
        }

    _run_cmd(["sysctl", "-w", f"vm.swappiness={max(1, min(100, int(swappiness)))}"])
    status = get_swap_status(swap_path)
    via = f" (via {loop_dev})" if loop_dev else ""
    return {
        "ok": True,
        "already_enabled": False,
        "message": f"已成功划出 {actual_gb} GB 磁盘开启 Linux Swap 虚拟内存{via} (总 Swap: {status['total_gb']} GB)",
        "swap": status,
    }
