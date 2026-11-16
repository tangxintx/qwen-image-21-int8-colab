"""
High-speed Hugging Face API model downloader with real-time byte-level progress tracking.
Uses `huggingface_hub.snapshot_download` + `hf_transfer` (when available) + `HF_TOKEN`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("qwen_easycolab.hf")

ProgressCallback = Callable[[int, int, float, str], None]


def _enable_hf_transfer_if_available() -> bool:
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        return False


def _repo_cache_blobs_dir(repo_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    folder_name = "models--" + repo_id.replace("/", "--")
    return hub_dir / folder_name / "blobs"


def _measure_blobs_bytes(blobs_dir: Path) -> int:
    if not blobs_dir.exists():
        return 0
    total = 0
    try:
        for entry in os.scandir(blobs_dir):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _fetch_repo_total_bytes(repo_id: str, token: Optional[str]) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token or None).model_info(repo_id=repo_id, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        return sum(getattr(f, "size", 0) or 0 for f in siblings)
    except Exception as exc:
        logger.warning("Could not query repo size for %s: %s", repo_id, exc)
        return 0


def download_model_fast(
    repo_id: str,
    token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
) -> str:
    """
    Download `repo_id` via Hugging Face API (`snapshot_download` + optional `hf_transfer`)
    and return the local snapshot directory path.
    """
    from huggingface_hub import snapshot_download

    resolved_token = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    if resolved_token:
        os.environ["HF_TOKEN"] = resolved_token

    hf_transfer_on = _enable_hf_transfer_if_available()
    total_bytes = _fetch_repo_total_bytes(repo_id, resolved_token)
    blobs_dir = _repo_cache_blobs_dir(repo_id)

    stop_event = threading.Event()
    t_start = time.time()
    initial_bytes = _measure_blobs_bytes(blobs_dir)

    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=total_bytes if total_bytes > 0 else None,
                initial=min(initial_bytes, total_bytes) if total_bytes > 0 else initial_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"HF API 下载 {repo_id.split('/')[-1]}" + (" [hf_transfer]" if hf_transfer_on else ""),
            )
        except Exception:
            pbar = None

    def _watcher() -> None:
        last_bytes = initial_bytes
        last_time = t_start
        while not stop_event.wait(0.45):
            cur_bytes = _measure_blobs_bytes(blobs_dir)
            now = time.time()
            dt = max(0.1, now - last_time)
            speed_mb_s = max(0.0, (cur_bytes - last_bytes) / (1024 * 1024) / dt)
            last_bytes = cur_bytes
            last_time = now

            if pbar is not None:
                delta = max(0, cur_bytes - pbar.n)
                if delta > 0:
                    if pbar.total and pbar.n + delta > pbar.total:
                        pbar.total = pbar.n + delta
                    pbar.update(delta)

            if progress_cb is not None:
                cur_gb = cur_bytes / (1024**3)
                tot_gb = total_bytes / (1024**3) if total_bytes > 0 else 0.0
                if total_bytes > 0:
                    pct = min(99, int((cur_bytes / total_bytes) * 100))
                    msg = f"HF API 下载 {repo_id.split('/')[-1]}: {cur_gb:.2f}/{tot_gb:.2f} GB ({pct}% · {speed_mb_s:.1f} MB/s)"
                else:
                    msg = f"HF API 下载 {repo_id.split('/')[-1]}: {cur_gb:.2f} GB ({speed_mb_s:.1f} MB/s)"
                progress_cb(cur_bytes, total_bytes, speed_mb_s, msg)

    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()

    try:
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                token=resolved_token,
                max_workers=8,
            )
        except Exception as exc:
            if hf_transfer_on:
                logger.warning("hf_transfer download failed (%s), retrying with standard HF API workers...", exc)
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    token=resolved_token,
                    max_workers=8,
                )
            else:
                raise
        return local_dir
    finally:
        stop_event.set()
        watcher_thread.join(timeout=1.0)
        if pbar is not None:
            final_bytes = _measure_blobs_bytes(blobs_dir)
            if pbar.total and final_bytes < pbar.total:
                pbar.update(max(0, pbar.total - pbar.n))
            pbar.close()
        if progress_cb is not None:
            final_bytes = total_bytes or _measure_blobs_bytes(blobs_dir)
            progress_cb(final_bytes, final_bytes, 0.0, f"模型 {repo_id.split('/')[-1]} 缓存就绪")
