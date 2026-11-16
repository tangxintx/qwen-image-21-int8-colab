"""
High-speed Hugging Face API model downloader with real-time byte-level progress tracking.
Uses `huggingface_hub.snapshot_download` / `hf_hub_download` + `hf_transfer` (when available) + `HF_TOKEN`.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("qwen_easycolab.hf")

ProgressCallback = Callable[[int, int, float, str], None]

GGUF_BASE_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
GGUF_REDIRECTED_REPO = "abenzerps/Qwen-Image-2.1-Turbo-Quantized"

GGUF_VARIANTS: dict[str, dict[str, Any]] = {
    "q8": {
        "label": "Qwen-Image-2.1 Uncensored GGUF Q8_0 (满血BF16文本编码器 + BF16 VAE)",
        "dit_repo": GGUF_BASE_REPO,
        "dit_patterns": ["qwen-image-2.1-turbo-Q8_0.gguf", "*Q8_0.gguf", "*q8_0.gguf"],
        "dit_diffusers_fallback_repo": "m9ter1a/Qwen-Image-2.1-Turbo-GGUF",
        "dit_diffusers_fallback_patterns": ["*Q8_0.gguf", "*q8_0.gguf"],
        "te_repo": GGUF_BASE_REPO,
        "te_patterns": ["text_encoders/qwen3vl_8b_bf16.safetensors", "*qwen3vl*bf16*.safetensors"],
        "te_alt_repo": "Comfy-Org/Qwen-Image-2.1",
        "te_alt_patterns": ["split_files/text_encoders/qwen_3_vl_8b.safetensors", "*qwen*vl*8b*.safetensors"],
        "mmproj_repo": None,
        "mmproj_patterns": [],
        "vae_repo": GGUF_BASE_REPO,
        "vae_patterns": ["vae/qwen_image_2.1_vae_bf16.safetensors", "*qwen_image_2.1_vae_bf16*.safetensors"],
        "vae_alt_repo": "Comfy-Org/Qwen-Image-2.1",
        "vae_alt_patterns": ["split_files/vae/qwen_image_2.1_vae_bf16.safetensors", "*qwen_image_2.1_vae_bf16*.safetensors"],
        "base_config_repo": "Qwen/Qwen-Image-2.1-Turbo",
    },
    "q4": {
        "label": "Qwen-Image-2.1 Uncensored GGUF Q4_K_M (Qwen3-VL-8B Q4_K_M.gguf + BF16 VAE)",
        "dit_repo": GGUF_BASE_REPO,
        "dit_patterns": ["qwen-image-2.1-turbo-Q4_K_M.gguf", "*Q4_K_M.gguf", "*q4_k_m.gguf"],
        "dit_diffusers_fallback_repo": "m9ter1a/Qwen-Image-2.1-Turbo-GGUF",
        "dit_diffusers_fallback_patterns": ["*Q4_K_M.gguf", "*q4_k_m.gguf"],
        "te_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "te_patterns": ["Qwen3VL-8B-Instruct-Q4_K_M.gguf", "*Q4_K_M.gguf", "*q4_k_m.gguf"],
        "te_alt_repo": None,
        "te_alt_patterns": [],
        "mmproj_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "mmproj_patterns": [
            "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
            "mmproj*Q8_0*.gguf",
            "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
            "mmproj*.gguf",
        ],
        "vae_repo": GGUF_BASE_REPO,
        "vae_patterns": ["vae/qwen_image_2.1_vae_bf16.safetensors", "*qwen_image_2.1_vae_bf16*.safetensors"],
        "vae_alt_repo": "Comfy-Org/Qwen-Image-2.1",
        "vae_alt_patterns": ["split_files/vae/qwen_image_2.1_vae_bf16.safetensors", "*qwen_image_2.1_vae_bf16*.safetensors"],
        "base_config_repo": "Qwen/Qwen-Image-2.1-Turbo",
    },
}


def strip_model_variant_suffix(model_id: str) -> tuple[str, Optional[str]]:
    """Split `<repo_id>:q8` / `<repo_id>:q4` into `(repo_id, variant)` cleanly."""
    clean = (model_id or "").strip()
    lower = clean.lower()
    for suffix in (":q8", ":q4"):
        if lower.endswith(suffix):
            return clean[: -len(suffix)], suffix[1:]
    return clean, None


def _resolve_token(token: Optional[str]) -> Optional[str]:
    resolved = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    if resolved:
        os.environ["HF_TOKEN"] = resolved
    return resolved


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


def _matches_patterns(rfilename: str, patterns: Optional[list[str]]) -> bool:
    if not patterns:
        return True
    low = rfilename.lower()
    base = Path(rfilename).name.lower()
    for pat in patterns:
        pat_low = pat.lower()
        if fnmatch.fnmatch(low, pat_low) or fnmatch.fnmatch(base, pat_low):
            return True
    return False


def _fetch_repo_total_bytes(
    repo_id: str,
    token: Optional[str],
    allow_patterns: Optional[list[str]] = None,
) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token or None).model_info(repo_id=repo_id, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        return sum(
            (getattr(f, "size", 0) or 0)
            for f in siblings
            if _matches_patterns(getattr(f, "rfilename", "") or "", allow_patterns)
        )
    except Exception as exc:
        logger.warning("Could not query repo size for %s: %s", repo_id, exc)
        return 0


def resolve_repo_file(
    repo_id: str,
    patterns: list[str],
    token: Optional[str] = None,
) -> tuple[str, str, int]:
    """
    Resolve the canonical repo ID (following HF redirects), matching filename, and byte size
    for the highest-priority pattern in `patterns`.
    """
    from huggingface_hub import HfApi

    resolved_token = _resolve_token(token)
    info = HfApi(token=resolved_token).model_info(repo_id=repo_id, files_metadata=True)
    canonical_repo = getattr(info, "id", None) or repo_id
    siblings = getattr(info, "siblings", None) or []

    for pat in patterns:
        pat_low = pat.lower()
        for f in siblings:
            rfile = getattr(f, "rfilename", "") or ""
            if rfile == pat or rfile.lower() == pat_low:
                return canonical_repo, rfile, int(getattr(f, "size", 0) or 0)
        for f in siblings:
            rfile = getattr(f, "rfilename", "") or ""
            if fnmatch.fnmatch(rfile.lower(), pat_low) or fnmatch.fnmatch(Path(rfile).name.lower(), pat_low):
                return canonical_repo, rfile, int(getattr(f, "size", 0) or 0)

    raise FileNotFoundError(f"在仓库 {repo_id} 中未找到匹配文件 {patterns}")


def download_hf_file_fast(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    expected_bytes: int = 0,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
) -> str:
    """
    Download a single file from Hugging Face Hub using `hf_hub_download` + `hf_transfer`,
    tracking byte deltas cleanly without pulling unrelated files in huge repos.
    """
    from huggingface_hub import hf_hub_download

    resolved_token = _resolve_token(token)
    hf_transfer_on = _enable_hf_transfer_if_available()
    blobs_dir = _repo_cache_blobs_dir(repo_id)

    stop_event = threading.Event()
    t_start = time.time()
    initial_bytes = _measure_blobs_bytes(blobs_dir)
    display_name = Path(filename).name

    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=expected_bytes if expected_bytes > 0 else None,
                initial=0,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"HF 下载 {display_name}" + (" [hf_transfer]" if hf_transfer_on else ""),
            )
        except Exception:
            pbar = None

    def _watcher() -> None:
        last_bytes = initial_bytes
        last_time = t_start
        while not stop_event.wait(0.45):
            cur_bytes = _measure_blobs_bytes(blobs_dir)
            delta_total = max(0, cur_bytes - initial_bytes)
            now = time.time()
            dt = max(0.1, now - last_time)
            speed_mb_s = max(0.0, (cur_bytes - last_bytes) / (1024 * 1024) / dt)
            last_bytes = cur_bytes
            last_time = now

            if pbar is not None:
                step = max(0, delta_total - pbar.n)
                if step > 0:
                    if pbar.total and pbar.n + step > pbar.total:
                        pbar.total = pbar.n + step
                    pbar.update(step)

            if progress_cb is not None:
                cur_gb = delta_total / (1024**3)
                tot_gb = expected_bytes / (1024**3) if expected_bytes > 0 else 0.0
                if expected_bytes > 0:
                    pct = min(99, int((delta_total / expected_bytes) * 100))
                    msg = f"HF 下载 {display_name}: {cur_gb:.2f}/{tot_gb:.2f} GB ({pct}% · {speed_mb_s:.1f} MB/s)"
                else:
                    msg = f"HF 下载 {display_name}: {cur_gb:.2f} GB ({speed_mb_s:.1f} MB/s)"
                progress_cb(delta_total, expected_bytes, speed_mb_s, msg)

    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()

    try:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=resolved_token,
            )
        except Exception as exc:
            if hf_transfer_on:
                logger.warning("hf_transfer file download failed (%s), retrying standard HF API...", exc)
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=resolved_token,
                )
            else:
                raise
        return local_path
    finally:
        stop_event.set()
        watcher_thread.join(timeout=1.0)
        if pbar is not None:
            if pbar.total and pbar.n < pbar.total:
                pbar.update(max(0, pbar.total - pbar.n))
            pbar.close()
        if progress_cb is not None:
            final_b = expected_bytes or max(0, _measure_blobs_bytes(blobs_dir) - initial_bytes)
            progress_cb(final_b, final_b, 0.0, f"{display_name} 下载就绪")


def download_model_fast(
    repo_id: str,
    token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> str:
    """
    Download `repo_id` via Hugging Face API (`snapshot_download` + optional `hf_transfer`)
    and return the local snapshot directory path.
    """
    from huggingface_hub import snapshot_download

    resolved_token = _resolve_token(token)
    hf_transfer_on = _enable_hf_transfer_if_available()
    total_bytes = _fetch_repo_total_bytes(repo_id, resolved_token, allow_patterns=allow_patterns)
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

    snap_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "token": resolved_token,
        "max_workers": 8,
    }
    if allow_patterns:
        snap_kwargs["allow_patterns"] = allow_patterns

    try:
        try:
            local_dir = snapshot_download(**snap_kwargs)
        except Exception as exc:
            if hf_transfer_on:
                logger.warning("hf_transfer download failed (%s), retrying with standard HF API workers...", exc)
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                local_dir = snapshot_download(**snap_kwargs)
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


def _resolve_with_alt(
    primary_repo: str,
    primary_patterns: list[str],
    alt_repo: Optional[str],
    alt_patterns: list[str],
    token: Optional[str],
) -> tuple[str, str, int]:
    try:
        return resolve_repo_file(primary_repo, primary_patterns, token=token)
    except Exception as primary_exc:
        if alt_repo and alt_patterns:
            logger.info("Falling back to alt repo %s (%s)", alt_repo, primary_exc)
            return resolve_repo_file(alt_repo, alt_patterns, token=token)
        raise


def download_gguf_bundle_fast(
    variant: str,
    token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
    include_diffusers_configs: bool = False,
    include_mmproj: bool = True,
) -> dict[str, Any]:
    """
    Download only the exact GGUF DiT, Text Encoder (`qwen3vl_8b_bf16` for Q8 or `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
    + `mmproj` for Q4), and `qwen_image_2.1_vae_bf16.safetensors` files needed for the chosen variant.
    """
    v = (variant or "q8").strip().lower()
    if v not in GGUF_VARIANTS:
        raise ValueError(f"不支持的 GGUF 量化档位: {variant}（可选 q8 / q4）")

    cfg = GGUF_VARIANTS[v]
    resolved_token = _resolve_token(token)

    dit_repo, dit_file, dit_size = resolve_repo_file(cfg["dit_repo"], cfg["dit_patterns"], token=resolved_token)
    te_repo, te_file, te_size = _resolve_with_alt(
        cfg["te_repo"],
        cfg["te_patterns"],
        cfg.get("te_alt_repo"),
        cfg.get("te_alt_patterns") or [],
        token=resolved_token,
    )
    vae_repo, vae_file, vae_size = _resolve_with_alt(
        cfg["vae_repo"],
        cfg["vae_patterns"],
        cfg.get("vae_alt_repo"),
        cfg.get("vae_alt_patterns") or [],
        token=resolved_token,
    )

    mmproj_repo: Optional[str] = None
    mmproj_file: Optional[str] = None
    mmproj_size = 0
    if include_mmproj and cfg.get("mmproj_repo") and cfg.get("mmproj_patterns"):
        try:
            mmproj_repo, mmproj_file, mmproj_size = resolve_repo_file(
                cfg["mmproj_repo"],
                cfg["mmproj_patterns"],
                token=resolved_token,
            )
        except Exception as exc:
            logger.warning("mmproj GGUF resolution skipped (%s)", exc)

    total_expected = dit_size + te_size + vae_size + mmproj_size
    completed_bytes = 0

    def _sub_progress(label: str):
        if progress_cb is None:
            return None

        def _inner(cur_b: int, _tot_b: int, spd: float, msg: str) -> None:
            agg_cur = completed_bytes + cur_b
            progress_cb(agg_cur, total_expected, spd, f"[{label}] {msg}")

        return _inner

    dit_path = download_hf_file_fast(
        dit_repo,
        dit_file,
        token=resolved_token,
        expected_bytes=dit_size,
        progress_cb=_sub_progress(f"DiT {v.upper()}"),
        use_tqdm=use_tqdm,
    )
    completed_bytes += dit_size

    te_path = download_hf_file_fast(
        te_repo,
        te_file,
        token=resolved_token,
        expected_bytes=te_size,
        progress_cb=_sub_progress("文本编码器"),
        use_tqdm=use_tqdm,
    )
    completed_bytes += te_size

    mmproj_path: Optional[str] = None
    if mmproj_repo and mmproj_file:
        mmproj_path = download_hf_file_fast(
            mmproj_repo,
            mmproj_file,
            token=resolved_token,
            expected_bytes=mmproj_size,
            progress_cb=_sub_progress("视觉投影 mmproj"),
            use_tqdm=use_tqdm,
        )
        completed_bytes += mmproj_size

    vae_path = download_hf_file_fast(
        vae_repo,
        vae_file,
        token=resolved_token,
        expected_bytes=vae_size,
        progress_cb=_sub_progress("BF16 VAE"),
        use_tqdm=use_tqdm,
    )
    completed_bytes += vae_size

    base_config_dir: Optional[str] = None
    if include_diffusers_configs and cfg.get("base_config_repo"):
        base_config_dir = download_model_fast(
            cfg["base_config_repo"],
            token=resolved_token,
            progress_cb=_sub_progress("Pipeline 配置"),
            use_tqdm=use_tqdm,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "processor/*",
                "transformer/config.json",
                "text_encoder/config.json",
                "vae/config.json",
            ],
        )

    if progress_cb is not None:
        progress_cb(total_expected, total_expected, 0.0, f"GGUF {v.upper()} 模型组件下载完成")

    return {
        "variant": v,
        "label": cfg["label"],
        "dit_repo": dit_repo,
        "dit_filename": dit_file,
        "dit_path": dit_path,
        "te_repo": te_repo,
        "te_filename": te_file,
        "te_path": te_path,
        "mmproj_repo": mmproj_repo,
        "mmproj_filename": mmproj_file,
        "mmproj_path": mmproj_path,
        "vae_repo": vae_repo,
        "vae_filename": vae_file,
        "vae_path": vae_path,
        "base_config_dir": base_config_dir,
    }

