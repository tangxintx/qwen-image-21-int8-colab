"""
High-speed Hugging Face API model downloader with real-time byte-level progress tracking.
Uses `huggingface_hub.snapshot_download` / `hf_hub_download` + `hf_transfer` (when available) + `HF_TOKEN`.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import importlib.util
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("qwen_easycolab.hf")

ProgressCallback = Callable[[int, int, float, str], None]

# Configure HF_HUB_ENABLE_HF_TRANSFER *before* importing huggingface_hub anywhere,
# since huggingface_hub evaluates constants at import time.
_HF_TRANSFER_AVAILABLE = importlib.util.find_spec("hf_transfer") is not None
if _HF_TRANSFER_AVAILABLE and not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
_HF_TRANSFER_INITIAL_ENABLED = (
    _HF_TRANSFER_AVAILABLE
    and os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0").strip().lower() in {"1", "true", "yes"}
)
_hf_transfer_lock = threading.Lock()
_hf_transfer_disabled = False

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
    "int8_convrot": {
        "label": "Qwen-Image-2.1-Turbo INT8-ConvRot (ComfyUI / comfy-kitchen 极速版 · 7.26GB DiT + 9.35GB TE)",
        "dit_repo": GGUF_BASE_REPO,
        "dit_patterns": [
            "qwen-image-2.1-turbo-int8_convrot.safetensors",
            "qwen_image_2.1_turbo_int8_convrot.safetensors",
            "*turbo*int8_convrot*.safetensors",
            "*int8_convrot*.safetensors",
        ],
        "dit_diffusers_fallback_repo": None,
        "dit_diffusers_fallback_patterns": [],
        "te_repo": GGUF_BASE_REPO,
        "te_patterns": [
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "*qwen3vl*int8_convrot*.safetensors",
        ],
        "te_alt_repo": GGUF_REDIRECTED_REPO,
        "te_alt_patterns": ["text_encoders/qwen3vl_8b_bf16.safetensors"],
        "mmproj_repo": None,
        "mmproj_patterns": [],
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
    return (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None


def _is_hf_transfer_active() -> bool:
    with _hf_transfer_lock:
        if _hf_transfer_disabled or not _HF_TRANSFER_INITIAL_ENABLED:
            return False
    try:
        import huggingface_hub.constants as hf_const

        return bool(getattr(hf_const, "HF_HUB_ENABLE_HF_TRANSFER", _HF_TRANSFER_INITIAL_ENABLED))
    except Exception:
        return _HF_TRANSFER_INITIAL_ENABLED


def _disable_hf_transfer_safely() -> None:
    """Disable hf_transfer through `huggingface_hub.constants` under a lock instead of mutating os.environ."""
    global _hf_transfer_disabled
    with _hf_transfer_lock:
        _hf_transfer_disabled = True
        try:
            import huggingface_hub.constants as hf_const

            if hasattr(hf_const, "HF_HUB_ENABLE_HF_TRANSFER"):
                hf_const.HF_HUB_ENABLE_HF_TRANSFER = False
        except Exception:
            pass


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


def _measure_target_blob_bytes(blobs_dir: Path, expected_sha256: Optional[str]) -> Optional[int]:
    """Measure bytes on disk for a single known LFS blob (complete or `.incomplete`) without repo cross-talk."""
    if not expected_sha256 or not blobs_dir.exists():
        return None
    sha_clean = expected_sha256.strip().lower()
    if not sha_clean:
        return None
    for candidate in (blobs_dir / sha_clean, blobs_dir / f"{sha_clean}.incomplete"):
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.stat().st_size
        except OSError:
            pass
    return 0


def verify_file_sha256(file_path: Path, expected_sha256: str) -> None:
    """Verify SHA256 digest of a downloaded file and unlink corrupted artifacts on mismatch."""
    target = expected_sha256.strip().lower()
    if not target:
        return
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest().lower()
    if actual != target:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"文件 {file_path.name} SHA256 校验失败 (预期 {target[:12]}..., 实际 {actual[:12]}...)，已自动清理损坏文件"
        )


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


def _extract_lfs_sha256(sibling: Any) -> Optional[str]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        val = lfs.get("sha256")
    else:
        val = getattr(lfs, "sha256", None)
    return str(val).strip().lower() if val else None


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


def resolve_repo_file_meta(
    repo_id: str,
    patterns: list[str],
    token: Optional[str] = None,
) -> tuple[str, str, int, Optional[str]]:
    """
    Resolve `(canonical_repo, filename, byte_size, lfs_sha256)` for the highest-priority pattern in `patterns`.
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
                return canonical_repo, rfile, int(getattr(f, "size", 0) or 0), _extract_lfs_sha256(f)
        for f in siblings:
            rfile = getattr(f, "rfilename", "") or ""
            if fnmatch.fnmatch(rfile.lower(), pat_low) or fnmatch.fnmatch(Path(rfile).name.lower(), pat_low):
                return canonical_repo, rfile, int(getattr(f, "size", 0) or 0), _extract_lfs_sha256(f)

    raise FileNotFoundError(f"在仓库 {repo_id} 中未找到匹配文件 {patterns}")


def resolve_repo_file(
    repo_id: str,
    patterns: list[str],
    token: Optional[str] = None,
) -> tuple[str, str, int]:
    """
    Resolve the canonical repo ID (following HF redirects), matching filename, and byte size
    for the highest-priority pattern in `patterns`.
    """
    canonical_repo, rfile, size, _ = resolve_repo_file_meta(repo_id, patterns, token=token)
    return canonical_repo, rfile, size


def download_hf_file_fast(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    expected_bytes: int = 0,
    expected_sha256: Optional[str] = None,
    verify_sha256: bool = True,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
) -> str:
    """
    Download a single file from Hugging Face Hub using `hf_hub_download`,
    tracking single-file blob progress (preserving resume offset) and verifying SHA256 when available.
    """
    from huggingface_hub import hf_hub_download

    resolved_token = _resolve_token(token)
    hf_transfer_on = _is_hf_transfer_active()
    blobs_dir = _repo_cache_blobs_dir(repo_id)

    stop_event = threading.Event()
    t_start = time.time()
    use_blob_sha = bool(expected_sha256)
    initial_bytes = 0 if use_blob_sha else _measure_blobs_bytes(blobs_dir)
    display_name = Path(filename).name

    def _current_file_bytes() -> int:
        if use_blob_sha:
            measured = _measure_target_blob_bytes(blobs_dir, expected_sha256)
            if measured is not None:
                return measured
        return max(0, _measure_blobs_bytes(blobs_dir) - initial_bytes)

    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            init_b = min(_current_file_bytes(), expected_bytes) if expected_bytes > 0 else _current_file_bytes()
            pbar = tqdm(
                total=expected_bytes if expected_bytes > 0 else None,
                initial=init_b,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"HF 下载 {display_name}" + (" [hf_transfer]" if hf_transfer_on else ""),
            )
        except Exception:
            pbar = None

    def _watcher() -> None:
        last_bytes = _current_file_bytes()
        last_time = t_start
        while not stop_event.wait(0.45):
            cur_bytes = _current_file_bytes()
            now = time.time()
            dt = max(0.1, now - last_time)
            speed_mb_s = max(0.0, (cur_bytes - last_bytes) / (1024 * 1024) / dt)
            last_bytes = cur_bytes
            last_time = now

            if pbar is not None:
                step = max(0, cur_bytes - pbar.n)
                if step > 0:
                    if pbar.total and pbar.n + step > pbar.total:
                        pbar.total = pbar.n + step
                    pbar.update(step)

            if progress_cb is not None:
                cur_gb = cur_bytes / (1024**3)
                tot_gb = expected_bytes / (1024**3) if expected_bytes > 0 else 0.0
                if expected_bytes > 0:
                    pct = min(99, int((cur_bytes / expected_bytes) * 100))
                    msg = f"HF 下载 {display_name}: {cur_gb:.2f}/{tot_gb:.2f} GB ({pct}% · {speed_mb_s:.1f} MB/s)"
                else:
                    msg = f"HF 下载 {display_name}: {cur_gb:.2f} GB ({speed_mb_s:.1f} MB/s)"
                progress_cb(cur_bytes, expected_bytes, speed_mb_s, msg)

    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()
    succeeded = False

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
                _disable_hf_transfer_safely()
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=resolved_token,
                )
            else:
                raise

        if verify_sha256 and expected_sha256:
            lp = Path(local_path)
            if lp.exists() and lp.is_file():
                verify_file_sha256(lp, expected_sha256)

        succeeded = True
        return local_path
    finally:
        stop_event.set()
        watcher_thread.join(timeout=1.0)
        if pbar is not None:
            if succeeded and pbar.total and pbar.n < pbar.total:
                pbar.update(max(0, pbar.total - pbar.n))
            pbar.close()
        if succeeded and progress_cb is not None:
            final_b = expected_bytes or _current_file_bytes()
            progress_cb(final_b, final_b, 0.0, f"{display_name} 下载就绪")


def download_model_fast(
    repo_id: str,
    token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> str:
    """
    Download `repo_id` via Hugging Face API (`snapshot_download`)
    and return the local snapshot directory path.
    """
    from huggingface_hub import snapshot_download

    resolved_token = _resolve_token(token)
    hf_transfer_on = _is_hf_transfer_active()
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

    succeeded = False
    try:
        try:
            local_dir = snapshot_download(**snap_kwargs)
        except Exception as exc:
            if hf_transfer_on:
                logger.warning("hf_transfer download failed (%s), retrying with standard HF API workers...", exc)
                _disable_hf_transfer_safely()
                local_dir = snapshot_download(**snap_kwargs)
            else:
                raise
        succeeded = True
        return local_dir
    finally:
        stop_event.set()
        watcher_thread.join(timeout=1.0)
        if pbar is not None:
            if succeeded:
                final_bytes = _measure_blobs_bytes(blobs_dir)
                if pbar.total and final_bytes < pbar.total:
                    pbar.update(max(0, pbar.total - pbar.n))
            pbar.close()
        if succeeded and progress_cb is not None:
            final_bytes = total_bytes or _measure_blobs_bytes(blobs_dir)
            progress_cb(final_bytes, final_bytes, 0.0, f"模型 {repo_id.split('/')[-1]} 缓存就绪")


def _resolve_with_alt_meta(
    primary_repo: str,
    primary_patterns: list[str],
    alt_repo: Optional[str],
    alt_patterns: list[str],
    token: Optional[str],
) -> tuple[str, str, int, Optional[str]]:
    try:
        return resolve_repo_file_meta(primary_repo, primary_patterns, token=token)
    except Exception as primary_exc:
        if alt_repo and alt_patterns:
            logger.info("Falling back to alt repo %s (%s)", alt_repo, primary_exc)
            return resolve_repo_file_meta(alt_repo, alt_patterns, token=token)
        raise


def _resolve_with_alt(
    primary_repo: str,
    primary_patterns: list[str],
    alt_repo: Optional[str],
    alt_patterns: list[str],
    token: Optional[str],
) -> tuple[str, str, int]:
    canonical_repo, rfile, size, _ = _resolve_with_alt_meta(
        primary_repo, primary_patterns, alt_repo, alt_patterns, token=token
    )
    return canonical_repo, rfile, size


def download_gguf_bundle_fast(
    variant: str,
    token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
    include_diffusers_configs: bool = False,
    include_mmproj: bool = True,
    include_te: bool = True,
    max_workers: int = 2,
) -> dict[str, Any]:
    """
    Download only the exact DiT, Text Encoder, optional mmproj, and VAE files needed
    for the chosen quantized variant (`q8`, `q4`, or `int8_convrot`).
    """
    v = (variant or "q8").strip().lower()
    if v not in GGUF_VARIANTS:
        raise ValueError(f"不支持的量化档位: {variant}（可选 {', '.join(GGUF_VARIANTS.keys())}）")

    cfg = GGUF_VARIANTS[v]
    resolved_token = _resolve_token(token)

    dit_repo, dit_file, dit_size, dit_sha = _resolve_with_alt_meta(
        cfg["dit_repo"],
        cfg["dit_patterns"],
        cfg.get("dit_diffusers_fallback_repo"),
        cfg.get("dit_diffusers_fallback_patterns") or [],
        token=resolved_token,
    )

    te_repo: Optional[str] = None
    te_file: Optional[str] = None
    te_size = 0
    te_sha: Optional[str] = None
    if include_te and cfg.get("te_repo") and cfg.get("te_patterns"):
        te_repo, te_file, te_size, te_sha = _resolve_with_alt_meta(
            cfg["te_repo"],
            cfg["te_patterns"],
            cfg.get("te_alt_repo"),
            cfg.get("te_alt_patterns") or [],
            token=resolved_token,
        )

    vae_repo, vae_file, vae_size, vae_sha = _resolve_with_alt_meta(
        cfg["vae_repo"],
        cfg["vae_patterns"],
        cfg.get("vae_alt_repo"),
        cfg.get("vae_alt_patterns") or [],
        token=resolved_token,
    )

    mmproj_repo: Optional[str] = None
    mmproj_file: Optional[str] = None
    mmproj_size = 0
    mmproj_sha: Optional[str] = None
    if include_mmproj and cfg.get("mmproj_repo") and cfg.get("mmproj_patterns"):
        try:
            mmproj_repo, mmproj_file, mmproj_size, mmproj_sha = resolve_repo_file_meta(
                cfg["mmproj_repo"],
                cfg["mmproj_patterns"],
                token=resolved_token,
            )
        except Exception as exc:
            logger.warning("mmproj GGUF resolution skipped (%s)", exc)

    total_expected = dit_size + te_size + vae_size + mmproj_size

    progress_lock = threading.Lock()
    progress_map: dict[str, int] = {}

    def _sub_progress(key: str, label: str, expected_sz: int):
        if progress_cb is None:
            return None

        def _inner(cur_b: int, _tot_b: int, spd: float, msg: str) -> None:
            with progress_lock:
                progress_map[key] = min(cur_b, expected_sz) if expected_sz > 0 else cur_b
                agg_cur = sum(progress_map.values())
            progress_cb(agg_cur, total_expected, spd, f"[{label}] {msg}")

        return _inner

    download_specs: list[tuple[str, str, str, str, int, Optional[str]]] = [
        ("dit", f"DiT {v.upper()}", dit_repo, dit_file, dit_size, dit_sha),
    ]
    if te_repo and te_file:
        download_specs.append(("te", "文本编码器", te_repo, te_file, te_size, te_sha))
    if mmproj_repo and mmproj_file:
        download_specs.append(("mmproj", "视觉投影 mmproj", mmproj_repo, mmproj_file, mmproj_size, mmproj_sha))
    download_specs.append(("vae", "BF16 VAE", vae_repo, vae_file, vae_size, vae_sha))

    results: dict[str, str] = {}
    eff_workers = 1 if use_tqdm else max(1, min(int(max_workers), len(download_specs)))
    if eff_workers == 1:
        for key, label, r_id, f_name, sz, sha in download_specs:
            results[key] = download_hf_file_fast(
                r_id,
                f_name,
                token=resolved_token,
                expected_bytes=sz,
                expected_sha256=sha,
                progress_cb=_sub_progress(key, label, sz),
                use_tqdm=use_tqdm,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as pool:
            future_map = {
                pool.submit(
                    download_hf_file_fast,
                    r_id,
                    f_name,
                    resolved_token,
                    sz,
                    sha,
                    True,
                    _sub_progress(key, label, sz),
                    False,
                ): key
                for key, label, r_id, f_name, sz, sha in download_specs
            }
            for fut in concurrent.futures.as_completed(future_map):
                key = future_map[fut]
                results[key] = fut.result()

    dit_path = results["dit"]
    te_path = results.get("te")
    mmproj_path = results.get("mmproj")
    vae_path = results["vae"]

    base_config_dir: Optional[str] = None
    if include_diffusers_configs and cfg.get("base_config_repo"):
        base_config_dir = download_model_fast(
            cfg["base_config_repo"],
            token=resolved_token,
            progress_cb=_sub_progress("cfg", "Pipeline 配置", 0),
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
        progress_cb(total_expected, total_expected, 0.0, f"{v.upper()} 模型组件下载完成")

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
