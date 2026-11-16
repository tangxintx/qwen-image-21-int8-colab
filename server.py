"""
Qwen-Image-2.1 INT8 Mobile Web Server for Google Colab.
Inspired by wy51ai/easycolab — lightweight, touch-first, mobile-friendly FastAPI backend.
"""

from __future__ import annotations

import base64
import ctypes
import gc
import inspect
import io
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import psutil
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from pydantic import BaseModel, Field

from comfy_launcher import ComfyUILauncher
from gpu_profiles import GPU_PROFILES, GPUProfile, match_gpu_profile
from hf_downloader import (
    GGUF_BASE_REPO,
    GGUF_REDIRECTED_REPO,
    GGUF_VARIANTS,
    download_gguf_bundle_fast,
    download_hf_file_fast,
    download_model_fast,
    resolve_repo_file,
    strip_model_variant_suffix,
)
from lora_manager import (
    LoraDownloadManager,
    delete_lora,
    get_loras_dir,
    list_loras,
    normalize_lora_state_dict,
)
from swap_manager import enable_linux_swap, get_swap_status
from terminal_manager import LogRingBuffer, TerminalManager

SERVER_START_TIME = time.time()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def trim_host_memory() -> None:
    """Force Python GC and glibc arena trim on Linux to return freed RAM to the OS."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def _parse_cc(compute_cap: str) -> tuple[int, int]:
    try:
        major_s, minor_s = (compute_cap or "0.0").split(".", 1)
        return int(major_s), int(minor_s)
    except Exception:
        return 0, 0


def build_pipeline_quant_config(quant_mode: str, profile: GPUProfile) -> tuple[Any, str]:
    """
    Build component-wise `PipelineQuantizationConfig` so BOTH `transformer` (DiT)
    and `text_encoder` (Qwen3-VL 8B) are quantized, avoiding a 17 GB BF16 text encoder,
    while keeping `vae` untouched in native BF16/FP16 for color & detail fidelity.
    """
    if quant_mode == "none":
        return None, "满血原生精度 (不量化)"

    cc_major, cc_minor = _parse_cc(profile.compute_cap)
    supports_fp8 = (cc_major > 8) or (cc_major == 8 and cc_minor >= 9)

    if quant_mode in {"fp8-int8", "fp8-all", "int8-torchao"}:
        from diffusers import PipelineQuantizationConfig
        from diffusers import TorchAoConfig as DiffusersTorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        want_dit_fp8 = quant_mode in {"fp8-int8", "fp8-all"}
        want_te_fp8 = quant_mode == "fp8-all"

        if (want_dit_fp8 or want_te_fp8) and not supports_fp8:
            logger.warning(
                "GPU %s (sm_%d%d) lacks Ada/Hopper FP8 Tensor Cores; degrading FP8 components to INT8.",
                profile.display_name,
                cc_major,
                cc_minor,
            )
            want_dit_fp8 = False
            want_te_fp8 = False

        if want_dit_fp8 or want_te_fp8:
            from torchao.quantization import Float8WeightOnlyConfig

            dit_ao = Float8WeightOnlyConfig() if want_dit_fp8 else Int8WeightOnlyConfig()
            te_ao = Float8WeightOnlyConfig() if want_te_fp8 else Int8WeightOnlyConfig()
        else:
            dit_ao = Int8WeightOnlyConfig()
            te_ao = Int8WeightOnlyConfig()

        cfg = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersTorchAoConfig(dit_ao),
                "text_encoder": TransformersTorchAoConfig(te_ao),
            }
        )
        desc = (
            f"DiT={'FP8' if want_dit_fp8 else 'INT8'} + "
            f"TextEncoder={'FP8' if want_te_fp8 else 'INT8'} + VAE={profile.dtype.upper()}"
        )
        return cfg, desc

    if quant_mode == "int8-bnb":
        from diffusers import PipelineQuantizationConfig
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        cfg = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersBnbConfig(load_in_8bit=True),
                "text_encoder": TransformersBnbConfig(load_in_8bit=True),
            }
        )
        return cfg, f"DiT=BnB8bit + TextEncoder=BnB8bit + VAE={profile.dtype.upper()}"

    return None, profile.dtype.upper()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log_buffer = LogRingBuffer(max_lines=400)
logging.getLogger().addHandler(log_buffer)
for _uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uvicorn_logger_name).addHandler(log_buffer)
logger = logging.getLogger("qwen_easycolab")

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", str(BASE_DIR / "outputs"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_INDEX_PATH = OUTPUT_DIR / "gallery.json"

DEFAULT_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo")
DEFAULT_QUANT_MODE = os.environ.get("QWEN_QUANT_MODE", "int8-torchao")
DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}
PORT = int(os.environ.get("PORT", "7860"))

DEFAULT_AI_ENDPOINT = os.environ.get("QWEN_AI_ENDPOINT", "https://api.pie-xian.com/v1/chat/completions")
DEFAULT_AI_API_KEY = os.environ.get("QWEN_AI_API_KEY", "")
DEFAULT_AI_MODEL = os.environ.get("QWEN_AI_MODEL", "qwen3.8-27b")
AI_CONFIG_PATH = BASE_DIR / ".ai_config.json"

MODEL_PRESETS = {
    "Qwen/Qwen-Image-2.1-Turbo": {
        "label": "Qwen-Image-2.1-Turbo (INT8 极速版)",
        "default_steps": 8,
        "default_cfg": 1.0,
        "description": "8步极速出图，适合T4/L4移动端高频交互",
    },
    "Qwen/Qwen-Image-2.1": {
        "label": "Qwen-Image-2.1 (满血基础版 · 40步精绘)",
        "default_steps": 40,
        "default_cfg": 1.0,
        "description": "满血高精细画质与文字海报渲染，支持40步精绘（可搭配不量化 BF16）",
    },
    "addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit": {
        "label": "Qwen-Image-2.1-Turbo 预量化 8bit",
        "default_steps": 8,
        "default_cfg": 1.0,
        "description": "预量化 TorchAO 8bit 权重，加载更快、显存更低",
    },
    "abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q8": {
        "label": "Qwen-Image-2.1 Uncensored GGUF Q8 满血版 (BF16 编码器 + BF16 VAE)",
        "default_steps": 8,
        "default_cfg": 1.0,
        "quant_mode": "gguf-q8-full",
        "description": "Q8_0 扩散权重直接拉满，搭配全精度 BF16 文本编码器与 qwen_image_2.1_vae_bf16",
    },
    "abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q4": {
        "label": "Qwen-Image-2.1 Uncensored GGUF Q4 轻量版 (Qwen3-VL-8B Q4_K_M + BF16 VAE)",
        "default_steps": 8,
        "default_cfg": 1.0,
        "quant_mode": "gguf-q4",
        "description": "Q4_K_M 扩散权重 + Qwen3-VL-8B Q4_K_M.gguf 文本编码器 + qwen_image_2.1_vae_bf16",
    },
}


def resolve_gguf_variant(model_id: str, quant_mode: str) -> Optional[str]:
    """Return `'q8'` or `'q4'` if the configured model/quant targets a GGUF variant, else `None`."""
    base_id, suffix = strip_model_variant_suffix(model_id)
    base_low = base_id.lower()
    qm = (quant_mode or "").strip().lower()
    is_gguf_repo = base_low in {
        GGUF_BASE_REPO.lower(),
        GGUF_REDIRECTED_REPO.lower(),
        "m9ter1a/qwen-image-2.1-turbo-gguf",
    } or "gguf" in base_low
    is_gguf_quant = qm in {"gguf-q8-full", "gguf-q4", "q8", "q4"}

    if not is_gguf_repo and not is_gguf_quant:
        return None
    if suffix in {"q8", "q4"}:
        return suffix
    if qm in {"gguf-q4", "q4"}:
        return "q4"
    if qm in {"gguf-q8-full", "q8", "none"}:
        return "q8"
    return "q8"



def align_dim(value: int, multiple: int = 16, min_val: int = 256, max_val: int = 3072) -> int:
    """Clamp and align dimension to a multiple required by diffusion VAEs."""
    clamped = max(min_val, min(max_val, int(value)))
    aligned = (clamped // multiple) * multiple
    return max(min_val, aligned)


_AT_REF_RE = re.compile(r"(?<![\w.])@(\d+(?:\s*[,，、]\s*\d+)*)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def resolve_multi_image_prompt(prompt: str, total_images: int) -> str:
    """
    Convert user `@0` (primary) and `@1,2` / `@1，2` (secondary) tags into explicit
    1-based image references (`图1`, `图2` or `Picture 1`, `Picture 2`) aligned with `image=[...]`.
    """
    if total_images <= 0 or "@" not in prompt:
        return prompt
    use_cjk = bool(_CJK_RE.search(prompt))

    def _repl(m: re.Match[str]) -> str:
        raw_nums = [int(x.strip()) for x in re.split(r"[,，、]", m.group(1)) if x.strip().isdigit()]
        valid = [n for n in raw_nums if 0 <= n < total_images]
        if not valid:
            return m.group(0)
        if use_cjk:
            return "、".join(f"图{n + 1}" for n in valid)
        labels = [f"Picture {n + 1}" for n in valid]
        return " and ".join(labels) if len(labels) <= 2 else ", ".join(labels[:-1]) + f", and {labels[-1]}"

    return _AT_REF_RE.sub(_repl, prompt)


def resize_preserve_aspect(img: Image.Image, max_edge: int = 1280) -> Image.Image:
    """Resize secondary reference image proportionally while aligning to 16px."""
    w, h = img.size
    if max(w, h) <= max_edge:
        return img.resize((align_dim(w, 16, 128, max_edge), align_dim(h, 16, 128, max_edge)), Image.Resampling.LANCZOS)
    scale = max_edge / max(w, h)
    nw = align_dim(int(round(w * scale)), 16, 128, max_edge)
    nh = align_dim(int(round(h * scale)), 16, 128, max_edge)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


SUPPORTED_ASPECT_RATIOS: list[tuple[str, int, int]] = [
    ("1:1", 1, 1),
    ("16:9", 16, 9),
    ("9:16", 9, 16),
    ("4:3", 4, 3),
    ("3:4", 3, 4),
    ("3:2", 3, 2),
    ("2:3", 2, 3),
]


def pick_closest_supported_ratio(w: int, h: int) -> tuple[str, int, int, float]:
    """Return `(label, rw, rh, log_diff)` for the closest Qwen-Image supported aspect ratio."""
    orig_ratio = max(1, w) / max(1, h)
    best: Optional[tuple[str, int, int, float]] = None
    for label, rw, rh in SUPPORTED_ASPECT_RATIOS:
        diff = abs(math.log((rw / rh) / orig_ratio))
        if best is None or diff < best[3]:
            best = (label, rw, rh, diff)
    assert best is not None
    return best


def auto_pad_transparent_to_supported(
    image: Image.Image,
    mask: Optional[Image.Image] = None,
    req_w: int = 1024,
    req_h: int = 1024,
) -> tuple[Image.Image, Optional[Image.Image], int, int, Optional[dict[str, Any]], Optional[str]]:
    """
    When the input image's aspect ratio is not one of the supported ratios (or differs
    from the requested target aspect ratio), pad it using a transparent RGBA canvas
    to a supported ratio, align target dimensions to 16px, and return crop metadata
    plus a user-facing notice so the generated output can be cropped back to original ratio.
    """
    orig_w, orig_h = max(1, image.width), max(1, image.height)
    orig_ratio = orig_w / orig_h
    req_ratio = max(1, req_w) / max(1, req_h)

    closest_label, rw, rh, orig_diff = pick_closest_supported_ratio(orig_w, orig_h)
    req_closest_label, req_rw, req_rh, req_diff = pick_closest_supported_ratio(req_w, req_h)

    # Check if original image already matches a supported ratio AND matches requested ratio
    ratio_mismatch = abs(math.log(req_ratio / orig_ratio)) > 0.025
    unsupported_orig = orig_diff > 0.025

    if not unsupported_orig and not ratio_mismatch:
        return image, mask, align_dim(req_w, 16), align_dim(req_h, 16), None, None

    # Use requested supported ratio if user explicitly picked one, otherwise closest supported ratio to original
    if not unsupported_orig and ratio_mismatch and req_diff <= 0.025:
        target_label, trw, trh = req_closest_label, req_rw, req_rh
    else:
        target_label, trw, trh = closest_label, rw, rh

    target_ratio = trw / trh
    if orig_ratio < target_ratio:
        pad_h = orig_h
        pad_w = max(orig_w, int(round(orig_h * target_ratio)))
    else:
        pad_w = orig_w
        pad_h = max(orig_h, int(round(orig_w / target_ratio)))

    off_x = max(0, (pad_w - orig_w) // 2)
    off_y = max(0, (pad_h - orig_h) // 2)

    # Transparent RGBA expansion canvas
    padded_img = Image.new("RGBA", (pad_w, pad_h), (0, 0, 0, 0))
    src_rgba = image.convert("RGBA")
    padded_img.paste(src_rgba, (off_x, off_y), src_rgba)

    padded_mask: Optional[Image.Image] = None
    if mask is not None:
        m_resized = mask if mask.size == (orig_w, orig_h) else mask.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        padded_mask = Image.new("L", (pad_w, pad_h), 0)
        padded_mask.paste(m_resized.convert("L"), (off_x, off_y))

    area = max(256 * 256, req_w * req_h)
    scale = math.sqrt(area / max(1, pad_w * pad_h))
    aligned_w = align_dim(int(round(pad_w * scale)), 16)
    aligned_h = align_dim(int(round(pad_h * scale)), 16)

    crop_info = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "off_x": off_x,
        "off_y": off_y,
        "ratio_label": target_label,
    }
    notice = (
        f"原图比例 ({orig_w}×{orig_h}) 与当前输出比例不匹配，已自动使用透明图层扩图至支持比例 "
        f"{target_label} ({aligned_w}×{aligned_h})，输出图片将自动裁剪还原为原比例。"
    )
    return padded_img, padded_mask, aligned_w, aligned_h, crop_info, notice


def crop_back_to_original_ratio(img: Image.Image, crop_info: Optional[dict[str, Any]]) -> Image.Image:
    """Crop a padded generation result back to the original unpadded image aspect ratio."""
    if not crop_info:
        return img
    rw, rh = img.size
    pad_w = max(1, int(crop_info["pad_w"]))
    pad_h = max(1, int(crop_info["pad_h"]))
    sx = rw / pad_w
    sy = rh / pad_h
    x0 = max(0, int(round(crop_info["off_x"] * sx)))
    y0 = max(0, int(round(crop_info["off_y"] * sy)))
    cw = max(16, int(round(crop_info["orig_w"] * sx)))
    ch = max(16, int(round(crop_info["orig_h"] * sy)))
    x1 = min(rw, x0 + cw)
    y1 = min(rh, y0 + ch)
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def get_dir_size_bytes(path: Path) -> int:
    """Safely compute total bytes of all files under a directory."""
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += (Path(root) / fname).stat().st_size
            except OSError:
                pass
    return total


def decode_data_url_image(data_url: str) -> Image.Image:
    """Decode a base64 data URL or raw base64 string into an RGB PIL Image."""
    if "," in data_url and data_url.startswith("data:"):
        _, b64_part = data_url.split(",", 1)
    else:
        b64_part = data_url
    raw = base64.b64decode(b64_part)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_data_url_mask(data_url: str) -> Image.Image:
    """Decode a mask data URL into an L-mode grayscale PIL Image (white=inpaint)."""
    if "," in data_url and data_url.startswith("data:"):
        _, b64_part = data_url.split(",", 1)
    else:
        b64_part = data_url
    raw = base64.b64decode(b64_part)
    img = Image.open(io.BytesIO(raw))
    if img.mode == "RGBA":
        # Support either painted alpha channel or white-on-black mask
        r, g, b, a = img.split()
        extrema = a.getextrema()
        if extrema[0] < 255:
            return a
        return Image.merge("RGB", (r, g, b)).convert("L")
    return img.convert("L")


# ---------------------------------------------------------------------------
# Gallery Persistence
# ---------------------------------------------------------------------------


@dataclass
class GalleryItem:
    id: str
    filename: str
    url: str
    mode: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    denoise: float
    true_cfg_scale: float
    seed: int
    model_id: str
    quant_mode: str
    elapsed_sec: float
    created_at: str
    size_bytes: int = 0
    hires_scale: float = 1.0
    ref_count: int = 0


class GalleryManager:
    """Thread-safe local gallery manager stored on disk + JSON index."""

    def __init__(self, output_dir: Path, index_path: Path) -> None:
        self.output_dir = output_dir
        self.index_path = index_path
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self.index_path.exists():
                try:
                    data = json.loads(self.index_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        # Filter out records whose image files were deleted externally
                        self._items = [
                            item
                            for item in data
                            if (self.output_dir / item.get("filename", "")).exists()
                        ]
                        return
                except Exception as exc:
                    logger.warning("Failed to load gallery index: %s", exc)
            self._items = []
            self._save_locked()

    def _save_locked(self) -> None:
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.index_path)

    def add_image(self, image: Image.Image, meta: dict[str, Any]) -> dict[str, Any]:
        item_id = meta.get("id") or uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_{meta['mode']}_{timestamp}_{item_id}.png"
        file_path = self.output_dir / filename
        image.save(file_path, format="PNG", optimize=True)
        size_bytes = file_path.stat().st_size

        item = GalleryItem(
            id=item_id,
            filename=filename,
            url=f"/outputs/{filename}",
            mode=meta["mode"],
            prompt=meta["prompt"],
            negative_prompt=meta.get("negative_prompt", ""),
            width=image.width,
            height=image.height,
            steps=meta["steps"],
            denoise=meta["denoise"],
            true_cfg_scale=meta["true_cfg_scale"],
            seed=meta["seed"],
            model_id=meta["model_id"],
            quant_mode=meta["quant_mode"],
            elapsed_sec=round(meta["elapsed_sec"], 2),
            created_at=datetime.now().isoformat(timespec="seconds"),
            size_bytes=size_bytes,
            hires_scale=meta.get("hires_scale", 1.0),
            ref_count=meta.get("ref_count", 0),
        )
        record = asdict(item)
        with self._lock:
            self._items.insert(0, record)
            self._save_locked()
        return record

    def list_items(
        self,
        mode: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            filtered = self._items
            if mode and mode != "all":
                filtered = [x for x in filtered if x.get("mode") == mode]
            if query:
                q = query.strip().lower()
                filtered = [
                    x
                    for x in filtered
                    if q in x.get("prompt", "").lower() or q in str(x.get("seed", ""))
                ]
            total = len(filtered)
            total_bytes = sum(x.get("size_bytes", 0) for x in self._items)
            return {
                "total": total,
                "total_all": len(self._items),
                "total_bytes": total_bytes,
                "items": filtered[offset : offset + limit],
            }

    def get_item(self, item_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            for item in self._items:
                if item.get("id") == item_id:
                    return dict(item)
        return None

    def delete_items(self, item_ids: list[str]) -> int:
        id_set = set(item_ids)
        removed = 0
        with self._lock:
            remaining = []
            for item in self._items:
                if item.get("id") in id_set:
                    file_path = self.output_dir / item.get("filename", "")
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
                    removed += 1
                else:
                    remaining.append(item)
            self._items = remaining
            self._save_locked()
        return removed


# ---------------------------------------------------------------------------
# Generation Job State & Engine
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    job_id: str
    status: Literal["queued", "loading_model", "running", "completed", "cancelled", "error"]
    mode: str
    batch_size: int
    current_batch: int = 0
    current_step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    message: str = "等待中..."
    notice: Optional[str] = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "mode": self.mode,
            "batch_size": self.batch_size,
            "current_batch": self.current_batch,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": round(self.progress, 4),
            "message": self.message,
            "notice": self.notice,
            "results": self.results,
            "error": self.error,
            "elapsed_sec": round((self.finished_at or time.time()) - self.created_at, 2),
        }


class QwenImage21Engine:
    """Manages Qwen-Image-2.1 INT8 pipeline lifecycle across 5 Colab GPU tiers."""

    def __init__(self) -> None:
        self.model_id: str = DEFAULT_MODEL_ID
        self.quant_mode: str = DEFAULT_QUANT_MODE
        self.gpu_profile_override: str = os.environ.get("QWEN_GPU_PROFILE", "auto")
        self.dtype_mode: str = os.environ.get("QWEN_DTYPE", "auto")
        self.offload_mode: str = os.environ.get("QWEN_OFFLOAD", "auto")
        self.vae_tiling_mode: str = os.environ.get("QWEN_VAE_TILING", "auto")
        self.hf_token: str = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        self.ai_endpoint: str = DEFAULT_AI_ENDPOINT
        self.ai_api_key: str = DEFAULT_AI_API_KEY
        self.ai_model: str = DEFAULT_AI_MODEL
        self._load_ai_config()
        self.demo_mode: bool = DEMO_MODE
        self.pipe: Any = None
        self.device: str = "cpu"
        self.loaded: bool = False
        self.loading: bool = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._pipe_supports_strength: Optional[bool] = None
        self._pipe_supports_mask: Optional[bool] = None
        self._active_lora_path: Optional[str] = None
        self._active_lora_weight: Optional[float] = None

    def _load_ai_config(self) -> None:
        if AI_CONFIG_PATH.exists():
            try:
                data = json.loads(AI_CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if data.get("ai_endpoint"):
                        self.ai_endpoint = str(data["ai_endpoint"]).strip()
                    if data.get("ai_api_key"):
                        self.ai_api_key = str(data["ai_api_key"]).strip()
                    if data.get("ai_model"):
                        self.ai_model = str(data["ai_model"]).strip()
            except Exception as exc:
                logger.warning("Failed to load AI config: %s", exc)

    def _save_ai_config(self) -> None:
        try:
            AI_CONFIG_PATH.write_text(
                json.dumps(
                    {
                        "ai_endpoint": self.ai_endpoint,
                        "ai_api_key": self.ai_api_key,
                        "ai_model": self.ai_model,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist AI config: %s", exc)

    def _detect_torch(self) -> tuple[Any, bool]:
        try:
            import torch

            return torch, torch.cuda.is_available()
        except ImportError:
            return None, False

    def resolve_gpu_profile(self) -> GPUProfile:
        """Resolve active GPU optimization profile (manual override or hardware auto-match)."""
        if self.gpu_profile_override != "auto" and self.gpu_profile_override in GPU_PROFILES:
            return GPU_PROFILES[self.gpu_profile_override]

        torch, has_cuda = self._detect_torch()
        if torch is not None and has_cuda:
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024**3)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            return match_gpu_profile(props.name, total_vram_gb, cc_major, cc_minor)

        return GPU_PROFILES["t4_turing"]

    def effective_settings(self) -> dict[str, Any]:
        profile = self.resolve_gpu_profile()
        eff_dtype = profile.dtype if self.dtype_mode == "auto" else self.dtype_mode
        gguf_var = resolve_gguf_variant(self.model_id, self.quant_mode)
        if self.offload_mode == "auto":
            # Unquantized (full-precision) weights exceed 16-24GB VRAM, so auto-upgrade T4/L4 to sequential offload
            if self.quant_mode == "none" and profile.vram_gb <= 24:
                eff_offload = "sequential_cpu_offload"
            elif gguf_var == "q8" and profile.vram_gb <= 16:
                eff_offload = "sequential_cpu_offload"
            elif gguf_var == "q8" and profile.vram_gb <= 24:
                eff_offload = "smart_stage_offload"
            elif gguf_var == "q4" and profile.vram_gb <= 16:
                eff_offload = "model_cpu_offload"
            else:
                eff_offload = profile.offload_mode
        else:
            eff_offload = self.offload_mode
        if self.vae_tiling_mode == "auto":
            eff_vae_tiling = profile.vae_tiling
        else:
            eff_vae_tiling = self.vae_tiling_mode == "on"
        return {
            "profile": profile,
            "dtype": eff_dtype,
            "offload_mode": eff_offload,
            "vae_tiling": eff_vae_tiling,
            "vae_slicing": eff_vae_tiling or profile.vae_slicing,
            "attention_slicing": profile.attention_slicing,
            "allow_tf32": profile.allow_tf32,
            "gguf_variant": gguf_var,
        }

    def _load_gguf_pipeline(
        self,
        gguf_variant: str,
        eff: dict[str, Any],
        dtype: Any,
        job: Optional[JobState],
        dl_progress: Any,
    ) -> Any:
        import torch
        from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration

        profile: GPUProfile = eff["profile"]
        token = self.hf_token or None
        cfg_variant = GGUF_VARIANTS[gguf_variant]
        base_repo = cfg_variant["base_config_repo"]

        bundle = download_gguf_bundle_fast(
            variant=gguf_variant,
            token=token,
            progress_cb=dl_progress,
            use_tqdm=False,
            include_diffusers_configs=True,
            include_mmproj=False,
        )
        if job and job.cancel_event.is_set():
            raise InterruptedError("用户已取消生成")

        base_cfg_dir = bundle["base_config_dir"]
        if job:
            job.status = "loading_model"
            job.progress = max(job.progress, 0.32)
            job.message = f"[{profile.display_name}] 正在加载 GGUF {gguf_variant.upper()} 扩散权重..."

        # 1) Load GGUF Diffusion Transformer (try user-specified abenzerps GGUF first, fallback to Diffusers-layout GGUF)
        transformer = None
        try:
            from diffusers import GGUFQuantizationConfig

            def _load_dit_single(file_path: str) -> Any:
                try:
                    return QwenImage21Transformer2DModel.from_single_file(
                        file_path,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                        torch_dtype=dtype,
                        config=base_cfg_dir,
                        subfolder="transformer",
                        local_files_only=True,
                    )
                except TypeError:
                    return QwenImage21Transformer2DModel.from_single_file(
                        file_path,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                        dtype=dtype,
                        config=base_cfg_dir,
                        subfolder="transformer",
                        local_files_only=True,
                    )

            try:
                transformer = _load_dit_single(bundle["dit_path"])
                logger.info("Loaded DiT directly from %s", bundle["dit_filename"])
            except Exception as primary_dit_exc:
                logger.warning(
                    "Direct Diffusers loading of %s failed (%s); falling back to Diffusers-formatted GGUF %s...",
                    bundle["dit_filename"],
                    primary_dit_exc,
                    cfg_variant["dit_diffusers_fallback_repo"],
                )
                fb_repo, fb_file, fb_size = resolve_repo_file(
                    cfg_variant["dit_diffusers_fallback_repo"],
                    cfg_variant["dit_diffusers_fallback_patterns"],
                    token=token,
                )
                fb_path = download_hf_file_fast(
                    fb_repo,
                    fb_file,
                    token=token,
                    expected_bytes=fb_size,
                    progress_cb=dl_progress,
                    use_tqdm=False,
                )
                transformer = _load_dit_single(fb_path)
                logger.info("Loaded Diffusers-compatible GGUF DiT from %s/%s", fb_repo, fb_file)
        except Exception as gguf_dit_exc:
            logger.warning("GGUF DiT loading failed (%s); falling back to %s transformer.", gguf_dit_exc, base_repo)
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                base_repo,
                subfolder="transformer",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                token=token,
            )

        # 2) Load Text Encoder: Q8 -> Full BF16 Qwen3-VL-8B ("直接拉满"); Q4 -> Qwen3-VL-8B Q4_K_M.gguf (w/ 4-bit fallback)
        if job:
            job.progress = max(job.progress, 0.55)
            job.message = (
                f"[{profile.display_name}] 正在加载全精度 BF16 Qwen3-VL 文本编码器..."
                if gguf_variant == "q8"
                else f"[{profile.display_name}] 正在加载 Qwen3-VL-8B Q4_K_M 文本编码器..."
            )

        text_encoder = None
        te_cfg = AutoConfig.from_pretrained(str(Path(base_cfg_dir) / "text_encoder"), local_files_only=True)
        if gguf_variant == "q8":
            try:
                from safetensors.torch import load_file as load_safetensors_file

                raw_sd = load_safetensors_file(bundle["te_path"], device="cpu")
                remapped_sd: dict[str, Any] = {}
                for k, v in raw_sd.items():
                    nk = k
                    if k.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")):
                        nk = "model.language_model." + k[len("model.") :]
                    elif k.startswith("visual."):
                        nk = "model." + k
                    remapped_sd[nk] = v
                del raw_sd
                if any(k.startswith("model.language_model.") for k in remapped_sd):
                    te_cand, load_info = Qwen3VLForConditionalGeneration.from_pretrained(
                        None,
                        config=te_cfg,
                        state_dict=remapped_sd,
                        torch_dtype=torch.bfloat16 if dtype == torch.bfloat16 else dtype,
                        low_cpu_mem_usage=True,
                        output_loading_info=True,
                    )
                    missing = [m for m in load_info.get("missing_keys", []) if "lm_head" not in m]
                    if not missing:
                        text_encoder = te_cand
                        logger.info("Loaded BF16 Qwen3-VL text encoder directly from %s", bundle["te_filename"])
                    else:
                        del te_cand
                del remapped_sd
            except Exception as te_bf16_exc:
                logger.info("Single-file BF16 TE load skipped (%s); loading HF BF16 shards from %s.", te_bf16_exc, base_repo)

            if text_encoder is None:
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    base_repo,
                    subfolder="text_encoder",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    token=token,
                )
        else:
            try:
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    bundle["te_repo"],
                    gguf_file=Path(bundle["te_path"]).name,
                    config=te_cfg,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    token=token,
                )
                logger.info("Loaded Qwen3-VL Q4_K_M GGUF text encoder from %s", bundle["te_filename"])
            except Exception as te_gguf_exc:
                logger.warning(
                    "Transformers Qwen3-VL GGUF load unavailable (%s); falling back to 4-bit quantized %s text_encoder.",
                    te_gguf_exc,
                    base_repo,
                )
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                        base_repo,
                        subfolder="text_encoder",
                        quantization_config=bnb_cfg,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        token=token,
                    )
                except Exception as bnb_exc:
                    logger.warning("4-bit BnB fallback unavailable (%s); falling back to TorchAO INT8 text encoder.", bnb_exc)
                    from torchao.quantization import Int8WeightOnlyConfig
                    from transformers import TorchAoConfig

                    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                        base_repo,
                        subfolder="text_encoder",
                        quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        token=token,
                    )

        # 3) Load qwen_image_2.1_vae_bf16.safetensors
        if job:
            job.progress = max(job.progress, 0.78)
            job.message = f"[{profile.display_name}] 正在加载 qwen_image_2.1_vae_bf16 VAE..."

        vae = None
        try:
            vae = AutoencoderKLQwenImage21.from_single_file(
                bundle["vae_path"],
                config=base_cfg_dir,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            logger.info("Loaded BF16 VAE from %s", bundle["vae_filename"])
        except Exception as vae_exc:
            logger.warning("Single-file VAE load failed (%s); falling back to %s vae.", vae_exc, base_repo)
            vae = AutoencoderKLQwenImage21.from_pretrained(
                base_repo,
                subfolder="vae",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                token=token,
            )

        try:
            return QwenImage21Pipeline.from_pretrained(
                base_cfg_dir,
                transformer=transformer,
                text_encoder=text_encoder,
                vae=vae,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except TypeError:
            return QwenImage21Pipeline.from_pretrained(
                base_cfg_dir,
                transformer=transformer,
                text_encoder=text_encoder,
                vae=vae,
                dtype=dtype,
                local_files_only=True,
            )

    def ensure_loaded(self, job: Optional[JobState] = None) -> None:
        if self.demo_mode:
            self.loaded = True
            self.device = f"demo ({self.resolve_gpu_profile().key})"
            return

        with self._load_lock:
            if self.loaded and self.pipe is not None:
                return

            torch, has_cuda = self._detect_torch()
            if torch is None or not has_cuda:
                raise RuntimeError(
                    "未检测到可用 CUDA GPU。请在 Colab 中切换为 G4/H100/A100/L4/T4 GPU 运行时，"
                    "或设置 DEMO_MODE=1 进行本地界面预览。"
                )

            eff = self.effective_settings()
            profile: GPUProfile = eff["profile"]
            gguf_variant: Optional[str] = eff.get("gguf_variant")
            self.loading = True
            if job:
                job.status = "loading_model"
                job.progress = 0.06
                job.message = f"[{profile.display_name}] 正在下载/加载 {self.model_id} ({self.quant_mode})..."

            try:
                if job and job.cancel_event.is_set():
                    raise InterruptedError("用户已取消生成")
                from diffusers import QwenImage21Pipeline

                def _dl_progress(cur_b: int, tot_b: int, _spd: float, msg: str) -> None:
                    if job:
                        job.status = "loading_model"
                        job.message = msg
                        if tot_b > 0:
                            job.progress = min(0.28, max(0.02, 0.02 + 0.26 * (cur_b / tot_b)))

                # Apply hardware-specific TensorCore / TF32 settings
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = eff["allow_tf32"]
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = eff["allow_tf32"]
                    torch.backends.cudnn.benchmark = True

                dtype = torch.bfloat16 if eff["dtype"] == "bfloat16" else torch.float16
                logger.info(
                    "GPU Profile=%s | dtype=%s | offload=%s | vae_tiling=%s | tf32=%s | gguf=%s",
                    profile.display_name,
                    eff["dtype"],
                    eff["offload_mode"],
                    eff["vae_tiling"],
                    eff["allow_tf32"],
                    gguf_variant,
                )

                if gguf_variant is not None:
                    pipe = self._load_gguf_pipeline(gguf_variant, eff, dtype, job, _dl_progress)
                else:
                    local_model_dir = download_model_fast(
                        repo_id=self.model_id,
                        token=self.hf_token or None,
                        progress_cb=_dl_progress,
                        use_tqdm=False,
                    )

                    if job and job.cancel_event.is_set():
                        raise InterruptedError("用户已取消生成")

                    if job:
                        job.status = "loading_model"
                        job.progress = max(job.progress, 0.30)
                        job.message = f"[{profile.display_name}] 正在载入模型权重 ({self.quant_mode})..."

                    load_kwargs: dict[str, Any] = {
                        "torch_dtype": dtype,
                        "low_cpu_mem_usage": True,
                    }

                    # Apply component-wise quantization (quantize BOTH DiT & Qwen3-VL text_encoder; keep VAE in BF16/FP16)
                    is_prequant = "8bit" in self.model_id.lower() or "int8" in self.model_id.lower() or "fp8" in self.model_id.lower()
                    if not is_prequant and self.quant_mode != "none":
                        try:
                            q_cfg, q_desc = build_pipeline_quant_config(self.quant_mode, profile)
                            if q_cfg is not None:
                                load_kwargs["quantization_config"] = q_cfg
                                logger.info("Using PipelineQuantizationConfig: %s", q_desc)
                                if job:
                                    job.message = f"[{profile.display_name}] 量化配置: {q_desc}..."
                        except Exception as q_exc:
                            logger.warning(
                                "PipelineQuantizationConfig failed (%s); falling back to Diffusers TorchAoConfig('int8wo').",
                                q_exc,
                            )
                            if self.quant_mode in {"fp8-int8", "fp8-all", "int8-torchao"}:
                                try:
                                    try:
                                        from diffusers import TorchAoConfig
                                    except ImportError:
                                        from diffusers.quantizers import TorchAoConfig  # type: ignore

                                    load_kwargs["quantization_config"] = TorchAoConfig("int8wo")
                                except Exception as inner_exc:
                                    logger.warning("TorchAoConfig fallback unavailable (%s).", inner_exc)

                    def _load_pipe_once(extra_kwargs: dict[str, Any]) -> Any:
                        try:
                            return QwenImage21Pipeline.from_pretrained(local_model_dir, local_files_only=True, **extra_kwargs)
                        except TypeError:
                            alt_kwargs = dict(extra_kwargs)
                            alt_kwargs.pop("torch_dtype", None)
                            alt_kwargs["dtype"] = dtype
                            return QwenImage21Pipeline.from_pretrained(local_model_dir, local_files_only=True, **alt_kwargs)

                    try:
                        pipe = _load_pipe_once(load_kwargs)
                    except Exception as load_exc:
                        err_text = str(load_exc)
                        if "Int8WeightOnlyConfig" in err_text or "Version mismatch" in err_text:
                            logger.warning("Detected torchao Int8WeightOnlyConfig version mismatch (%s); attempting version=2 fallback.", err_text)
                            recovered = False
                            try:
                                from torchao.quantization import Int8WeightOnlyConfig

                                try:
                                    from diffusers import PipelineQuantizationConfig, TorchAoConfig
                                except ImportError:
                                    from diffusers.quantizers import TorchAoConfig  # type: ignore
                                    PipelineQuantizationConfig = None  # type: ignore

                                retry_kwargs = dict(load_kwargs)
                                if PipelineQuantizationConfig is not None:
                                    retry_kwargs["quantization_config"] = PipelineQuantizationConfig(
                                        quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2))}
                                    )
                                else:
                                    retry_kwargs["quantization_config"] = TorchAoConfig(Int8WeightOnlyConfig(version=2))
                                pipe = _load_pipe_once(retry_kwargs)
                                recovered = True
                            except Exception as retry_exc:
                                logger.error("torchao version=2 fallback failed: %s", retry_exc)
                            if not recovered:
                                raise RuntimeError(
                                    f"TorchAO 量化版本不兼容 ({err_text})。"
                                    "请先在 Colab 重新运行第 1 步（或执行 `pip install -U torchao` 升级 torchao），"
                                    "或在设置中将模型切换为 `Qwen/Qwen-Image-2.1-Turbo`（使用运行时 INT8 量化）。"
                                ) from load_exc
                        else:
                            raise

                if hasattr(pipe, "vae"):
                    if eff["vae_tiling"] and hasattr(pipe.vae, "enable_tiling"):
                        pipe.vae.enable_tiling()
                    if eff["vae_slicing"] and hasattr(pipe.vae, "enable_slicing"):
                        pipe.vae.enable_slicing()

                if eff["attention_slicing"] and hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing("auto")

                if eff["offload_mode"] == "sequential_cpu_offload":
                    pipe.enable_sequential_cpu_offload()
                    self.device = "cuda (sequential_offload)"
                elif eff["offload_mode"] == "smart_stage_offload":
                    pipe.enable_model_cpu_offload()
                    self.device = "cuda (smart_stage_offload)"
                elif eff["offload_mode"] == "model_cpu_offload":
                    pipe.enable_model_cpu_offload()
                    self.device = "cuda (model_offload)"
                else:
                    pipe = pipe.to("cuda")
                    self.device = "cuda (full)"

                trim_host_memory()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sig = inspect.signature(pipe.__call__)
                self._pipe_supports_strength = "strength" in sig.parameters
                self._pipe_supports_mask = "mask_image" in sig.parameters

                self.pipe = pipe
                self.loaded = True
                if job:
                    job.status = "running"
                    job.progress = 0.0
                    job.message = f"[{profile.display_name}] 模型已驻留，开始推理..."
                logger.info("Pipeline loaded successfully on %s", self.device)
            finally:
                self.loading = False

    def _unload_active_lora(self) -> None:
        if self.pipe is not None and self._active_lora_path is not None:
            try:
                if hasattr(self.pipe, "unload_lora_weights"):
                    self.pipe.unload_lora_weights()
            except Exception as exc:
                logger.warning("unload_lora_weights warning: %s", exc)
        self._active_lora_path = None
        self._active_lora_weight = None
        trim_host_memory()
        torch, has_cuda = self._detect_torch()
        if torch is not None and has_cuda:
            torch.cuda.empty_cache()

    def sync_lora(self, lora_enabled: bool, lora_name: Optional[str], lora_weight: float) -> None:
        """Hot-load, remap keys with strict=False, adjust weight, or cleanly unload LoRA."""
        if self.pipe is None:
            return
        if not lora_enabled or not lora_name:
            if self._active_lora_path is not None:
                logger.info("Disabling LoRA and freeing weights/cache...")
                self._unload_active_lora()
            return

        target_path = (LORAS_DIR / Path(lora_name).name).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"未找到 LoRA 文件: {target_path}")

        target_str = str(target_path)
        if self._active_lora_path != target_str:
            self._unload_active_lora()
            state_dict, info = normalize_lora_state_dict(target_path)
            logger.info(
                "Loading LoRA %s (keys=%d, stripped=%s, weight=%.2f)",
                target_path.name,
                info["mapped_keys"],
                info["stripped_prefixes"],
                lora_weight,
            )
            try:
                sig = inspect.signature(self.pipe.load_lora_weights)
                load_kw: dict[str, Any] = {}
                if "adapter_name" in sig.parameters:
                    load_kw["adapter_name"] = "user_lora"
                if "strict" in sig.parameters:
                    load_kw["strict"] = False
                if "low_cpu_mem_usage" in sig.parameters:
                    load_kw["low_cpu_mem_usage"] = True
                try:
                    self.pipe.load_lora_weights(state_dict, **load_kw)
                except TypeError:
                    load_kw.pop("strict", None)
                    self.pipe.load_lora_weights(state_dict, **load_kw)
                if hasattr(self.pipe, "set_adapters"):
                    self.pipe.set_adapters(["user_lora"], adapter_weights=[float(lora_weight)])
                self._active_lora_path = target_str
                self._active_lora_weight = float(lora_weight)
            finally:
                del state_dict
                trim_host_memory()
                torch, has_cuda = self._detect_torch()
                if torch is not None and has_cuda:
                    torch.cuda.empty_cache()
        elif abs((self._active_lora_weight if self._active_lora_weight is not None else 0.85) - float(lora_weight)) > 1e-4:
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["user_lora"], adapter_weights=[float(lora_weight)])
            self._active_lora_weight = float(lora_weight)

    def unload(self) -> None:
        with self._load_lock:
            self._active_lora_path = None
            self._active_lora_weight = None
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            self.loaded = False
            trim_host_memory()
            torch, has_cuda = self._detect_torch()
            if torch is not None and has_cuda:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def reconfigure(
        self,
        model_id: Optional[str] = None,
        quant_mode: Optional[str] = None,
        gpu_profile_override: Optional[str] = None,
        dtype_mode: Optional[str] = None,
        offload_mode: Optional[str] = None,
        vae_tiling_mode: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: Optional[bool] = None,
        vae_tiling: Optional[bool] = None,
        ai_endpoint: Optional[str] = None,
        ai_api_key: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> None:
        needs_reload = False
        ai_changed = False
        if ai_endpoint is not None and ai_endpoint.strip():
            self.ai_endpoint = ai_endpoint.strip()
            ai_changed = True
        if ai_api_key is not None and ai_api_key.strip():
            self.ai_api_key = ai_api_key.strip()
            ai_changed = True
        if ai_model is not None and ai_model.strip():
            self.ai_model = ai_model.strip()
            ai_changed = True
        if ai_changed:
            self._save_ai_config()

        if hf_token is not None and hf_token.strip():
            self.hf_token = hf_token.strip()
            os.environ["HF_TOKEN"] = self.hf_token
        if model_id and model_id != self.model_id:
            self.model_id = model_id
            needs_reload = True
        if quant_mode and quant_mode != self.quant_mode:
            self.quant_mode = quant_mode
            needs_reload = True
        if gpu_profile_override and gpu_profile_override != self.gpu_profile_override:
            self.gpu_profile_override = gpu_profile_override
            needs_reload = True
        if dtype_mode and dtype_mode != self.dtype_mode:
            self.dtype_mode = dtype_mode
            needs_reload = True
        if offload_mode and offload_mode != self.offload_mode:
            self.offload_mode = offload_mode
            needs_reload = True
        elif cpu_offload is not None:
            mapped = "model_cpu_offload" if cpu_offload else "none"
            if mapped != self.offload_mode:
                self.offload_mode = mapped
                needs_reload = True
        if vae_tiling_mode and vae_tiling_mode != self.vae_tiling_mode:
            self.vae_tiling_mode = vae_tiling_mode
            needs_reload = True
        elif vae_tiling is not None:
            mapped_t = "on" if vae_tiling else "off"
            if mapped_t != self.vae_tiling_mode:
                self.vae_tiling_mode = mapped_t
                needs_reload = True

        if needs_reload and self.loaded:
            self.unload()

    def _run_demo_image(
        self,
        mode: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        denoise: float,
        seed: int,
        init_image: Optional[Image.Image],
        mask_image: Optional[Image.Image],
        extra_ref_images: list[Image.Image],
        lora_summary: str,
        job: JobState,
        batch_idx: int,
    ) -> Image.Image:
        """Simulate realistic step progression and image transformations when DEMO_MODE=1."""
        rng = random.Random(seed)
        for step in range(1, steps + 1):
            if job.cancel_event.is_set():
                raise InterruptedError("用户已取消生成")
            time.sleep(0.05)
            job.current_step = step
            base_progress = batch_idx / job.batch_size
            step_progress = (step / max(1, steps)) / job.batch_size
            job.progress = min(0.99, base_progress + step_progress)
            job.message = f"[演示模式] 第 {batch_idx + 1}/{job.batch_size} 张 · Step {step}/{steps}"

        if init_image is not None:
            canvas = init_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).copy()
            if mode == "hires":
                canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.8, percent=140, threshold=2))
            elif mode == "inpaint" and mask_image is not None:
                m = mask_image.resize((width, height), Image.Resampling.BILINEAR).filter(
                    ImageFilter.GaussianBlur(radius=8)
                )
                overlay = Image.new("RGB", (width, height), (rng.randint(80, 220), rng.randint(60, 200), rng.randint(120, 255)))
                tint = Image.blend(canvas, overlay, alpha=max(0.2, min(0.85, denoise)))
                canvas = Image.composite(tint, canvas, m)
            else:
                tint = ImageEnhance.Color(canvas).enhance(1.0 + 0.35 * denoise)
                tint = ImageEnhance.Contrast(tint).enhance(1.0 + 0.2 * denoise)
                canvas = Image.blend(canvas, tint, alpha=denoise)
                if extra_ref_images:
                    thumb_sz = max(48, min(96, width // 6))
                    for idx_r, rimg in enumerate(extra_ref_images[:4]):
                        thumb = rimg.convert("RGB").resize((thumb_sz, thumb_sz), Image.Resampling.LANCZOS)
                        tx = 14 + idx_r * (thumb_sz + 8)
                        ty = 14
                        if tx + thumb_sz < width:
                            canvas.paste(thumb, (tx, ty))
                            d_t = ImageDraw.Draw(canvas)
                            d_t.rectangle([tx, ty, tx + thumb_sz, ty + thumb_sz], outline=(34, 211, 238), width=2)
                            d_t.text((tx + 4, ty + 4), f"@{idx_r + 1}", fill=(255, 255, 255))
        else:
            canvas = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(canvas)
            c1 = (rng.randint(25, 95), rng.randint(30, 110), rng.randint(90, 210))
            c2 = (rng.randint(110, 240), rng.randint(50, 160), rng.randint(130, 255))
            for y in range(height):
                t = y / max(1, height - 1)
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            for _ in range(7):
                cx = rng.randint(0, width)
                cy = rng.randint(0, height)
                rad = rng.randint(width // 10, width // 3)
                fill = (rng.randint(120, 255), rng.randint(100, 255), rng.randint(150, 255))
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=fill, width=4)

        draw = ImageDraw.Draw(canvas)
        refs_tag = f" | Refs:{1 + len(extra_ref_images)}" if extra_ref_images else ""
        lora_tag = f" | {lora_summary}" if lora_summary else ""
        badge = f"Qwen-Image-2.1 DEMO | {mode.upper()}{refs_tag}{lora_tag} | {width}x{height} | Steps:{steps} | Seed:{seed}"
        draw.rectangle([14, height - 46, min(width - 14, 14 + len(badge) * 7 + 16), height - 14], fill=(12, 16, 28))
        draw.text((22, height - 38), badge, fill=(230, 240, 255))
        return canvas

    def generate_one(
        self,
        mode: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        denoise: float,
        true_cfg_scale: float,
        seed: int,
        init_image: Optional[Image.Image],
        mask_image: Optional[Image.Image],
        extra_ref_images: Optional[list[Image.Image]],
        hires_scale: float,
        mask_blur: int,
        job: JobState,
        batch_idx: int,
        lora_enabled: bool = False,
        lora_name: Optional[str] = None,
        lora_weight: float = 0.85,
        lora_trigger: str = "",
        lora_action_mode: str = "standard",
        lora_control_image: Optional[Image.Image] = None,
        lora_control_strength: float = 0.85,
    ) -> Image.Image:
        width = align_dim(width, 16)
        height = align_dim(height, 16)
        refs_list = (extra_ref_images or [])[:9]
        total_refs = (1 if init_image is not None else 0) + len(refs_list)
        working_prompt = prompt
        if lora_enabled and lora_trigger.strip():
            trig = lora_trigger.strip()
            if trig.lower() not in working_prompt.lower():
                working_prompt = f"{trig}, {working_prompt}"
        resolved_prompt = (
            resolve_multi_image_prompt(working_prompt, total_refs) if mode == "i2i" and total_refs > 0 else working_prompt
        )

        lora_summary = ""
        if lora_enabled and lora_name:
            lora_summary = f"LoRA:{lora_name[:18]}({lora_weight:.2f})"

        using_lora_control = bool(lora_enabled and lora_action_mode == "control" and lora_control_image is not None)

        if self.demo_mode:
            if lora_enabled:
                logger.info(
                    "[演示模式] LoRA 模拟参数: file=%s | weight=%.2f | trigger=%r | mode=%s | ctrl_strength=%.2f | has_ctrl_img=%s",
                    lora_name,
                    lora_weight,
                    lora_trigger,
                    lora_action_mode,
                    lora_control_strength,
                    lora_control_image is not None,
                )
            demo_init = init_image if init_image is not None else (
                lora_control_image if (lora_enabled and lora_action_mode == "control") else None
            )
            demo_mask = mask_image
            crop_info: Optional[dict[str, Any]] = None
            target_w = align_dim(int(width * hires_scale), 16) if mode == "hires" and demo_init else width
            target_h = align_dim(int(height * hires_scale), 16) if mode == "hires" and demo_init else height
            if demo_init is not None:
                req_base_w = align_dim(int(demo_init.width * hires_scale), 16) if mode == "hires" else width
                req_base_h = align_dim(int(demo_init.height * hires_scale), 16) if mode == "hires" else height
                demo_init, demo_mask, target_w, target_h, crop_info, pad_notice = auto_pad_transparent_to_supported(
                    demo_init, demo_mask, req_w=req_base_w, req_h=req_base_h
                )
                if pad_notice:
                    job.notice = pad_notice
            demo_out = self._run_demo_image(
                mode=mode,
                prompt=resolved_prompt,
                width=target_w,
                height=target_h,
                steps=steps,
                denoise=denoise,
                seed=seed,
                init_image=demo_init,
                mask_image=demo_mask,
                extra_ref_images=refs_list if mode == "i2i" else [],
                lora_summary=lora_summary,
                job=job,
                batch_idx=batch_idx,
            )
            return crop_back_to_original_ratio(demo_out, crop_info)

        if job.cancel_event.is_set():
            raise InterruptedError("用户已取消生成")
        self.ensure_loaded(job)
        if job.cancel_event.is_set():
            raise InterruptedError("用户已取消生成")
        torch, _ = self._detect_torch()

        with self._infer_lock:
            if job.cancel_event.is_set():
                raise InterruptedError("用户已取消生成")
            self.sync_lora(lora_enabled=lora_enabled, lora_name=lora_name, lora_weight=lora_weight)
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

            target_w, target_h = width, height
            ref_image = init_image
            prepared_mask: Optional[Image.Image] = None
            prepared_extras: list[Image.Image] = []
            crop_info = None

            src_primary = init_image if init_image is not None else (
                lora_control_image if (mode == "t2i" and using_lora_control) else None
            )
            if src_primary is not None:
                req_base_w = align_dim(int(src_primary.width * hires_scale), 16) if mode == "hires" else width
                req_base_h = align_dim(int(src_primary.height * hires_scale), 16) if mode == "hires" else height
                padded_img, padded_mask, target_w, target_h, crop_info, pad_notice = auto_pad_transparent_to_supported(
                    src_primary,
                    mask_image if mode == "inpaint" else None,
                    req_w=req_base_w,
                    req_h=req_base_h,
                )
                if pad_notice:
                    job.notice = pad_notice
                    logger.info("Auto-pad applied for job %s: %s", job.job_id, pad_notice)

                ref_image = padded_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                if mode == "inpaint" and padded_mask is not None:
                    prepared_mask = padded_mask.resize((target_w, target_h), Image.Resampling.BILINEAR)
                if mode == "i2i":
                    if refs_list:
                        max_edge = max(target_w, target_h, 1024)
                        prepared_extras = [resize_preserve_aspect(rimg, max_edge=max_edge) for rimg in refs_list]
                    if using_lora_control and lora_control_image is not None:
                        prepared_extras.append(resize_preserve_aspect(lora_control_image, max_edge=max(target_w, target_h, 1024)))

            eff_now = self.effective_settings()
            is_smart_stage = eff_now["offload_mode"] == "smart_stage_offload"

            def step_callback(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                if job.cancel_event.is_set():
                    raise InterruptedError("用户已取消生成")
                pipe_total = getattr(_pipe, "_num_timesteps", None)
                if not isinstance(pipe_total, int) or pipe_total <= 0:
                    ts = getattr(_pipe, "timesteps", None)
                    try:
                        pipe_total = len(ts) if ts is not None else steps
                    except TypeError:
                        pipe_total = steps
                cur = step_index + 1
                actual_total = max(int(pipe_total or steps), cur, 1)
                if is_smart_stage and torch.cuda.is_available() and (step_index == 0 or cur == actual_total):
                    # Stage boundary trim: flush text_encoder blocks at step 0, and DiT blocks before VAE decode
                    torch.cuda.empty_cache()
                job.total_steps = actual_total
                job.current_step = cur
                base_progress = batch_idx / job.batch_size
                step_progress = (cur / actual_total) / job.batch_size
                job.progress = min(0.99, base_progress + step_progress)
                job.message = f"第 {batch_idx + 1}/{job.batch_size} 张 · Step {cur}/{actual_total}"
                return callback_kwargs

            call_kwargs: dict[str, Any] = {
                "prompt": resolved_prompt,
                "width": target_w,
                "height": target_h,
                "num_inference_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "generator": generator,
                "callback_on_step_end": step_callback,
            }
            if true_cfg_scale > 1.0 and negative_prompt.strip():
                call_kwargs["negative_prompt"] = negative_prompt.strip()

            if ref_image is not None and (mode in {"i2i", "hires", "inpaint"} or (mode == "t2i" and using_lora_control)):
                call_kwargs["image"] = [ref_image, *prepared_extras] if prepared_extras else ref_image
                if self._pipe_supports_strength:
                    eff_strength = lora_control_strength if (mode == "t2i" and using_lora_control) else denoise
                    call_kwargs["strength"] = max(0.05, min(1.0, float(eff_strength)))
                if mode == "inpaint" and prepared_mask is not None and self._pipe_supports_mask:
                    call_kwargs["mask_image"] = prepared_mask

            # Register fast mid-step abort hooks so clicking Stop cancels immediately
            hook_handles: list[Any] = []

            def _abort_pre_hook(*_args: Any, **_kwargs: Any) -> None:
                if job.cancel_event.is_set():
                    raise InterruptedError("用户已取消生成")

            try:
                if hasattr(self.pipe, "_interrupt"):
                    setattr(self.pipe, "_interrupt", False)
                for comp_name in ("transformer", "text_encoder", "vae"):
                    comp = getattr(self.pipe, comp_name, None)
                    if comp is not None and hasattr(comp, "register_forward_pre_hook"):
                        hook_handles.append(comp.register_forward_pre_hook(_abort_pre_hook))
                transformer = getattr(self.pipe, "transformer", None)
                blocks = getattr(transformer, "transformer_blocks", None) if transformer is not None else None
                if blocks is not None:
                    for blk in blocks:
                        if hasattr(blk, "register_forward_pre_hook"):
                            hook_handles.append(blk.register_forward_pre_hook(_abort_pre_hook))

                if is_smart_stage and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    out = self.pipe(**call_kwargs)
                except Exception as pipe_exc:
                    # If pipeline rejects RGBA 4-channel transparent canvas, retry cleanly with RGB
                    if ref_image is not None and ref_image.mode == "RGBA" and not job.cancel_event.is_set():
                        err_s = str(pipe_exc).lower()
                        if any(k in err_s for k in ("channel", "rgba", "shape", "size", "mode", "rgb", "mismatch")):
                            logger.warning("Pipeline rejected RGBA canvas (%s); falling back to RGB canvas.", pipe_exc)
                            rgb_ref = ref_image.convert("RGB")
                            call_kwargs["image"] = [rgb_ref, *prepared_extras] if prepared_extras else rgb_ref
                            out = self.pipe(**call_kwargs)
                        else:
                            raise
                    else:
                        raise
            finally:
                for h in hook_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                if is_smart_stage and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    trim_host_memory()

            result_img: Image.Image = out.images[0].convert("RGB")

            # Guarantee exact size & RGB mode match before blend/composite (prevents PIL "images do not match")
            if ref_image is not None:
                ref_rgb = ref_image.convert("RGB")
                if result_img.size != ref_rgb.size:
                    result_img = result_img.resize(ref_rgb.size, Image.Resampling.LANCZOS)
                if mode == "inpaint" and prepared_mask is not None:
                    if prepared_mask.size != ref_rgb.size:
                        prepared_mask = prepared_mask.resize(ref_rgb.size, Image.Resampling.BILINEAR)
                    feather = prepared_mask.convert("L")
                    if mask_blur > 0:
                        feather = feather.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                    if denoise < 0.99:
                        blended = Image.blend(ref_rgb, result_img, alpha=max(0.05, min(1.0, denoise)))
                        result_img = Image.composite(blended, ref_rgb, feather)
                    else:
                        result_img = Image.composite(result_img, ref_rgb, feather)
                elif (
                    mode in {"i2i", "hires"}
                    and not prepared_extras
                    and not self._pipe_supports_strength
                    and denoise < 0.99
                ):
                    alpha = max(0.1, min(1.0, denoise))
                    result_img = Image.blend(ref_rgb, result_img, alpha=alpha)

            return crop_back_to_original_ratio(result_img, crop_info)


# ---------------------------------------------------------------------------
# Global Singletons & FastAPI App
# ---------------------------------------------------------------------------

LORAS_DIR = get_loras_dir(BASE_DIR)
gallery = GalleryManager(OUTPUT_DIR, GALLERY_INDEX_PATH)
engine = QwenImage21Engine()
terminal = TerminalManager(BASE_DIR)
lora_downloads = LoraDownloadManager(LORAS_DIR)
comfy_launcher = ComfyUILauncher(LORAS_DIR)
jobs_lock = threading.Lock()
jobs: dict[str, JobState] = {}
_hf_cache_stat_lock = threading.Lock()
_hf_cache_stat_cache: tuple[float, int] = (0.0, 0)

app = FastAPI(title="Qwen-Image-2.1 INT8 Mobile Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    mode: Literal["t2i", "i2i", "hires", "inpaint"] = "t2i"
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = Field(default=1024, ge=256, le=3072)
    height: int = Field(default=1024, ge=256, le=3072)
    steps: int = Field(default=8, ge=1, le=80)
    denoise: float = Field(default=0.75, ge=0.05, le=1.0)
    true_cfg_scale: float = Field(default=1.0, ge=1.0, le=12.0)
    seed: int = Field(default=-1)
    batch_size: int = Field(default=1, ge=1, le=8)
    hires_scale: float = Field(default=1.5, ge=1.0, le=2.5)
    mask_blur: int = Field(default=8, ge=0, le=48)
    init_image_b64: Optional[str] = None
    ref_images_b64: list[str] = Field(default_factory=list, max_length=9)
    mask_image_b64: Optional[str] = None
    lora_enabled: bool = False
    lora_name: Optional[str] = None
    lora_weight: float = Field(default=0.85, ge=0.0, le=1.5)
    lora_trigger: str = Field(default="", max_length=500)
    lora_action_mode: Literal["standard", "control"] = "standard"
    lora_control_image_b64: Optional[str] = None
    lora_control_strength: float = Field(default=0.85, ge=0.0, le=1.5)


class ConfigUpdateRequest(BaseModel):
    model_id: Optional[str] = None
    quant_mode: Optional[str] = None
    gpu_profile_override: Optional[str] = None
    dtype_mode: Optional[str] = None
    offload_mode: Optional[str] = None
    vae_tiling_mode: Optional[str] = None
    hf_token: Optional[str] = None
    cpu_offload: Optional[bool] = None
    vae_tiling: Optional[bool] = None
    ai_endpoint: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_model: Optional[str] = None


class AIChatRequest(BaseModel):
    scene: Literal["studio", "terminal", "error"] = "studio"
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)


class TerminalRunRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4000)


class BatchDeleteRequest(BaseModel):
    ids: list[str]


class ShutdownRequest(BaseModel):
    unassign_colab: bool = False


class SwapRequest(BaseModel):
    size_gb: int = Field(default=8, ge=1, le=32)


class LoraDownloadRequest(BaseModel):
    url: str = Field(..., min_length=4, max_length=2000)
    token: Optional[str] = Field(default=None, max_length=500)
    filename: Optional[str] = Field(default=None, max_length=200)


class ComfyLaunchRequest(BaseModel):
    gguf_variant: Literal["q8", "q4", "both", "none"] = "q4"


def collect_models_inventory() -> dict[str, Any]:
    """Collect currently loaded pipeline/LoRA and downloaded HF / LoRA models on disk."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))

    cur_base_repo, _ = strip_model_variant_suffix(engine.model_id)
    cur_aliases = {cur_base_repo.lower()}
    if cur_base_repo.lower() == GGUF_BASE_REPO.lower():
        cur_aliases.add(GGUF_REDIRECTED_REPO.lower())

    downloaded_hf: list[dict[str, Any]] = []
    if hub_dir.exists():
        try:
            for entry in hub_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("models--"):
                    repo_id = entry.name[len("models--") :].replace("--", "/", 1)
                    snaps_dir = entry / "snapshots"
                    has_snapshot = snaps_dir.exists() and any(snaps_dir.iterdir())
                    size_b = get_dir_size_bytes(entry)
                    is_cur = repo_id.lower() in cur_aliases
                    downloaded_hf.append(
                        {
                            "repo_id": repo_id,
                            "size_gb": round(size_b / (1024**3), 2),
                            "size_mb": round(size_b / (1024**2), 1),
                            "ready": has_snapshot,
                            "is_current": is_cur,
                            "is_loaded": engine.loaded and is_cur,
                        }
                    )
        except OSError:
            pass
    downloaded_hf.sort(key=lambda x: (not x["is_current"], -x["size_gb"], x["repo_id"]))

    active_lora_name = Path(engine._active_lora_path).name if engine._active_lora_path else None
    loras = list_loras(LORAS_DIR)
    for item in loras:
        item["is_loaded"] = bool(active_lora_name and item["name"] == active_lora_name)

    return {
        "loaded": {
            "is_loaded": engine.loaded,
            "is_loading": engine.loading,
            "demo_mode": engine.demo_mode,
            "model_id": engine.model_id,
            "quant_mode": engine.quant_mode,
            "gguf_variant": resolve_gguf_variant(engine.model_id, engine.quant_mode),
            "device": engine.device,
            "active_lora": active_lora_name,
            "active_lora_weight": engine._active_lora_weight,
        },
        "downloaded_hf": downloaded_hf,
        "downloaded_loras": loras,
        "comfyui_installed": comfy_launcher.get_status().get("installed", False),
    }


def collect_system_stats() -> dict[str, Any]:
    """Collect RAM, VRAM, Disk, Gallery, GPU Profile, and HF cache usage."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(OUTPUT_DIR))
    outputs_bytes = get_dir_size_bytes(OUTPUT_DIR)

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    now_ts = time.time()
    with _hf_cache_stat_lock:
        if now_ts - _hf_cache_stat_cache[0] > 15.0:
            hf_cache_bytes = get_dir_size_bytes(hf_home)
            globals()["_hf_cache_stat_cache"] = (now_ts, hf_cache_bytes)
        else:
            hf_cache_bytes = _hf_cache_stat_cache[1]

    eff = engine.effective_settings()
    active_profile: GPUProfile = eff["profile"]
    gguf_variant: Optional[str] = eff.get("gguf_variant")

    gpu_info: dict[str, Any] = {
        "available": False,
        "name": f"演示模式 ({active_profile.display_name})" if engine.demo_mode else "No CUDA GPU",
        "vram_used_gb": 0.0,
        "vram_total_gb": float(active_profile.vram_gb) if engine.demo_mode else 0.0,
        "vram_percent": 0.0,
        "torch_allocated_gb": 0.0,
        "profile_key": active_profile.key,
        "profile_name": active_profile.display_name,
        "arch": active_profile.arch,
        "notes": active_profile.notes,
    }

    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            used_b = total_b - free_b
            alloc_b = torch.cuda.memory_allocated(0)
            gpu_info = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_used_gb": round(used_b / (1024**3), 2),
                "vram_total_gb": round(total_b / (1024**3), 2),
                "vram_percent": round((used_b / max(1, total_b)) * 100, 1),
                "torch_allocated_gb": round(alloc_b / (1024**3), 2),
                "profile_key": active_profile.key,
                "profile_name": active_profile.display_name,
                "arch": active_profile.arch,
                "notes": active_profile.notes,
            }
    except Exception:
        pass

    return {
        "ram": {
            "used_gb": round(vm.used / (1024**3), 2),
            "total_gb": round(vm.total / (1024**3), 2),
            "percent": round(vm.percent, 1),
        },
        "swap": get_swap_status(),
        "disk": {
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "percent": round((disk.used / max(1, disk.total)) * 100, 1),
        },
        "storage": {
            "outputs_mb": round(outputs_bytes / (1024**2), 2),
            "hf_cache_gb": round(hf_cache_bytes / (1024**3), 2),
            "output_dir": str(OUTPUT_DIR),
        },
        "gpu": gpu_info,
        "models": collect_models_inventory(),
        "ai": {
            "endpoint": engine.ai_endpoint,
            "api_key": engine.ai_api_key,
            "model": engine.ai_model,
            "has_api_key": bool(engine.ai_api_key),
        },
        "engine": {
            "model_id": engine.model_id,
            "quant_mode": engine.quant_mode,
            "gguf_variant": gguf_variant,
            "is_gguf": gguf_variant is not None,
            "gpu_profile_override": engine.gpu_profile_override,
            "dtype_mode": engine.dtype_mode,
            "offload_mode": engine.offload_mode,
            "vae_tiling_mode": engine.vae_tiling_mode,
            "effective_dtype": eff["dtype"],
            "effective_offload": eff["offload_mode"],
            "effective_vae_tiling": eff["vae_tiling"],
            "effective_tf32": eff["allow_tf32"],
            "loaded": engine.loaded,
            "loading": engine.loading,
            "device": engine.device,
            "demo_mode": engine.demo_mode,
            "has_hf_token": bool(engine.hf_token),
            "is_full_precision": engine.quant_mode == "none",
            "default_steps": MODEL_PRESETS.get(engine.model_id, {}).get("default_steps", 8),
            "uptime_sec": round(time.time() - SERVER_START_TIME, 1),
            "cpu_offload": eff["offload_mode"] != "none",
            "vae_tiling": eff["vae_tiling"],
        },
    }


def _run_generation_job(job_id: str, req: GenerateRequest) -> None:
    with jobs_lock:
        job = jobs[job_id]

    try:
        init_img = decode_data_url_image(req.init_image_b64) if req.init_image_b64 else None
        extra_refs = [decode_data_url_image(b) for b in (req.ref_images_b64 or [])[:9] if b]
        mask_img = decode_data_url_mask(req.mask_image_b64) if req.mask_image_b64 else None
        lora_ctrl_img = decode_data_url_image(req.lora_control_image_b64) if req.lora_control_image_b64 else None

        if req.mode in {"i2i", "hires", "inpaint"} and init_img is None:
            raise ValueError(f"模式 {req.mode} 需要先上传或选择参考底图")
        if req.mode == "inpaint" and mask_img is None:
            raise ValueError("局部重绘模式需要在画布上涂抹重绘区域 (Mask)")

        job.status = "running"
        job.total_steps = req.steps
        base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)

        for idx in range(req.batch_size):
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.message = "已取消生成"
                job.finished_at = time.time()
                return

            item_seed = base_seed + idx if req.seed >= 0 else random.randint(0, 2**31 - 1)
            job.current_batch = idx + 1
            job.current_step = 0
            job.message = f"准备生成第 {idx + 1}/{req.batch_size} 张..."

            t0 = time.time()
            img = engine.generate_one(
                mode=req.mode,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                denoise=req.denoise,
                true_cfg_scale=req.true_cfg_scale,
                seed=item_seed,
                init_image=init_img,
                mask_image=mask_img,
                extra_ref_images=extra_refs,
                hires_scale=req.hires_scale,
                mask_blur=req.mask_blur,
                job=job,
                batch_idx=idx,
                lora_enabled=req.lora_enabled,
                lora_name=req.lora_name,
                lora_weight=req.lora_weight,
                lora_trigger=req.lora_trigger,
                lora_action_mode=req.lora_action_mode,
                lora_control_image=lora_ctrl_img,
                lora_control_strength=req.lora_control_strength,
            )
            elapsed = time.time() - t0

            record = gallery.add_image(
                img,
                {
                    "mode": req.mode,
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "steps": req.steps,
                    "denoise": req.denoise,
                    "true_cfg_scale": req.true_cfg_scale,
                    "seed": item_seed,
                    "model_id": engine.model_id,
                    "quant_mode": "DEMO" if engine.demo_mode else engine.quant_mode,
                    "elapsed_sec": elapsed,
                    "hires_scale": req.hires_scale if req.mode == "hires" else 1.0,
                    "ref_count": (1 + len(extra_refs)) if (req.mode == "i2i" and init_img is not None) else (1 if init_img is not None else 0),
                },
            )
            job.results.append(record)

        job.progress = 1.0
        job.status = "completed"
        job.message = f"生成完成，共 {len(job.results)} 张"
        job.finished_at = time.time()

    except InterruptedError:
        job.status = "cancelled"
        job.message = "生成已手动停止"
        job.finished_at = time.time()
    except Exception as exc:
        logger.exception("Generation job %s failed", job_id)
        job.status = "error"
        job.error = str(exc)
        job.message = f"生成失败: {exc}"
        job.finished_at = time.time()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_id": engine.model_id,
        "quant_mode": engine.quant_mode,
        "demo_mode": engine.demo_mode,
        "loaded": engine.loaded,
        "presets": MODEL_PRESETS,
        "gpu_profiles": {k: v.to_dict() for k, v in GPU_PROFILES.items()},
        "active_gpu_profile": engine.resolve_gpu_profile().to_dict(),
    }


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    return collect_system_stats()


@app.post("/api/config")
def api_update_config(req: ConfigUpdateRequest) -> dict[str, Any]:
    engine.reconfigure(
        model_id=req.model_id,
        quant_mode=req.quant_mode,
        gpu_profile_override=req.gpu_profile_override,
        dtype_mode=req.dtype_mode,
        offload_mode=req.offload_mode,
        vae_tiling_mode=req.vae_tiling_mode,
        hf_token=req.hf_token,
        cpu_offload=req.cpu_offload,
        vae_tiling=req.vae_tiling,
        ai_endpoint=req.ai_endpoint,
        ai_api_key=req.ai_api_key,
        ai_model=req.ai_model,
    )
    return {"ok": True, "stats": collect_system_stats()}


@app.get("/api/terminal/cwd")
def api_terminal_cwd() -> dict[str, Any]:
    return {"cwd": terminal.cwd}


@app.post("/api/terminal/run")
def api_terminal_run(req: TerminalRunRequest) -> dict[str, Any]:
    task = terminal.run_command(req.command)
    return {
        "cmd_id": task.cmd_id,
        "cwd": terminal.cwd,
        "running": task.running,
    }


@app.get("/api/terminal/poll/{cmd_id}")
def api_terminal_poll(cmd_id: str, offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    data = terminal.poll(cmd_id, offset=offset)
    if data is None:
        raise HTTPException(status_code=404, detail="Terminal task not found")
    return data


@app.post("/api/terminal/kill/{cmd_id}")
def api_terminal_kill(cmd_id: str) -> dict[str, Any]:
    killed = terminal.kill(cmd_id)
    return {"ok": killed}


@app.get("/api/logs")
def api_server_logs(limit: int = Query(default=150, ge=10, le=500)) -> dict[str, Any]:
    lines = log_buffer.get_lines(limit=limit)
    log_file = Path("/tmp/qwen_server.log")
    if os.name != "nt" and log_file.exists():
        try:
            raw_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            if raw_lines:
                lines = raw_lines[-limit:]
        except OSError:
            pass
    return {"lines": lines, "ts": time.time()}


@app.get("/api/loras")
def api_list_loras() -> dict[str, Any]:
    return {"loras_dir": str(LORAS_DIR), "items": list_loras(LORAS_DIR)}


@app.delete("/api/loras/{lora_name}")
def api_delete_lora(lora_name: str) -> dict[str, Any]:
    if engine._active_lora_path and Path(engine._active_lora_path).name == Path(lora_name).name:
        engine._unload_active_lora()
    ok = delete_lora(LORAS_DIR, lora_name)
    if not ok:
        raise HTTPException(status_code=404, detail="LoRA file not found")
    return {"ok": True, "items": list_loras(LORAS_DIR)}


def _extract_ai_actions(reply_text: str) -> Optional[dict[str, Any]]:
    """Extract the last ```json ... ``` action block from AI reply if present."""
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", reply_text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```\s*(\{.*?\})\s*```", reply_text, flags=re.DOTALL)
    for raw_json in reversed(matches):
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _build_ai_system_prompt(scene: str, context: dict[str, Any]) -> str:
    stats = collect_system_stats()
    eng = stats["engine"]
    gpu = stats["gpu"]
    loras = [f"{x['name']} ({x['size_mb']}MB, {'有效' if x.get('valid', True) else '损坏'})" for x in stats["models"]["downloaded_loras"]]
    base_info = (
        f"当前系统环境：\n"
        f"- GPU: {gpu.get('name')} (VRAM {gpu.get('vram_used_gb')}/{gpu.get('vram_total_gb')} GB, 档位 {gpu.get('profile_name')})\n"
        f"- 当前模型: {eng.get('model_id')} | 量化: {eng.get('quant_mode')} | 显存调度: {eng.get('effective_offload')} | 已加载: {eng.get('loaded')}\n"
        f"- 可用 LoRA 列表: {', '.join(loras) if loras else '无'}\n"
        f"- 终端当前目录: {terminal.cwd}\n"
        f"- 支持的仓库预设: Qwen/Qwen-Image-2.1-Turbo (推荐8步), Qwen/Qwen-Image-2.1 (推荐40步), addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit, abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q8 (Q8满血, 推荐8步), abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q4 (Q4轻量, 推荐8步)\n"
        f"- 支持的量化模式: fp8-int8, fp8-all, int8-torchao, int8-bnb, gguf-q8-full, gguf-q4, none\n"
        f"- 支持的显存调度: auto, smart_stage_offload, none, model_cpu_offload, sequential_cpu_offload\n"
        f"- 支持的生成比例: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3\n"
    )
    action_schema = (
        "如果需要修改提示词、生成参数、引擎设置或建议在终端执行命令，请在回答末尾附加一个 ```json ... ``` 代码块（仅包含需要修改的字段）：\n"
        "```json\n"
        "{\n"
        '  "prompt": "新的正向提示词（可选）",\n'
        '  "negative_prompt": "反向提示词（可选）",\n'
        '  "mode": "t2i | i2i | hires | inpaint（可选）",\n'
        '  "ratio": "1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3（可选）",\n'
        '  "tier": "1k | 1.5k | 2k（可选）",\n'
        '  "width": 1024,\n'
        '  "height": 1024,\n'
        '  "steps": 8,\n'
        '  "denoise": 0.75,\n'
        '  "true_cfg_scale": 1.0,\n'
        '  "seed": -1,\n'
        '  "batch_size": 1,\n'
        '  "hires_scale": 1.5,\n'
        '  "mask_blur": 8,\n'
        '  "lora_enabled": false,\n'
        '  "lora_name": "文件名（可选）",\n'
        '  "lora_weight": 0.85,\n'
        '  "lora_trigger": "触发词（可选）",\n'
        '  "settings": {\n'
        '    "model_id": "...",\n'
        '    "quant_mode": "...",\n'
        '    "offload_mode": "...",\n'
        '    "dtype_mode": "auto | bfloat16 | float16",\n'
        '    "vae_tiling_mode": "auto | on | off"\n'
        "  },\n"
        '  "terminal_commands": ["可一键执行的Linux/Python命令（可选）"]\n'
        "}\n"
        "```\n"
    )
    if scene == "terminal":
        extra = f"终端近期输出/上下文：\n{json.dumps(context, ensure_ascii=False)[:2500]}\n"
        return (
            "你是 Google Colab Qwen-Image-2.1 WebUI 的内置终端 AI 助手。请用简洁中文解答命令问题、诊断环境报错，并在需要运行代码或修复依赖时在 ```json``` 中提供 `terminal_commands` 数组。\n"
            + base_info
            + extra
            + action_schema
        )
    if scene == "error":
        recent_logs = "\n".join(log_buffer.get_lines(limit=35))
        extra = f"生图报错上下文：\n{json.dumps(context, ensure_ascii=False)[:2500]}\n最近服务日志：\n{recent_logs[-2000:]}\n"
        return (
            "你是 Google Colab Qwen-Image-2.1 WebUI 的故障排查专家。用户生图发生报错，请直接指出原因、给出清晰修复步骤，并尽可能通过 ```json``` 提供可直接应用的修复参数/设置或终端修复命令。\n"
            + base_info
            + extra
            + action_schema
        )
    extra = f"当前创作台参数：\n{json.dumps(context, ensure_ascii=False)[:2500]}\n"
    return (
        "你是 Google Colab Qwen-Image-2.1 WebUI 的创作与调参 AI 助手。你可以帮用户撰写或润色高质量中英文提示词（支持图生图 @0 主图、@1 次图语法）、推荐分辨率比例/步数/Denoise/CFG/LoRA 参数，或修改系统设置。请简明回答，并在需要一键应用时附带 ```json``` 块。\n"
        + base_info
        + extra
        + action_schema
    )


def _call_ai_endpoint(messages: list[dict[str, str]]) -> str:
    endpoint = (engine.ai_endpoint or "").strip()
    api_key = (engine.ai_api_key or "").strip()
    model = (engine.ai_model or "").strip()
    if not endpoint or not model:
        raise RuntimeError("请先在设置中配置 AI 接口地址与模型名称")

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1600,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Qwen-Image-2.1-EasyColab/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"AI 接口返回 HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        raise RuntimeError(f"连接 AI 接口失败: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"AI 返回格式异常: {str(body)[:300]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content") or choices[0].get("text") or ""
    return str(content).strip()


@app.post("/api/ai/chat")
def api_ai_chat(req: AIChatRequest) -> dict[str, Any]:
    sys_prompt = _build_ai_system_prompt(req.scene, req.context)
    messages: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    for item in (req.history or [])[-10:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)[:4000]})
    messages.append({"role": "user", "content": req.message})

    try:
        reply = _call_ai_endpoint(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    actions = _extract_ai_actions(reply)
    return {"ok": True, "reply": reply, "actions": actions, "model": engine.ai_model}


@app.post("/api/loras/download")
def api_download_lora(req: LoraDownloadRequest) -> dict[str, Any]:
    task_id = lora_downloads.start_download(
        url=req.url,
        token=req.token,
        filename_override=req.filename,
        demo_mode=engine.demo_mode,
    )
    return {"ok": True, "task_id": task_id, "loras_dir": str(LORAS_DIR)}


@app.get("/api/loras/download/{task_id}")
def api_download_lora_status(task_id: str) -> dict[str, Any]:
    st = lora_downloads.get_status(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Download task not found")
    return st


@app.get("/api/comfyui/status")
def api_comfyui_status() -> dict[str, Any]:
    return comfy_launcher.get_status()


@app.post("/api/comfyui/launch")
def api_comfyui_launch(req: Optional[ComfyLaunchRequest] = None) -> dict[str, Any]:
    gguf_variant = req.gguf_variant if req is not None else "q4"
    return comfy_launcher.launch(
        demo_mode=engine.demo_mode,
        gguf_variant=gguf_variant,
        hf_token=engine.hf_token or None,
    )


@app.post("/api/comfyui/stop")
def api_comfyui_stop() -> dict[str, Any]:
    return comfy_launcher.stop()


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    job = JobState(
        job_id=job_id,
        status="queued",
        mode=req.mode,
        batch_size=req.batch_size,
        total_steps=req.steps,
        message="任务已加入队列...",
    )
    with jobs_lock:
        jobs[job_id] = job
        if len(jobs) > 50:
            done_keys = [
                k for k, v in sorted(jobs.items(), key=lambda kv: kv[1].created_at)
                if v.status in {"completed", "cancelled", "error"}
            ]
            for old_k in done_keys[:-30]:
                jobs.pop(old_k, None)

    worker = threading.Thread(target=_run_generation_job, args=(job_id, req), daemon=True)
    worker.start()
    return {"job_id": job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.message = "已取消排队"
        else:
            job.message = "正在立即中断生成..."
    if engine.pipe is not None and hasattr(engine.pipe, "_interrupt"):
        try:
            setattr(engine.pipe, "_interrupt", True)
        except Exception:
            pass
    return {"ok": True}


@app.get("/api/gallery")
def api_gallery(
    mode: Optional[str] = Query(default="all"),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return gallery.list_items(mode=mode, query=q, limit=limit, offset=offset)


@app.delete("/api/gallery/{item_id}")
def api_delete_gallery_item(item_id: str) -> dict[str, Any]:
    removed = gallery.delete_items([item_id])
    if removed == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True, "removed": removed}


@app.post("/api/gallery/batch-delete")
def api_batch_delete_gallery(req: BatchDeleteRequest) -> dict[str, Any]:
    removed = gallery.delete_items(req.ids)
    return {"ok": True, "removed": removed}


@app.get("/api/gallery/export")
def api_export_gallery(ids: Optional[str] = Query(default=None)) -> StreamingResponse:
    """Bundle selected or all gallery images + metadata JSON into a single ZIP."""
    target_ids = set(ids.split(",")) if ids else None
    listing = gallery.list_items(limit=500)["items"]
    if target_ids:
        listing = [x for x in listing if x["id"] in target_ids]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in listing:
            fpath = OUTPUT_DIR / item["filename"]
            if fpath.exists():
                zf.write(fpath, arcname=item["filename"])
        zf.writestr("metadata.json", json.dumps(listing, ensure_ascii=False, indent=2))
    buf.seek(0)
    zip_bytes = buf.getvalue()
    filename = f"qwen_gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


@app.post("/api/system/enable-swap")
def api_enable_swap(req: Optional[SwapRequest] = None) -> dict[str, Any]:
    size_gb = req.size_gb if req else 8
    result = enable_linux_swap(size_gb=size_gb)
    return {**result, "stats": collect_system_stats()}


@app.post("/api/system/free-vram")
def api_free_vram(unload_model: bool = Query(default=False)) -> dict[str, Any]:
    if unload_model:
        engine.unload()
    else:
        trim_host_memory()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return {"ok": True, "stats": collect_system_stats()}


@app.post("/api/system/clear-hf-cache")
def api_clear_hf_cache() -> dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"
    if hub_dir.exists():
        shutil.rmtree(hub_dir, ignore_errors=True)
    with _hf_cache_stat_lock:
        globals()["_hf_cache_stat_cache"] = (0.0, 0)
    return {"ok": True, "stats": collect_system_stats()}


@app.post("/api/shutdown")
def api_shutdown(req: ShutdownRequest) -> JSONResponse:
    """Gracefully unload model, stop tunnel/server, and optionally unassign Colab runtime."""
    def _shutdown_sequence() -> None:
        time.sleep(0.6)
        logger.info("Executing one-click shutdown (unassign_colab=%s)...", req.unassign_colab)
        try:
            comfy_launcher.stop()
            engine.unload()
        except Exception as exc:
            logger.warning("Unload warning: %s", exc)

        # Terminate external tunnel pid if registered by Colab launcher
        tunnel_pid = os.environ.get("CLOUDFLARED_PID")
        if tunnel_pid and tunnel_pid.isdigit():
            try:
                os.kill(int(tunnel_pid), signal.SIGTERM)
            except OSError:
                pass

        if req.unassign_colab:
            try:
                from google.colab import runtime  # type: ignore

                runtime.unassign()
            except Exception as exc:
                logger.info("Colab runtime.unassign() skipped/unavailable: %s", exc)

        os._exit(0)

    threading.Thread(target=_shutdown_sequence, daemon=True).start()
    return JSONResponse(
        {
            "ok": True,
            "message": "正在卸载模型并关闭服务..." + ("（已请求释放 Colab GPU 运行时）" if req.unassign_colab else ""),
        }
    )


# Mount generated outputs & static frontend
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
