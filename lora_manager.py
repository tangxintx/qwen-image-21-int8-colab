"""
LoRA Manager for Qwen-Image-2.1 INT8 / FP8 Colab Studio.
- Manages `/content/loras` (Colab) or local fallback directory
- Downloads LoRA weights from direct URLs, Hugging Face, and Civitai (with API page resolution + token)
- Validates `.safetensors` headers (and optional tensor finiteness) before saving
- Auto-remaps non-standard prefixes (ComfyUI `diffusion_model.`, Kohya `lora_unet_` hierarchy, `.alpha` scaling) for Diffusers
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("qwen_easycolab.lora")

ALLOWED_LORA_EXTS = {".safetensors", ".pt", ".bin", ".ckpt"}
_MIN_VALID_LORA_BYTES = 16 * 1024  # 16 KB minimum sanity floor for real weight files
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB safety cap

# Known multi-word snake_case module & attribute tokens in Qwen-Image / DiT / Flux LoRA keys
_KOHYA_COMPOUND_TOKENS = (
    "single_transformer_blocks",
    "transformer_blocks",
    "double_blocks",
    "single_blocks",
    "time_text_embed",
    "timestep_embedder",
    "context_embedder",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "norm_q",
    "norm_k",
    "adaLN_modulation",
    "proj_in",
    "final_layer",
    "to_out",
    "to_q",
    "to_k",
    "to_v",
    "img_mlp",
    "txt_mlp",
    "img_mod",
    "txt_mod",
    "img_in",
    "txt_in",
    "x_embedder",
    "proj_out",
    "norm_out",
    "ff_context",
    "gate_up",
    "gate_layer",
    "linear_1",
    "linear_2",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


def get_loras_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve `/content/loras` on Colab/Linux when writable, otherwise `<base_dir>/loras`."""
    env_dir = os.environ.get("QWEN_LORAS_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    colab_root = Path("/content")
    if os.name != "nt" and colab_root.exists() and os.access(colab_root, os.W_OK):
        p = colab_root / "loras"
        p.mkdir(parents=True, exist_ok=True)
        return p

    fallback_base = base_dir or Path(__file__).resolve().parent
    p = (fallback_base / "loras").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _check_state_dict_finite(raw_sd: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Verify no floating-point tensors contain NaN or Inf values."""
    try:
        import torch
    except ImportError:
        return True, None

    for name, val in raw_sd.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            if not bool(torch.isfinite(val).all()):
                return False, f"权重张量 {name} 包含 NaN 或 Inf，文件可能已损坏"
    return True, None


def validate_lora_file(lora_path: Path, check_tensors: bool = False) -> tuple[bool, Optional[str]]:
    """
    Validate that a LoRA file is a genuine binary weight file and not a saved HTML/JSON error page.
    When `check_tensors=True`, also loads weights on CPU to verify there are no NaN/Inf tensors.
    Returns `(is_valid, issue_reason)`.
    """
    if not lora_path.exists() or not lora_path.is_file():
        return False, "文件不存在"
    try:
        size = lora_path.stat().st_size
    except OSError as exc:
        return False, f"无法读取文件大小: {exc}"

    if size == 0:
        return False, "文件为空 (0 B)"

    try:
        with open(lora_path, "rb") as f:
            prefix = f.read(32)
    except OSError as exc:
        return False, f"无法读取文件头: {exc}"

    stripped = prefix.lstrip()
    if stripped.startswith((b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML", b"<?xml")):
        return False, "文件内容为 HTML 网页（可能缺少 Civitai Token 或下载链接不是权重直链）"

    is_demo_placeholder = prefix.startswith(b"DEMO_LORA_PLACEHOLDER")
    if lora_path.suffix.lower() == ".safetensors":
        if is_demo_placeholder:
            return True, None
        if len(prefix) < 8:
            return False, "safetensors 文件头过短"
        header_len = struct.unpack("<Q", prefix[:8])[0]
        if header_len < 2 or header_len > _MAX_SAFETENSORS_HEADER_BYTES or (8 + header_len) > size:
            return (
                False,
                f"safetensors 文件头异常 (header_len={header_len}, size={size})，文件可能已损坏或为网页错误页",
            )
        try:
            with open(lora_path, "rb") as f:
                f.seek(8)
                header_bytes = f.read(min(header_len, 4096))
                if not header_bytes.lstrip().startswith(b"{"):
                    return False, "safetensors 元数据非 JSON 格式"
                if header_len <= 1024 * 1024:
                    f.seek(8)
                    full_hdr = f.read(header_len).decode("utf-8")
                    parsed = json.loads(full_hdr)
                    if not isinstance(parsed, dict):
                        return False, "safetensors 元数据结构异常"
        except Exception:
            return False, "safetensors 文件头解析失败，请重新下载"

        if check_tensors:
            try:
                from safetensors.torch import load_file

                sd = load_file(str(lora_path), device="cpu")
                ok, reason = _check_state_dict_finite(sd)
                if not ok:
                    return False, reason
            except Exception as exc:
                return False, f"safetensors 张量读取校验失败: {exc}"
    else:
        if size < _MIN_VALID_LORA_BYTES and not is_demo_placeholder:
            return False, f"权重文件过小 ({size} B)，可能下载不完整"
        if check_tensors and not is_demo_placeholder:
            try:
                import torch

                raw = torch.load(str(lora_path), map_location="cpu", weights_only=True)
                sd = _unwrap_state_dict(raw)
                ok, reason = _check_state_dict_finite(sd)
                if not ok:
                    return False, reason
            except Exception as exc:
                return False, f"权重张量校验失败: {exc}"

    return True, None


def list_loras(loras_dir: Path) -> list[dict[str, Any]]:
    """List available LoRA files under `loras_dir` sorted by newest first."""
    if not loras_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for entry in loras_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in ALLOWED_LORA_EXTS:
            st = entry.stat()
            valid, issue = validate_lora_file(entry, check_tensors=False)
            if not valid and st.st_size < 128:
                try:
                    if entry.read_bytes().startswith(b"DEMO_LORA_PLACEHOLDER"):
                        valid, issue = True, None
                except OSError:
                    pass
            items.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "size_mb": round(st.st_size / (1024 * 1024), 2),
                    "size_bytes": st.st_size,
                    "mtime": st.st_mtime,
                    "valid": valid,
                    "issue": issue,
                }
            )
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


def delete_lora(loras_dir: Path, filename: str) -> bool:
    """Safely delete a LoRA file inside `loras_dir`."""
    safe_name = Path(filename).name
    if not safe_name:
        return False
    target = (loras_dir / safe_name).resolve()
    if target.parent != loras_dir.resolve() or not target.exists() or not target.is_file():
        return False
    target.unlink()
    return True


def normalize_download_url(url: str, token: Optional[str] = None) -> tuple[str, dict[str, str]]:
    """Normalize Hugging Face `/blob/` links and Civitai download auth cleanly."""
    clean_url = url.strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Qwen-Image-2.1-EasyColab/1.0",
    }
    tok = (token or "").strip()

    if "huggingface.co/" in clean_url and "/blob/" in clean_url:
        clean_url = clean_url.replace("/blob/", "/resolve/")

    if "civitai.com" in clean_url:
        parsed = urllib.parse.urlparse(clean_url)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path.startswith("/models/") and "/api/download/" not in parsed.path:
            ver_id = qs.get("modelVersionId", [None])[0]
            if ver_id:
                clean_url = f"https://civitai.com/api/download/models/{ver_id}"
        elif parsed.path.startswith("/model-versions/"):
            ver_m = re.match(r"^/model-versions/(\d+)", parsed.path)
            if ver_m:
                clean_url = f"https://civitai.com/api/download/models/{ver_m.group(1)}"
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
            sep = "&" if "?" in clean_url else "?"
            if "token=" not in clean_url:
                clean_url = f"{clean_url}{sep}token={urllib.parse.quote(tok)}"
    elif tok:
        headers["Authorization"] = f"Bearer {tok}"

    return clean_url, headers


def resolve_download_target(url: str, token: Optional[str] = None) -> tuple[str, dict[str, str], Optional[str]]:
    """
    Resolve Civitai model page URLs lacking `?modelVersionId=...` (such as
    `https://civitai.com/models/3027392/poselineartlora`) via Civitai's v1 API
    so we download the actual model weights instead of an HTML web page.
    Returns `(download_url, headers, suggested_filename)`.
    """
    req_url, headers = normalize_download_url(url, token=token)
    suggested_name: Optional[str] = None

    parsed = urllib.parse.urlparse(req_url)
    if "civitai.com" in parsed.netloc.lower() and "/api/download/" not in parsed.path:
        model_match = re.match(r"^/models/(\d+)", parsed.path)
        if model_match:
            model_id = model_match.group(1)
            api_url = f"https://civitai.com/api/v1/models/{model_id}"
            api_headers = {
                "User-Agent": headers["User-Agent"],
                "Accept": "application/json",
            }
            if "Authorization" in headers:
                api_headers["Authorization"] = headers["Authorization"]
            api_req = urllib.request.Request(api_url, headers=api_headers)
            try:
                with urllib.request.urlopen(api_req, timeout=20) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                raise RuntimeError(
                    f"解析 Civitai 模型页面失败 ({exc})。请检查链接或改用带 modelVersionId 的下载直链（如需登录请填写 Civitai Token）"
                ) from exc

            versions = payload.get("modelVersions") or []
            if not versions:
                raise RuntimeError("Civitai API 未返回可用的模型版本，可能需要填写 Civitai API Token")
            ver = versions[0]
            ver_id = ver.get("id")
            files = ver.get("files") or []
            primary_file = next(
                (f for f in files if f.get("primary") or f.get("type") == "Model"),
                files[0] if files else {},
            )
            dl_url = primary_file.get("downloadUrl") or ver.get("downloadUrl")
            if not dl_url and ver_id:
                dl_url = f"https://civitai.com/api/download/models/{ver_id}"
            if not dl_url:
                raise RuntimeError("未在 Civitai 模型版本中找到下载链接")
            suggested_name = primary_file.get("name")
            req_url, headers = normalize_download_url(dl_url, token=token)

    return req_url, headers, suggested_name


def _extract_filename(url: str, content_disp: str) -> str:
    if content_disp:
        m_utf = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", content_disp, re.IGNORECASE)
        if m_utf:
            name = urllib.parse.unquote(m_utf.group(1)).strip().strip('"\'')
            if name:
                return Path(name).name
        m_std = re.search(r'filename\s*=\s*"?([^";]+)"?', content_disp, re.IGNORECASE)
        if m_std:
            name = m_std.group(1).strip().strip('"\'')
            if name:
                return Path(name).name

    parsed = urllib.parse.urlparse(url)
    candidate = Path(urllib.parse.unquote(parsed.path)).name
    if candidate and candidate not in {"download", "models", ""}:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
        if not Path(safe).suffix:
            safe += ".safetensors"
        return safe
    return f"lora_{int(time.time())}.safetensors"


def _sanitize_lora_filename(raw_name: str) -> str:
    name = Path(raw_name).name.strip()
    if not name:
        name = f"lora_{int(time.time())}.safetensors"
    if not Path(name).suffix:
        name += ".safetensors"
    name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]", "_", name)
    # Prevent hidden dot-files or `..` names
    name = name.lstrip(".") or f"lora_{int(time.time())}.safetensors"
    return name


class LoraDownloadManager:
    """Manages async LoRA file downloads into `LORAS_DIR` with real-time progress and bounded task history."""

    def __init__(self, loras_dir: Path) -> None:
        self.loras_dir = loras_dir
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def _prune_tasks_locked(self, max_items: int = 50, ttl_sec: float = 3600.0) -> None:
        now = time.time()
        expired = [
            k for k, v in self._tasks.items()
            if v.get("status") in {"completed", "error"}
            and (now - float(v.get("created_at", now))) > ttl_sec
        ]
        for k in expired:
            self._tasks.pop(k, None)

        if len(self._tasks) > max_items:
            done = [
                k for k, v in sorted(self._tasks.items(), key=lambda kv: float(kv[1].get("created_at", 0.0)))
                if v.get("status") in {"completed", "error"}
            ]
            for old_k in done[: max(0, len(self._tasks) - max_items)]:
                self._tasks.pop(old_k, None)

    def start_download(
        self,
        url: str,
        token: Optional[str] = None,
        filename_override: Optional[str] = None,
        demo_mode: bool = False,
    ) -> str:
        task_id = uuid.uuid4().hex[:10]
        state = {
            "task_id": task_id,
            "status": "downloading",
            "progress": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "speed_mb_s": 0.0,
            "filename": "",
            "message": "正在解析并连接下载地址...",
            "error": None,
            "created_at": time.time(),
        }
        with self._lock:
            self._tasks[task_id] = state
            self._prune_tasks_locked()

        def _worker() -> None:
            tmp_dest: Optional[Path] = None
            try:
                if demo_mode:
                    raw_name = filename_override if (filename_override and filename_override.strip()) else _extract_filename(url, "")
                    fname = _sanitize_lora_filename(raw_name)
                    for i in range(1, 6):
                        time.sleep(0.12)
                        with self._lock:
                            state["progress"] = i * 0.2
                            state["downloaded_mb"] = round(i * 25.6, 1)
                            state["total_mb"] = 128.0
                            state["speed_mb_s"] = 48.5
                            state["filename"] = fname
                            state["message"] = f"[演示模式] 正在下载 {fname}: {i * 20}%"
                    target = self.loras_dir / fname
                    target.write_bytes(b"DEMO_LORA_PLACEHOLDER")
                    with self._lock:
                        state["status"] = "completed"
                        state["progress"] = 1.0
                        state["message"] = f"下载完成: {fname}"
                    return

                req_url, headers, suggested_name = resolve_download_target(url, token=token)
                req = urllib.request.Request(req_url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    final_url = resp.geturl()
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    if "text/html" in ctype or "application/json" in ctype or "text/plain" in ctype:
                        raise RuntimeError(
                            f"下载地址返回了非模型内容 (Content-Type: {ctype})。"
                            "若为 Civitai 模型，请在 Token 框填入 Civitai API Key 或检查链接是否正确"
                        )

                    cd = resp.headers.get("Content-Disposition", "")
                    total_bytes = int(resp.headers.get("Content-Length") or 0)
                    if filename_override and filename_override.strip():
                        raw_fname = filename_override
                    elif cd:
                        raw_fname = _extract_filename(final_url, cd)
                    elif suggested_name:
                        raw_fname = suggested_name
                    else:
                        raw_fname = _extract_filename(final_url, cd)

                    fname = _sanitize_lora_filename(raw_fname)
                    self.loras_dir.mkdir(parents=True, exist_ok=True)
                    dest = self.loras_dir / fname
                    tmp_dest = self.loras_dir / f".{fname}.{task_id}.part"

                    downloaded = 0
                    t_last = time.time()
                    b_last = 0
                    with open(tmp_dest, "wb") as f:
                        while True:
                            chunk = resp.read(262_144)
                            if not chunk:
                                break
                            if downloaded == 0:
                                head_trim = chunk[:32].lstrip()
                                if head_trim.startswith((b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML")):
                                    raise RuntimeError(
                                        "服务器返回了 HTML 登录/拦截页面而非 LoRA 权重。请在 Token 框填写 Civitai / HF API Token 后重试"
                                    )
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - t_last >= 0.25:
                                dt = max(0.05, now - t_last)
                                spd = (downloaded - b_last) / (1024 * 1024) / dt
                                b_last = downloaded
                                t_last = now
                                with self._lock:
                                    state["filename"] = fname
                                    state["downloaded_mb"] = round(downloaded / (1024 * 1024), 2)
                                    state["total_mb"] = round(total_bytes / (1024 * 1024), 2)
                                    state["speed_mb_s"] = round(spd, 1)
                                    state["progress"] = (
                                        min(0.99, downloaded / total_bytes) if total_bytes > 0 else 0.0
                                    )
                                    pct_str = f"{int(state['progress'] * 100)}%" if total_bytes > 0 else f"{state['downloaded_mb']} MB"
                                    state["message"] = f"下载 {fname}: {pct_str} ({spd:.1f} MB/s)"

                    check_path = self.loras_dir / f".validate_{task_id}_{fname}"
                    tmp_dest.replace(check_path)
                    tmp_dest = check_path
                    valid, issue = validate_lora_file(check_path, check_tensors=True)
                    if not valid:
                        raise RuntimeError(issue or "下载的 LoRA 权重文件校验失败")

                    check_path.replace(dest)
                    tmp_dest = None
                    with self._lock:
                        state["filename"] = fname
                        state["status"] = "completed"
                        state["progress"] = 1.0
                        state["downloaded_mb"] = round(downloaded / (1024 * 1024), 2)
                        if total_bytes <= 0:
                            state["total_mb"] = state["downloaded_mb"]
                        state["message"] = f"已保存至 {dest}"
            except Exception as exc:
                logger.warning("LoRA download failed (%s): %s", url, exc)
                if tmp_dest is not None and tmp_dest.exists():
                    try:
                        tmp_dest.unlink()
                    except OSError:
                        pass
                with self._lock:
                    state["status"] = "error"
                    state["error"] = str(exc)
                    state["message"] = f"下载失败: {exc}"

        threading.Thread(target=_worker, daemon=True).start()
        return task_id

    def get_status(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            item = self._tasks.get(task_id)
            return dict(item) if item else None


def _unwrap_state_dict(raw_obj: Any) -> dict[str, Any]:
    """Unpack nested `.pt` / `.bin` LoRA checkpoints (`state_dict`, `unet`, `transformer`, `model`, `lora`)."""
    if not isinstance(raw_obj, dict):
        raise RuntimeError(f"无法解包非字典格式的 LoRA 权重 ({type(raw_obj).__name__})")

    cur: dict[str, Any] = raw_obj
    for _ in range(3):
        unwrapped = False
        for wrapper_key in ("state_dict", "unet", "transformer", "model", "lora", "module"):
            sub = cur.get(wrapper_key)
            if isinstance(sub, dict) and sub:
                # Only replace wholesale if there aren't sibling tensor/dict components
                other_dicts = {
                    k: v for k, v in cur.items()
                    if k != wrapper_key and isinstance(v, dict) and v
                }
                if not other_dicts:
                    cur = sub
                    unwrapped = True
                    break
        if not unwrapped:
            break

    # Flatten nested sub-dictionaries (e.g. {"unet": {...}, "text_encoder": {...}})
    flat: dict[str, Any] = {}
    for k, v in cur.items():
        if isinstance(v, dict):
            prefix = "transformer" if k in {"unet", "diffusion_model"} else k
            for sub_k, sub_v in v.items():
                flat[f"{prefix}.{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def _restore_kohya_unet_key(body: str) -> str:
    """
    Restore Kohya `lora_unet_<underscored_hierarchy>.<attr>` back to Diffusers dotted module paths
    using architecture-aware compound token boundaries (preserving tokens like `transformer_blocks`,
    `to_out`, `img_mlp`, `gate_up`, `add_q_proj`, etc.).
    """
    if "." in body:
        mod_part, attr_part = body.split(".", 1)
    else:
        mod_part, attr_part = body, ""

    # Protect known multi-word snake_case identifiers with null-byte placeholders
    protected = mod_part
    placeholders: dict[str, str] = {}
    for idx, tok in enumerate(_KOHYA_COMPOUND_TOKENS):
        if tok in protected:
            marker = f"\x00T{idx}\x00"
            placeholders[marker] = tok
            protected = protected.replace(tok, marker)

    # Split remaining module hierarchy underscores into dots
    dotted = protected.replace("_", ".")
    for marker, orig_tok in placeholders.items():
        dotted = dotted.replace(marker, orig_tok)

    dotted = re.sub(r"\.+", ".", dotted).strip(".")
    return f"transformer.{dotted}.{attr_part}" if attr_part else f"transformer.{dotted}"


def normalize_lora_state_dict(lora_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a LoRA `.safetensors` / `.pt` / `.bin` file on CPU, verify tensor finiteness,
    remap ComfyUI & Kohya prefixes/hierarchies, apply `.alpha / rank` scaling to `lora_B.weight`,
    and report key collisions for Diffusers `load_lora_weights(..., strict=False)`.
    """
    valid, issue = validate_lora_file(lora_path, check_tensors=False)
    if not valid:
        raise RuntimeError(
            f"LoRA 文件 {lora_path.name} 无效或已损坏 ({issue})。请在 LoRA 设置或监控页删除后重新下载（Civitai 链接请附带 API Token）"
        )

    if lora_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        raw_sd = load_file(str(lora_path), device="cpu")
    else:
        import torch

        loaded_obj = torch.load(str(lora_path), map_location="cpu", weights_only=True)
        raw_sd = _unwrap_state_dict(loaded_obj)

    ok_finite, finite_issue = _check_state_dict_finite(raw_sd)
    if not ok_finite:
        raise RuntimeError(f"LoRA 文件 {lora_path.name} 校验失败: {finite_issue}")

    remapped: dict[str, Any] = {}
    alphas: dict[str, Any] = {}
    stripped_prefixes: set[str] = set()
    collisions: list[str] = []

    for orig_key, tensor in raw_sd.items():
        k = orig_key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "base_model.model."):
            if k.startswith(prefix):
                k = "transformer." + k[len(prefix) :]
                stripped_prefixes.add(prefix)
                break

        if ".lora_down." in k:
            k = k.replace(".lora_down.", ".lora_A.")
        elif ".lora_up." in k:
            k = k.replace(".lora_up.", ".lora_B.")

        if k.startswith("lora_unet_"):
            stripped_prefixes.add("lora_unet_")
            k = _restore_kohya_unet_key(k[len("lora_unet_") :])

        if k.endswith(".alpha"):
            base_key = k[: -len(".alpha")]
            if base_key in alphas:
                logger.warning("Duplicate LoRA alpha key '%s' from original key '%s'", k, orig_key)
                collisions.append(k)
            alphas[base_key] = tensor
            continue

        if k in remapped:
            logger.warning("LoRA key collision detected: '%s' overwritten by '%s'", k, orig_key)
            collisions.append(k)
        remapped[k] = tensor

    # Fold Kohya/ComfyUI `.alpha / rank` scaling into `lora_B.weight` so Diffusers PEFT
    # matches exact Kohya/ComfyUI LoRA strength instead of defaulting alpha == rank.
    alpha_scaled_layers = 0
    for base_key, alpha_tensor in alphas.items():
        a_key = f"{base_key}.lora_A.weight"
        b_key = f"{base_key}.lora_B.weight"
        a_tensor = remapped.get(a_key)
        b_tensor = remapped.get(b_key)
        if a_tensor is not None and b_tensor is not None and hasattr(a_tensor, "shape") and len(a_tensor.shape) >= 1:
            rank = int(a_tensor.shape[0])
            if rank > 0:
                try:
                    alpha_val = float(alpha_tensor.item() if hasattr(alpha_tensor, "item") else alpha_tensor)
                    scale = alpha_val / float(rank)
                    if abs(scale - 1.0) > 1e-6:
                        remapped[b_key] = (b_tensor.float() * scale).to(b_tensor.dtype)
                        alpha_scaled_layers += 1
                    continue
                except Exception as exc:
                    logger.warning("Failed to apply LoRA alpha scaling for %s: %s", base_key, exc)
        # Preserve unmatched `.alpha` tensor rather than silently dropping it
        remapped[f"{base_key}.alpha"] = alpha_tensor

    info = {
        "orig_keys": len(raw_sd),
        "mapped_keys": len(remapped),
        "stripped_prefixes": sorted(stripped_prefixes),
        "collisions": collisions,
        "alpha_scaled_layers": alpha_scaled_layers,
    }
    return remapped, info
