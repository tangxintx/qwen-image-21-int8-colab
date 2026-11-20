"""
On-demand ComfyUI Launcher for Google Colab.
Never clones, downloads, or starts ComfyUI automatically on startup.
Only runs when explicitly triggered by the user from Settings.
Includes automatic installation of `city96/ComfyUI-GGUF` and Q8/Q4 GGUF model staging.
"""

from __future__ import annotations

import json
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

COMFY_GGUF_URL = "https://github.com/city96/ComfyUI-GGUF"
COMFY_GGUF_GIT = f"{COMFY_GGUF_URL}.git"
COMFY_GGUF_QWEN3VL_TE_GIT = "https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE.git"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink `src` -> `dst` on POSIX (fallback to copy on Windows or cross-device failure)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except OSError:
            pass
        dst.unlink()
    elif dst.exists():
        return

    if os.name != "nt":
        try:
            os.symlink(str(src), str(dst))
            return
        except OSError:
            pass
    shutil.copy2(str(src), str(dst))


def build_gguf_workflow(
    variant: str,
    dit_filename: str,
    te_filename: str,
    vae_filename: str = "qwen_image_2.1_vae_bf16.safetensors",
) -> dict[str, Any]:
    """
    Build a ready-to-import ComfyUI workflow JSON for `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`
    using `city96/ComfyUI-GGUF` (`UnetLoaderGGUF` + `CLIPLoader`/`CLIPLoaderGGUF`) and native `TextEncodeQwenImage21`.
    """
    v = variant.lower().strip()
    is_q4 = v == "q4"

    te_node = (
        {
            "id": 2,
            "type": "CLIPLoaderGGUF",
            "pos": [60, 230],
            "size": [340, 82],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2], "slot_index": 0}],
            "properties": {"Node name for S&R": "CLIPLoaderGGUF"},
            "widgets_values": [te_filename, "qwen_image"],
        }
        if is_q4
        else {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [60, 230],
            "size": [340, 106],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2], "slot_index": 0}],
            "properties": {"Node name for S&R": "CLIPLoader"},
            "widgets_values": [te_filename, "qwen_image", "default"],
        }
    )

    nodes = [
        {
            "id": 1,
            "type": "UnetLoaderGGUF",
            "pos": [60, 80],
            "size": [340, 58],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "properties": {"Node name for S&R": "UnetLoaderGGUF"},
            "widgets_values": [dit_filename],
        },
        te_node,
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [60, 390],
            "size": [340, 58],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [3, 12], "slot_index": 0}],
            "properties": {"Node name for S&R": "VAELoader"},
            "widgets_values": [vae_filename],
        },
        {
            "id": 4,
            "type": "TextEncodeQwenImage21",
            "pos": [450, 210],
            "size": [420, 240],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "vae", "type": "VAE", "link": 12, "shape": 7},
            ],
            "outputs": [
                {"name": "positive", "type": "CONDITIONING", "links": [4], "slot_index": 0},
                {"name": "negative", "type": "CONDITIONING", "links": [5], "slot_index": 1},
                {"name": "latent", "type": "LATENT", "links": [], "slot_index": 2},
            ],
            "properties": {"Node name for S&R": "TextEncodeQwenImage21"},
            "widgets_values": [
                "cinematic photograph, glowing neon Tokyo street at night after rain, ultra detailed, 8k",
                "",
                1024,
            ],
        },
        {
            "id": 5,
            "type": "EmptyLatentImage",
            "pos": [450, 490],
            "size": [315, 106],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6], "slot_index": 0}],
            "properties": {"Node name for S&R": "EmptyLatentImage"},
            "widgets_values": [1024, 1024, 1],
        },
        {
            "id": 6,
            "type": "RandomNoise",
            "pos": [920, 80],
            "size": [280, 82],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "NOISE", "type": "NOISE", "links": [7], "slot_index": 0}],
            "properties": {"Node name for S&R": "RandomNoise"},
            "widgets_values": [42, "randomize"],
        },
        {
            "id": 7,
            "type": "CFGGuider",
            "pos": [920, 200],
            "size": [280, 98],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "positive", "type": "CONDITIONING", "link": 4},
                {"name": "negative", "type": "CONDITIONING", "link": 5},
            ],
            "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [8], "slot_index": 0}],
            "properties": {"Node name for S&R": "CFGGuider"},
            "widgets_values": [1.0],
        },
        {
            "id": 8,
            "type": "KSamplerSelect",
            "pos": [920, 340],
            "size": [280, 58],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": [9], "slot_index": 0}],
            "properties": {"Node name for S&R": "KSamplerSelect"},
            "widgets_values": ["euler"],
        },
        {
            "id": 9,
            "type": "ManualSigmas",
            "pos": [920, 440],
            "size": [320, 82],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [10], "slot_index": 0}],
            "properties": {"Node name for S&R": "ManualSigmas"},
            "widgets_values": ["1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0", False],
        },
        {
            "id": 10,
            "type": "SamplerCustomAdvanced",
            "pos": [1280, 180],
            "size": [300, 124],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [
                {"name": "noise", "type": "NOISE", "link": 7},
                {"name": "guider", "type": "GUIDER", "link": 8},
                {"name": "sampler", "type": "SAMPLER", "link": 9},
                {"name": "sigmas", "type": "SIGMAS", "link": 10},
                {"name": "latent_image", "type": "LATENT", "link": 6},
            ],
            "outputs": [
                {"name": "output", "type": "LATENT", "links": [11], "slot_index": 0},
                {"name": "denoised_output", "type": "LATENT", "links": [], "slot_index": 1},
            ],
            "properties": {"Node name for S&R": "SamplerCustomAdvanced"},
        },
        {
            "id": 11,
            "type": "VAEDecode",
            "pos": [1280, 350],
            "size": [210, 46],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 11},
                {"name": "vae", "type": "VAE", "link": 3},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [13], "slot_index": 0}],
            "properties": {"Node name for S&R": "VAEDecode"},
        },
        {
            "id": 12,
            "type": "SaveImage",
            "pos": [1540, 180],
            "size": [320, 270],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 13}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": [f"Qwen21_GGUF_{v.upper()}"],
        },
    ]

    links = [
        [1, 1, 0, 7, 0, "MODEL"],
        [2, 2, 0, 4, 0, "CLIP"],
        [3, 3, 0, 11, 1, "VAE"],
        [4, 4, 0, 7, 1, "CONDITIONING"],
        [5, 4, 1, 7, 2, "CONDITIONING"],
        [6, 5, 0, 10, 4, "LATENT"],
        [7, 6, 0, 10, 0, "NOISE"],
        [8, 7, 0, 10, 1, "GUIDER"],
        [9, 8, 0, 10, 2, "SAMPLER"],
        [10, 9, 0, 10, 3, "SIGMAS"],
        [11, 10, 0, 11, 0, "LATENT"],
        [12, 3, 0, 4, 1, "VAE"],
        [13, 11, 0, 12, 0, "IMAGE"],
    ]

    return {
        "last_node_id": 12,
        "last_link_id": 13,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.85, "offset": [0, 0]},
            "qwen_gguf_variant": v,
        },
        "version": 0.4,
    }


def write_gguf_workflows(
    comfy_dir: Path,
    staged_meta: Optional[dict[str, dict[str, str]]] = None,
) -> list[str]:
    """Write ready-to-use Q8 and Q4 ComfyUI-GGUF workflow JSONs into `user/default/workflows/`."""
    wf_dir = comfy_dir / "user" / "default" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "q8": {
            "dit": "qwen-image-2.1-turbo-Q8_0.gguf",
            "te": "qwen3vl_8b_bf16.safetensors",
            "vae": "qwen_image_2.1_vae_bf16.safetensors",
            "out": "Qwen_Image_2.1_Uncensored_GGUF_Q8.json",
        },
        "q4": {
            "dit": "qwen-image-2.1-turbo-Q4_K_M.gguf",
            "te": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            "vae": "qwen_image_2.1_vae_bf16.safetensors",
            "out": "Qwen_Image_2.1_Uncensored_GGUF_Q4.json",
        },
    }
    written: list[str] = []
    for variant, info in defaults.items():
        meta = (staged_meta or {}).get(variant, {})
        wf = build_gguf_workflow(
            variant=variant,
            dit_filename=meta.get("dit_filename") or info["dit"],
            te_filename=meta.get("te_filename") or info["te"],
            vae_filename=meta.get("vae_filename") or info["vae"],
        )
        out_path = wf_dir / info["out"]
        out_path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(info["out"])
    return written


_UNSET: Any = object()


class ComfyUILauncher:
    """Manages on-demand ComfyUI installation, ComfyUI-GGUF plugin, GGUF model staging, LoRA symlinking, and tunnel."""

    def __init__(self, loras_dir: Path) -> None:
        self.loras_dir = loras_dir
        self._lock = threading.Lock()
        self._status: str = "idle"  # idle | installing | starting | running | error
        self._progress: float = 0.0
        self._message: str = "未启动（按需手动启动 · 含 city96/ComfyUI-GGUF 插件）"
        self._public_url: Optional[str] = None
        self._local_url: str = f"http://127.0.0.1:{COMFY_PORT}"
        self._error: Optional[str] = None
        self._proc: Optional[subprocess.Popen[Any]] = None
        self._tunnel_proc: Optional[subprocess.Popen[str]] = None
        self._log_file: Optional[Any] = None
        self._gguf_variant: str = "q4"
        self._staged_files: list[str] = []
        self._workflows: list[str] = []

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            if self._status == "running" and self._proc is not None and self._proc.poll() is not None:
                self._status = "idle"
                self._message = "ComfyUI 进程已退出"
                self._public_url = None
            gguf_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
            te_helper_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF-Qwen3VL-TE"
            return {
                "status": self._status,
                "progress": round(self._progress, 3),
                "message": self._message,
                "public_url": self._public_url,
                "local_url": self._local_url,
                "error": self._error,
                "installed": COMFY_DIR.exists(),
                "gguf_plugin_installed": gguf_dir.exists(),
                "gguf_qwen3vl_te_installed": te_helper_dir.exists(),
                "gguf_plugin_url": COMFY_GGUF_URL,
                "gguf_variant": self._gguf_variant,
                "staged_files": list(self._staged_files),
                "workflows": list(self._workflows),
            }

    def _set_state(
        self,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        public_url: Any = _UNSET,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if status is not None:
                self._status = status
            if progress is not None:
                self._progress = progress
            if message is not None:
                self._message = message
            if public_url is not _UNSET:
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

    def _install_gguf_plugins(self) -> None:
        custom_nodes = COMFY_DIR / "custom_nodes"
        custom_nodes.mkdir(parents=True, exist_ok=True)

        gguf_dir = custom_nodes / "ComfyUI-GGUF"
        if not gguf_dir.exists():
            self._set_state("installing", 0.36, "正在安装 GGUF 加载器插件 city96/ComfyUI-GGUF...")
            res = subprocess.run(
                ["git", "clone", "--depth", "1", COMFY_GGUF_GIT, str(gguf_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"git clone ComfyUI-GGUF 失败: {(res.stdout or '')[-300:]}")
        else:
            subprocess.run(["git", "-C", str(gguf_dir), "pull", "--ff-only"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

        gguf_req = gguf_dir / "requirements.txt"
        self._set_state("installing", 0.44, "正在安装 ComfyUI-GGUF 依赖 (gguf / protobuf / sentencepiece)...")
        if gguf_req.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(gguf_req)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gguf>=0.13.0", "sentencepiece>=0.2.0", "protobuf>=4.25.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Best-effort Qwen3-VL TE GGUF compatibility node (auto no-op if upstream ComfyUI-GGUF already supports it)
        te_helper_dir = custom_nodes / "ComfyUI-GGUF-Qwen3VL-TE"
        if not te_helper_dir.exists():
            try:
                res_te = subprocess.run(
                    ["git", "clone", "--depth", "1", COMFY_GGUF_QWEN3VL_TE_GIT, str(te_helper_dir)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                )
                if res_te.returncode != 0:
                    logger.warning("Optional ComfyUI-GGUF-Qwen3VL-TE clone skipped: %s", (res_te.stdout or "")[-200:])
            except Exception as exc:
                logger.warning("Optional ComfyUI-GGUF-Qwen3VL-TE clone skipped: %s", exc)

    def _stage_gguf_models(self, gguf_variant: str, hf_token: Optional[str] = None) -> None:
        from hf_downloader import download_gguf_bundle_fast

        norm = (gguf_variant or "q4").strip().lower()
        targets = ["q8", "q4"] if norm == "both" else ([norm] if norm in {"q8", "q4"} else [])
        staged_names: list[str] = []
        staged_meta: dict[str, dict[str, str]] = {}

        for idx, var in enumerate(targets):
            base_pct = 0.50 + idx * 0.14

            def _dl_cb(cur_b: int, tot_b: int, _spd: float, msg: str, _base: float = base_pct) -> None:
                frac = (cur_b / tot_b) if tot_b > 0 else 0.3
                self._set_state("installing", min(0.76, _base + 0.12 * frac), msg)

            bundle = download_gguf_bundle_fast(
                variant=var,
                token=hf_token,
                progress_cb=_dl_cb,
                use_tqdm=False,
                include_diffusers_configs=False,
                include_mmproj=True,
            )

            dit_src = Path(bundle["dit_path"])
            te_src = Path(bundle["te_path"])
            vae_src = Path(bundle["vae_path"])

            _link_or_copy(dit_src, COMFY_DIR / "models" / "diffusion_models" / dit_src.name)
            _link_or_copy(dit_src, COMFY_DIR / "models" / "unet" / dit_src.name)
            _link_or_copy(te_src, COMFY_DIR / "models" / "text_encoders" / te_src.name)
            _link_or_copy(te_src, COMFY_DIR / "models" / "clip" / te_src.name)
            _link_or_copy(vae_src, COMFY_DIR / "models" / "vae" / vae_src.name)

            staged_names.extend([dit_src.name, te_src.name, vae_src.name])
            if bundle.get("mmproj_path"):
                mm_src = Path(bundle["mmproj_path"])
                _link_or_copy(mm_src, COMFY_DIR / "models" / "text_encoders" / mm_src.name)
                _link_or_copy(mm_src, COMFY_DIR / "models" / "clip" / mm_src.name)
                staged_names.append(mm_src.name)

            staged_meta[var] = {
                "dit_filename": dit_src.name,
                "te_filename": te_src.name,
                "vae_filename": vae_src.name,
            }

        workflows = write_gguf_workflows(COMFY_DIR, staged_meta=staged_meta)
        with self._lock:
            self._staged_files = list(dict.fromkeys(staged_names))
            self._workflows = workflows

    def _ensure_cloudflared(self) -> Optional[Path]:
        if CLOUDFLARED_BIN.exists() and CLOUDFLARED_BIN.stat().st_size > 1_000_000:
            return CLOUDFLARED_BIN
        if os.name == "nt":
            return None
        part_path = CLOUDFLARED_BIN.with_suffix(".comfy.part")
        try:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            req = urllib.request.Request(url, headers={"User-Agent": "Qwen-Image-2.1-EasyColab/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "wb") as f:
                shutil.copyfileobj(resp, f, length=262_144)
            if part_path.exists() and part_path.stat().st_size > 1_000_000:
                part_path.replace(CLOUDFLARED_BIN)
                CLOUDFLARED_BIN.chmod(CLOUDFLARED_BIN.stat().st_mode | 0o111)
                return CLOUDFLARED_BIN
            part_path.unlink(missing_ok=True)
            return None
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            logger.warning("Failed to download cloudflared for ComfyUI: %s", exc)
            return None

    def launch(
        self,
        demo_mode: bool = False,
        gguf_variant: str = "q4",
        hf_token: Optional[str] = None,
    ) -> dict[str, Any]:
        norm_var = (gguf_variant or "q4").strip().lower()
        if norm_var not in {"q8", "q4", "both", "none"}:
            norm_var = "q4"

        with self._lock:
            if self._status in {"installing", "starting", "running"}:
                return self.get_status()
            self._gguf_variant = norm_var
            self._status = "installing"
            self._progress = 0.05
            self._message = f"准备启动 ComfyUI（含 city96/ComfyUI-GGUF · 档位 {norm_var.upper()}）..."
            self._public_url = None
            self._error = None

        def _worker() -> None:
            try:
                if demo_mode:
                    demo_staged: list[str] = []
                    if norm_var in {"q8", "both"}:
                        demo_staged.extend(["qwen-image-2.1-turbo-Q8_0.gguf", "qwen3vl_8b_bf16.safetensors"])
                    if norm_var in {"q4", "both"}:
                        demo_staged.extend(["qwen-image-2.1-turbo-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"])
                    if norm_var != "none":
                        demo_staged.append("qwen_image_2.1_vae_bf16.safetensors")

                    for pct, msg in [
                        (0.20, "[演示模式] 模拟克隆 ComfyUI 仓库..."),
                        (0.42, "[演示模式] 模拟安装 city96/ComfyUI-GGUF 加载器插件..."),
                        (0.68, f"[演示模式] 模拟准备 GGUF 模型与 VAE ({norm_var.upper()})..."),
                        (0.88, "[演示模式] 模拟写入工作流并启动 ComfyUI (8188)..."),
                    ]:
                        time.sleep(0.18)
                        self._set_state("installing", pct, msg)

                    with self._lock:
                        self._staged_files = list(dict.fromkeys(demo_staged))
                        self._workflows = [
                            "Qwen_Image_2.1_Uncensored_GGUF_Q8.json",
                            "Qwen_Image_2.1_Uncensored_GGUF_Q4.json",
                        ]
                    self._set_state(
                        status="running",
                        progress=1.0,
                        message=f"ComfyUI + ComfyUI-GGUF 演示服务已就绪 ({norm_var.upper()})",
                        public_url=self._local_url,
                    )
                    return

                # 1) Clone ComfyUI if not present
                if not (COMFY_DIR / "main.py").exists():
                    self._set_state("installing", 0.14, "正在克隆 ComfyUI 仓库...")
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
                    self._set_state("installing", 0.28, "正在安装 ComfyUI 依赖...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                # 3) Install city96/ComfyUI-GGUF plugin + dependencies
                self._install_gguf_plugins()

                # 4) Link LoRAs & stage GGUF DiT / Text Encoder / VAE + write workflows
                self._link_loras_dir()
                self._stage_gguf_models(norm_var, hf_token=hf_token)

                self._set_state("starting", 0.78, f"正在启动 ComfyUI + GGUF 加载器 (端口 {COMFY_PORT})...")

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
                    self._set_state("starting", 0.78 + min(0.10, i * 0.002), f"等待 ComfyUI 监听端口 {COMFY_PORT}...")
                    time.sleep(1)

                if not ready:
                    raise RuntimeError(f"ComfyUI 在 60 秒内未能监听端口 {COMFY_PORT}")

                # 5) Establish Cloudflared public tunnel
                public_url = self._local_url
                cf_bin = self._ensure_cloudflared()
                if cf_bin is not None:
                    self._set_state("starting", 0.90, "正在建立 ComfyUI Cloudflare 公网隧道...")
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
                        proc_ref = self._tunnel_proc
                        if proc_ref is None or proc_ref.stdout is None:
                            done_evt.set()
                            return
                        # Keep draining stdout for the entire lifetime of cloudflared to avoid pipe buffer deadlock
                        for line in proc_ref.stdout:
                            if not found:
                                m = pat.search(line)
                                if m:
                                    found.append(m.group(0))
                                    done_evt.set()
                        done_evt.set()

                    threading.Thread(target=_read_tunnel, daemon=True).start()
                    done_evt.wait(timeout=35)
                    if found:
                        public_url = found[0]

                self._set_state(
                    status="running",
                    progress=1.0,
                    message=f"ComfyUI + ComfyUI-GGUF 已启动就绪 ({norm_var.upper()})",
                    public_url=public_url,
                )
            except Exception as exc:
                logger.exception("ComfyUI launch failed")
                self.stop()
                self._set_state(
                    status="error",
                    progress=0.0,
                    message=f"ComfyUI 启动失败: {exc}",
                    public_url=None,
                    error=str(exc),
                )

        threading.Thread(target=_worker, daemon=True).start()
        return self.get_status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            tunnel_proc = self._tunnel_proc
            main_proc = self._proc
            log_file = self._log_file
            self._proc = None
            self._tunnel_proc = None
            self._log_file = None

        for proc in (tunnel_proc, main_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        self._set_state(status="idle", progress=0.0, message="ComfyUI 已停止", public_url=None, error=None)
        return self.get_status()

