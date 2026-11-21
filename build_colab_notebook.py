"""
Build a clean Google Colab notebook (`Qwen_Image_21_INT8_Colab.ipynb`) that clones/syncs
directly from GitHub (`https://github.com/tangxintx/qwen-image-21-int8-colab.git`),
with smart incremental dependency checks, high-speed HF API downloads, opt-in AI auto-run,
native tqdm progress bars, and QR rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_NB = BASE_DIR / "Qwen_Image_21_INT8_Colab.ipynb"
REPO_URL = "https://github.com/tangxintx/qwen-image-21-int8-colab.git"


def main() -> None:
    cell1_source = '''#@title 1️⃣ 环境自检与智能增量依赖安装（重复运行自动跳过已就绪依赖）
import importlib.util
import subprocess
import sys

FORCE_REINSTALL = False  #@param {type:"boolean"}

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"], check=False)
from tqdm.auto import tqdm


def _has_qwen_diffusers() -> bool:
    try:
        import diffusers

        return hasattr(diffusers, "QwenImage21Pipeline")
    except Exception:
        return False


def _has_mods(*names: str) -> bool:
    return all(importlib.util.find_spec(n) is not None for n in names)


steps = [("检查 GPU 硬件与 CUDA 状态", ["nvidia-smi"])]

if FORCE_REINSTALL or not _has_qwen_diffusers():
    steps.append((
        "安装 Diffusers (Qwen-Image-2.1 支持)",
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "git+https://github.com/huggingface/diffusers.git"],
    ))
else:
    print("[跳过] Diffusers (QwenImage21Pipeline) 已就绪")

if FORCE_REINSTALL or not _has_mods("transformers", "accelerate", "safetensors", "torchao", "bitsandbytes", "gguf", "huggingface_hub", "hf_transfer"):
    steps.append((
        "安装 Transformers / TorchAO / GGUF / HF 极速下载",
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade",
            "transformers>=5.17.0", "accelerate", "safetensors", "torchao>=0.14.0",
            "bitsandbytes", "gguf>=0.13.0", "protobuf>=4.25.0", "sentencepiece>=0.2.0",
            "huggingface_hub", "hf_transfer",
        ],
    ))
else:
    print("[跳过] Transformers / TorchAO / GGUF / hf_transfer 已就绪")

if FORCE_REINSTALL or not _has_mods("fastapi", "uvicorn", "multipart", "psutil", "qrcode", "PIL"):
    steps.append((
        "安装 Web 服务与二维码组件",
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade",
            "fastapi", "uvicorn[standard]", "python-multipart", "psutil", "qrcode", "pillow",
        ],
    ))
else:
    print("[跳过] FastAPI / Uvicorn / QRCode 组件已就绪")

pbar = tqdm(total=len(steps), desc="环境初始化", unit="步")
for label, cmd in steps:
    pbar.set_description(f"正在: {label}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cmd[0] == "nvidia-smi" and res.stdout:
        print(res.stdout)
    elif res.returncode != 0:
        print(f"[警告] {label} 返回码 {res.returncode}:\\n{res.stdout[-800:]}")
    pbar.update(1)
pbar.set_description("环境初始化完成")
pbar.close()

try:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"✅ CUDA 就绪: {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM) · PyTorch {torch.__version__}")
    else:
        print("⚠️ 未检测到 CUDA GPU，建议在「代码执行程序 -> 更改运行时类型」中切换至 L4 / T4 / A100 GPU。")
except Exception as exc:
    print(f"⚠️ PyTorch 检测异常: {exc}")
print("环境准备完毕，请运行第 2 步启动工作台！")
'''

    cell2_source = f'''#@title 2️⃣ 同步最新代码并启动 WebUI（支持手机扫码与桌面端双栏工作台）
import importlib
import os
import subprocess
import sys
from pathlib import Path
from tqdm.auto import tqdm

#@markdown ### 1. 模型方案与硬件预设
EDITION = "Uncensored GGUF Q8 满血版 (Turbo-Quantized Q8_0 DiT + 全精度BF16编码器 + BF16 VAE)"  #@param ["Uncensored GGUF Q8 满血版 (Turbo-Quantized Q8_0 DiT + 全精度BF16编码器 + BF16 VAE)", "Uncensored GGUF Q4 轻量版 (Q4_K_M DiT + Qwen3-VL-8B Q4_K_M + BF16 VAE)", "L4 24GB 黄金方案 · Turbo FP8 DiT + INT8 编码器 + BF16 VAE", "L4 24GB 精绘方案 · Qwen-2.1 FP8 DiT + INT8 编码器 + BF16 VAE 40步", "INT8 轻量省显存版 (Turbo · 推荐 T4/L4)", "满血版 · Turbo 原生BF16/FP16不量化 (适合 A100/H100/G4)", "满血旗舰版 · Qwen-2.1 原生BF16/FP16 40步 (适合 A100-80G/H100/G4)", "自定义下方模型与量化参数"]
GPU_PROFILE = "auto"  #@param ["auto", "g4_rtx_pro_6000", "h100_sxm", "a100_sxm", "l4_ada", "t4_turing"]
MODEL_ID = "abenzerps/Qwen-Image-2.1-Turbo-Quantized:q8"  #@param ["abenzerps/Qwen-Image-2.1-Turbo-Quantized:q8", "abenzerps/Qwen-Image-2.1-Turbo-Quantized:q4", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q8", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF:q4", "Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1", "addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit"] {{allow-input: true}}
QUANT_MODE = "gguf-q8-full"  #@param ["gguf-q8-full", "gguf-q4", "fp8-int8", "fp8-all", "int8-torchao", "int8-bnb", "none"]

#@markdown ### 2. 下载加速、Google 云盘挂载与内存选项
HF_TOKEN = ""  #@param {{type:"string"}}
PRELOAD_MODEL = True  #@param {{type:"boolean"}}
MOUNT_GDRIVE = False  #@param {{type:"boolean"}}
GDRIVE_DIR = "/content/drive/MyDrive/qwen_image_21_outputs"  #@param {{type:"string"}}
GDRIVE_DIRECT_OUTPUT = False  #@param {{type:"boolean"}}
SWAP_OPTION = "跳过 (默认关闭)"  #@param ["跳过 (默认关闭)", "尝试开启 8GB Swap (仅支持虚拟机)", "尝试开启 16GB Swap (仅支持虚拟机)"]

#@markdown ### 3. 内置 AI 助手与自动运行代码（可选）
AI_ENDPOINT = ""  #@param {{type:"string"}}
AI_MODEL = "qwen3.8-27b"  #@param {{type:"string"}}
AI_API_KEY = ""  #@param {{type:"string"}}
AI_AUTO_RUN = False  #@param {{type:"boolean"}}

#@markdown ### 4. 仓库源与调试模式
REPO_URL = "{REPO_URL}"  #@param {{type:"string"}}
BRANCH = "main"  #@param {{type:"string"}}
DEMO_MODE = False  #@param {{type:"boolean"}}

# 1) 从 GitHub 拉取/更新项目代码到 /content/qwen_studio（保留 outputs/、loras/、.ai_config.json）
work_dir = Path("/content/qwen_studio")
with tqdm(total=2, desc="同步 GitHub 仓库", unit="步") as pbar:
    if (work_dir / ".git").exists():
        subprocess.run(["git", "-C", str(work_dir), "remote", "set-url", "origin", REPO_URL], check=True)
        pbar.update(1)
        subprocess.run(["git", "-C", str(work_dir), "fetch", "--depth", "1", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(work_dir), "reset", "--hard", f"origin/{{BRANCH}}"], check=True)
        pbar.update(1)
    elif work_dir.exists() and any(work_dir.iterdir()):
        subprocess.run(["git", "init", "-b", BRANCH, str(work_dir)], check=True)
        subprocess.run(["git", "-C", str(work_dir), "remote", "add", "origin", REPO_URL], check=True)
        pbar.update(1)
        subprocess.run(["git", "-C", str(work_dir), "fetch", "--depth", "1", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(work_dir), "reset", "--hard", f"origin/{{BRANCH}}"], check=True)
        pbar.update(1)
    else:
        pbar.update(1)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(work_dir)], check=True)
        pbar.update(1)

print(f"已同步最新代码 ({{REPO_URL}} @ {{BRANCH}}) -> {{work_dir}}")

os.chdir(work_dir)
if str(work_dir) not in sys.path:
    sys.path.insert(0, str(work_dir))

# 2) 解析版本预设
EDITION_MAP = [
    ("Uncensored GGUF Q8", ("abenzerps/Qwen-Image-2.1-Turbo-Quantized:q8", "gguf-q8-full", "auto")),
    ("Uncensored GGUF Q4", ("abenzerps/Qwen-Image-2.1-Turbo-Quantized:q4", "gguf-q4", "auto")),
    ("L4 24GB 黄金方案", ("Qwen/Qwen-Image-2.1-Turbo", "fp8-int8", "smart_stage_offload")),
    ("L4 24GB 精绘方案", ("Qwen/Qwen-Image-2.1", "fp8-int8", "smart_stage_offload")),
    ("满血旗舰版", ("Qwen/Qwen-Image-2.1", "none", "auto")),
    ("满血版", ("Qwen/Qwen-Image-2.1-Turbo", "none", "auto")),
    ("INT8", ("Qwen/Qwen-Image-2.1-Turbo", "int8-torchao", "auto")),
]
eff_model_id, eff_quant_mode, eff_offload_mode = MODEL_ID, QUANT_MODE, "auto"
for key, cfg in EDITION_MAP:
    if key in EDITION:
        eff_model_id, eff_quant_mode, eff_offload_mode = cfg
        break

eff_swap_gb = "16" if "16GB" in SWAP_OPTION else ("8" if "8GB" in SWAP_OPTION else "0")

os.environ["QWEN_MODEL_ID"] = eff_model_id
os.environ["QWEN_GPU_PROFILE"] = GPU_PROFILE
os.environ["QWEN_QUANT_MODE"] = eff_quant_mode
os.environ["QWEN_OFFLOAD"] = eff_offload_mode
os.environ["QWEN_PRELOAD_MODEL"] = "1" if PRELOAD_MODEL else "0"
os.environ["QWEN_ENABLE_SWAP_GB"] = eff_swap_gb
os.environ["DEMO_MODE"] = "1" if DEMO_MODE else "0"
os.environ["QWEN_AI_AUTO_RUN"] = "1" if AI_AUTO_RUN else "0"

if AI_ENDPOINT.strip():
    os.environ["QWEN_AI_ENDPOINT"] = AI_ENDPOINT.strip()
if AI_API_KEY.strip():
    os.environ["QWEN_AI_API_KEY"] = AI_API_KEY.strip()
if AI_MODEL.strip():
    os.environ["QWEN_AI_MODEL"] = AI_MODEL.strip()
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN.strip()

# 3) 自选挂载 Google 云盘（支持作品库一键保存到云盘，或直接绑定为默认输出目录）
if MOUNT_GDRIVE:
    with tqdm(total=2, desc="挂载 Google 云盘", unit="步") as gbar:
        from google.colab import drive
        drive.mount("/content/drive")
        gbar.update(1)
        out_dir = (GDRIVE_DIR or "").strip() or "/content/drive/MyDrive/qwen_image_21_outputs"
        os.makedirs(out_dir, exist_ok=True)
        os.environ["QWEN_GDRIVE_DIR"] = out_dir
        if GDRIVE_DIRECT_OUTPUT:
            os.environ["QWEN_OUTPUT_DIR"] = out_dir
        gbar.update(1)
    if GDRIVE_DIRECT_OUTPUT:
        print(f"Google 云盘已挂载并直接设为默认作品目录: {{out_dir}}")
    else:
        print(f"Google 云盘已挂载，可在 WebUI 作品库中一键保存至: {{out_dir}}")

# 4) 启动统一服务与公网穿透隧道（含残留清理、自动重试与二维码卡片）
import colab_launch
importlib.reload(colab_launch)
colab_launch.launch_in_colab()
'''

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header-md"},
                "source": [
                    "# Qwen-Image-2.1 Studio · Google Colab 工作台\n",
                    "\n",
                    "> **说明**：本项目代码由 AI 辅助构建，云端轻量化部署理念参考 [wy51ai/easycolab](https://github.com/wy51ai/easycolab)。\n",
                    "> **GitHub 仓库**：[tangxintx/qwen-image-21-int8-colab](https://github.com/tangxintx/qwen-image-21-int8-colab)\n",
                    "\n",
                    "每次运行第 2 步会自动同步最新代码并保留本地作品、LoRA 与 AI 配置：\n",
                    "- **默认启动下载 Quantized Q8 版**：`abenzerps/Qwen-Image-2.1-Turbo-Quantized` Q8_0 + BF16 编码器 + BF16 VAE，L4 自动逐层智能调度防爆显存\n",
                    "- **智能增量依赖安装**：Cell 1 自动检测已有环境，重复运行数秒完成\n",
                    "- **Google 云盘自选挂载 & 作品库一键备份**：勾选 `MOUNT_GDRIVE` 后，可在网页作品库中一键保存到 Google 云盘\n",
                    "- **四大创作模式固定悬浮切换**：文生图 / 图生图（`@0..@9` 多图引用） / 高清修复 / 多图涂抹局部重绘（支持非标比例透明扩图还原与前后对比滑块）\n",
                    "- **五档 GPU 自动适配**：`RTX PRO 6000 (96GB)` / `H100 (80GB)` / `A100 (80G/40G)` / `L4 (24GB)` / `T4 (16GB)`",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"id": "cell-1-setup"},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in cell1_source.splitlines()],
            },
            {
                "cell_type": "code",
                "metadata": {"id": "cell-2-launch"},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in cell2_source.splitlines()],
            },
        ],
    }

    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated GitHub-synced notebook: {OUT_NB} ({OUT_NB.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
