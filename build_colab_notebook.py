"""
Build a self-contained Google Colab notebook (`Qwen_Image_21_INT8_Colab.ipynb`)
with high-speed Hugging Face API download (`hf_transfer` + `HF_TOKEN`),
native in-notebook tqdm progress bars, and direct QR code rendering.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_NB = BASE_DIR / "Qwen_Image_21_INT8_Colab.ipynb"

BUNDLE_FILES = [
    "colab_launch.py",
    "hf_downloader.py",
    "swap_manager.py",
    "lora_manager.py",
    "comfy_launcher.py",
    "server.py",
    "gpu_profiles.py",
    "terminal_manager.py",
    "requirements.txt",
    "static/index.html",
    "static/styles.css",
    "static/app.js",
]


def build_bundle_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in BUNDLE_FILES:
            fpath = BASE_DIR / rel
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required file for notebook bundle: {fpath}")
            zf.write(fpath, arcname=rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    bundle_b64 = build_bundle_b64()
    chunks = [bundle_b64[i : i + 100] for i in range(0, len(bundle_b64), 100)]
    b64_python_literal = "(\n    " + "\n    ".join(f'"{c}"' for c in chunks) + "\n)"

    cell1_source = '''#@title 1️⃣ 检查 GPU 并安装依赖环境（含 hf_transfer 极速下载与分步进度条）
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"], check=False)
from tqdm.auto import tqdm

steps = [
    ("检查 GPU 硬件状态", ["nvidia-smi"]),
    ("更新 pip 包管理器", [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "pip"]),
    ("安装 Diffusers (Qwen-Image-2.1 支持)", [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "git+https://github.com/huggingface/diffusers.git"]),
    ("安装 Transformers / TorchAO INT8 / HF 极速下载", [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "transformers>=5.17.0", "accelerate", "safetensors", "torchao>=0.14.0", "bitsandbytes", "huggingface_hub", "hf_transfer"]),
    ("安装 Web 服务与二维码组件", [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "fastapi", "uvicorn[standard]", "python-multipart", "psutil", "qrcode", "pillow"]),
]

pbar = tqdm(total=len(steps), desc="环境初始化", unit="步")
for label, cmd in steps:
    pbar.set_description(f"正在: {label}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cmd[0] == "nvidia-smi" and res.stdout:
        print(res.stdout)
    elif res.returncode != 0:
        print(f"[警告] {label} 返回码 {res.returncode}:\\n{res.stdout[-800:]}")
    pbar.update(1)
pbar.set_description("环境安装完成")
pbar.close()
print("全部依赖安装完成，请运行第 2 步！")
'''

    cell2_source = f'''#@title 2️⃣ 配置模型、HF/AI Token 与 GPU 档位并启动（含实时下载进度条与二维码）
import base64
import importlib
import io
import os
import sys
import zipfile
from pathlib import Path
from tqdm.auto import tqdm

EDITION = "L4 24GB 黄金方案 · Turbo FP8 DiT + INT8 编码器 + BF16 VAE (推荐 L4)"  #@param ["L4 24GB 黄金方案 · Turbo FP8 DiT + INT8 编码器 + BF16 VAE (推荐 L4)", "L4 24GB 精绘方案 · Qwen-2.1 FP8 DiT + INT8 编码器 + BF16 VAE 40步", "INT8 轻量省显存版 (Turbo · 推荐 T4/L4)", "满血版 · Turbo 原生BF16/FP16不量化 (适合 A100/H100/G4)", "满血旗舰版 · Qwen-2.1 原生BF16/FP16 40步 (适合 A100-80G/H100/G4)", "自定义下方模型与量化参数"]
MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"  #@param ["Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1", "addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit"] {{allow-input: true}}
HF_TOKEN = ""  #@param {{type:"string"}}
AI_API_KEY = ""  #@param {{type:"string"}}
AI_MODEL = "qwen3.8-27b"  #@param {{type:"string"}}
GPU_PROFILE = "auto"  #@param ["auto", "g4_rtx_pro_6000", "h100_sxm", "a100_sxm", "l4_ada", "t4_turing"]
QUANT_MODE = "fp8-int8"  #@param ["fp8-int8", "fp8-all", "int8-torchao", "int8-bnb", "none"]
SWAP_OPTION = "跳过 (默认关闭)"  #@param ["跳过 (默认关闭)", "尝试开启 8GB Swap (仅支持虚拟机)", "尝试开启 16GB Swap (仅支持虚拟机)"]
PRELOAD_MODEL = True  #@param {{type:"boolean"}}
MOUNT_GDRIVE = False  #@param {{type:"boolean"}}
DEMO_MODE = False  #@param {{type:"boolean"}}

APP_BUNDLE_B64 = {b64_python_literal}

# 1) 自动释放/同步最新内置项目代码到 /content/qwen_studio（保留已有 outputs 作品目录与本地 .ai_config.json）
work_dir = Path("/content/qwen_studio")
work_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(APP_BUNDLE_B64))) as zf:
    members = zf.infolist()
    for info in tqdm(members, desc="释放/更新项目文件", unit="文件"):
        zf.extract(info, work_dir)
print(f"项目文件已同步至: {{work_dir}}")

os.chdir(work_dir)
if str(work_dir) not in sys.path:
    sys.path.insert(0, str(work_dir))

eff_offload_mode = "auto"
if "L4 24GB 黄金方案" in EDITION:
    eff_model_id = "Qwen/Qwen-Image-2.1-Turbo"
    eff_quant_mode = "fp8-int8"
    eff_offload_mode = "smart_stage_offload"
elif "L4 24GB 精绘方案" in EDITION:
    eff_model_id = "Qwen/Qwen-Image-2.1"
    eff_quant_mode = "fp8-int8"
    eff_offload_mode = "smart_stage_offload"
elif "满血旗舰版" in EDITION:
    eff_model_id = "Qwen/Qwen-Image-2.1"
    eff_quant_mode = "none"
elif "满血版" in EDITION:
    eff_model_id = "Qwen/Qwen-Image-2.1-Turbo"
    eff_quant_mode = "none"
elif "INT8" in EDITION:
    eff_model_id = "Qwen/Qwen-Image-2.1-Turbo"
    eff_quant_mode = "int8-torchao"
else:
    eff_model_id = MODEL_ID
    eff_quant_mode = QUANT_MODE

if "16GB" in SWAP_OPTION:
    eff_swap_gb = "16"
elif "8GB" in SWAP_OPTION:
    eff_swap_gb = "8"
else:
    eff_swap_gb = "0"

os.environ["QWEN_MODEL_ID"] = eff_model_id
os.environ["QWEN_GPU_PROFILE"] = GPU_PROFILE
os.environ["QWEN_QUANT_MODE"] = eff_quant_mode
os.environ["QWEN_OFFLOAD"] = eff_offload_mode
os.environ["QWEN_PRELOAD_MODEL"] = "1" if PRELOAD_MODEL else "0"
os.environ["QWEN_ENABLE_SWAP_GB"] = eff_swap_gb
os.environ["DEMO_MODE"] = "1" if DEMO_MODE else "0"
if AI_API_KEY.strip():
    os.environ["QWEN_AI_API_KEY"] = AI_API_KEY.strip()
if AI_MODEL.strip():
    os.environ["QWEN_AI_MODEL"] = AI_MODEL.strip()
print(f"[版本] {{EDITION}} -> 模型={{eff_model_id}}, 量化={{eff_quant_mode}}, 调度={{eff_offload_mode}}, Swap={{eff_swap_gb}}GB")
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN.strip()

# 2) 可选挂载 Google Drive（带进度条）
if MOUNT_GDRIVE:
    with tqdm(total=2, desc="挂载 Google Drive", unit="步") as gbar:
        from google.colab import drive
        drive.mount("/content/drive")
        gbar.update(1)
        out_dir = "/content/drive/MyDrive/qwen_image_21_outputs"
        os.makedirs(out_dir, exist_ok=True)
        os.environ["QWEN_OUTPUT_DIR"] = out_dir
        gbar.update(1)
    print(f"作品库已绑定至 Google Drive: {{out_dir}}")

# 3) 直接在当前 Notebook 进程中启动（自动清理残留端口、渲染 tqdm 进度条与手机扫码二维码）
import colab_launch
importlib.reload(colab_launch)
colab_launch.launch_in_colab()
'''

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "header-md"},
                "source": [
                    "# Qwen-Image-2.1 INT8 · 手机网页创作台 (Google Colab)\n",
                    "\n",
                    "> **说明**：本项目代码由 AI 生成，交互设计与云端部署理念参考 [wy51ai/easycolab](https://github.com/wy51ai/easycolab)。\n",
                    "\n",
                    "单文件开箱即用，每一步均带实时进度条：\n",
                    "- **HF API 极速下载**：内置 `hf_transfer` + `HF_TOKEN` 支持，预下载显示实时 `GB / MB/s` 进度条\n",
                    "- **GPU 硬件自适应**：`G4 · RTX PRO 6000 (96GB)` / `H100 · Hopper SXM (80GB)` / `A100 · Ampere SXM (80GB)` / `L4 · Ada Lovelace (24GB)` / `T4 · Turing (16GB)`\n",
                    "- **四大创作模式**：文生图 / 图生图（多图 `@0,@1`） / 高清修复 / 涂抹局部重绘（支持非标准比例透明图层自动扩图与裁剪还原）\n",
                    "- **内置 AI 助手 & LoRA 管理 & 手机端 Web 终端 & 资源监控 & 一键关闭**",
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
    print(f"Generated self-contained notebook: {OUT_NB} ({OUT_NB.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
