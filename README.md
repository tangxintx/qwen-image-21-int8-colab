# Qwen-Image-2.1 INT8 · Google Colab 手机网页创作台

> **说明**：本项目代码由 **AI 生成**，轻量化云端交互与部署思路参考 [wy51ai/easycolab](https://github.com/wy51ai/easycolab)。

专为 **Google Colab** 与 **手机端浏览器** 打造的 `Qwen-Image-2.1`（FP8 / INT8 / 满血原生）图像生成与编辑 Web 应用。

## 核心功能

1. **四大创作模式 & 非标准比例自适应**
   - **文生图 (T2I)**：支持 `Qwen/Qwen-Image-2.1-Turbo`（8 步极速）、`Qwen/Qwen-Image-2.1`（40 步精绘）、预量化 8bit 权重，以及 [`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) 的 **Q8 满血版**（`Q8_0` DiT + 全精度 BF16 文本编码器 + `qwen_image_2.1_vae_bf16.safetensors`）与 **Q4 轻量版**（`Q4_K_M` DiT + `Qwen3-VL-8B` `Q4_K_M.gguf` + `qwen_image_2.1_vae_bf16.safetensors`）。
   - **图生图 (I2I)**：支持主图 `@0` + 最多 9 张次图 `@1..@9` 多图引用，结合 `Denoise` 控制重绘幅度。
   - **高清修复 (Hires Fix)**：支持 `1.0x ~ 2.5x` 倍率放大与二次精绘去噪，自动启用 VAE Tiling/Slicing 防爆显存。
   - **局部重绘 (Inpaint)**：内置手机触控涂抹遮罩画布（画笔、橡皮、笔刷粗细、撤销、防误触锁屏、羽化融合）。
   - **自动透明扩图 & 裁剪还原**：上传非模型预设比例图片时，自动使用透明图层扩图至支持比例并弹出提醒，生成完成后自动裁剪还原为原图比例。

2. **内置 AI 智能助手（支持 OpenAI 兼容接口）**
   - **创作台 AI 帮写/调参**：一键撰写/润色提示词，并支持一键应用分辨率、步数、CFG、LoRA 与系统设置修改。
   - **终端「询问AI」**：结合当前目录与命令行输出解答问题，支持将 AI 建议命令一键发送到终端执行。
   - **生图报错「一键询问 AI」**：生成报错时自动打包报错信息、当前参数与服务日志，快速定位并给出修复方案。

3. **LoRA 下载与管理**
   - 支持 Hugging Face、Civitai 模型页/直链自动解析与校验下载（拦截 HTML 错误页），支持损坏检测与一键删除。

4. **五档 Colab GPU 自动适配**（`gpu_profiles.py`）
   - **G4 · RTX PRO 6000 (96GB, Blackwell `sm_120`)**：BF16 + TF32 全显存驻留。
   - **H100 · Hopper SXM (80GB, `sm_90`)**：BF16 + TF32 全显存驻留。
   - **A100 · Ampere SXM (80GB/40GB, `sm_80`)**：BF16 + TF32。
   - **L4 · Ada Lovelace (24GB, `sm_89`)**：FP8 DiT + INT8 编码器 + BF16 VAE + 智能分阶段显存调度。
   - **T4 · Turing (16GB, `sm_75`)**：FP16 + Model CPU Offload + VAE/注意力切片。

5. **按需 ComfyUI + GGUF 加载器插件 (`city96/ComfyUI-GGUF`)**
   - 在「设置 -> ComfyUI 启动器」中可一键按需安装 ComfyUI 与 [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) 扩展插件，自动按所选档位（Q4 / Q8 / 全部）高速拉取并挂载 DiT `.gguf`、文本编码器与 `qwen_image_2.1_vae_bf16.safetensors`，同步生成即开即用的 ComfyUI 工作流（`Qwen_Image_2.1_Uncensored_GGUF_Q8.json` / `Qwen_Image_2.1_Uncensored_GGUF_Q4.json`）。

6. **监控、Web 终端、作品库与一键关闭**
   - **资源与模型监控**：实时监控 VRAM、RAM、Swap、磁盘占用，并展示当前已加载模型/LoRA、本地 HF 缓存模型与 LoRA 列表。
   - **Web 终端 & 作品库**：持久 `cd` 工作目录、服务日志流、作品 ZIP 打包、Lightbox 大图与参数一键复用。
   - **一键关闭**：支持在手机端一键卸载模型、关闭隧道与服务，并可选调用 `google.colab.runtime.unassign()` 释放 Colab 计算单元。

## 快速启动

### 方式一：Google Colab 一键运行
上传并打开仓库中的 [`Qwen_Image_21_INT8_Colab.ipynb`](Qwen_Image_21_INT8_Colab.ipynb)，依次运行第 1 步（安装依赖）和第 2 步（选择版本与启动），扫码或点击生成的 `trycloudflare.com` 链接即可在手机端使用。

### 方式二：统一启动脚本（Colab / 本地均适用）
```bash
pip install -r requirements.txt

# 本地无 GPU 演示模式（不启动公网隧道）
python colab_launch.py --local --demo

# 本地有 GPU 直接运行
python colab_launch.py --local

# Colab / Linux 环境下启动并生成 Cloudflare 公网隧道与二维码
python colab_launch.py
```

启动后本地默认访问 `http://127.0.0.1:7860`。

## 文件结构

- `Qwen_Image_21_INT8_Colab.ipynb` — Google Colab 自包含一键运行笔记本
- `build_colab_notebook.py` — 打包构建 `.ipynb` 的脚本
- `colab_launch.py` — 统一启动器（端口残留清理、`cloudflared` 隧道、Colab 二维码与本地 `--local/--demo` 模式）
- `server.py` — FastAPI 后端、`QwenImage21Pipeline` 推理引擎、透明扩图裁剪与 AI 接口代理
- `lora_manager.py` — Civitai / HF LoRA 解析下载、header 校验与前缀重映射
- `gpu_profiles.py` — 五档 Colab GPU 自动检测与显存/精度优化配置
- `hf_downloader.py` — Hugging Face API 极速下载器（`hf_transfer` + GGUF Q8/Q4 定向组件下载）
- `terminal_manager.py` — 持久化工作目录 Web 终端与实时服务日志缓冲
- `swap_manager.py` — Linux Swap 虚拟内存配置工具
- `comfy_launcher.py` — 按需手动启动 ComfyUI 辅助启动器（内置 `city96/ComfyUI-GGUF` 与工作流生成）
- `static/` — 移动优先暗色毛玻璃 Web 前端（`index.html`, `styles.css`, `app.js`）
- `test_server.py` — 自动化接口与引擎冒烟测试脚本

## 致谢与声明
- **说明**：本项目全部代码均由 AI 生成。
- **参考项目**：[https://github.com/wy51ai/easycolab](https://github.com/wy51ai/easycolab)
