# Qwen-Image-2.1 Studio · Google Colab & 跨端网页创作台

> **声明**：本项目代码由 AI 辅助生成，云端轻量化部署思路参考 [wy51ai/easycolab](https://github.com/wy51ai/easycolab)。

专为 **Google Colab** 与 **桌面/手机端浏览器** 打造的 `Qwen-Image-2.1`（FP8 / INT8 / GGUF Q8·Q4 / 原生 BF16）图像生成与编辑工作台。采用克制清爽的**柔和浅色工作室界面**（支持一键切换暗色模式），桌面端提供双栏工作台布局，手机端自适应单栏触控体验。

---

## 一、核心功能一览

### 1. 四大创作模式与画质增强
- **文生图 (T2I)**：支持 `Qwen/Qwen-Image-2.1-Turbo`（8 步极速）、`Qwen/Qwen-Image-2.1`（40 步精绘）、预量化 8-bit 权重，以及 [`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) 的 **Q8 满血版** 与 **Q4 轻量版**。
- **图生图 (I2I)**：支持主图 `@0` + 最多 9 张次图 `@1..@9` 多图引用（自动将 `@0` / `@1,2` 映射为模型可理解的 `图1` / `Picture 1` 语法），配合 `Denoise` 精细控制重绘幅度。
- **高清修复 (Hires Fix)**：支持 `1.0x ~ 2.5x` 倍率放大与二次精绘去噪，自动开启 VAE Tiling / Slicing 防爆显存。
- **多图涂抹局部重绘 (Inpaint)**：支持一次选择多张图并独立涂抹遮罩（画笔 / 橡皮 / 笔刷大小 / 15 步撤销 / 防误触锁屏 / 边缘羽化），可选开启**「局部重绘区域加速」**仅裁剪重绘区周边推理后无缝回贴原图。
- **非标比例自动透明扩图 & 还原裁剪**：上传非模型预设比例（`1:1 / 16:9 / 9:16 / 4:3 / 3:4 / 3:2 / 2:3`）的图片时，自动用透明 RGBA 图层扩展至最接近的支持比例推理，出图后自动裁剪还原为原图比例。

### 2. 舒适排版与高效率交互体验
- **柔和浅色 / 中性暗色双主题**：默认明亮清爽的浅色工作室风格，右上角一键切换中性暗色主题并本地记忆。
- **桌面双栏 / 手机自适应**：宽度 `>= 900px` 自动切换为「左侧参数控制 + 右侧实时创作画布」双栏布局与顶部居中导航；手机端自动折叠为轻量单栏与底部导航。
- **右下角固定悬浮「开始生成」键 (FAB)**：无论页面滚动到任何位置均可一键触发生成，推理期间平滑展开进度胶囊与停止按钮。
- **拖拽与 `Ctrl+V` 粘贴上传**：支持将图片直接拖入参考区，或在创作台任意位置 `Ctrl+V` 粘贴剪贴板截图（文生图模式下粘贴会自动切换至图生图并挂载）。
- **参数本地记忆 & 提示词历史/收藏**：自动保存常用参数至浏览器 `localStorage`，记录最近 20 条提示词并支持一键收藏常用模板。
- **前后对比滑块 & 一键复制参数**：在图生图、高清修复、局部重绘的成品详情中可切换「前后对比」拖动滑块直观对比原图与成品，并支持一键复制提示词或全套生成参数；支持 `Ctrl/Cmd + Enter` 生成、`Esc` 关闭弹窗。

### 3. 内置 AI 助手 & 自选「AI 自行运行代码」
- **多场景独立对话**：创作台（写词/调参/支持可选识图）、Web 终端（环境体检/写命令）、生图报错（一键打包日志诊断）三场景历史隔离、互不串台。
- **自选 AI 自动运行代码（默认关闭）**：
  - 用户可在 **Colab 第 2 步 (`AI_AUTO_RUN`)**、**设置 -> AI 助手配置** 或 **AI 弹窗顶部开关** 按需开启。
  - 开启后，AI 在 JSON 中返回的 `terminal_commands` 会在终端顺序自动执行，实时展示控制台输出并将 `stdout / stderr / 退出码` 回传给 AI 进行多步排查与修复（单次上限 5 轮，随时可点「停止自动执行」立即终止）。
  - **内置高危命令安全护栏**：自动拦截 `rm -rf /`、磁盘格式化 (`mkfs`/`fdisk`)、`dd of=/dev/...`、Fork Bomb、关机重启、`curl ... | bash` 等高危指令，命中时强制暂停并交由用户手动决定「仍要手动执行」或「跳过」。

### 4. 五档 Colab GPU 自适应 & LoRA / ComfyUI / 监控
- **五档硬件自动适配**（[gpu_profiles.py](gpu_profiles.py)）：
  - `G4 · RTX PRO 6000 (96GB, sm_120)` / `H100 (80GB, sm_90)` / `A100 (80GB/40GB, sm_80)`：原生 BF16 + TF32 全驻留。
  - `L4 (24GB, sm_89)`：默认推荐 **`Qwen/Qwen-Image-2.1-Turbo`（TorchAO FP8 DiT + INT8 编码器 + BF16 VAE，全显存常驻，禁用 `attention_slicing`）**，支持 Prompt 编码结果 LRU 缓存；使用 BF16 编码器时自动开启真分阶段调度（仅将 TE 卸载至 CPU）。
  - `T4 (16GB, sm_75)`：FP16 + Model CPU Offload + VAE/注意力切片防 OOM。
- **LoRA 管理**：支持 Hugging Face / Civitai 页面链接自动解析直链下载、`.safetensors` 文件头与 NaN/Inf 校验、Kohya `lora_unet_` 层级还原与 `.alpha` 缩放自动折算，支持画风模式与参考图控制模式。
- **按需 ComfyUI + INT8-ConvRot / GGUF 插件**：在设置页手动按需一键部署 ComfyUI，支持原生 **`INT8-ConvRot`**（7.26 GB DiT + 9.35 GB TE，最高效 L4 工作流）及 [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) Q4/Q8 工作流，并按显卡显存自动注入 `--normalvram` / `--lowvram` 与端口冲突自适应。
- **资源监控、Web 终端与一键关闭**：实时监测 VRAM / RAM / Swap / 磁盘占用、已加载与缓存模型清单、服务日志流、ZIP 打包导出，支持手机端一键卸载模型关闭服务并可选调用 `google.colab.runtime.unassign()` 释放 Colab 计算单元。

---

## 二、快速启动

### 方式一：Google Colab 一键运行（推荐）
1. 在 Google Colab 中打开仓库内的 [`Qwen_Image_21_INT8_Colab.ipynb`](Qwen_Image_21_INT8_Colab.ipynb)（推荐在「代码执行程序 -> 更改运行时类型」选择 **L4** 或更高 GPU）。
2. 运行 **Cell 1**：检查 GPU 状态并执行**智能增量依赖安装**（重复运行时会自动跳过已安装的 `diffusers` / `transformers` / `torchao` 等核心包）。
3. 配置 **Cell 2** 表单参数（选择模型方案、可选填 `HF_TOKEN` / `AI_API_KEY` / 是否开启 `AI_AUTO_RUN` / 是否挂载 Google Drive），运行后将自动同步 GitHub 最新代码、建立 Cloudflare 公网隧道并渲染**手机扫码二维码**。

### 方式二：本地运行 / 无 GPU 演示模式
```bash
pip install -r requirements.txt

# 本地无 GPU 演示模式（不建公网隧道，用于界面体验与开发测试）
python colab_launch.py --local --demo

# 本地有 CUDA GPU 直接运行
python colab_launch.py --local

# 指定端口并默认开启 AI 自动运行代码模式
python colab_launch.py --local --port 7860 --ai-auto-run
```
启动后浏览器访问 `http://127.0.0.1:7860`。

---

## 三、程序架构与文件结构

```text
qwen-image-21-int8-colab/
├── Qwen_Image_21_INT8_Colab.ipynb   # Google Colab 两步一键启动笔记本（由 build_colab_notebook.py 生成）
├── build_colab_notebook.py          # 自动构建并同步更新 .ipynb 的打包脚本
├── colab_launch.py                  # 统一启动器（端口/残留进程清理、Swap、HF 预下载、Cloudflare 隧道重试与二维码）
├── server.py                        # FastAPI 后端服务、QwenImage21Engine 推理引擎、作品库管理与 AI 代理接口
├── gpu_profiles.py                  # 五档 Colab GPU 自动识别与 dtype / offload / VAE 切片策略表
├── hf_downloader.py                 # Hugging Face API 极速下载器（hf_transfer + GGUF Q8/Q4 定向切片拉取）
├── lora_manager.py                  # Civitai / HF LoRA 链接解析下载、safetensors 文件头校验与前缀重映射
├── terminal_manager.py              # 持久化工作目录 Web 终端、日志环形缓冲与 AI 高危命令安全护栏
├── swap_manager.py                  # Linux Swap 虚拟内存 4MB 预探与 OverlayFS loopback 自动挂载工具
├── comfy_launcher.py                # 按需 ComfyUI + city96/ComfyUI-GGUF 扩展部署与 Q8/Q4 工作流生成器
├── static/
│   ├── index.html                   # 语义化前端结构（双端导航、双栏创作台、右下角悬浮 FAB、对比 Lightbox、AI 弹窗）
│   ├── styles.css                   # 柔和浅色默认主题 + 中性暗色主题变量与响应式排版样式
│   └── app.js                       # 前端交互、串行防重入轮询、本地持久化、前后对比滑块与 AI 自动运行闭环
├── test_server.py                   # 自动化冒烟与集成测试脚本（含 GPU 档位、GGUF、安全护栏、对比图、Notebook 校验）
├── requirements.txt                 # Python 依赖清单
└── README.md                        # 本自述与使用注意点文档
```

### 核心数据流与目录说明
- **生成作品存储**：默认位于 `outputs/`（若 Colab 开启 `MOUNT_GDRIVE`，则绑定至 `/content/drive/MyDrive/qwen_image_21_outputs`）；`gallery.json` 记录作品索引，`outputs/.sources/` 存储图生图/高清/重绘的轻量化原图缩略图以支持跨刷新前后对比，删除作品时会自动连带清理。
- **LoRA 目录**：Colab 环境默认 `/content/loras`，本地环境默认 `<项目目录>/loras`（可通过 `QWEN_LORAS_DIR` 覆盖）。
- **AI 配置文件**：保存在项目根目录 `.ai_config.json`（已加入 `.gitignore`，同步 GitHub 代码时保留、不会上传提交）。

---

## 四、使用注意点与避坑指南

1. **硬件与方案选型建议**
   - **T4 (16GB)**：推荐使用 `Turbo INT8 通用版` 或 `GGUF Q4 轻量版`，输出分辨率建议保持 `1K` 档位；不要在 T4 上强制关闭 CPU Offload 运行原生 BF16。
   - **L4 (24GB)**：默认 **`Turbo FP8 + INT8`（FP8 DiT + INT8 编码器 + BF16 VAE，全显存常驻无 PCIe 往返，禁用 `attention_slicing`）**，或 `addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit` / `GGUF Q8 + INT8 编码`；追求极速推荐先生成 `1K`（`1344×768`）再高清放大（`1.5K 16:9` 原生为 `1920×1088` 像素近翻倍）。
   - **A100 (80GB) / H100 / RTX PRO 6000**：可直接选择原生 BF16 不量化满血版，开启 `1.5K / 2K` 直出与多图连抽。
2. **隐私、鉴权与 API 密钥安全**
   - `/api/stats` 仅向前端返回 `has_api_key` 与脱敏掩码（如 `sk-a***9f2b`），绝不回传完整明文密钥。
   - **一次性管理令牌保护**：Web 终端（`/api/terminal/*`）、系统运维（`/api/system/*`）、配置修改（`/api/config`）、删除、LoRA 下载与关机接口均受随机生成的 **Bearer Token (`QWEN_ADMIN_TOKEN`)** 鉴权；通过 Colab 输出的 `#token=...` 链接/二维码打开会自动在本地保存并清除地址栏令牌，且手工及 AI 终端命令统一经过高危指令安全护栏校验。
3. **自选「AI 自动运行代码」安全边界**
   - 该功能默认**关闭**，仅在您明确信任所用大模型并需要自动排查依赖/环境问题时开启。
   - 虽然内置了高危命令正则护栏与 5 轮上限 / 60 秒单命令超时保护，但仍建议在运行期间留意 AI 弹窗中的实时输出，随时可点击「停止自动执行」强行中断。
4. **ComfyUI 与主 WebUI 显存互斥**
   - 在 L4 / T4 等中小显存显卡上，若准备在设置页启动按需 ComfyUI，建议先在「系统监控」中点击 **「卸载当前模型」** 释放显存，避免两个推理进程同时驻留触发 OOM。
5. **LoRA 下载与 Civitai 拦截**
   - 部分 Civitai 模型需要登录权限才能下载，若未填写 API Token 会返回 HTML 登录页。本程序已内置下载前/后校验，会直接拦截并提示填写 Civitai API Token。
6. **Colab 磁盘空间与断连保护**
   - 免费/标准 Colab 容器磁盘有限（约 70~100 GB），同时缓存多个全精度大模型容易写满磁盘；切换大底模前可在「系统监控」中点击「清空 HF 磁盘缓存」释放空间。
   - 重要作品建议开启 `MOUNT_GDRIVE` 或定期在作品库点击「导出 ZIP」备份到本地；用完后务必点击右上角「关闭」并勾选断开运行时以节省计算单元。
7. **Windows 本地开发与测试提示**
   - Windows 环境默认关闭 Cloudflare 隧道与 Linux Swap 配置，仅用于 `--local --demo` 前端预览或本机 CUDA 推理。
   - 修改 `build_colab_notebook.py` 后务必重新运行 `python build_colab_notebook.py` 以同步刷新 `Qwen_Image_21_INT8_Colab.ipynb`。

---

## 五、测试与校验

运行内置自动化测试套件（覆盖文生图/图生图/高清/局部重绘、非标比例透明扩图还原、加速裁剪、5 档 GPU 适配、GGUF Q8/Q4 下载解析、ComfyUI 工作流生成、API Key 脱敏、AI 高危命令护栏、前后对比图清理与 Notebook 完整性检查）：

```bash
python test_server.py
```
