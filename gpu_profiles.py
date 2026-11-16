"""
GPU Hardware Auto-Detection & Optimization Profiles for Google Colab.
Supports:
  - G4 · NVIDIA RTX PRO 6000 Blackwell (96GB)
  - H100 · Hopper SXM (80GB)
  - A100 · Ampere SXM (80GB / 40GB)
  - L4 · Ada Lovelace (24GB)
  - T4 · Turing (16GB)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GPUProfile:
    key: str
    display_name: str
    arch: str
    vram_gb: int
    compute_cap: str
    dtype: str  # "bfloat16" | "float16"
    offload_mode: str  # "none" | "model_cpu_offload" | "sequential_cpu_offload"
    vae_tiling: bool
    vae_slicing: bool
    attention_slicing: bool
    allow_tf32: bool
    recommended_tier: str  # "1k" | "1.5k" | "2k"
    recommended_batch: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GPU_PROFILES: dict[str, GPUProfile] = {
    "g4_rtx_pro_6000": GPUProfile(
        key="g4_rtx_pro_6000",
        display_name="G4 · RTX PRO 6000 (96GB)",
        arch="Blackwell (sm_120)",
        vram_gb=96,
        compute_cap="12.0",
        dtype="bfloat16",
        offload_mode="none",
        vae_tiling=False,
        vae_slicing=False,
        attention_slicing=False,
        allow_tf32=True,
        recommended_tier="2k",
        recommended_batch=4,
        notes="96GB 超大显存全驻留，原生 BF16 + TF32，2K 直出与多图连抽无需卸载",
    ),
    "h100_sxm": GPUProfile(
        key="h100_sxm",
        display_name="H100 · Hopper SXM (80GB)",
        arch="Hopper (sm_90)",
        vram_gb=80,
        compute_cap="9.0",
        dtype="bfloat16",
        offload_mode="none",
        vae_tiling=False,
        vae_slicing=False,
        attention_slicing=False,
        allow_tf32=True,
        recommended_tier="2k",
        recommended_batch=4,
        notes="80GB HBM3 全驻留，开启 TF32 与原生 BF16，极速 2K 推理",
    ),
    "a100_sxm": GPUProfile(
        key="a100_sxm",
        display_name="A100 · Ampere SXM (80GB)",
        arch="Ampere (sm_80)",
        vram_gb=80,
        compute_cap="8.0",
        dtype="bfloat16",
        offload_mode="none",
        vae_tiling=False,
        vae_slicing=False,
        attention_slicing=False,
        allow_tf32=True,
        recommended_tier="2k",
        recommended_batch=4,
        notes="80GB HBM2e 全驻留（40GB 版自动开启 VAE 分块），BF16 + TF32 高效运行",
    ),
    "l4_ada": GPUProfile(
        key="l4_ada",
        display_name="L4 · Ada Lovelace (24GB)",
        arch="Ada Lovelace (sm_89)",
        vram_gb=24,
        compute_cap="8.9",
        dtype="bfloat16",
        offload_mode="smart_stage_offload",
        vae_tiling=True,
        vae_slicing=True,
        attention_slicing=False,
        allow_tf32=True,
        recommended_tier="1.5k",
        recommended_batch=2,
        notes="24GB + 53GB 内存推荐 FP8 DiT + INT8 编码器 + BF16 VAE，智能分阶段卸载峰值约 11-13GB",
    ),
    "t4_turing": GPUProfile(
        key="t4_turing",
        display_name="T4 · Turing (16GB)",
        arch="Turing (sm_75)",
        vram_gb=16,
        compute_cap="7.5",
        dtype="float16",
        offload_mode="model_cpu_offload",
        vae_tiling=True,
        vae_slicing=True,
        attention_slicing=True,
        allow_tf32=False,
        recommended_tier="1k",
        recommended_batch=1,
        notes="16GB 显存启用 FP16 张量核心 + Model CPU Offload + VAE/注意力切片防 OOM",
    ),
}


def match_gpu_profile(device_name: str, total_vram_gb: float, cc_major: int = 0, cc_minor: int = 0) -> GPUProfile:
    """Match detected CUDA GPU against the 5 Colab GPU tiers."""
    name_upper = (device_name or "").upper()

    if ("PRO 6000" in name_upper) or ("RTX 6000" in name_upper and total_vram_gb >= 85) or total_vram_gb >= 88:
        return GPU_PROFILES["g4_rtx_pro_6000"]
    if "H100" in name_upper or (cc_major == 9 and total_vram_gb >= 70):
        return GPU_PROFILES["h100_sxm"]
    if "A100" in name_upper or (cc_major == 8 and cc_minor == 0 and total_vram_gb >= 36):
        profile = GPU_PROFILES["a100_sxm"]
        # 40GB A100 variant: enable VAE tiling for safe 2K decoding
        if total_vram_gb < 55:
            return GPUProfile(
                **{
                    **profile.to_dict(),
                    "display_name": "A100 · Ampere (40GB)",
                    "vram_gb": 40,
                    "vae_tiling": True,
                    "vae_slicing": True,
                    "recommended_batch": 2,
                }
            )
        return profile
    if "L4" in name_upper or (cc_major == 8 and cc_minor == 9 and 20 <= total_vram_gb <= 32):
        return GPU_PROFILES["l4_ada"]
    if "T4" in name_upper or (cc_major == 7 and total_vram_gb <= 18) or total_vram_gb <= 18:
        return GPU_PROFILES["t4_turing"]

    # Fallback heuristic by VRAM & compute capability
    if total_vram_gb >= 70:
        return GPU_PROFILES["h100_sxm"]
    if total_vram_gb >= 36:
        return GPU_PROFILES["a100_sxm"]
    if total_vram_gb >= 20:
        return GPU_PROFILES["l4_ada"]
    return GPU_PROFILES["t4_turing"]
