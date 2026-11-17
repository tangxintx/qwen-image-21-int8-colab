"""Smoke tests for Qwen-Image-2.1 INT8 Mobile Web Server (runs in DEMO_MODE)."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import base64
import io

os.environ["DEMO_MODE"] = "1"
_TMP_DIR = tempfile.mkdtemp(prefix="qwen_test_outputs_")
_TMP_LORAS = tempfile.mkdtemp(prefix="qwen_test_loras_")
os.environ["QWEN_OUTPUT_DIR"] = _TMP_DIR
os.environ["QWEN_LORAS_DIR"] = _TMP_LORAS

from PIL import Image, ImageDraw  # noqa: E402

import server  # noqa: E402

server.AI_CONFIG_PATH = server.Path(_TMP_DIR) / ".ai_config.json"


def _make_b64_png(w: int = 256, h: int = 256, color=(80, 120, 200)) -> str:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _make_b64_mask(w: int = 256, h: int = 256) -> str:
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([64, 64, 192, 192], fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _can_use_socketpair(timeout: float = 0.6) -> bool:
    """Detect whether asyncio/TestClient socketpair works in this sandbox."""
    if os.name != "nt":
        return True
    import socket

    ok = [False]

    def _probe() -> None:
        try:
            a, b = socket.socketpair()
            a.close()
            b.close()
            ok[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return ok[0]


def _run_http_tests() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(server.app)

    def _wait_job(job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = client.get(f"/api/jobs/{job_id}")
            assert res.status_code == 200
            data = res.json()
            if data["status"] in {"completed", "error", "cancelled"}:
                return data
            time.sleep(0.08)
        raise TimeoutError(f"Job {job_id} timed out")

    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["ok"] is True and health["demo_mode"] is True

    stats = client.get("/api/stats").json()
    assert "ram" in stats and "swap" in stats and "disk" in stats and "gpu" in stats and "storage" in stats

    r = client.post(
        "/api/generate",
        json={
            "mode": "t2i",
            "prompt": "cyberpunk street neon rain",
            "width": 512,
            "height": 512,
            "steps": 4,
            "denoise": 0.75,
            "seed": 123,
            "batch_size": 2,
        },
    )
    assert r.status_code == 200
    job = _wait_job(r.json()["job_id"])
    assert job["status"] == "completed" and len(job["results"]) == 2
    assert job["results"][0]["seed"] == 123 and job["results"][1]["seed"] == 124

    init_b64 = _make_b64_png(320, 320)
    mask_b64 = _make_b64_mask(320, 320)

    for payload, expected_w in [
        ({"mode": "i2i", "prompt": "把 @1，2 的元素放到 @0 中", "width": 320, "height": 320, "steps": 3, "denoise": 0.65, "seed": 77, "batch_size": 1, "init_image_b64": init_b64, "ref_images_b64": [init_b64, _make_b64_png(256, 384, (200, 90, 90))]}, 320),
        ({"mode": "hires", "prompt": "ultra detailed texture", "width": 320, "height": 320, "steps": 3, "denoise": 0.4, "hires_scale": 1.5, "seed": -1, "batch_size": 1, "init_image_b64": init_b64}, 480),
        ({"mode": "inpaint", "prompt": "red neon hat", "width": 320, "height": 320, "steps": 3, "denoise": 0.85, "seed": 99, "batch_size": 1, "init_image_b64": init_b64, "mask_image_b64": mask_b64}, 320),
    ]:
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200
        j = _wait_job(resp.json()["job_id"])
        assert j["status"] == "completed", j
        assert j["results"][0]["width"] == expected_w

    gal = client.get("/api/gallery").json()
    assert gal["total_all"] == 5

    zip_res = client.get("/api/gallery/export")
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"
    assert int(zip_res.headers.get("content-length", "0")) > 0

    first_id = gal["items"][0]["id"]
    assert client.delete(f"/api/gallery/{first_id}").status_code == 200
    assert client.get("/api/gallery").json()["total_all"] == 4

    _assert_profiles_and_modes(
        lambda body: client.post("/api/config", json=body).json()["stats"]["engine"]
    )

    assert client.get("/api/terminal/cwd").status_code == 200
    cmd_id = client.post("/api/terminal/run", json={"command": "echo hello_easycolab"}).json()["cmd_id"]
    term_out = ""
    for _ in range(60):
        pr = client.get(f"/api/terminal/poll/{cmd_id}?offset=0").json()
        term_out += pr["chunk"]
        if not pr["running"]:
            assert pr["exit_code"] == 0
            break
        time.sleep(0.05)
    assert "hello_easycolab" in term_out

    assert client.post("/api/terminal/run", json={"command": "cd static"}).json()["cwd"].endswith("static")
    chain_id = client.post("/api/terminal/run", json={"command": "cd .. && echo chain_ok"}).json()["cmd_id"]
    chain_out = ""
    for _ in range(40):
        pr = client.get(f"/api/terminal/poll/{chain_id}?offset=0").json()
        if not pr["running"]:
            chain_out = pr["chunk"]
            assert pr["exit_code"] == 0, pr
            break
        time.sleep(0.05)
    assert "chain_ok" in chain_out

    assert isinstance(client.get("/api/logs").json()["lines"], list)
    assert client.post("/api/system/free-vram").status_code == 200
    swap_data = client.post("/api/system/enable-swap", json={"size_gb": 8}).json()
    assert "ok" in swap_data and "message" in swap_data and "swap" in swap_data and "stats" in swap_data


def _run_direct_inprocess_tests() -> None:
    """Run identical endpoint & engine checks without asyncio socketpair when blocked by sandbox."""
    assert (server.STATIC_DIR / "index.html").exists()

    def _wait_job(job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = server.api_get_job(job_id)
            if data["status"] in {"completed", "error", "cancelled"}:
                return data
            time.sleep(0.05)
        raise TimeoutError(f"Job {job_id} timed out")

    health = server.api_health()
    assert health["ok"] is True and health["demo_mode"] is True

    stats = server.api_stats()
    assert "ram" in stats and "swap" in stats and "disk" in stats and "gpu" in stats and "storage" in stats
    assert "uptime_sec" in stats["engine"] and stats["engine"]["uptime_sec"] >= 0

    j_res = server.api_generate(
        server.GenerateRequest(
            mode="t2i",
            prompt="cyberpunk street neon rain",
            width=512,
            height=512,
            steps=4,
            denoise=0.75,
            seed=123,
            batch_size=2,
        )
    )
    job = _wait_job(j_res["job_id"])
    assert job["status"] == "completed" and len(job["results"]) == 2
    assert job["results"][0]["seed"] == 123 and job["results"][1]["seed"] == 124

    init_b64 = _make_b64_png(320, 320)
    mask_b64 = _make_b64_mask(320, 320)

    for req_obj, expected_w in [
        (server.GenerateRequest(mode="i2i", prompt="把 @1，2 的元素放到 @0 中", width=320, height=320, steps=3, denoise=0.65, seed=77, batch_size=1, init_image_b64=init_b64, ref_images_b64=[init_b64, _make_b64_png(256, 384, (200, 90, 90))]), 320),
        (server.GenerateRequest(mode="hires", prompt="ultra detailed texture", width=320, height=320, steps=3, denoise=0.4, hires_scale=1.5, seed=-1, batch_size=1, init_image_b64=init_b64), 480),
        (server.GenerateRequest(mode="inpaint", prompt="red neon hat", width=320, height=320, steps=3, denoise=0.85, seed=99, batch_size=1, init_image_b64=init_b64, mask_image_b64=mask_b64), 320),
    ]:
        j = _wait_job(server.api_generate(req_obj)["job_id"])
        assert j["status"] == "completed", j
        assert j["results"][0]["width"] == expected_w
        if req_obj.mode == "i2i":
            assert j["results"][0]["ref_count"] == 3

    gal = server.api_gallery(mode="all", q=None, limit=100, offset=0)
    assert gal["total_all"] == 5

    zip_resp = server.api_export_gallery(ids=None)
    assert zip_resp.media_type == "application/zip"
    assert int(zip_resp.headers.get("content-length", "0")) > 0

    first_id = gal["items"][0]["id"]
    assert server.api_delete_gallery_item(first_id)["ok"] is True
    assert server.api_gallery(mode="all", q=None, limit=100, offset=0)["total_all"] == 4

    _assert_profiles_and_modes(
        lambda body: server.api_update_config(server.ConfigUpdateRequest(**body))["stats"]["engine"]
    )

    assert "cwd" in server.api_terminal_cwd()
    cmd_id = server.api_terminal_run(server.TerminalRunRequest(command="echo hello_easycolab"))["cmd_id"]
    term_out = ""
    for _ in range(60):
        pr = server.api_terminal_poll(cmd_id, offset=0)
        term_out += pr["chunk"]
        if not pr["running"]:
            assert pr["exit_code"] == 0
            break
        time.sleep(0.05)
    assert "hello_easycolab" in term_out

    assert server.api_terminal_run(server.TerminalRunRequest(command="cd static"))["cwd"].endswith("static")
    chain_id = server.api_terminal_run(server.TerminalRunRequest(command="cd .. && echo chain_ok"))["cmd_id"]
    chain_out = ""
    for _ in range(40):
        pr = server.api_terminal_poll(chain_id, offset=0)
        if not pr["running"]:
            chain_out = pr["chunk"]
            assert pr["exit_code"] == 0, pr
            break
        time.sleep(0.05)
    assert "chain_ok" in chain_out

    assert isinstance(server.api_server_logs(limit=50)["lines"], list)
    assert server.api_free_vram(unload_model=False)["ok"] is True
    swap_data = server.api_enable_swap(server.SwapRequest(size_gb=8))
    assert "ok" in swap_data and "message" in swap_data and "swap" in swap_data and "stats" in swap_data

    # LoRA download URL normalization, download API, and dual-mode LoRA generation in Demo Mode
    from lora_manager import normalize_download_url

    norm_u, hdrs = normalize_download_url("https://civitai.com/models/123?modelVersionId=456", token="tok_abc")
    assert "civitai.com/api/download/models/456" in norm_u and "token=tok_abc" in norm_u and "Bearer tok_abc" in hdrs["Authorization"]

    dl_res = server.api_download_lora(server.LoraDownloadRequest(url="https://huggingface.co/test/demo/resolve/main/style_v1.safetensors"))
    assert dl_res["ok"] is True
    for _ in range(30):
        st = server.api_download_lora_status(dl_res["task_id"])
        if st["status"] == "completed":
            break
        time.sleep(0.08)
    loras_listing = server.api_list_loras()["items"]
    assert any(x["name"] == "style_v1.safetensors" for x in loras_listing)

    lora_std = _wait_job(
        server.api_generate(
            server.GenerateRequest(
                mode="t2i",
                prompt="studio portrait",
                width=256,
                height=256,
                steps=2,
                seed=11,
                lora_enabled=True,
                lora_name="style_v1.safetensors",
                lora_weight=0.85,
                lora_trigger="my_trigger",
                lora_action_mode="standard",
            )
        )["job_id"]
    )
    assert lora_std["status"] == "completed"

    lora_ctrl = _wait_job(
        server.api_generate(
            server.GenerateRequest(
                mode="t2i",
                prompt="pose transfer",
                width=256,
                height=256,
                steps=2,
                seed=12,
                lora_enabled=True,
                lora_name="style_v1.safetensors",
                lora_weight=0.85,
                lora_action_mode="control",
                lora_control_image_b64=init_b64,
                lora_control_strength=0.9,
            )
        )["job_id"]
    )
    assert lora_ctrl["status"] == "completed"

    # ComfyUI on-demand launcher (idle by default -> launch -> stop)
    assert server.api_comfyui_status()["status"] == "idle"
    server.api_comfyui_launch()
    for _ in range(30):
        cst = server.api_comfyui_status()
        if cst["status"] == "running":
            break
        time.sleep(0.08)
    assert server.api_comfyui_status()["status"] == "running"
    assert server.api_comfyui_stop()["status"] == "idle"

    # New: Auto transparent padding + notice + auto crop back to original aspect ratio
    odd_b64 = _make_b64_png(500, 900, (90, 140, 210))
    odd_mask_b64 = _make_b64_mask(500, 900)
    odd_job = _wait_job(
        server.api_generate(
            server.GenerateRequest(
                mode="inpaint",
                prompt="fix odd ratio",
                width=512,
                height=512,
                steps=3,
                denoise=0.85,
                seed=55,
                init_image_b64=odd_b64,
                mask_image_b64=odd_mask_b64,
            )
        )["job_id"]
    )
    assert odd_job["status"] == "completed", odd_job
    assert odd_job["notice"] and "透明图层扩图" in odd_job["notice"]
    out_w = odd_job["results"][0]["width"]
    out_h = odd_job["results"][0]["height"]
    assert abs((out_w / out_h) - (500 / 900)) < 0.05, (out_w, out_h)

    # New: Corrupted HTML LoRA detection & deletion
    from pathlib import Path
    from lora_manager import validate_lora_file

    bad_lora = Path(_TMP_LORAS) / "broken_html.safetensors"
    bad_lora.write_text("<!DOCTYPE html><html><body>Civitai login</body></html>", encoding="utf-8")
    is_valid, issue = validate_lora_file(bad_lora)
    assert is_valid is False and issue
    assert any(x["name"] == "broken_html.safetensors" and x["valid"] is False for x in server.api_list_loras()["items"])
    assert server.api_delete_lora("broken_html.safetensors")["ok"] is True
    assert not bad_lora.exists()

    # New: Monitor models inventory & AI default settings & AI chat action extraction
    st_now = server.api_stats()
    assert "models" in st_now and "loaded" in st_now["models"] and "downloaded_loras" in st_now["models"]
    assert st_now["ai"]["model"] == "qwen3.8-27b"
    assert "api.pie-xian.com" in st_now["ai"]["endpoint"]
    server.api_update_config(server.ConfigUpdateRequest(ai_api_key="sk-test-placeholder-key"))
    assert server.api_stats()["ai"]["has_api_key"] is True

    orig_call_ai = server._call_ai_endpoint
    try:
        server._call_ai_endpoint = lambda _msgs: (
            '建议调高步数：\n```json\n{"prompt": "neon cyber city", "steps": 20, "ratio": "3:4"}\n```'
        )
        ai_res = server.api_ai_chat(server.AIChatRequest(scene="studio", message="帮我写提示词"))
        assert ai_res["ok"] is True
        assert ai_res["actions"]["prompt"] == "neon cyber city"
        assert ai_res["actions"]["steps"] == 20
    finally:
        server._call_ai_endpoint = orig_call_ai

    # New: Fast cancellation
    cancel_job_id = server.api_generate(
        server.GenerateRequest(mode="t2i", prompt="cancel test", width=256, height=256, steps=20, seed=1)
    )["job_id"]
    time.sleep(0.05)
    assert server.api_cancel_job(cancel_job_id)["ok"] is True
    cancelled_job = _wait_job(cancel_job_id)
    assert cancelled_job["status"] == "cancelled", cancelled_job

    # New: Inpaint Fast Crop ("局部重绘加速") crops around mask and stitches back aligned to original
    full_img_b64 = _make_b64_png(640, 640, (40, 90, 160))
    small_mask = Image.new("L", (640, 640), 0)
    ImageDraw.Draw(small_mask).rectangle([280, 280, 360, 360], fill=255)
    small_mask_buf = io.BytesIO()
    small_mask.save(small_mask_buf, format="PNG")
    small_mask_b64 = "data:image/png;base64," + base64.b64encode(small_mask_buf.getvalue()).decode()

    server.api_update_config(server.ConfigUpdateRequest(inpaint_fast_crop=True))
    assert server.api_stats()["engine"]["inpaint_fast_crop"] is True
    fast_inpaint_job = _wait_job(
        server.api_generate(
            server.GenerateRequest(
                mode="inpaint",
                prompt="add gold emblem to @0 using @1",
                width=640,
                height=640,
                steps=3,
                denoise=0.9,
                seed=77,
                init_image_b64=full_img_b64,
                ref_images_b64=[init_b64],
                mask_image_b64=small_mask_b64,
                inpaint_fast_crop=True,
            )
        )["job_id"]
    )
    assert fast_inpaint_job["status"] == "completed", fast_inpaint_job
    assert fast_inpaint_job["notice"] and "局部重绘加速" in fast_inpaint_job["notice"]
    assert fast_inpaint_job["results"][0]["width"] == 640 and fast_inpaint_job["results"][0]["height"] == 640
    stitched_out = Image.open(Path(_TMP_DIR) / fast_inpaint_job["results"][0]["filename"])
    # Top-left corner outside crop box must stay aligned with original background color (40, 90, 160)
    assert stitched_out.getpixel((5, 5)) == (40, 90, 160)

    # New: Auto-select downloaded HF model on first load unless user modified in Settings
    tmp_hf_hub = Path(_TMP_DIR) / "hf_hub_test"
    cached_snap = tmp_hf_hub / "models--Qwen--Qwen-Image-2.1" / "snapshots" / "rev1"
    cached_snap.mkdir(parents=True, exist_ok=True)
    (cached_snap / "model_index.json").write_text("{}", encoding="utf-8")
    old_hf_hub = os.environ.get("HF_HUB_CACHE")
    try:
        os.environ["HF_HUB_CACHE"] = str(tmp_hf_hub)
        server.engine._user_modified_model = False
        server.engine.model_id = "Qwen/Qwen-Image-2.1-Turbo"
        picked = server.engine._auto_select_downloaded_model()
        assert picked == "Qwen/Qwen-Image-2.1" and server.engine.model_id == "Qwen/Qwen-Image-2.1"

        # If user explicitly changes model in Settings, auto-select must respect user choice
        server.api_update_config(server.ConfigUpdateRequest(model_id="Qwen/Qwen-Image-2.1-Turbo"))
        assert server.engine._user_modified_model is True
        assert server.engine._auto_select_downloaded_model() is None
        assert server.engine.model_id == "Qwen/Qwen-Image-2.1-Turbo"
    finally:
        if old_hf_hub is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = old_hf_hub

    # New: AI Vision toggle sends multimodal image_url when enabled
    server.api_update_config(server.ConfigUpdateRequest(ai_vision_enabled=True))
    assert server.api_stats()["ai"]["vision_enabled"] is True
    captured_msgs: list = []
    orig_call_ai2 = server._call_ai_endpoint
    try:
        def _capture_ai(msgs):
            captured_msgs.append(msgs)
            return '视觉分析完成：\n```json\n{"prompt": "detailed neon portrait"}\n```'
        server._call_ai_endpoint = _capture_ai
        vis_res = server.api_ai_chat(
            server.AIChatRequest(scene="studio", message="根据图片强化提示词", images=[init_b64])
        )
        assert vis_res["used_vision"] is True
        assert isinstance(captured_msgs[0][-1]["content"], list)
        assert any(part.get("type") == "image_url" for part in captured_msgs[0][-1]["content"])
    finally:
        server._call_ai_endpoint = orig_call_ai2


def _assert_profiles_and_modes(update_fn) -> None:
    assert server.resolve_multi_image_prompt("把 @1，2 放进 @0", 3) == "把 图2、图3 放进 图1"
    assert server.resolve_multi_image_prompt("put @1, 2 into @0", 3) == "put Picture 2 and Picture 3 into Picture 1"

    from gpu_profiles import match_gpu_profile

    assert match_gpu_profile("NVIDIA RTX PRO 6000 Blackwell", 96.0, 12, 0).key == "g4_rtx_pro_6000"
    assert match_gpu_profile("NVIDIA H100 80GB HBM3", 80.0, 9, 0).key == "h100_sxm"
    assert match_gpu_profile("NVIDIA A100-SXM4-80GB", 80.0, 8, 0).key == "a100_sxm"
    assert match_gpu_profile("NVIDIA L4", 24.0, 8, 9).key == "l4_ada"
    assert match_gpu_profile("Tesla T4", 15.0, 7, 5).key == "t4_turing"

    for profile_key, expected_dtype, expected_offload in [
        ("g4_rtx_pro_6000", "bfloat16", "none"),
        ("h100_sxm", "bfloat16", "none"),
        ("a100_sxm", "bfloat16", "none"),
        ("l4_ada", "bfloat16", "smart_stage_offload"),
        ("t4_turing", "float16", "model_cpu_offload"),
    ]:
        eng = update_fn(
            {
                "gpu_profile_override": profile_key,
                "dtype_mode": "auto",
                "offload_mode": "auto",
                "vae_tiling_mode": "auto",
            }
        )
        assert eng["effective_dtype"] == expected_dtype
        assert eng["effective_offload"] == expected_offload

    # Full-precision ("满血版本") config & T4 auto sequential_cpu_offload safety
    full_cfg = update_fn(
        {
            "model_id": "Qwen/Qwen-Image-2.1",
            "quant_mode": "none",
            "gpu_profile_override": "t4_turing",
            "offload_mode": "auto",
        }
    )
    assert full_cfg["is_full_precision"] is True
    assert full_cfg["default_steps"] == 40
    assert full_cfg["effective_offload"] == "sequential_cpu_offload"

    # L4 FP8 DiT + INT8 TextEncoder + BF16 VAE + smart_stage_offload
    l4_cfg = update_fn(
        {
            "model_id": "Qwen/Qwen-Image-2.1-Turbo",
            "quant_mode": "fp8-int8",
            "gpu_profile_override": "l4_ada",
            "offload_mode": "auto",
            "dtype_mode": "auto",
        }
    )
    assert l4_cfg["quant_mode"] == "fp8-int8"
    assert l4_cfg["effective_dtype"] == "bfloat16"
    assert l4_cfg["effective_offload"] == "smart_stage_offload"


def run_tests() -> None:
    if _can_use_socketpair():
        mode = "http_testclient"
        _run_http_tests()
    else:
        mode = "direct_inprocess"
        _run_direct_inprocess_tests()
    print(f"ALL TESTS PASSED ({mode} | 5 generation assertions + 5 GPU profiles + L4 FP8 + terminal + gallery in {_TMP_DIR})")


if __name__ == "__main__":
    run_tests()
