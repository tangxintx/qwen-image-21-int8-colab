(() => {
  "use strict";

  // -------------------------------------------------------------------------
  // Resolution & Aspect Ratio Presets (all divisible by 16 for Qwen-Image-2.1)
  // -------------------------------------------------------------------------
  const RESOLUTION_PRESETS = {
    "1k": {
      "1:1": [1024, 1024],
      "16:9": [1344, 768],
      "9:16": [768, 1344],
      "4:3": [1152, 864],
      "3:4": [864, 1152],
      "3:2": [1248, 832],
      "2:3": [832, 1248],
    },
    "1.5k": {
      "1:1": [1536, 1536],
      "16:9": [1920, 1088],
      "9:16": [1088, 1920],
      "4:3": [1664, 1248],
      "3:4": [1248, 1664],
      "3:2": [1920, 1280],
      "2:3": [1280, 1920],
    },
    "2k": {
      "1:1": [2048, 2048],
      "16:9": [2752, 1536],
      "9:16": [1536, 2752],
      "4:3": [2304, 1728],
      "3:4": [1728, 2304],
      "3:2": [2496, 1664],
      "2:3": [1664, 2496],
    },
  };

  const MODE_LABELS = {
    t2i: "文生图",
    i2i: "图生图",
    hires: "高清修复",
    inpaint: "局部重绘",
  };

  // -------------------------------------------------------------------------
  // App State
  // -------------------------------------------------------------------------
  const state = {
    mode: "t2i",
    tier: "1k",
    ratio: "1:1",
    width: 1024,
    height: 1024,
    batchSize: 1,
    lastSeed: null,
    activeJobId: null,
    pollTimer: null,
    statsTimer: null,
    initImageB64: null,
    initImageObj: null,
    extraRefImages: [],
    loraControlB64: null,
    comfyPollTimer: null,
    comfyOpenedOnce: false,
    galleryItems: [],
    galleryFilter: "all",
    gallerySearch: "",
    gallerySelecting: false,
    selectedIds: new Set(),
    lightboxItem: null,
    // Inpaint canvas state
    inpaintTool: "brush",
    brushSize: 36,
    drawing: false,
    lastPoint: null,
    undoStack: [],
    lorasItems: [],
    lastGenError: null,
    lastGenPayload: null,
    notifiedJobId: null,
  };

  // Offscreen exact-resolution mask canvas (black bg, white strokes)
  const maskDataCanvas = document.createElement("canvas");
  const maskDataCtx = maskDataCanvas.getContext("2d");

  // -------------------------------------------------------------------------
  // DOM Helpers
  // -------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function showToast(msg, duration = 2600) {
    const el = $("toastEl");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => el.classList.add("hidden"), duration);
  }

  function align16(val) {
    const n = Math.max(256, Math.min(3072, Math.round(Number(val) || 1024)));
    return Math.max(256, Math.floor(n / 16) * 16);
  }

  let _topBusyCount = 0;
  function setTopBusy(busy) {
    _topBusyCount = Math.max(0, _topBusyCount + (busy ? 1 : -1));
    $("topProgressTrack").classList.toggle("hidden", _topBusyCount === 0);
  }

  async function apiFetch(url, options = {}, showTopBar = false) {
    if (showTopBar) setTopBusy(true);
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          detail = err.detail || err.message || detail;
        } catch (_) {}
        throw new Error(detail);
      }
      return await res.json();
    } finally {
      if (showTopBar) setTopBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Navigation & Tabs
  // -------------------------------------------------------------------------
  function switchTab(tabName) {
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    $(`tab-${tabName}`).classList.add("active");
    document.querySelector(`.nav-item[data-tab="${tabName}"]`).classList.add("active");

    if (tabName === "gallery") loadGallery();
    if (tabName === "monitor") refreshStats();
    if (tabName === "terminal") onEnterTerminalTab();
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // -------------------------------------------------------------------------
  // Mode Switching
  // -------------------------------------------------------------------------
  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".mode-tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.mode === mode);
    });

    const needsImage = mode !== "t2i";
    $("imageInputCard").classList.toggle("hidden", !needsImage);

    const titles = {
      i2i: "图生图 · 多图参考 (@0主图)",
      hires: "高清修复 · 原图",
      inpaint: "局部重绘 · 遮罩",
    };
    if (titles[mode]) $("imageInputTitle").textContent = titles[mode];

    const denoiseSlider = $("denoiseSlider");
    if (mode === "hires") {
      denoiseSlider.value = "0.40";
    } else if (mode === "i2i") {
      denoiseSlider.value = "0.65";
    } else if (mode === "inpaint") {
      denoiseSlider.value = "0.85";
    }
    $("denoiseVal").textContent = Number(denoiseSlider.value).toFixed(2);
    $("denoiseBlock").style.opacity = mode === "t2i" ? "0.55" : "1";

    refreshImageStage();
  }

  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  // -------------------------------------------------------------------------
  // Resolution & Aspect Ratio Controls
  // -------------------------------------------------------------------------
  function applyResolution(w, h) {
    state.width = align16(w);
    state.height = align16(h);
    $("widthInput").value = state.width;
    $("heightInput").value = state.height;
    $("currentResBadge").textContent = `${state.width} × ${state.height}`;
  }

  const TIER_BASE_PIXELS = {
    "1k": 1024 * 1024,
    "1.5k": 1536 * 1536,
    "2k": 2048 * 2048,
  };

  function updateRatioChipUI() {
    document.querySelectorAll("#ratioSelector .ratio-chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.ratio === state.ratio);
    });
  }

  function applyOriginalAspectResolution(img) {
    if (!img || !img.width || !img.height) return false;
    const targetArea = TIER_BASE_PIXELS[state.tier] || (1024 * 1024);
    const scale = Math.sqrt(targetArea / (img.width * img.height));
    applyResolution(img.width * scale, img.height * scale);
    return true;
  }

  function syncPresetResolution() {
    if (state.ratio === "custom") return;
    if (state.ratio === "original") {
      if (!applyOriginalAspectResolution(state.initImageObj)) {
        showToast("请先选择主图 (@0)");
        state.ratio = "1:1";
        updateRatioChipUI();
        const fallback = RESOLUTION_PRESETS[state.tier]?.["1:1"];
        if (fallback) applyResolution(fallback[0], fallback[1]);
      }
      return;
    }
    const preset = RESOLUTION_PRESETS[state.tier]?.[state.ratio];
    if (preset) applyResolution(preset[0], preset[1]);
  }

  document.querySelectorAll("#tierSelector .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.tier = btn.dataset.tier;
      document.querySelectorAll("#tierSelector .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (state.ratio === "custom") state.ratio = "1:1";
      updateRatioChipUI();
      syncPresetResolution();
    });
  });

  document.querySelectorAll("#ratioSelector .ratio-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.ratio = btn.dataset.ratio;
      updateRatioChipUI();
      syncPresetResolution();
    });
  });

  function onManualDimInput() {
    state.ratio = "custom";
    updateRatioChipUI();
    applyResolution($("widthInput").value, $("heightInput").value);
  }

  $("widthInput").addEventListener("change", onManualDimInput);
  $("heightInput").addEventListener("change", onManualDimInput);
  $("swapDimBtn").addEventListener("click", () => {
    state.ratio = "custom";
    updateRatioChipUI();
    applyResolution($("heightInput").value, $("widthInput").value);
  });

  // -------------------------------------------------------------------------
  // Reference Image & Interactive Touch Inpaint Canvas
  // -------------------------------------------------------------------------
  function insertPromptToken(token) {
    const box = $("promptInput");
    const start = box.selectionStart ?? box.value.length;
    const end = box.selectionEnd ?? box.value.length;
    const before = box.value.slice(0, start);
    const after = box.value.slice(end);
    const padL = before && !/[\s，,、]$/.test(before) ? " " : "";
    const padR = after && !/^[\s，,、]/.test(after) ? " " : "";
    box.value = `${before}${padL}${token}${padR}${after}`;
    const cursor = before.length + padL.length + token.length;
    box.focus();
    box.setSelectionRange(cursor, cursor);
  }

  function renderExtraRefs() {
    const isI2I = state.mode === "i2i";
    $("extraRefsSection").classList.toggle("hidden", !isI2I);

    const extras = state.extraRefImages;
    $("extraRefsCount").textContent = `${extras.length}/9`;
    const grid = $("extraRefsGrid");
    grid.classList.toggle("hidden", !isI2I || extras.length === 0);
    grid.innerHTML = "";

    extras.forEach((item, idx) => {
      const refNum = idx + 1;
      const card = document.createElement("div");
      card.className = "extra-ref-card";
      card.innerHTML = `
        <img src="${item.b64}" alt="@${refNum}" />
        <span class="extra-ref-tag" title="插入 @${refNum}">@${refNum}</span>
        <div class="extra-ref-actions">
          <button class="extra-ref-mini-btn" data-act="promote" title="设为主图 (@0)">
            <svg class="icon"><use href="#ic-arrow-up" /></svg>
          </button>
          <button class="extra-ref-mini-btn" data-act="remove" title="移除">
            <svg class="icon"><use href="#ic-close" /></svg>
          </button>
        </div>
      `;
      card.querySelector("img").addEventListener("click", () => insertPromptToken(`@${refNum}`));
      card.querySelector(".extra-ref-tag").addEventListener("click", () => insertPromptToken(`@${refNum}`));
      card.querySelector('[data-act="promote"]').addEventListener("click", () => {
        const oldPrimary = state.initImageB64
          ? { b64: state.initImageB64, width: state.initImageObj?.width || 1024, height: state.initImageObj?.height || 1024 }
          : null;
        state.extraRefImages.splice(idx, 1);
        if (oldPrimary) state.extraRefImages.unshift(oldPrimary);
        loadInitImageFromDataUrl(item.b64);
      });
      card.querySelector('[data-act="remove"]').addEventListener("click", () => {
        state.extraRefImages.splice(idx, 1);
        renderExtraRefs();
      });
      grid.appendChild(card);
    });

    // Quick @0 / @1 / @1,2 mention chips above prompt quick tags
    const bar = $("refMentionBar");
    const showMentions = isI2I && (Boolean(state.initImageObj) || extras.length > 0);
    bar.classList.toggle("hidden", !showMentions);
    bar.innerHTML = "";
    if (showMentions) {
      const tokens = [];
      if (state.initImageObj) tokens.push({ label: "@0 主图", val: "@0" });
      extras.forEach((_, idx) => tokens.push({ label: `@${idx + 1}`, val: `@${idx + 1}` }));
      if (extras.length >= 2) {
        const multi = extras.map((_, idx) => idx + 1).join(",");
        tokens.push({ label: `@${multi}`, val: `@${multi}` });
      }
      tokens.forEach((t) => {
        const b = document.createElement("button");
        b.className = "tag-chip ref-mention";
        b.textContent = t.label;
        b.addEventListener("click", () => insertPromptToken(t.val));
        bar.appendChild(b);
      });
    }
  }

  function refreshImageStage() {
    const hasImg = Boolean(state.initImageObj);
    $("uploadPlaceholder").classList.toggle("hidden", hasImg);
    $("clearImageBtn").classList.toggle("hidden", !hasImg && state.extraRefImages.length === 0);

    if (!hasImg) {
      $("refPreviewWrap").classList.add("hidden");
      $("inpaintEditor").classList.add("hidden");
      renderExtraRefs();
      return;
    }

    const isInpaint = state.mode === "inpaint";
    $("refPreviewWrap").classList.toggle("hidden", isInpaint);
    $("inpaintEditor").classList.toggle("hidden", !isInpaint);

    if (!isInpaint) {
      $("refPreviewImg").src = state.initImageB64;
      $("refImageMeta").textContent = `${state.initImageObj.width} × ${state.initImageObj.height}`;
    } else {
      setupInpaintCanvases(state.initImageObj);
    }
    renderExtraRefs();
  }

  function resetMaskCanvas() {
    maskDataCanvas.width = 0;
    maskDataCanvas.height = 0;
    const overlayCanvas = $("maskOverlayCanvas");
    if (overlayCanvas) {
      const ovCtx = overlayCanvas.getContext("2d");
      ovCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  const SUPPORTED_RATIOS = [1, 16 / 9, 9 / 16, 4 / 3, 3 / 4, 3 / 2, 2 / 3];
  function isSupportedAspectRatio(w, h) {
    if (!w || !h) return true;
    const r = w / h;
    return SUPPORTED_RATIOS.some((sr) => Math.abs(Math.log(sr / r)) <= 0.025);
  }

  function loadInitImageFromDataUrl(dataUrl) {
    const img = new Image();
    img.onload = () => {
      state.initImageB64 = dataUrl;
      state.initImageObj = img;
      state.undoStack = [];
      resetMaskCanvas();
      state.ratio = "original";
      updateRatioChipUI();
      applyOriginalAspectResolution(img);
      refreshImageStage();
      if (!isSupportedAspectRatio(img.width, img.height)) {
        showToast("原图比例非模型预设比例，生成时将自动透明扩图并在输出后自动裁剪还原", 3800);
      }
    };
    img.src = dataUrl;
  }

  function addExtraRefFromDataUrl(dataUrl) {
    if (state.extraRefImages.length >= 9) {
      showToast("最多支持 9 张次图 (@1~@9)");
      return;
    }
    const img = new Image();
    img.onload = () => {
      state.extraRefImages.push({ b64: dataUrl, width: img.width, height: img.height });
      renderExtraRefs();
    };
    img.src = dataUrl;
  }

  async function addExtraRefFromUrl(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onload = () => addExtraRefFromDataUrl(reader.result);
    reader.readAsDataURL(blob);
  }

  async function loadInitImageFromUrl(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onload = () => loadInitImageFromDataUrl(reader.result);
    reader.readAsDataURL(blob);
  }

  $("primaryRefBadge").addEventListener("click", () => insertPromptToken("@0"));

  $("imageFileInput").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadInitImageFromDataUrl(reader.result);
    reader.readAsDataURL(file);
    e.target.value = "";
  });

  $("extraImageFileInput").addEventListener("change", (e) => {
    const files = Array.from(e.target.files || []);
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => addExtraRefFromDataUrl(reader.result);
      reader.readAsDataURL(file);
    });
    e.target.value = "";
  });

  $("clearImageBtn").addEventListener("click", () => {
    state.initImageB64 = null;
    state.initImageObj = null;
    state.extraRefImages = [];
    state.undoStack = [];
    resetMaskCanvas();
    if (state.ratio === "original") {
      state.ratio = "1:1";
      updateRatioChipUI();
      syncPresetResolution();
    }
    refreshImageStage();
  });

  $("pickFromGalleryBtn").addEventListener("click", () => {
    showToast("在作品库点击图片，即可一键送入当前模式");
    switchTab("gallery");
  });

  function setupInpaintCanvases(img) {
    const bgCanvas = $("bgCanvas");
    const overlayCanvas = $("maskOverlayCanvas");
    const bgCtx = bgCanvas.getContext("2d");
    const ovCtx = overlayCanvas.getContext("2d");

    // Display size up to 640px wide while keeping exact aspect ratio
    const maxDisplayW = Math.min(640, $("canvasStage").clientWidth || 360);
    const scale = maxDisplayW / img.width;
    const dispW = Math.max(200, Math.round(img.width * scale));
    const dispH = Math.max(120, Math.round(img.height * scale));

    bgCanvas.width = dispW;
    bgCanvas.height = dispH;
    overlayCanvas.width = dispW;
    overlayCanvas.height = dispH;

    bgCtx.clearRect(0, 0, dispW, dispH);
    bgCtx.drawImage(img, 0, 0, dispW, dispH);

    // Exact pixel mask canvas
    if (maskDataCanvas.width !== img.width || maskDataCanvas.height !== img.height) {
      maskDataCanvas.width = img.width;
      maskDataCanvas.height = img.height;
      maskDataCtx.fillStyle = "#000000";
      maskDataCtx.fillRect(0, 0, img.width, img.height);
      ovCtx.clearRect(0, 0, dispW, dispH);
    } else {
      syncOverlayFromMaskData();
    }
  }

  function syncOverlayFromMaskData() {
    const overlayCanvas = $("maskOverlayCanvas");
    const ovCtx = overlayCanvas.getContext("2d");
    ovCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Render white pixels of maskDataCanvas as translucent violet overlay
    const temp = document.createElement("canvas");
    temp.width = overlayCanvas.width;
    temp.height = overlayCanvas.height;
    const tCtx = temp.getContext("2d");
    tCtx.drawImage(maskDataCanvas, 0, 0, temp.width, temp.height);
    const imgData = tCtx.getImageData(0, 0, temp.width, temp.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const lum = d[i];
      if (lum > 30) {
        d[i] = 168;
        d[i + 1] = 85;
        d[i + 2] = 247;
        d[i + 3] = Math.round((lum / 255) * 155);
      } else {
        d[i + 3] = 0;
      }
    }
    tCtx.putImageData(imgData, 0, 0);
    ovCtx.drawImage(temp, 0, 0);
  }

  function pushUndoSnapshot() {
    if (!maskDataCanvas.width || !maskDataCanvas.height) return;
    const snap = maskDataCtx.getImageData(0, 0, maskDataCanvas.width, maskDataCanvas.height);
    state.undoStack.push(snap);
    if (state.undoStack.length > 15) state.undoStack.shift();
  }

  function getCanvasPoint(evt) {
    const overlay = $("maskOverlayCanvas");
    const rect = overlay.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * overlay.width;
    const y = ((evt.clientY - rect.top) / rect.height) * overlay.height;
    return { x, y };
  }

  function drawStrokeSegment(p1, p2) {
    const overlay = $("maskOverlayCanvas");
    const ovCtx = overlay.getContext("2d");
    const scaleX = maskDataCanvas.width / overlay.width;
    const scaleY = maskDataCanvas.height / overlay.height;
    const isEraser = state.inpaintTool === "eraser";

    // Draw on exact maskDataCanvas
    maskDataCtx.save();
    maskDataCtx.lineCap = "round";
    maskDataCtx.lineJoin = "round";
    maskDataCtx.lineWidth = state.brushSize * scaleX;
    maskDataCtx.strokeStyle = isEraser ? "#000000" : "#ffffff";
    maskDataCtx.beginPath();
    maskDataCtx.moveTo(p1.x * scaleX, p1.y * scaleY);
    maskDataCtx.lineTo(p2.x * scaleX, p2.y * scaleY);
    maskDataCtx.stroke();
    maskDataCtx.restore();

    // Draw visual overlay
    ovCtx.save();
    ovCtx.lineCap = "round";
    ovCtx.lineJoin = "round";
    ovCtx.lineWidth = state.brushSize;
    if (isEraser) {
      ovCtx.globalCompositeOperation = "destination-out";
      ovCtx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      ovCtx.globalCompositeOperation = "source-over";
      ovCtx.strokeStyle = "rgba(168, 85, 247, 0.58)";
    }
    ovCtx.beginPath();
    ovCtx.moveTo(p1.x, p1.y);
    ovCtx.lineTo(p2.x, p2.y);
    ovCtx.stroke();
    ovCtx.restore();
  }

  const overlayEl = $("maskOverlayCanvas");
  overlayEl.style.touchAction = $("scrollLockToggle").checked ? "pan-y" : "none";

  overlayEl.addEventListener("pointerdown", (e) => {
    if ($("scrollLockToggle").checked) return;
    e.preventDefault();
    overlayEl.setPointerCapture(e.pointerId);
    pushUndoSnapshot();
    state.drawing = true;
    const pt = getCanvasPoint(e);
    state.lastPoint = pt;
    drawStrokeSegment(pt, { x: pt.x + 0.01, y: pt.y + 0.01 });
  });

  overlayEl.addEventListener("pointermove", (e) => {
    if (!state.drawing) return;
    e.preventDefault();
    const pt = getCanvasPoint(e);
    drawStrokeSegment(state.lastPoint, pt);
    state.lastPoint = pt;
  });

  const endStroke = () => {
    state.drawing = false;
    state.lastPoint = null;
  };
  overlayEl.addEventListener("pointerup", endStroke);
  overlayEl.addEventListener("pointercancel", endStroke);

  $("scrollLockToggle").addEventListener("change", (e) => {
    const locked = e.target.checked;
    overlayEl.style.touchAction = locked ? "pan-y" : "none";
    showToast(locked ? "已锁屏（禁止涂抹，可上下滑动页面）" : "已解锁（可涂抹遮罩区域）");
  });

  $("toolBrushBtn").addEventListener("click", () => {
    state.inpaintTool = "brush";
    $("toolBrushBtn").classList.add("active");
    $("toolEraserBtn").classList.remove("active");
  });

  $("toolEraserBtn").addEventListener("click", () => {
    state.inpaintTool = "eraser";
    $("toolEraserBtn").classList.add("active");
    $("toolBrushBtn").classList.remove("active");
  });

  $("brushSizeInput").addEventListener("input", (e) => {
    state.brushSize = Number(e.target.value);
    $("brushSizeVal").textContent = state.brushSize;
  });

  $("toolUndoBtn").addEventListener("click", () => {
    const snap = state.undoStack.pop();
    if (!snap) return showToast("没有可撤销的笔画");
    maskDataCtx.putImageData(snap, 0, 0);
    syncOverlayFromMaskData();
  });

  $("toolClearMaskBtn").addEventListener("click", () => {
    pushUndoSnapshot();
    maskDataCtx.fillStyle = "#000000";
    maskDataCtx.fillRect(0, 0, maskDataCanvas.width, maskDataCanvas.height);
    syncOverlayFromMaskData();
  });

  // -------------------------------------------------------------------------
  // Prompt & Parameter Controls
  // -------------------------------------------------------------------------
  $("clearPromptBtn").addEventListener("click", () => {
    $("promptInput").value = "";
  });

  document.querySelectorAll("#quickTags .tag-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const box = $("promptInput");
      const tag = chip.dataset.tag;
      box.value = box.value.trim() ? `${box.value.trim()}, ${tag}` : tag;
    });
  });

  $("stepsSlider").addEventListener("input", (e) => {
    $("stepsVal").textContent = e.target.value;
  });
  document.querySelectorAll(".mini-pill[data-steps]").forEach((p) => {
    p.addEventListener("click", () => {
      $("stepsSlider").value = p.dataset.steps;
      $("stepsVal").textContent = p.dataset.steps;
    });
  });

  $("denoiseSlider").addEventListener("input", (e) => {
    $("denoiseVal").textContent = Number(e.target.value).toFixed(2);
  });

  $("cfgSlider").addEventListener("input", (e) => {
    $("cfgVal").textContent = Number(e.target.value).toFixed(1);
  });

  $("hiresScaleSlider").addEventListener("input", (e) => {
    $("hiresScaleVal").textContent = `${Number(e.target.value).toFixed(2)}x`;
  });

  $("maskBlurSlider").addEventListener("input", (e) => {
    $("maskBlurVal").textContent = `${e.target.value}px`;
  });

  $("randomSeedBtn").addEventListener("click", () => {
    $("seedInput").value = -1;
    showToast("已设为随机种子 (-1)");
  });

  $("lockLastSeedBtn").addEventListener("click", () => {
    if (state.lastSeed === null) return showToast("暂无可锁定的上次种子");
    $("seedInput").value = state.lastSeed;
    showToast(`已锁定 Seed: ${state.lastSeed}`);
  });

  $("batchDecBtn").addEventListener("click", () => {
    state.batchSize = Math.max(1, state.batchSize - 1);
    $("batchVal").textContent = state.batchSize;
  });
  $("batchIncBtn").addEventListener("click", () => {
    state.batchSize = Math.min(8, state.batchSize + 1);
    $("batchVal").textContent = state.batchSize;
  });

  $("resetParamsBtn").addEventListener("click", () => {
    $("stepsSlider").value = 8;
    $("stepsVal").textContent = "8";
    $("denoiseSlider").value = 0.75;
    $("denoiseVal").textContent = "0.75";
    $("cfgSlider").value = 1.0;
    $("cfgVal").textContent = "1.0";
    $("seedInput").value = -1;
    state.batchSize = 1;
    $("batchVal").textContent = "1";
    showToast("已恢复默认采样参数");
  });

  // -------------------------------------------------------------------------
  // Generation Flow & Job Polling
  // -------------------------------------------------------------------------
  function hasNonEmptyMask() {
    if (!maskDataCanvas.width || !maskDataCanvas.height) return false;
    const data = maskDataCtx.getImageData(0, 0, maskDataCanvas.width, maskDataCanvas.height).data;
    for (let i = 0; i < data.length; i += 16) {
      if (data[i] > 20) return true;
    }
    return false;
  }

  function setGeneratingUI(active) {
    $("generateBtn").disabled = active;
    $("generateBtn").innerHTML = active
      ? '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>生成中...</span>'
      : '<svg class="icon"><use href="#ic-play" /></svg><span>开始生成</span>';
    $("cancelGenBtn").classList.toggle("hidden", !active);
    $("cancelGenBtn").disabled = false;
    $("cancelGenBtn").innerHTML = '<svg class="icon"><use href="#ic-stop" /></svg><span>停止</span>';
    $("progressWrap").classList.toggle("hidden", !active);
    if (!active) {
      $("progressBarFill").style.width = "0%";
    }
  }

  function showGenErrorBanner(errMsg) {
    state.lastGenError = errMsg;
    $("genErrorText").textContent = `生成失败: ${errMsg}`;
    $("genErrorBanner").classList.remove("hidden");
  }

  function hideGenBanners() {
    state.lastGenError = null;
    $("genErrorBanner").classList.add("hidden");
    $("genNoticeBanner").classList.add("hidden");
  }

  $("generateBtn").addEventListener("click", async () => {
    const prompt = $("promptInput").value.trim();
    if (!prompt) {
      showToast("请先输入提示词 (Prompt)");
      $("promptInput").focus();
      return;
    }

    if (state.mode !== "t2i" && !state.initImageB64) {
      showToast("当前模式需要先上传或导入参考底图");
      return;
    }
    if (state.mode === "inpaint" && !hasNonEmptyMask()) {
      showToast("请在画面上涂抹需要重绘的区域");
      return;
    }

    hideGenBanners();
    state.notifiedJobId = null;

    const payload = {
      mode: state.mode,
      prompt,
      negative_prompt: $("negativePromptInput").value.trim(),
      width: align16($("widthInput").value),
      height: align16($("heightInput").value),
      steps: Number($("stepsSlider").value),
      denoise: Number($("denoiseSlider").value),
      true_cfg_scale: Number($("cfgSlider").value),
      seed: Number($("seedInput").value || -1),
      batch_size: state.batchSize,
      hires_scale: Number($("hiresScaleSlider").value),
      mask_blur: Number($("maskBlurSlider").value),
      init_image_b64: state.mode === "t2i" ? null : state.initImageB64,
      ref_images_b64: state.mode === "i2i" ? state.extraRefImages.map((x) => x.b64) : [],
      mask_image_b64: state.mode === "inpaint" ? maskDataCanvas.toDataURL("image/png") : null,
      lora_enabled: $("loraEnableToggle").checked,
      lora_name: $("loraSelect").value || null,
      lora_weight: Number($("loraWeightSlider").value),
      lora_trigger: $("loraTriggerInput").value.trim(),
      lora_action_mode: $("loraActionMode").value,
      lora_control_image_b64: $("loraActionMode").value === "control" ? state.loraControlB64 : null,
      lora_control_strength: Number($("loraCtrlStrengthSlider").value),
    };
    state.lastGenPayload = { ...payload, init_image_b64: Boolean(payload.init_image_b64), mask_image_b64: Boolean(payload.mask_image_b64), ref_images_b64: payload.ref_images_b64.length };

    try {
      setGeneratingUI(true);
      $("progressMsg").textContent = "提交生成任务...";
      $("progressPct").textContent = "0%";
      const resp = await apiFetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.activeJobId = resp.job_id;
      startPollingJob(resp.job_id);
    } catch (err) {
      setGeneratingUI(false);
      showGenErrorBanner(err.message);
      showToast(`启动失败: ${err.message}`, 3500);
    }
  });

  $("cancelGenBtn").addEventListener("click", async () => {
    if (!state.activeJobId) return;
    $("cancelGenBtn").disabled = true;
    $("cancelGenBtn").innerHTML = '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>停止中...</span>';
    $("progressMsg").textContent = "正在立即中断生成...";
    try {
      await apiFetch(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST" });
      showToast("已发送中断指令，正在停止...");
    } catch (_) {}
  });

  function startPollingJob(jobId) {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const job = await apiFetch(`/api/jobs/${jobId}`);
        const isQueued = job.status === "queued";
        const pct = Math.round((job.progress || 0) * 100);
        const showIndeterminate = isQueued || (job.status === "loading_model" && pct <= 2);
        $("progressMsg").textContent = job.message || "生成中...";
        $("progressPct").textContent = showIndeterminate ? "准备中" : `${pct}%`;
        $("progressBarFill").classList.toggle("indeterminate", showIndeterminate);
        if (!showIndeterminate) {
          $("progressBarFill").style.width = `${pct}%`;
        }

        if (job.notice) {
          $("genNoticeBanner").textContent = job.notice;
          $("genNoticeBanner").classList.remove("hidden");
          if (state.notifiedJobId !== jobId) {
            state.notifiedJobId = jobId;
            showToast(job.notice, 4500);
          }
        }

        if (["completed", "cancelled", "error"].includes(job.status)) {
          clearInterval(state.pollTimer);
          state.activeJobId = null;
          setGeneratingUI(false);

          if (job.status === "completed") {
            renderLatestResults(job.results, job.elapsed_sec);
            if (job.results?.length) {
              state.lastSeed = job.results[job.results.length - 1].seed;
            }
            showToast(`生成完成 (${job.results.length} 张 · ${job.elapsed_sec}s)`);
            refreshStats();
          } else if (job.status === "cancelled") {
            if (job.results?.length) renderLatestResults(job.results, job.elapsed_sec);
            showToast("生成已停止");
          } else {
            const errText = job.error || "未知错误";
            showGenErrorBanner(errText);
            showToast(`生成失败: ${errText}`, 8000);
          }
        }
      } catch (err) {
        clearInterval(state.pollTimer);
        state.activeJobId = null;
        setGeneratingUI(false);
        showToast(`轮询中断: ${err.message}`);
      }
    }, 350);
  }

  function renderLatestResults(items, totalElapsed) {
    const card = $("latestResultCard");
    const grid = $("latestResultGrid");
    if (!items || !items.length) return;
    card.classList.remove("hidden");
    $("latestElapsedText").textContent = `总耗时 ${totalElapsed}s`;
    grid.innerHTML = "";

    items.forEach((item) => {
      const el = document.createElement("div");
      el.className = "img-card";
      el.innerHTML = `
        <img src="${item.url}" alt="result" loading="lazy" />
        <div class="img-card-foot">
          <span>${item.width}×${item.height}</span>
          <span>Seed ${item.seed}</span>
        </div>
        <div class="img-card-actions">
          <button class="mini-action" data-act="i2i">图生图</button>
          <button class="mini-action" data-act="hires">高清</button>
          <button class="mini-action" data-act="inpaint">重绘</button>
          <a class="mini-action" href="${item.url}" download="${item.filename}" style="text-decoration:none">存图</a>
        </div>
      `;
      el.querySelector("img").addEventListener("click", () => openLightbox(item));
      el.querySelector('[data-act="i2i"]').addEventListener("click", () => sendItemToMode(item, "i2i"));
      el.querySelector('[data-act="hires"]').addEventListener("click", () => sendItemToMode(item, "hires"));
      el.querySelector('[data-act="inpaint"]').addEventListener("click", () => sendItemToMode(item, "inpaint"));
      grid.appendChild(el);
    });
  }

  async function sendItemToMode(item, targetMode) {
    setMode(targetMode);
    switchTab("studio");
    await loadInitImageFromUrl(item.url);
    if (item.prompt) $("promptInput").value = item.prompt;
    closeLightbox();
    showToast(`已导入到「${MODE_LABELS[targetMode]}」`);
  }

  // -------------------------------------------------------------------------
  // Gallery
  // -------------------------------------------------------------------------
  async function loadGallery() {
    try {
      const params = new URLSearchParams({
        mode: state.galleryFilter,
        q: state.gallerySearch,
        limit: "150",
      });
      const data = await apiFetch(`/api/gallery?${params.toString()}`);
      state.galleryItems = data.items || [];
      const mb = ((data.total_bytes || 0) / (1024 * 1024)).toFixed(1);
      $("gallerySummary").textContent = `${data.total_all || 0} 张 · ${mb} MB`;
      renderGallery();
    } catch (err) {
      showToast(`加载作品库失败: ${err.message}`);
    }
  }

  function renderGallery() {
    const grid = $("galleryGrid");
    grid.innerHTML = "";
    $("galleryEmptyState").classList.toggle("hidden", state.galleryItems.length > 0);

    state.galleryItems.forEach((item) => {
      const card = document.createElement("div");
      const isSelected = state.selectedIds.has(item.id);
      card.className = `img-card ${isSelected ? "selected" : ""}`;
      card.innerHTML = `
        ${state.gallerySelecting ? `<div class="select-badge">${isSelected ? '<svg class="icon"><use href="#ic-check" /></svg>' : ""}</div>` : ""}
        <img src="${item.url}" alt="${item.filename}" loading="lazy" />
        <div class="img-card-foot">
          <span>${MODE_LABELS[item.mode] || item.mode}</span>
          <span>${item.width}×${item.height}</span>
        </div>
      `;
      card.addEventListener("click", () => {
        if (state.gallerySelecting) {
          if (state.selectedIds.has(item.id)) state.selectedIds.delete(item.id);
          else state.selectedIds.add(item.id);
          renderGallery();
        } else {
          openLightbox(item);
        }
      });
      grid.appendChild(card);
    });
  }

  document.querySelectorAll("#galleryFilterChips .filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.galleryFilter = chip.dataset.filter;
      document.querySelectorAll("#galleryFilterChips .filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      loadGallery();
    });
  });

  $("gallerySearchInput").addEventListener("input", (e) => {
    state.gallerySearch = e.target.value.trim();
    clearTimeout(loadGallery._t);
    loadGallery._t = setTimeout(loadGallery, 250);
  });

  $("galleryRefreshBtn").addEventListener("click", () => {
    setTopBusy(true);
    loadGallery().finally(() => setTopBusy(false));
  });

  $("galleryExportBtn").addEventListener("click", async () => {
    const ids = state.selectedIds.size ? `?ids=${Array.from(state.selectedIds).join(",")}` : "";
    const wrap = $("galleryExportProgressWrap");
    const fill = $("galleryExportBarFill");
    const msg = $("galleryExportMsg");
    const pctEl = $("galleryExportPct");

    wrap.classList.remove("hidden");
    fill.classList.add("indeterminate");
    msg.textContent = "正在打包 ZIP...";
    pctEl.textContent = "准备中";

    try {
      const res = await fetch(`/api/gallery/export${ids}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const total = Number(res.headers.get("content-length") || 0);
      const reader = res.body?.getReader();
      const chunks = [];
      let received = 0;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.byteLength;
          if (total > 0) {
            fill.classList.remove("indeterminate");
            const pct = Math.min(100, Math.round((received / total) * 100));
            fill.style.width = `${pct}%`;
            pctEl.textContent = `${pct}%`;
          } else {
            pctEl.textContent = `${(received / (1024 * 1024)).toFixed(1)} MB`;
          }
        }
      } else {
        const buf = await res.arrayBuffer();
        chunks.push(new Uint8Array(buf));
      }

      fill.classList.remove("indeterminate");
      fill.style.width = "100%";
      pctEl.textContent = "100%";
      msg.textContent = "打包完成，正在保存...";

      const blob = new Blob(chunks, { type: "application/zip" });
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = `qwen_gallery_${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
      showToast("作品库 ZIP 下载成功");
    } catch (err) {
      showToast(`打包导出失败: ${err.message}`);
    } finally {
      setTimeout(() => {
        wrap.classList.add("hidden");
        fill.style.width = "0%";
      }, 900);
    }
  });

  $("gallerySelectToggleBtn").addEventListener("click", () => {
    state.gallerySelecting = !state.gallerySelecting;
    if (!state.gallerySelecting) state.selectedIds.clear();
    $("gallerySelectToggleBtn").classList.toggle("active", state.gallerySelecting);
    $("galleryBatchDeleteBtn").classList.toggle("hidden", !state.gallerySelecting);
    renderGallery();
  });

  $("galleryBatchDeleteBtn").addEventListener("click", async () => {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return showToast("请先勾选要删除的图片");
    try {
      await apiFetch("/api/gallery/batch-delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids }),
      });
      state.selectedIds.clear();
      showToast(`已删除 ${ids.length} 张作品`);
      loadGallery();
      refreshStats();
    } catch (err) {
      showToast(`删除失败: ${err.message}`);
    }
  });

  // -------------------------------------------------------------------------
  // Lightbox Modal
  // -------------------------------------------------------------------------
  function openLightbox(item) {
    state.lightboxItem = item;
    $("lightboxTitle").textContent = `${MODE_LABELS[item.mode] || item.mode} · ${item.width}×${item.height}`;
    $("lightboxImg").src = item.url;
    $("lbDownloadBtn").href = item.url;
    $("lbDownloadBtn").download = item.filename;
    $("lightboxMeta").innerHTML = `
      <div><strong>Prompt:</strong> ${escapeHtml(item.prompt)}</div>
      <div style="margin-top:4px">
        Steps: ${item.steps} · Denoise: ${item.denoise} · CFG: ${item.true_cfg_scale} · Seed: <code>${item.seed}</code> · 耗时: ${item.elapsed_sec}s
      </div>
    `;
    $("lightboxModal").classList.remove("hidden");
  }

  function closeLightbox() {
    $("lightboxModal").classList.add("hidden");
    state.lightboxItem = null;
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  $("lightboxCloseBtn").addEventListener("click", closeLightbox);
  $("lightboxBackdrop").addEventListener("click", closeLightbox);

  $("lbReuseParamsBtn").addEventListener("click", () => {
    const item = state.lightboxItem;
    if (!item) return;
    setMode(item.mode || "t2i");
    $("promptInput").value = item.prompt || "";
    $("negativePromptInput").value = item.negative_prompt || "";
    applyResolution(item.width, item.height);
    $("stepsSlider").value = item.steps;
    $("stepsVal").textContent = item.steps;
    $("denoiseSlider").value = item.denoise;
    $("denoiseVal").textContent = Number(item.denoise).toFixed(2);
    $("cfgSlider").value = item.true_cfg_scale;
    $("cfgVal").textContent = Number(item.true_cfg_scale).toFixed(1);
    $("seedInput").value = item.seed;
    closeLightbox();
    switchTab("studio");
    showToast("已复用该作品全套参数");
  });

  $("lbToI2IBtn").addEventListener("click", () => state.lightboxItem && sendItemToMode(state.lightboxItem, "i2i"));
  $("lbAddExtraRefBtn").addEventListener("click", async () => {
    if (!state.lightboxItem) return;
    setMode("i2i");
    switchTab("studio");
    await addExtraRefFromUrl(state.lightboxItem.url);
    const nextIdx = state.extraRefImages.length + 1;
    closeLightbox();
    showToast(`已添加为次图 (@${Math.min(9, nextIdx)})`);
  });
  $("lbToHiresBtn").addEventListener("click", () => state.lightboxItem && sendItemToMode(state.lightboxItem, "hires"));
  $("lbToInpaintBtn").addEventListener("click", () => state.lightboxItem && sendItemToMode(state.lightboxItem, "inpaint"));

  $("lbDeleteBtn").addEventListener("click", async () => {
    if (!state.lightboxItem) return;
    try {
      await apiFetch(`/api/gallery/${state.lightboxItem.id}`, { method: "DELETE" });
      closeLightbox();
      showToast("已从作品库删除");
      loadGallery();
      refreshStats();
    } catch (err) {
      showToast(`删除失败: ${err.message}`);
    }
  });

  // -------------------------------------------------------------------------
  // System Monitor & Maintenance & One-Click Shutdown
  // -------------------------------------------------------------------------
  function renderMonitorModels(models) {
    const loadedEl = $("monitorLoadedModels");
    const hfEl = $("monitorDownloadedHf");
    const lorasEl = $("monitorDownloadedLoras");
    if (!loadedEl || !hfEl || !lorasEl) return;

    const loaded = (models && models.loaded) || {};
    const statusBadge = loaded.demo_mode
      ? '<span class="model-badge info">演示模式</span>'
      : loaded.is_loaded
        ? '<span class="model-badge ok">已驻留显存</span>'
        : loaded.is_loading
          ? '<span class="model-badge warn">加载中...</span>'
          : '<span class="model-badge warn">待加载(按需)</span>';

    let loadedHtml = `
      <div class="model-row">
        <div class="model-row-info">
          <div class="model-row-name">${escapeHtml(loaded.model_id || "--")}</div>
          <div class="model-row-sub">量化: ${escapeHtml(loaded.quant_mode || "--")} · 设备: ${escapeHtml(loaded.device || "cpu")}</div>
        </div>
        <div class="model-row-actions">${statusBadge}</div>
      </div>`;
    if (loaded.active_lora) {
      loadedHtml += `
        <div class="model-row">
          <div class="model-row-info">
            <div class="model-row-name">LoRA: ${escapeHtml(loaded.active_lora)}</div>
            <div class="model-row-sub">当前挂载权重: ${Number(loaded.active_lora_weight ?? 0.85).toFixed(2)}</div>
          </div>
          <div class="model-row-actions"><span class="model-badge ok">已加载</span></div>
        </div>`;
    }
    loadedEl.innerHTML = loadedHtml;

    const hfList = (models && models.downloaded_hf) || [];
    if (!hfList.length) {
      hfEl.innerHTML = '<div class="model-row"><div class="model-row-sub">暂无本地 HF 缓存模型（首次生成时自动高速下载）</div></div>';
    } else {
      hfEl.innerHTML = "";
      hfList.forEach((m) => {
        const row = document.createElement("div");
        row.className = "model-row";
        const sizeLabel = m.size_gb >= 0.1 ? `${m.size_gb} GB` : `${m.size_mb} MB`;
        const badge = m.is_loaded
          ? '<span class="model-badge ok">已加载</span>'
          : m.is_current
            ? '<span class="model-badge info">当前选中</span>'
            : m.ready
              ? '<span class="model-badge info">已缓存</span>'
              : '<span class="model-badge warn">缓存中</span>';
        row.innerHTML = `
          <div class="model-row-info">
            <div class="model-row-name">${escapeHtml(m.repo_id)}</div>
            <div class="model-row-sub">占用空间: ${sizeLabel}</div>
          </div>
          <div class="model-row-actions">
            ${badge}
            ${!m.is_current ? '<button class="btn-chip" data-act="use">切换</button>' : ""}
          </div>`;
        const useBtn = row.querySelector('[data-act="use"]');
        if (useBtn) {
          useBtn.addEventListener("click", () => {
            const sel = $("modelSelect");
            if (!Array.from(sel.options).some((o) => o.value === m.repo_id)) {
              const opt = document.createElement("option");
              opt.value = m.repo_id;
              opt.textContent = m.repo_id;
              sel.appendChild(opt);
            }
            sel.value = m.repo_id;
            saveEngineConfig();
          });
        }
        hfEl.appendChild(row);
      });
    }

    const loraList = (models && models.downloaded_loras) || [];
    if (!loraList.length) {
      lorasEl.innerHTML = '<div class="model-row"><div class="model-row-sub">暂无已下载的 LoRA 文件</div></div>';
    } else {
      lorasEl.innerHTML = "";
      loraList.forEach((item) => {
        const row = document.createElement("div");
        row.className = "model-row";
        const validBadge = item.valid === false
          ? `<span class="model-badge err" title="${escapeHtml(item.issue || "文件损坏")}">损坏</span>`
          : item.is_loaded
            ? '<span class="model-badge ok">已加载</span>'
            : '<span class="model-badge info">可用</span>';
        row.innerHTML = `
          <div class="model-row-info">
            <div class="model-row-name">${escapeHtml(item.name)}</div>
            <div class="model-row-sub">${item.size_mb} MB${item.issue ? " · " + escapeHtml(item.issue) : ""}</div>
          </div>
          <div class="model-row-actions">
            ${validBadge}
            <button class="btn-chip danger" data-act="del">删除</button>
          </div>`;
        row.querySelector('[data-act="del"]').addEventListener("click", () => deleteLoraByName(item.name));
        lorasEl.appendChild(row);
      });
    }
  }

  async function refreshStats() {
    try {
      const s = await apiFetch("/api/stats");
      const gpu = s.gpu;
      const ram = s.ram;
      const disk = s.disk;
      const eng = s.engine;

      const statusMode = eng.demo_mode ? "演示模式" : eng.loaded ? "模型已驻留" : "待加载(按需)";
      let quantTag = "INT8";
      if (eng.quant_mode === "none") quantTag = `${eng.effective_dtype.toUpperCase()} 满血`;
      else if (eng.quant_mode === "gguf-q8-full" || eng.gguf_variant === "q8") quantTag = "GGUF Q8 满血";
      else if (eng.quant_mode === "gguf-q4" || eng.gguf_variant === "q4") quantTag = "GGUF Q4";
      else if (eng.quant_mode === "fp8-int8") quantTag = "FP8+INT8";
      else if (eng.quant_mode === "fp8-all") quantTag = "FP8";
      $("headerQuantBadge").textContent = quantTag;
      $("engineStatusText").textContent = `${eng.model_id.split("/").pop()} · ${statusMode}`;

      // Monitor gauges
      $("gpuNameText").textContent = gpu.name;
      $("vramText").textContent = gpu.available
        ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB (${gpu.vram_percent}%)`
        : "未启用 CUDA (DEMO)";
      $("vramBar").style.width = `${gpu.vram_percent || 0}%`;

      $("ramText").textContent = `${ram.used_gb} / ${ram.total_gb} GB (${ram.percent}%)`;
      $("ramBar").style.width = `${ram.percent}%`;

      const swap = s.swap || {};
      if (swap.enabled) {
        $("swapText").textContent = `${swap.used_gb} / ${swap.total_gb} GB (${swap.percent}%)`;
        $("swapSubText").textContent = `已挂载虚拟内存 (${swap.free_gb} GB 可用)`;
        $("swapBar").style.width = `${swap.percent || 0}%`;
      } else {
        $("swapText").textContent = swap.supported ? "未开启 (容器默认受限)" : "仅 Linux 支持";
        $("swapSubText").textContent = swap.supported
          ? "标准 Colab 容器内核通常禁止 swapon；已启用 low_cpu_mem + 内存回收"
          : "本地非 Linux 环境";
        $("swapBar").style.width = "0%";
      }

      $("diskText").textContent = `${disk.used_gb} / ${disk.total_gb} GB (${disk.percent}%)`;
      $("diskFreeText").textContent = `剩余可用 ${disk.free_gb} GB`;
      $("diskBar").style.width = `${disk.percent}%`;

      $("outputsSizeText").textContent = `${s.storage.outputs_mb} MB`;
      $("hfCacheSizeText").textContent = `${s.storage.hf_cache_gb} GB`;
      $("modelLoadedText").textContent = eng.demo_mode ? "演示模式" : eng.loaded ? "已加载" : "未加载";
      $("uptimeText").textContent = formatUptime(eng.uptime_sec || 0);

      // GPU Hardware Profile Box
      $("gpuProfileBadge").textContent = gpu.arch || "GPU 适配";
      $("gpuProfileTitle").textContent = gpu.profile_name || "未知 GPU";
      $("gpuProfileMeta").textContent =
        `精度: ${eng.effective_dtype.toUpperCase()} · 卸载: ${eng.effective_offload} · VAE分块: ${eng.effective_vae_tiling ? "开" : "关"} · TF32: ${eng.effective_tf32 ? "开" : "关"}`;
      $("gpuProfileNotes").textContent = gpu.notes || "";

      if (s.models) renderMonitorModels(s.models);
      if (s.ai && $("aiModelBadge")) {
        $("aiModelBadge").textContent = s.ai.model || "qwen3.8-27b";
      }
    } catch (_) {
      $("engineStatusText").textContent = "连接已断开";
    }
  }

  async function runOpsWithProgress(label, fn) {
    const wrap = $("opsProgressWrap");
    $("opsProgressMsg").textContent = label;
    $("opsProgressPct").textContent = "处理中";
    wrap.classList.remove("hidden");
    setTopBusy(true);
    try {
      await fn();
    } finally {
      wrap.classList.add("hidden");
      setTopBusy(false);
    }
  }

  $("refreshStatsBtn").addEventListener("click", () => runOpsWithProgress("正在刷新系统资源...", refreshStats));

  $("enableSwapBtn").addEventListener("click", () =>
    runOpsWithProgress("正在划出 8 GB 磁盘开启 Linux Swap 虚拟内存...", async () => {
      const res = await apiFetch("/api/system/enable-swap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size_gb: 8 }),
      });
      showToast(res.message || "Swap 操作完成", 4500);
      await refreshStats();
    })
  );

  $("freeVramBtn").addEventListener("click", () =>
    runOpsWithProgress("正在释放 CUDA 显存碎片...", async () => {
      await apiFetch("/api/system/free-vram", { method: "POST" });
      showToast("已释放 CUDA 显存碎片");
      await refreshStats();
    })
  );

  $("unloadModelBtn").addEventListener("click", () =>
    runOpsWithProgress("正在卸载模型释放显存...", async () => {
      await apiFetch("/api/system/free-vram?unload_model=true", { method: "POST" });
      showToast("已卸载模型释放显存");
      await refreshStats();
    })
  );

  $("clearHfCacheBtn").addEventListener("click", () =>
    runOpsWithProgress("正在清理 HuggingFace 模型缓存...", async () => {
      await apiFetch("/api/system/clear-hf-cache", { method: "POST" });
      showToast("已清理 HuggingFace 模型缓存");
      await refreshStats();
    })
  );

  // -------------------------------------------------------------------------
  // Mobile Web Terminal & Server Logs
  // -------------------------------------------------------------------------
  const termState = {
    mode: "shell", // "shell" | "logs"
    activeCmdId: null,
    pollTimer: null,
    logsTimer: null,
    offset: 0,
    history: [],
    histIdx: -1,
    shellBuffer: "Web 终端就绪\n",
  };

  function formatUptime(sec) {
    const s = Math.max(0, Math.floor(Number(sec) || 0));
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const r = String(s % 60).padStart(2, "0");
    return `${h}:${m}:${r}`;
  }

  function appendTerminalText(txt) {
    termState.shellBuffer += txt;
    if (termState.shellBuffer.length > 120000) {
      termState.shellBuffer = termState.shellBuffer.slice(-100000);
    }
    if (termState.mode === "shell") {
      const scr = $("terminalScreen");
      scr.textContent = termState.shellBuffer;
      scr.scrollTop = scr.scrollHeight;
    }
  }

  function setTermRunningUI(running, cmdLabel = "") {
    $("termRunBtn").classList.toggle("hidden", running);
    $("termKillBtn").classList.toggle("hidden", !running);
    $("termProgressWrap").classList.toggle("hidden", !running);
    if (running) {
      $("termProgressMsg").textContent = cmdLabel ? `执行中: ${cmdLabel}` : "命令执行中...";
      $("termProgressElapsed").textContent = "0.0s";
    }
  }

  async function refreshTerminalCwd() {
    try {
      const data = await apiFetch("/api/terminal/cwd");
      $("terminalCwdText").textContent = data.cwd || "~";
    } catch (_) {}
  }

  async function fetchServerLogs(forceBottom = false) {
    if (termState.mode !== "logs") return;
    try {
      const data = await apiFetch("/api/logs?limit=200");
      const scr = $("terminalScreen");
      const wasNearBottom = scr.scrollHeight - scr.scrollTop - scr.clientHeight < 64;
      scr.textContent = (data.lines || []).join("\n") || "暂无日志输出...";
      if (forceBottom || wasNearBottom) {
        scr.scrollTop = scr.scrollHeight;
      }
    } catch (_) {}
  }

  function setTerminalViewMode(mode) {
    termState.mode = mode;
    $("termModeShellBtn").classList.toggle("active", mode === "shell");
    $("termModeLogsBtn").classList.toggle("active", mode === "logs");
    $("terminalInputBar").classList.toggle("hidden", mode !== "shell");
    $("termQuickCmds").classList.toggle("hidden", mode !== "shell");
    clearInterval(termState.logsTimer);

    if (mode === "logs") {
      fetchServerLogs(true);
      termState.logsTimer = setInterval(() => fetchServerLogs(false), 700);
    } else {
      const scr = $("terminalScreen");
      scr.textContent = termState.shellBuffer;
      scr.scrollTop = scr.scrollHeight;
      refreshTerminalCwd();
    }
  }

  function onEnterTerminalTab() {
    if (termState.mode === "shell") refreshTerminalCwd();
    else fetchServerLogs(true);
  }

  async function runTerminalCommand(rawCmd) {
    const cmd = (rawCmd || "").trim();
    if (!cmd) return;
    if (termState.mode !== "shell") setTerminalViewMode("shell");
    if (termState.activeCmdId) {
      showToast("当前有命令正在运行，请先等待或点击中断");
      return;
    }

    termState.history.unshift(cmd);
    if (termState.history.length > 40) termState.history.pop();
    termState.histIdx = -1;
    $("terminalCmdInput").value = "";

    const cwdNow = $("terminalCwdText").textContent || "~";
    appendTerminalText(`\n[${cwdNow}]$ ${cmd}\n`);

    try {
      const res = await apiFetch("/api/terminal/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
      $("terminalCwdText").textContent = res.cwd || cwdNow;
      termState.activeCmdId = res.cmd_id;
      termState.offset = 0;
      setTermRunningUI(true, cmd);
      pollTerminalTask(res.cmd_id);
    } catch (err) {
      appendTerminalText(`[请求失败] ${err.message}\n`);
    }
  }

  function pollTerminalTask(cmdId) {
    clearInterval(termState.pollTimer);
    const step = async () => {
      try {
        const data = await apiFetch(`/api/terminal/poll/${cmdId}?offset=${termState.offset}`);
        if (data.chunk) appendTerminalText(data.chunk);
        termState.offset = data.offset;
        if (data.cwd) $("terminalCwdText").textContent = data.cwd;
        if (data.elapsed_sec !== undefined) {
          $("termProgressElapsed").textContent = `${Number(data.elapsed_sec).toFixed(1)}s`;
        }

        if (!data.running) {
          clearInterval(termState.pollTimer);
          termState.activeCmdId = null;
          setTermRunningUI(false);
          if (data.exit_code !== 0 && data.exit_code !== null) {
            appendTerminalText(`[退出码 ${data.exit_code} · ${data.elapsed_sec}s]\n`);
          }
        }
      } catch (err) {
        clearInterval(termState.pollTimer);
        termState.activeCmdId = null;
        setTermRunningUI(false);
        appendTerminalText(`[轮询中断] ${err.message}\n`);
      }
    };
    step();
    termState.pollTimer = setInterval(step, 180);
  }

  $("termModeShellBtn").addEventListener("click", () => setTerminalViewMode("shell"));
  $("termModeLogsBtn").addEventListener("click", () => setTerminalViewMode("logs"));
  $("termClearBtn").addEventListener("click", () => {
    if (termState.mode === "shell") {
      termState.shellBuffer = "";
      $("terminalScreen").textContent = "";
    } else {
      fetchServerLogs();
    }
  });

  $("termRunBtn").addEventListener("click", () => runTerminalCommand($("terminalCmdInput").value));
  $("terminalCmdInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runTerminalCommand($("terminalCmdInput").value);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (termState.history.length && termState.histIdx + 1 < termState.history.length) {
        termState.histIdx += 1;
        $("terminalCmdInput").value = termState.history[termState.histIdx];
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (termState.histIdx > 0) {
        termState.histIdx -= 1;
        $("terminalCmdInput").value = termState.history[termState.histIdx];
      } else {
        termState.histIdx = -1;
        $("terminalCmdInput").value = "";
      }
    }
  });

  $("termHistUpBtn").addEventListener("click", () => {
    if (!termState.history.length) return showToast("暂无命令历史");
    termState.histIdx = (termState.histIdx + 1) % termState.history.length;
    $("terminalCmdInput").value = termState.history[termState.histIdx];
  });

  $("termKillBtn").addEventListener("click", async () => {
    if (!termState.activeCmdId) return;
    try {
      await apiFetch(`/api/terminal/kill/${termState.activeCmdId}`, { method: "POST" });
    } catch (_) {}
  });

  document.querySelectorAll("#termQuickCmds .tag-chip[data-cmd]").forEach((chip) => {
    chip.addEventListener("click", () => runTerminalCommand(chip.dataset.cmd));
  });
  $("refreshModelsBtn").addEventListener("click", () => runOpsWithProgress("正在刷新模型与缓存清单...", refreshStats));

  // Settings save & Edition Quick Switch
  async function saveEngineConfig(recommendedSteps = null) {
    try {
      await apiFetch(
        "/api/config",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: $("modelSelect").value,
            quant_mode: $("quantSelect").value,
            gpu_profile_override: $("gpuProfileSelect").value,
            dtype_mode: $("dtypeSelect").value,
            offload_mode: $("offloadSelect").value,
            vae_tiling_mode: $("vaeTilingSelect").value,
            hf_token: $("hfTokenInput").value.trim() || null,
            ai_endpoint: $("aiEndpointInput").value.trim() || null,
            ai_model: $("aiModelInput").value.trim() || null,
            ai_api_key: $("aiApiKeyInput").value.trim() || null,
          }),
        },
        true
      );
      if (recommendedSteps) {
        $("stepsSlider").value = String(recommendedSteps);
        $("stepsVal").textContent = String(recommendedSteps);
      }
      const qVal = $("quantSelect").value;
      if ($("comfyGgufSelect") && !$("comfyGgufSelect").disabled) {
        if (qVal === "gguf-q8-full") $("comfyGgufSelect").value = "q8";
        else if (qVal === "gguf-q4") $("comfyGgufSelect").value = "q4";
      }
      const isFull = qVal === "none" || qVal === "gguf-q8-full";
      showToast(isFull ? "已切换至满血高精度配置（下次生成生效）" : "模型与系统配置已保存");
      refreshStats();
    } catch (err) {
      showToast(`保存失败: ${err.message}`);
    }
  }

  async function saveAiConfigOnly() {
    try {
      await apiFetch(
        "/api/config",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ai_endpoint: $("aiEndpointInput").value.trim() || null,
            ai_model: $("aiModelInput").value.trim() || null,
            ai_api_key: $("aiApiKeyInput").value.trim() || null,
          }),
        },
        true
      );
      $("aiModelBadge").textContent = $("aiModelInput").value.trim() || "qwen3.8-27b";
      showToast("AI 接口配置已保存");
    } catch (err) {
      showToast(`保存 AI 配置失败: ${err.message}`);
    }
  }

  document.querySelectorAll("#editionQuickBtns .tag-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("modelSelect").value = btn.dataset.model;
      $("quantSelect").value = btn.dataset.quant;
      if (btn.dataset.offload) $("offloadSelect").value = btn.dataset.offload;
      saveEngineConfig(Number(btn.dataset.steps || 8));
    });
  });

  $("saveConfigBtn").addEventListener("click", () => saveEngineConfig());
  $("saveAiConfigBtn").addEventListener("click", saveAiConfigOnly);
  $("testAiConfigBtn").addEventListener("click", async () => {
    await saveAiConfigOnly();
    try {
      $("testAiConfigBtn").disabled = true;
      const res = await apiFetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene: "studio", message: "请回复：连接正常", history: [], context: {} }),
      }, true);
      showToast(`AI 连接成功 (${res.model}): ${(res.reply || "").slice(0, 40)}`, 4000);
    } catch (err) {
      showToast(`AI 测试失败: ${err.message}`, 5000);
    } finally {
      $("testAiConfigBtn").disabled = false;
    }
  });

  // One-click Shutdown Modal
  function openShutdownModal() {
    $("shutdownModal").classList.remove("hidden");
  }
  function closeShutdownModal() {
    $("shutdownModal").classList.add("hidden");
  }

  $("headerShutdownBtn").addEventListener("click", openShutdownModal);
  $("monitorShutdownBtn").addEventListener("click", openShutdownModal);
  $("shutdownCancelBtn").addEventListener("click", closeShutdownModal);
  $("shutdownDismissBtn").addEventListener("click", closeShutdownModal);
  $("shutdownBackdrop").addEventListener("click", closeShutdownModal);

  $("shutdownConfirmBtn").addEventListener("click", async () => {
    const unassignColab = $("unassignColabCheck").checked;
    try {
      const res = await apiFetch("/api/shutdown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ unassign_colab: unassignColab }),
      });
      closeShutdownModal();
      clearInterval(state.statsTimer);
      $("engineStatusText").textContent = "服务已安全关闭";
      showToast(res.message || "服务已关闭", 6000);
    } catch (err) {
      showToast(`关闭请求异常: ${err.message}`);
    }
  });

  // -------------------------------------------------------------------------
  // LoRA Management, Download Modal & ComfyUI Launcher
  // -------------------------------------------------------------------------
  function updateSelectedLoraWarning() {
    const selName = $("loraSelect").value;
    const item = (state.lorasItems || []).find((x) => x.name === selName);
    $("loraDeleteBtn").classList.toggle("hidden", !selName);
    if (item && item.valid === false) {
      $("loraWarnText").textContent = `警告：${item.issue || "LoRA 文件无效或已损坏"}，请点击右侧垃圾桶删除后重新下载。`;
      $("loraWarnText").classList.remove("hidden");
    } else {
      $("loraWarnText").classList.add("hidden");
    }
  }

  async function deleteLoraByName(name) {
    if (!name) return;
    try {
      await apiFetch(`/api/loras/${encodeURIComponent(name)}`, { method: "DELETE" });
      showToast(`已删除 LoRA: ${name}`);
      await refreshLoraList();
      refreshStats();
    } catch (err) {
      showToast(`删除失败: ${err.message}`);
    }
  }

  async function refreshLoraList(selectFilename = null) {
    try {
      const data = await apiFetch("/api/loras");
      state.lorasItems = data.items || [];
      const sel = $("loraSelect");
      const prev = selectFilename || sel.value;
      sel.innerHTML = '<option value="">请选择 LoRA (.safetensors)...</option>';
      state.lorasItems.forEach((item) => {
        const opt = document.createElement("option");
        opt.value = item.name;
        const badTag = item.valid === false ? " [文件损坏]" : "";
        opt.textContent = `${item.name} (${item.size_mb} MB)${badTag}`;
        sel.appendChild(opt);
      });
      if (prev && Array.from(sel.options).some((o) => o.value === prev)) {
        sel.value = prev;
      }
      updateSelectedLoraWarning();
    } catch (_) {}
  }

  $("loraSelect").addEventListener("change", updateSelectedLoraWarning);
  $("loraDeleteBtn").addEventListener("click", () => deleteLoraByName($("loraSelect").value));

  async function startLoraDownload(url, token, ui) {
    const cleanUrl = (url || "").trim();
    if (!cleanUrl) {
      showToast("请输入 LoRA 下载链接");
      return;
    }
    ui.wrap.classList.remove("hidden");
    ui.fill.classList.add("indeterminate");
    ui.fill.style.width = "0%";
    ui.msg.textContent = "正在请求下载...";
    ui.pct.textContent = "准备中";
    if (ui.btn) ui.btn.disabled = true;

    try {
      const res = await apiFetch("/api/loras/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: cleanUrl, token: (token || "").trim() || null }),
      });
      const taskId = res.task_id;
      const timer = setInterval(async () => {
        try {
          const st = await apiFetch(`/api/loras/download/${taskId}`);
          const pct = Math.round((st.progress || 0) * 100);
          const indet = st.status === "downloading" && (!st.total_mb || pct === 0);
          ui.fill.classList.toggle("indeterminate", indet);
          if (!indet) ui.fill.style.width = `${pct}%`;
          ui.msg.textContent = st.message || "下载中...";
          ui.pct.textContent = indet ? `${st.downloaded_mb || 0} MB` : `${pct}%`;

          if (st.status === "completed" || st.status === "error") {
            clearInterval(timer);
            if (ui.btn) ui.btn.disabled = false;
            if (st.status === "completed") {
              ui.fill.classList.remove("indeterminate");
              ui.fill.style.width = "100%";
              ui.pct.textContent = "100%";
              showToast(`LoRA 已下载: ${st.filename}`);
              await refreshLoraList(st.filename);
              refreshStats();
              if (ui.onComplete) ui.onComplete(st.filename);
            } else {
              showToast(`LoRA 下载失败: ${st.error || "未知错误"}`, 6000);
            }
          }
        } catch (err) {
          clearInterval(timer);
          if (ui.btn) ui.btn.disabled = false;
          showToast(`下载进度查询失败: ${err.message}`);
        }
      }, 350);
    } catch (err) {
      if (ui.btn) ui.btn.disabled = false;
      ui.wrap.classList.add("hidden");
      showToast(`下载启动失败: ${err.message}`);
    }
  }

  $("loraEnableToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("loraPanelBody").classList.toggle("hidden", !on);
    if (on) refreshLoraList();
  });
  $("loraRefreshBtn").addEventListener("click", () => refreshLoraList());
  $("loraWeightSlider").addEventListener("input", (e) => {
    $("loraWeightVal").textContent = Number(e.target.value).toFixed(2);
  });
  $("loraActionMode").addEventListener("change", (e) => {
    $("loraControlPanel").classList.toggle("hidden", e.target.value !== "control");
  });
  $("loraCtrlStrengthSlider").addEventListener("input", (e) => {
    $("loraCtrlStrengthVal").textContent = Number(e.target.value).toFixed(2);
  });
  $("loraCtrlFileInput").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      state.loraControlB64 = reader.result;
      $("loraCtrlPreviewImg").src = reader.result;
      $("loraCtrlPlaceholder").classList.add("hidden");
      $("loraCtrlPreviewWrap").classList.remove("hidden");
      $("loraCtrlClearBtn").classList.remove("hidden");
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  });
  $("loraCtrlClearBtn").addEventListener("click", () => {
    state.loraControlB64 = null;
    $("loraCtrlPreviewImg").src = "";
    $("loraCtrlPlaceholder").classList.remove("hidden");
    $("loraCtrlPreviewWrap").classList.add("hidden");
    $("loraCtrlClearBtn").classList.add("hidden");
  });
  $("loraPanelDownloadBtn").addEventListener("click", () =>
    startLoraDownload($("loraPanelUrlInput").value, $("loraPanelTokenInput").value, {
      wrap: $("loraPanelDlProgressWrap"),
      fill: $("loraPanelDlFill"),
      msg: $("loraPanelDlMsg"),
      pct: $("loraPanelDlPct"),
      btn: $("loraPanelDownloadBtn"),
    })
  );

  // Terminal "下载LORA" modal
  function openLoraModal() {
    $("loraDownloadModal").classList.remove("hidden");
  }
  function closeLoraModal() {
    $("loraDownloadModal").classList.add("hidden");
  }
  $("termDownloadLoraBtn").addEventListener("click", openLoraModal);
  $("loraModalCloseBtn").addEventListener("click", closeLoraModal);
  $("loraModalCancelBtn").addEventListener("click", closeLoraModal);
  $("loraModalBackdrop").addEventListener("click", closeLoraModal);
  $("loraModalConfirmBtn").addEventListener("click", () =>
    startLoraDownload($("modalLoraUrlInput").value, $("modalLoraTokenInput").value, {
      wrap: $("modalLoraProgressWrap"),
      fill: $("modalLoraProgressFill"),
      msg: $("modalLoraProgressMsg"),
      pct: $("modalLoraProgressPct"),
      btn: $("loraModalConfirmBtn"),
      onComplete: () => setTimeout(closeLoraModal, 600),
    })
  );

  // -------------------------------------------------------------------------
  // AI Assistant (Studio Prompts/Params/Settings, Terminal Commands, Error Fix)
  // -------------------------------------------------------------------------
  const aiState = {
    scene: "studio", // "studio" | "terminal" | "error"
    history: [],     // { role, content, actions }
    busy: false,
  };

  const AI_SCENE_META = {
    studio: {
      title: "AI 创作与调参助手",
      quick: [
        { label: "润色当前提示词", text: "帮我扩写并优化当前的提示词，提升光影、构图与细节描述，并返回可直接应用的 JSON。" },
        { label: "推荐高画质参数", text: "根据我当前的 GPU 与模型配置，推荐最适合的步数、CFG 和分辨率比例并帮我改好。" },
        { label: "生成赛博海报方案", text: "帮我写一段高质量中文赛博朋克霓虹海报提示词，设置适合竖屏海报的比例和参数。" },
        { label: "切换 L4 黄金方案", text: "帮我把设置切换为 L4 黄金方案 (FP8+INT8+智能分阶段调度)，并调整步数为 8。" },
      ],
    },
    terminal: {
      title: "AI 终端代码助手",
      quick: [
        { label: "诊断环境与显存", text: "请给我检查 GPU 显存、PyTorch CUDA 和磁盘占用的终端命令。" },
        { label: "查看最近报错日志", text: "帮我分析最近的终端输出或服务日志，如果有报错请给出一键修复命令。" },
        { label: "清理缓存释放空间", text: "给我安全清理 pip 缓存和临时文件的命令。" },
      ],
    },
    error: {
      title: "AI 报错诊断与一键修复",
      quick: [
        { label: "一键分析并给修复方案", text: "请分析刚才生图报错的根本原因，给出可直接应用的参数/设置修改或终端命令。" },
      ],
    },
  };

  function collectAiContext(scene) {
    const ctx = {
      mode: state.mode,
      prompt: $("promptInput").value,
      negative_prompt: $("negativePromptInput").value,
      width: Number($("widthInput").value),
      height: Number($("heightInput").value),
      ratio: state.ratio,
      tier: state.tier,
      steps: Number($("stepsSlider").value),
      denoise: Number($("denoiseSlider").value),
      true_cfg_scale: Number($("cfgSlider").value),
      seed: Number($("seedInput").value),
      batch_size: state.batchSize,
      lora_enabled: $("loraEnableToggle").checked,
      lora_name: $("loraSelect").value,
      lora_weight: Number($("loraWeightSlider").value),
      model_id: $("modelSelect").value,
      quant_mode: $("quantSelect").value,
      offload_mode: $("offloadSelect").value,
    };
    if (scene === "terminal") {
      ctx.cwd = $("terminalCwdText").textContent;
      ctx.recent_terminal_output = (termState.shellBuffer || "").slice(-1500);
    }
    if (scene === "error") {
      ctx.error = state.lastGenError;
      ctx.last_payload = state.lastGenPayload;
    }
    return ctx;
  }

  function summarizeAiActions(actions) {
    if (!actions || typeof actions !== "object") return "";
    const parts = [];
    if (actions.prompt) parts.push("正向提示词");
    if (actions.negative_prompt !== undefined) parts.push("反向提示词");
    if (actions.mode) parts.push(`模式:${MODE_LABELS[actions.mode] || actions.mode}`);
    if (actions.width && actions.height) parts.push(`尺寸:${actions.width}×${actions.height}`);
    else if (actions.ratio) parts.push(`比例:${actions.ratio}`);
    if (actions.steps !== undefined) parts.push(`步数:${actions.steps}`);
    if (actions.denoise !== undefined) parts.push(`Denoise:${actions.denoise}`);
    if (actions.true_cfg_scale !== undefined) parts.push(`CFG:${actions.true_cfg_scale}`);
    if (actions.lora_enabled !== undefined) parts.push(`LoRA:${actions.lora_enabled ? "开" : "关"}`);
    if (actions.settings && typeof actions.settings === "object") parts.push("系统设置");
    return parts.join(" · ");
  }

  async function applyAiActions(actions) {
    if (!actions || typeof actions !== "object") return;
    if (actions.mode && ["t2i", "i2i", "hires", "inpaint"].includes(actions.mode)) {
      setMode(actions.mode);
    }
    if (typeof actions.prompt === "string" && actions.prompt.trim()) {
      $("promptInput").value = actions.prompt.trim();
    }
    if (typeof actions.negative_prompt === "string") {
      $("negativePromptInput").value = actions.negative_prompt;
    }
    if (actions.tier && ["1k", "1.5k", "2k"].includes(actions.tier)) {
      state.tier = actions.tier;
      document.querySelectorAll("#tierSelector .seg-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.tier === state.tier);
      });
    }
    if (actions.ratio && actions.ratio !== "custom") {
      state.ratio = actions.ratio;
      updateRatioChipUI();
      syncPresetResolution();
    }
    if (actions.width && actions.height) {
      applyResolution(actions.width, actions.height);
    }
    if (actions.steps !== undefined) {
      const st = Math.max(1, Math.min(60, Math.round(Number(actions.steps))));
      $("stepsSlider").value = String(st);
      $("stepsVal").textContent = String(st);
    }
    if (actions.denoise !== undefined) {
      const dn = Math.max(0.05, Math.min(1.0, Number(actions.denoise)));
      $("denoiseSlider").value = String(dn);
      $("denoiseVal").textContent = dn.toFixed(2);
    }
    if (actions.true_cfg_scale !== undefined) {
      const cfg = Math.max(1.0, Math.min(8.0, Number(actions.true_cfg_scale)));
      $("cfgSlider").value = String(cfg);
      $("cfgVal").textContent = cfg.toFixed(1);
    }
    if (actions.seed !== undefined) {
      $("seedInput").value = String(Math.round(Number(actions.seed)));
    }
    if (actions.batch_size !== undefined) {
      state.batchSize = Math.max(1, Math.min(8, Math.round(Number(actions.batch_size))));
      $("batchVal").textContent = String(state.batchSize);
    }
    if (actions.hires_scale !== undefined) {
      const hs = Math.max(1.0, Math.min(2.5, Number(actions.hires_scale)));
      $("hiresScaleSlider").value = String(hs);
      $("hiresScaleVal").textContent = `${hs.toFixed(2)}x`;
    }
    if (actions.mask_blur !== undefined) {
      const mb = Math.max(0, Math.min(32, Math.round(Number(actions.mask_blur))));
      $("maskBlurSlider").value = String(mb);
      $("maskBlurVal").textContent = `${mb}px`;
    }
    if (actions.lora_enabled !== undefined) {
      const on = Boolean(actions.lora_enabled);
      $("loraEnableToggle").checked = on;
      $("loraPanelBody").classList.toggle("hidden", !on);
      if (on) await refreshLoraList(actions.lora_name || null);
    }
    if (typeof actions.lora_name === "string" && actions.lora_name) {
      if (Array.from($("loraSelect").options).some((o) => o.value === actions.lora_name)) {
        $("loraSelect").value = actions.lora_name;
        updateSelectedLoraWarning();
      }
    }
    if (actions.lora_weight !== undefined) {
      const lw = Math.max(0.0, Math.min(1.5, Number(actions.lora_weight)));
      $("loraWeightSlider").value = String(lw);
      $("loraWeightVal").textContent = lw.toFixed(2);
    }
    if (typeof actions.lora_trigger === "string") {
      $("loraTriggerInput").value = actions.lora_trigger;
    }

    if (actions.settings && typeof actions.settings === "object") {
      const s = actions.settings;
      if (s.model_id) $("modelSelect").value = s.model_id;
      if (s.quant_mode) $("quantSelect").value = s.quant_mode;
      if (s.offload_mode) $("offloadSelect").value = s.offload_mode;
      if (s.dtype_mode) $("dtypeSelect").value = s.dtype_mode;
      if (s.vae_tiling_mode) $("vaeTilingSelect").value = s.vae_tiling_mode;
      await saveEngineConfig(actions.steps ? Number(actions.steps) : null);
    }

    showToast("已一键应用 AI 修改的提示词与参数");
  }

  function renderAiChat() {
    const list = $("aiChatList");
    list.innerHTML = "";
    if (!aiState.history.length) {
      list.innerHTML = `<div class="ai-msg assistant"><div class="ai-msg-body">你好！我是你的 Qwen-Image-2.1 AI 助手（${escapeHtml($("aiModelInput").value || "qwen3.8-27b")}）。你可以直接向我提问、让我帮你写提示词与调参、修改系统设置，或在终端中帮你编写和执行命令。</div></div>`;
      return;
    }

    aiState.history.forEach((msg) => {
      const wrap = document.createElement("div");
      wrap.className = `ai-msg ${msg.role}`;
      const cleanText = msg.role === "assistant"
        ? String(msg.content || "").replace(/```json[\s\S]*?```/gi, "").trim() || msg.content
        : msg.content;
      const body = document.createElement("div");
      body.className = "ai-msg-body";
      body.textContent = cleanText;
      wrap.appendChild(body);

      if (msg.role === "assistant" && msg.actions && typeof msg.actions === "object") {
        const actBox = document.createElement("div");
        actBox.className = "ai-actions-box";
        const summary = summarizeAiActions(msg.actions);
        if (summary) {
          const sumEl = document.createElement("div");
          sumEl.className = "ai-actions-summary";
          sumEl.textContent = `可应用改动：${summary}`;
          actBox.appendChild(sumEl);

          const btnsRow = document.createElement("div");
          btnsRow.className = "ai-actions-btns";
          const applyBtn = document.createElement("button");
          applyBtn.className = "btn-chip primary";
          applyBtn.type = "button";
          applyBtn.textContent = "一键应用全部修改";
          applyBtn.addEventListener("click", () => applyAiActions(msg.actions));
          btnsRow.appendChild(applyBtn);
          actBox.appendChild(btnsRow);
        }

        const cmds = Array.isArray(msg.actions.terminal_commands) ? msg.actions.terminal_commands : [];
        cmds.forEach((cmd) => {
          if (!cmd || typeof cmd !== "string") return;
          const cmdRow = document.createElement("div");
          cmdRow.className = "ai-cmd-row";
          cmdRow.innerHTML = `<code>${escapeHtml(cmd)}</code><button class="btn-chip primary ai-cmd-btn" type="button">运行</button>`;
          cmdRow.querySelector("button").addEventListener("click", () => {
            closeAiModal();
            switchTab("terminal");
            runTerminalCommand(cmd);
          });
          actBox.appendChild(cmdRow);
        });

        if (actBox.children.length > 0) wrap.appendChild(actBox);
      }

      list.appendChild(wrap);
    });
    list.scrollTop = list.scrollHeight;
  }

  function openAiModal(scene = "studio", autoPrompt = null) {
    aiState.scene = scene;
    const meta = AI_SCENE_META[scene] || AI_SCENE_META.studio;
    $("aiModalTitle").textContent = meta.title;
    const qWrap = $("aiQuickPrompts");
    qWrap.innerHTML = "";
    meta.quick.forEach((q) => {
      const b = document.createElement("button");
      b.className = "tag-chip";
      b.type = "button";
      b.textContent = q.label;
      b.addEventListener("click", () => sendAiMessage(q.text));
      qWrap.appendChild(b);
    });

    $("aiAssistantModal").classList.remove("hidden");
    renderAiChat();
    if (autoPrompt) {
      sendAiMessage(autoPrompt);
    } else {
      $("aiChatInput").focus();
    }
  }

  function closeAiModal() {
    $("aiAssistantModal").classList.add("hidden");
  }

  async function sendAiMessage(explicitText = null) {
    if (aiState.busy) return;
    const input = $("aiChatInput");
    const text = (explicitText !== null ? explicitText : input.value).trim();
    if (!text) return;
    if (explicitText === null) input.value = "";

    aiState.history.push({ role: "user", content: text });
    renderAiChat();

    aiState.busy = true;
    $("aiChatSendBtn").disabled = true;
    $("aiChatSendBtn").innerHTML = '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>思考中</span>';

    try {
      const res = await apiFetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene: aiState.scene,
          message: text,
          history: aiState.history.slice(-10, -1).map((m) => ({ role: m.role, content: m.content })),
          context: collectAiContext(aiState.scene),
        }),
      }, true);
      aiState.history.push({
        role: "assistant",
        content: res.reply || "",
        actions: res.actions || null,
      });
      renderAiChat();
    } catch (err) {
      aiState.history.push({
        role: "assistant",
        content: `请求 AI 失败：${err.message}\n请检查「设置 -> AI 助手 API 配置」中的接口地址与 API Key。`,
      });
      renderAiChat();
    } finally {
      aiState.busy = false;
      $("aiChatSendBtn").disabled = false;
      $("aiChatSendBtn").innerHTML = '<svg class="icon"><use href="#ic-sparkles" /></svg><span>发送</span>';
    }
  }

  $("studioAiBtn").addEventListener("click", () => openAiModal("studio"));
  $("termAskAiBtn").addEventListener("click", () => openAiModal("terminal"));
  $("errorAskAiBtn").addEventListener("click", () => {
    const errText = state.lastGenError || "未知生图错误";
    openAiModal("error", `刚才生图报错了：${errText}\n请帮我分析原因并给出修复方案（如需改参数/设置/跑命令请直接给出 JSON）。`);
  });
  $("aiModalCloseBtn").addEventListener("click", closeAiModal);
  $("aiModalBackdrop").addEventListener("click", closeAiModal);
  $("aiClearChatBtn").addEventListener("click", () => {
    aiState.history = [];
    renderAiChat();
  });
  $("aiChatSendBtn").addEventListener("click", () => sendAiMessage());
  $("aiChatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendAiMessage();
    }
  });

  // ComfyUI On-Demand Launcher
  function renderComfyState(st) {
    const badgeMap = { idle: "未启动", installing: "下载安装中", starting: "启动中", running: "运行中", error: "失败" };
    $("comfyStatusBadge").textContent = badgeMap[st.status] || st.status;
    const busy = st.status === "installing" || st.status === "starting";
    const running = st.status === "running";
    $("comfyProgressWrap").classList.toggle("hidden", !busy && st.status !== "error");
    if (busy || st.status === "error") {
      const pct = Math.round((st.progress || 0) * 100);
      $("comfyProgressMsg").textContent = st.message || "";
      $("comfyProgressPct").textContent = st.status === "error" ? "失败" : `${pct}%`;
      $("comfyProgressFill").style.width = `${pct}%`;
    }
    if ($("comfyGgufSelect")) {
      $("comfyGgufSelect").disabled = busy || running;
      if ((busy || running) && st.gguf_variant && Array.from($("comfyGgufSelect").options).some((o) => o.value === st.gguf_variant)) {
        $("comfyGgufSelect").value = st.gguf_variant;
      }
    }
    if ($("comfyGgufMeta")) {
      const pluginOk = st.gguf_plugin_installed ? "已安装 city96/ComfyUI-GGUF" : "将自动安装 city96/ComfyUI-GGUF";
      const stagedCount = Array.isArray(st.staged_files) ? st.staged_files.length : 0;
      const wfCount = Array.isArray(st.workflows) ? st.workflows.length : 0;
      if (running || stagedCount > 0 || wfCount > 0) {
        $("comfyGgufMeta").innerHTML = `${escapeHtml(pluginOk)} · 档位 ${(escapeHtml((st.gguf_variant || "q4").toUpperCase()))}` +
          (stagedCount ? ` · 已挂载 ${stagedCount} 个模型文件` : "") +
          (wfCount ? ` · 已生成 ${wfCount} 份工作流` : "");
      } else {
        $("comfyGgufMeta").innerHTML = '启动时自动安装 <a href="https://github.com/city96/ComfyUI-GGUF" target="_blank" style="color:var(--cyan)">city96/ComfyUI-GGUF</a> 扩展插件并生成 Qwen-Image-2.1 工作流';
      }
    }
    $("comfyLaunchBtn").disabled = busy || running;
    $("comfyLaunchBtn").classList.toggle("hidden", running);
    $("comfyStopBtn").classList.toggle("hidden", !running);
    const link = $("comfyOpenLink");
    link.classList.toggle("hidden", !running || !st.public_url);
    if (running && st.public_url) {
      link.href = st.public_url;
      if (!state.comfyOpenedOnce) {
        state.comfyOpenedOnce = true;
        window.open(st.public_url, "_blank");
      }
    }
  }

  async function pollComfyStatus() {
    try {
      const st = await apiFetch("/api/comfyui/status");
      renderComfyState(st);
      if (st.status !== "installing" && st.status !== "starting") {
        clearInterval(state.comfyPollTimer);
        state.comfyPollTimer = null;
      }
    } catch (_) {}
  }

  $("comfyLaunchBtn").addEventListener("click", async () => {
    try {
      state.comfyOpenedOnce = false;
      const ggufVariant = $("comfyGgufSelect") ? $("comfyGgufSelect").value : "q4";
      const st = await apiFetch("/api/comfyui/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gguf_variant: ggufVariant }),
      });
      renderComfyState(st);
      clearInterval(state.comfyPollTimer);
      state.comfyPollTimer = setInterval(pollComfyStatus, 900);
    } catch (err) {
      showToast(`ComfyUI 启动失败: ${err.message}`);
    }
  });

  $("comfyStopBtn").addEventListener("click", async () => {
    try {
      const st = await apiFetch("/api/comfyui/stop", { method: "POST" });
      clearInterval(state.comfyPollTimer);
      state.comfyPollTimer = null;
      renderComfyState(st);
      showToast("ComfyUI 已停止");
    } catch (_) {}
  });

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------
  async function init() {
    setMode("t2i");
    applyResolution(1024, 1024);
    try {
      const stats = await apiFetch("/api/stats");
      const eng = stats.engine || {};
      if (eng.model_id) {
        const msel = $("modelSelect");
        if (!Array.from(msel.options).some((o) => o.value === eng.model_id)) {
          const opt = document.createElement("option");
          opt.value = eng.model_id;
          opt.textContent = eng.model_id;
          msel.appendChild(opt);
        }
        msel.value = eng.model_id;
      }
      if (eng.quant_mode) $("quantSelect").value = eng.quant_mode;
      if (eng.gguf_variant && $("comfyGgufSelect") && (eng.gguf_variant === "q8" || eng.gguf_variant === "q4")) {
        $("comfyGgufSelect").value = eng.gguf_variant;
      }
      if (eng.gpu_profile_override) $("gpuProfileSelect").value = eng.gpu_profile_override;
      if (eng.dtype_mode) $("dtypeSelect").value = eng.dtype_mode;
      if (eng.offload_mode) $("offloadSelect").value = eng.offload_mode;
      if (eng.vae_tiling_mode) $("vaeTilingSelect").value = eng.vae_tiling_mode;
      if (eng.default_steps) {
        $("stepsSlider").value = String(eng.default_steps);
        $("stepsVal").textContent = String(eng.default_steps);
      }
      if (eng.has_hf_token) {
        $("hfTokenInput").placeholder = "已配置 HF Token (留空保持不变)";
      }
      if (stats.ai) {
        if (stats.ai.endpoint) $("aiEndpointInput").value = stats.ai.endpoint;
        if (stats.ai.model) {
          $("aiModelInput").value = stats.ai.model;
          $("aiModelBadge").textContent = stats.ai.model;
        }
        if (stats.ai.api_key) $("aiApiKeyInput").value = stats.ai.api_key;
      }
    } catch (_) {}
    refreshStats();
    refreshLoraList();
    pollComfyStatus();
    state.statsTimer = setInterval(refreshStats, 3500);
  }

  init();
})();
