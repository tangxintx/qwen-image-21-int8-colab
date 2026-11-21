(() => {
  "use strict";

  // -------------------------------------------------------------------------
  // Constants & Presets
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

  const TIER_BASE_PIXELS = {
    "1k": 1024 * 1024,
    "1.5k": 1536 * 1536,
    "2k": 2048 * 2048,
  };

  const MODE_LABELS = {
    t2i: "文生图",
    i2i: "图生图",
    hires: "高清修复",
    inpaint: "局部重绘",
  };

  const SUPPORTED_RATIOS = [1, 16 / 9, 9 / 16, 4 / 3, 3 / 4, 3 / 2, 2 / 3];
  const STORAGE_KEYS = {
    theme: "qwen_studio_theme_v2",
    accent: "qwen_studio_accent_v2",
    prefs: "qwen_studio_prefs_v2",
    history: "qwen_prompt_history_v2",
    favorites: "qwen_prompt_favorites_v2",
  };
  const ACCENT_THEMES = {
    blue: {
      light: { accent: "#2563eb", hover: "#1d4ed8", soft: "#eff6ff", border: "#bfdbfe", text: "#1d4ed8", shadow: "0 6px 18px rgba(37, 99, 235, 0.25)" },
      dark: { accent: "#3b82f6", hover: "#2563eb", soft: "rgba(59, 130, 246, 0.14)", border: "rgba(59, 130, 246, 0.34)", text: "#93c5fd", shadow: "0 6px 18px rgba(37, 99, 235, 0.34)" },
    },
    emerald: {
      light: { accent: "#059669", hover: "#047857", soft: "#ecfdf5", border: "#a7f3d0", text: "#047857", shadow: "0 6px 18px rgba(5, 150, 105, 0.25)" },
      dark: { accent: "#10b981", hover: "#059669", soft: "rgba(16, 185, 129, 0.14)", border: "rgba(16, 185, 129, 0.34)", text: "#6ee7b7", shadow: "0 6px 18px rgba(16, 185, 129, 0.34)" },
    },
    violet: {
      light: { accent: "#7c3aed", hover: "#6d28d9", soft: "#f5f3ff", border: "#ddd6fe", text: "#6d28d9", shadow: "0 6px 18px rgba(124, 58, 237, 0.25)" },
      dark: { accent: "#8b5cf6", hover: "#7c3aed", soft: "rgba(139, 92, 246, 0.15)", border: "rgba(139, 92, 246, 0.36)", text: "#c4b5fd", shadow: "0 6px 18px rgba(139, 92, 246, 0.34)" },
    },
    amber: {
      light: { accent: "#d97706", hover: "#b45309", soft: "#fffbeb", border: "#fde68a", text: "#b45309", shadow: "0 6px 18px rgba(217, 119, 6, 0.25)" },
      dark: { accent: "#f59e0b", hover: "#d97706", soft: "rgba(245, 158, 11, 0.15)", border: "rgba(245, 158, 11, 0.36)", text: "#fcd34d", shadow: "0 6px 18px rgba(245, 158, 11, 0.34)" },
    },
    rose: {
      light: { accent: "#e11d48", hover: "#be123c", soft: "#fff1f2", border: "#fecdd3", text: "#be123c", shadow: "0 6px 18px rgba(225, 29, 72, 0.25)" },
      dark: { accent: "#f43f5e", hover: "#e11d48", soft: "rgba(244, 63, 94, 0.15)", border: "rgba(244, 63, 94, 0.36)", text: "#fda4af", shadow: "0 6px 18px rgba(244, 63, 94, 0.34)" },
    },
    graphite: {
      light: { accent: "#3f3f46", hover: "#27272a", soft: "#f4f4f5", border: "#d4d4d8", text: "#27272a", shadow: "0 6px 18px rgba(39, 39, 42, 0.22)" },
      dark: { accent: "#71717a", hover: "#52525b", soft: "rgba(161, 161, 170, 0.16)", border: "rgba(161, 161, 170, 0.35)", text: "#e4e4e7", shadow: "0 6px 18px rgba(0, 0, 0, 0.35)" },
    },
  };
  const MAX_AUTO_RUN_ROUNDS = 5;

  // -------------------------------------------------------------------------
  // App State
  // -------------------------------------------------------------------------
  const state = {
    activeTab: "studio",
    mode: "t2i",
    tier: "1k",
    ratio: "1:1",
    width: 1024,
    height: 1024,
    batchSize: 1,
    modelDefaultSteps: 8,
    lastSeed: null,
    activeJobId: null,
    generating: false,
    statsTimer: null,
    statsInFlight: false,
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
    // Inpaint multi-image state
    inpaintItems: [], // { id, b64, width, height, maskB64, maskPreviewB64, hasMask }
    editingInpaintIdx: -1,
    inpaintTool: "brush",
    brushSize: 36,
    drawing: false,
    lastPoint: null,
    undoStack: [],
    lorasItems: [],
    lastGenError: null,
    lastGenPayload: null,
    notifiedJobId: null,
    promptDrawerTab: "history",
    aiAutoRun: false,
    accentColor: "blue",
  };

  // Offscreen exact-resolution (capped) mask canvas
  const maskDataCanvas = document.createElement("canvas");
  const maskDataCtx = maskDataCanvas.getContext("2d");

  // -------------------------------------------------------------------------
  // DOM & Utility Helpers
  // -------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function showToast(msg, duration = 2600) {
    const el = $("toastEl");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => el.classList.add("hidden"), duration);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function align16(val) {
    const n = Math.max(256, Math.min(3072, Math.round(Number(val) || 1024)));
    return Math.max(256, Math.floor(n / 16) * 16);
  }

  function parseSeed() {
    const raw = String($("seedInput").value ?? "").trim();
    if (raw === "") return -1;
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : -1;
  }

  async function copyToClipboard(text, successMsg = "已复制到剪贴板") {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      showToast(successMsg);
    } catch (_) {
      showToast("复制失败，请手动长按选择");
    }
  }

  function readJsonStorage(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return fallback;
      const parsed = JSON.parse(raw);
      return parsed ?? fallback;
    } catch (_) {
      return fallback;
    }
  }

  function writeJsonStorage(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (_) {}
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

  function apiPostJsonWithUploadProgress(url, payload, onUploadProgress = null) {
    return new Promise((resolve, reject) => {
      const bodyStr = JSON.stringify(payload);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);
      xhr.setRequestHeader("Content-Type", "application/json");
      if (onUploadProgress && xhr.upload) {
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable && ev.total > 0) {
            onUploadProgress(Math.min(1, ev.loaded / ev.total));
          }
        };
      }
      xhr.onload = () => {
        let parsed = null;
        try {
          parsed = JSON.parse(xhr.responseText || "{}");
        } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(parsed || {});
        } else {
          const detail = parsed?.detail || parsed?.message || `HTTP ${xhr.status}`;
          reject(new Error(detail));
        }
      };
      xhr.onerror = () => reject(new Error("网络连接失败，请检查服务是否在线"));
      xhr.send(bodyStr);
    });
  }

  function loadImageObj(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("图片加载失败"));
      img.src = src;
    });
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("文件读取失败"));
      reader.readAsDataURL(file);
    });
  }

  async function compressImageDataUrl(dataUrl, maxEdge = 1600, quality = 0.90) {
    const img = await loadImageObj(dataUrl);
    const maxDim = Math.max(img.width, img.height, 1);
    const scale = Math.min(1, maxEdge / maxDim);
    const w = Math.max(64, Math.round(img.width * scale));
    const h = Math.max(64, Math.round(img.height * scale));

    // Skip re-encoding if already small and compact
    if (scale === 1 && dataUrl.length <= 450_000) {
      return { dataUrl, img };
    }

    const cvs = document.createElement("canvas");
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    ctx.drawImage(img, 0, 0, w, h);

    // Check sampled alpha to preserve PNG only when genuine transparency exists
    let hasAlpha = false;
    try {
      const sample = ctx.getImageData(0, 0, w, h).data;
      const step = Math.max(4, Math.floor(sample.length / 4096) * 4);
      for (let i = 3; i < sample.length; i += step) {
        if (sample[i] < 250) {
          hasAlpha = true;
          break;
        }
      }
    } catch (_) {}

    const compressed = hasAlpha ? cvs.toDataURL("image/png") : cvs.toDataURL("image/jpeg", quality);
    const finalUrl = (scale === 1 && compressed.length > dataUrl.length) ? dataUrl : compressed;
    const finalImg = finalUrl === dataUrl ? img : await loadImageObj(finalUrl);
    return { dataUrl: finalUrl, img: finalImg };
  }

  // -------------------------------------------------------------------------
  // Theme & Accent Color Settings (Moved from Header into Settings Tab)
  // -------------------------------------------------------------------------
  function applyAccentColor(accentKey) {
    const key = ACCENT_THEMES[accentKey] ? accentKey : "blue";
    state.accentColor = key;
    const mode = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const pal = ACCENT_THEMES[key][mode];
    const root = document.documentElement.style;
    root.setProperty("--accent", pal.accent);
    root.setProperty("--accent-hover", pal.hover);
    root.setProperty("--accent-soft", pal.soft);
    root.setProperty("--accent-border", pal.border);
    root.setProperty("--accent-text", pal.text);
    root.setProperty("--shadow-fab", pal.shadow);

    document.querySelectorAll("#accentColorSelector .accent-swatch").forEach((b) => {
      b.classList.toggle("active", b.dataset.accent === key);
    });
    try {
      localStorage.setItem(STORAGE_KEYS.accent, key);
    } catch (_) {}
  }

  function applyTheme(theme) {
    const t = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    const meta = $("themeColorMeta");
    if (meta) meta.setAttribute("content", t === "dark" ? "#111214" : "#f5f5f7");
    document.querySelectorAll("#themeModeSelector .theme-mode-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.themeChoice === t);
    });
    applyAccentColor(state.accentColor || "blue");
    try {
      localStorage.setItem(STORAGE_KEYS.theme, t);
    } catch (_) {}
  }

  document.querySelectorAll("#themeModeSelector .theme-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyTheme(btn.dataset.themeChoice);
    });
  });

  document.querySelectorAll("#accentColorSelector .accent-swatch").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyAccentColor(btn.dataset.accent);
    });
  });

  document.querySelectorAll("#settingsSubNav .subnav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.snav;
      document.querySelectorAll("#settingsSubNav .subnav-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".settings-subpanel").forEach((p) => {
        p.classList.toggle("active", p.id === `settings-panel-${target}`);
      });
    });
  });

  // -------------------------------------------------------------------------
  // Local Studio Parameter Persistence + Prompt History & Favorites
  // -------------------------------------------------------------------------
  let _savePrefsTimer = null;
  function scheduleSaveStudioPrefs() {
    clearTimeout(_savePrefsTimer);
    _savePrefsTimer = setTimeout(() => {
      writeJsonStorage(STORAGE_KEYS.prefs, {
        mode: state.mode,
        tier: state.tier,
        ratio: state.ratio,
        width: state.width,
        height: state.height,
        batchSize: state.batchSize,
        prompt: $("promptInput").value,
        negative_prompt: $("negativePromptInput").value,
        steps: Number($("stepsSlider").value),
        denoise: Number($("denoiseSlider").value),
        cfg: Number($("cfgSlider").value),
        hires_scale: Number($("hiresScaleSlider").value),
        mask_blur: Number($("maskBlurSlider").value),
        lora_enabled: $("loraEnableToggle").checked,
        lora_weight: Number($("loraWeightSlider").value),
        lora_trigger: $("loraTriggerInput").value,
      });
    }, 250);
  }

  function restoreStudioPrefs() {
    const saved = readJsonStorage(STORAGE_KEYS.prefs, null);
    if (!saved || typeof saved !== "object") return false;
    if (saved.mode && ["t2i", "i2i", "hires", "inpaint"].includes(saved.mode)) {
      setMode(saved.mode, true);
    }
    if (saved.tier && ["1k", "1.5k", "2k"].includes(saved.tier)) {
      state.tier = saved.tier;
      document.querySelectorAll("#tierSelector .seg-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.tier === state.tier);
      });
    }
    if (saved.ratio && saved.ratio !== "original") {
      state.ratio = saved.ratio;
      updateRatioChipUI();
    }
    if (saved.width && saved.height) {
      applyResolution(saved.width, saved.height, true);
    }
    if (typeof saved.prompt === "string") $("promptInput").value = saved.prompt;
    if (typeof saved.negative_prompt === "string") $("negativePromptInput").value = saved.negative_prompt;
    if (Number.isFinite(saved.steps)) {
      $("stepsSlider").value = String(saved.steps);
      $("stepsVal").textContent = String(saved.steps);
    }
    if (Number.isFinite(saved.denoise)) {
      $("denoiseSlider").value = String(saved.denoise);
      $("denoiseVal").textContent = Number(saved.denoise).toFixed(2);
    }
    if (Number.isFinite(saved.cfg)) {
      $("cfgSlider").value = String(saved.cfg);
      $("cfgVal").textContent = Number(saved.cfg).toFixed(1);
    }
    if (Number.isFinite(saved.hires_scale)) {
      $("hiresScaleSlider").value = String(saved.hires_scale);
      $("hiresScaleVal").textContent = `${Number(saved.hires_scale).toFixed(2)}x`;
    }
    if (Number.isFinite(saved.mask_blur)) {
      $("maskBlurSlider").value = String(saved.mask_blur);
      $("maskBlurVal").textContent = `${saved.mask_blur}px`;
    }
    if (Number.isFinite(saved.batchSize)) {
      state.batchSize = Math.max(1, Math.min(8, saved.batchSize));
      $("batchVal").textContent = String(state.batchSize);
    }
    if (typeof saved.lora_enabled === "boolean") {
      $("loraEnableToggle").checked = saved.lora_enabled;
      $("loraPanelBody").classList.toggle("hidden", !saved.lora_enabled);
    }
    if (Number.isFinite(saved.lora_weight)) {
      $("loraWeightSlider").value = String(saved.lora_weight);
      $("loraWeightVal").textContent = Number(saved.lora_weight).toFixed(2);
    }
    if (typeof saved.lora_trigger === "string") {
      $("loraTriggerInput").value = saved.lora_trigger;
    }
    return true;
  }

  function getPromptHistory() {
    const arr = readJsonStorage(STORAGE_KEYS.history, []);
    return Array.isArray(arr) ? arr : [];
  }

  function getPromptFavorites() {
    const arr = readJsonStorage(STORAGE_KEYS.favorites, []);
    return Array.isArray(arr) ? arr : [];
  }

  function recordPromptHistory(promptText) {
    const clean = (promptText || "").trim();
    if (!clean) return;
    const list = getPromptHistory().filter((x) => x !== clean);
    list.unshift(clean);
    writeJsonStorage(STORAGE_KEYS.history, list.slice(0, 20));
    if (!$("promptDrawer").classList.contains("hidden")) renderPromptDrawer();
  }

  function togglePromptFavorite(promptText) {
    const clean = (promptText || "").trim();
    if (!clean) return showToast("当前提示词为空");
    const favs = getPromptFavorites();
    const idx = favs.indexOf(clean);
    if (idx >= 0) {
      favs.splice(idx, 1);
      showToast("已取消收藏");
    } else {
      favs.unshift(clean);
      showToast("已收藏至常用模板");
    }
    writeJsonStorage(STORAGE_KEYS.favorites, favs.slice(0, 30));
    renderPromptDrawer();
  }

  function renderPromptDrawer() {
    const listEl = $("promptDrawerList");
    const isFav = state.promptDrawerTab === "favorites";
    const items = isFav ? getPromptFavorites() : getPromptHistory();
    const favSet = new Set(getPromptFavorites());
    listEl.innerHTML = "";

    if (!items.length) {
      listEl.innerHTML = `<div class="card-sub" style="padding:8px 4px">${isFav ? "暂无收藏模板，点击右上角「收藏当前」保存常用提示词" : "暂无历史记录，生成图片后将自动记录最近提示词"}</div>`;
      return;
    }

    items.forEach((text) => {
      const row = document.createElement("div");
      row.className = "prompt-item-row";
      const starred = favSet.has(text);
      row.innerHTML = `
        <span class="prompt-item-text" title="${escapeHtml(text)}">${escapeHtml(text)}</span>
        <div class="prompt-item-btns">
          <button class="prompt-star-btn ${starred ? "starred" : ""}" data-act="star" title="${starred ? "取消收藏" : "收藏"}" type="button">
            <svg class="icon"><use href="#ic-star" /></svg>
          </button>
          <button class="prompt-star-btn" data-act="del" title="移除" type="button">
            <svg class="icon"><use href="#ic-close" /></svg>
          </button>
        </div>
      `;
      row.querySelector(".prompt-item-text").addEventListener("click", () => {
        $("promptInput").value = text;
        scheduleSaveStudioPrefs();
        showToast("已填入提示词");
      });
      row.querySelector('[data-act="star"]').addEventListener("click", (ev) => {
        ev.stopPropagation();
        togglePromptFavorite(text);
      });
      row.querySelector('[data-act="del"]').addEventListener("click", (ev) => {
        ev.stopPropagation();
        const key = isFav ? STORAGE_KEYS.favorites : STORAGE_KEYS.history;
        const cur = (isFav ? getPromptFavorites() : getPromptHistory()).filter((x) => x !== text);
        writeJsonStorage(key, cur);
        renderPromptDrawer();
      });
      listEl.appendChild(row);
    });
  }

  $("promptHistoryToggleBtn").addEventListener("click", () => {
    const drawer = $("promptDrawer");
    const opening = drawer.classList.contains("hidden");
    drawer.classList.toggle("hidden", !opening);
    $("promptHistoryToggleBtn").classList.toggle("active", opening);
    if (opening) renderPromptDrawer();
  });

  document.querySelectorAll("#promptDrawerTabs .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.promptDrawerTab = btn.dataset.pdrawer;
      document.querySelectorAll("#promptDrawerTabs .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderPromptDrawer();
    });
  });

  $("saveCurrentPromptFavBtn").addEventListener("click", () => {
    togglePromptFavorite($("promptInput").value);
  });

  // -------------------------------------------------------------------------
  // Navigation & Tabs (Desktop Top Nav + Mobile Bottom Nav Synced)
  // -------------------------------------------------------------------------
  function updateFabVisibility() {
    const isStudio = state.activeTab === "studio";
    $("generateFabDock").classList.toggle("hidden", !isStudio);
    if ($("headerGenStatusBtn")) {
      $("headerGenStatusBtn").classList.toggle("hidden", isStudio || !state.generating);
    }
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".nav-item, .dnav-item").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabName);
    });
    const panel = $(`tab-${tabName}`);
    if (panel) panel.classList.add("active");
    updateFabVisibility();

    if (tabName === "gallery") loadGallery();
    if (tabName === "monitor") refreshStats();
    if (tabName === "terminal") onEnterTerminalTab();
  }

  document.querySelectorAll(".nav-item, .dnav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  $("brandStatusTrigger").addEventListener("click", () => switchTab("monitor"));
  if ($("headerGenStatusBtn")) {
    $("headerGenStatusBtn").addEventListener("click", () => switchTab("studio"));
  }
  $("headerAiBtn").addEventListener("click", () => {
    openAiModal(state.activeTab === "terminal" ? "terminal" : "studio");
  });

  // -------------------------------------------------------------------------
  // Mode Switching
  // -------------------------------------------------------------------------
  function setMode(mode, skipDenoiseReset = false) {
    state.mode = mode;
    document.querySelectorAll(".mode-tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.mode === mode);
    });

    const needsImage = mode !== "t2i";
    $("imageInputCard").classList.toggle("hidden", !needsImage);

    const titles = {
      i2i: "图生图 · 多图参考 (@0 主图)",
      hires: "高清修复 · 原始底图",
      inpaint: "局部重绘 · 点击图片涂抹遮罩",
    };
    if (titles[mode]) $("imageInputTitle").textContent = titles[mode];
    if ($("uploadPrimaryLabel")) {
      $("uploadPrimaryLabel").textContent = mode === "inpaint" ? "添加多图" : "上传主图";
    }
    if ($("uploadHintText")) {
      $("uploadHintText").textContent = mode === "inpaint"
        ? "支持多选上传、拖拽或 Ctrl+V 粘贴，点击缩略图涂抹重绘区域"
        : "点击上传、拖拽图片到此处，或直接 Ctrl+V 粘贴截图";
    }

    if (!skipDenoiseReset) {
      const denoiseSlider = $("denoiseSlider");
      if (mode === "hires") denoiseSlider.value = "0.40";
      else if (mode === "i2i") denoiseSlider.value = "0.65";
      else if (mode === "inpaint") denoiseSlider.value = "0.85";
      $("denoiseVal").textContent = Number(denoiseSlider.value).toFixed(2);
    }
    $("denoiseBlock").classList.toggle("hidden", mode === "t2i");

    refreshImageStage();
    if (!skipDenoiseReset) scheduleSaveStudioPrefs();
  }

  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  // -------------------------------------------------------------------------
  // Resolution & Aspect Ratio Controls
  // -------------------------------------------------------------------------
  function applyResolution(w, h, skipSave = false) {
    state.width = align16(w);
    state.height = align16(h);
    $("widthInput").value = state.width;
    $("heightInput").value = state.height;
    $("currentResBadge").textContent = `${state.width} × ${state.height}`;
    if (!skipSave) scheduleSaveStudioPrefs();
  }

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
        showToast("请先上传参考主图 (@0)");
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
      if (state.ratio === "custom" && $("manualResDetails")) {
        $("manualResDetails").open = true;
      }
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
  // Reference Images, Drag & Drop, Ctrl+V Paste & Inpaint Canvas
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
    scheduleSaveStudioPrefs();
  }

  function isSupportedAspectRatio(w, h) {
    if (!w || !h) return true;
    const r = w / h;
    return SUPPORTED_RATIOS.some((sr) => Math.abs(Math.log(sr / r)) <= 0.025);
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
          <button class="extra-ref-mini-btn" data-act="promote" title="设为主图 (@0)" type="button">
            <svg class="icon"><use href="#ic-arrow-up" /></svg>
          </button>
          <button class="extra-ref-mini-btn" data-act="remove" title="移除" type="button">
            <svg class="icon"><use href="#ic-close" /></svg>
          </button>
        </div>
      `;
      card.querySelector("img").addEventListener("click", () => insertPromptToken(`@${refNum}`));
      card.querySelector(".extra-ref-tag").addEventListener("click", () => insertPromptToken(`@${refNum}`));
      card.querySelector('[data-act="promote"]').addEventListener("click", async () => {
        const oldPrimary = state.initImageB64
          ? { b64: state.initImageB64, width: state.initImageObj?.width || 1024, height: state.initImageObj?.height || 1024 }
          : null;
        state.extraRefImages.splice(idx, 1);
        if (oldPrimary) state.extraRefImages.unshift(oldPrimary);
        await loadInitImageFromDataUrl(item.b64);
      });
      card.querySelector('[data-act="remove"]').addEventListener("click", () => {
        state.extraRefImages.splice(idx, 1);
        renderExtraRefs();
      });
      grid.appendChild(card);
    });

    const bar = $("refMentionBar");
    const inpaintRefCount = state.mode === "inpaint" ? Math.max(0, state.inpaintItems.length - 1) : 0;
    const showMentions =
      (isI2I && (Boolean(state.initImageObj) || extras.length > 0)) ||
      (state.mode === "inpaint" && state.inpaintItems.length > 0);
    bar.classList.toggle("hidden", !showMentions);
    bar.innerHTML = "";
    if (showMentions) {
      const tokens = [];
      if (state.mode === "inpaint") {
        tokens.push({ label: "@0 主图", val: "@0" });
        for (let i = 1; i <= inpaintRefCount; i++) tokens.push({ label: `@${i}`, val: `@${i}` });
      } else {
        if (state.initImageObj) tokens.push({ label: "@0 主图", val: "@0" });
        extras.forEach((_, idx) => tokens.push({ label: `@${idx + 1}`, val: `@${idx + 1}` }));
        if (extras.length >= 2) {
          const multi = extras.map((_, idx) => idx + 1).join(",");
          tokens.push({ label: `@${multi}`, val: `@${multi}` });
        }
      }
      tokens.forEach((t) => {
        const b = document.createElement("button");
        b.className = "tag-chip ref-mention";
        b.type = "button";
        b.textContent = t.label;
        b.addEventListener("click", () => insertPromptToken(t.val));
        bar.appendChild(b);
      });
    }
  }

  function renderInpaintMultiGrid() {
    const isInpaint = state.mode === "inpaint";
    const items = state.inpaintItems;
    $("inpaintMultiSection").classList.toggle("hidden", !isInpaint || items.length === 0);
    $("inpaintCountBadge").textContent = `${items.length} 张`;
    const grid = $("inpaintMultiGrid");
    grid.innerHTML = "";
    if (!isInpaint || !items.length) return;

    items.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = `inpaint-item-card ${idx === 0 ? "primary" : ""}`;
      const roleLabel = idx === 0 ? "@0 主图" : `@${idx} 参考`;
      card.innerHTML = `
        <div class="inpaint-thumb-wrap" title="点击进入涂抹遮罩">
          <img class="inpaint-thumb-img" src="${item.b64}" alt="inpaint-${idx}" />
          ${item.hasMask && item.maskPreviewB64 ? `<img class="inpaint-mask-overlay-img" src="${item.maskPreviewB64}" alt="mask" />` : ""}
          <span class="ref-role-badge">${roleLabel}</span>
          ${item.hasMask ? '<span class="inpaint-mask-badge">已涂抹</span>' : ""}
          <div class="inpaint-tap-hint">点击涂抹遮罩</div>
        </div>
        <div class="inpaint-item-foot">
          <span>${item.width}×${item.height}</span>
          <div style="display:flex;gap:4px">
            ${idx !== 0 ? '<button class="mini-action" data-act="primary" type="button">主图</button>' : ""}
            <button class="mini-action" data-act="del" type="button">删</button>
          </div>
        </div>
      `;
      card.querySelector(".inpaint-thumb-wrap").addEventListener("click", () => openInpaintModal(idx));
      const priBtn = card.querySelector('[data-act="primary"]');
      if (priBtn) {
        priBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          const [picked] = state.inpaintItems.splice(idx, 1);
          state.inpaintItems.unshift(picked);
          await syncPrimaryFromInpaintTop();
          refreshImageStage();
        });
      }
      card.querySelector('[data-act="del"]').addEventListener("click", async (ev) => {
        ev.stopPropagation();
        state.inpaintItems.splice(idx, 1);
        await syncPrimaryFromInpaintTop();
        refreshImageStage();
      });
      grid.appendChild(card);
    });
  }

  async function syncPrimaryFromInpaintTop() {
    const top = state.inpaintItems[0];
    if (!top) {
      if (state.mode === "inpaint") {
        state.initImageB64 = null;
        state.initImageObj = null;
      }
      return;
    }
    const img = await loadImageObj(top.b64);
    state.initImageB64 = top.b64;
    state.initImageObj = img;
    if (state.ratio === "original") applyOriginalAspectResolution(img);
  }

  function refreshImageStage() {
    const isInpaint = state.mode === "inpaint";
    const hasImg = isInpaint ? state.inpaintItems.length > 0 : Boolean(state.initImageObj);
    $("uploadPlaceholder").classList.toggle("hidden", hasImg);
    $("clearImageBtn").classList.toggle(
      "hidden",
      isInpaint ? state.inpaintItems.length === 0 : (!hasImg && state.extraRefImages.length === 0)
    );

    if (!hasImg) {
      $("refPreviewWrap").classList.add("hidden");
      $("inpaintMultiSection").classList.add("hidden");
      renderExtraRefs();
      return;
    }

    $("refPreviewWrap").classList.toggle("hidden", isInpaint);
    if (!isInpaint) {
      $("refPreviewImg").src = state.initImageB64;
      $("refImageMeta").textContent = `${state.initImageObj.width} × ${state.initImageObj.height}`;
    } else {
      renderInpaintMultiGrid();
    }
    renderExtraRefs();
  }

  function resetMaskCanvas() {
    maskDataCanvas.width = 0;
    maskDataCanvas.height = 0;
    const overlayCanvas = $("maskOverlayCanvas");
    if (overlayCanvas) {
      overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  async function addInpaintImageFromDataUrl(dataUrl) {
    if (state.inpaintItems.length >= 10) {
      showToast("局部重绘最多支持 10 张图片");
      return;
    }
    const { dataUrl: optUrl, img } = await compressImageDataUrl(dataUrl, 1920, 0.91);
    state.inpaintItems.push({
      id: `${Date.now()}_${Math.random().toString(16).slice(2, 7)}`,
      b64: optUrl,
      width: img.width,
      height: img.height,
      maskB64: null,
      maskPreviewB64: null,
      hasMask: false,
    });
    if (state.inpaintItems.length === 1) {
      state.initImageB64 = optUrl;
      state.initImageObj = img;
      state.ratio = "original";
      updateRatioChipUI();
      applyOriginalAspectResolution(img);
    }
    refreshImageStage();
    if (!isSupportedAspectRatio(img.width, img.height)) {
      showToast("非预设比例图片将自动透明扩图并在输出后裁剪还原", 3500);
    }
  }

  async function loadInitImageFromDataUrl(dataUrl) {
    if (state.mode === "inpaint") {
      await addInpaintImageFromDataUrl(dataUrl);
      return;
    }
    const { dataUrl: optUrl, img } = await compressImageDataUrl(dataUrl, 1920, 0.91);
    state.initImageB64 = optUrl;
    state.initImageObj = img;
    state.undoStack = [];
    resetMaskCanvas();
    state.ratio = "original";
    updateRatioChipUI();
    applyOriginalAspectResolution(img);
    refreshImageStage();
    if (!isSupportedAspectRatio(img.width, img.height)) {
      showToast("原图比例非模型预设比例，生成时将自动透明扩图并还原裁剪", 3500);
    }
  }

  async function addExtraRefFromDataUrl(dataUrl) {
    if (state.extraRefImages.length >= 9) {
      showToast("最多支持 9 张次图 (@1~@9)");
      return state.extraRefImages.length;
    }
    const { dataUrl: optUrl, img } = await compressImageDataUrl(dataUrl, 1400, 0.88);
    state.extraRefImages.push({ b64: optUrl, width: img.width, height: img.height });
    renderExtraRefs();
    return state.extraRefImages.length;
  }

  async function addExtraRefFromUrl(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const dataUrl = await readFileAsDataUrl(blob);
    return await addExtraRefFromDataUrl(dataUrl);
  }

  async function loadInitImageFromUrl(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const dataUrl = await readFileAsDataUrl(blob);
    await loadInitImageFromDataUrl(dataUrl);
  }

  async function handleMultiFileSelection(fileList, asExtra = false) {
    const files = Array.from(fileList || []).filter((f) => f && f.type.startsWith("image/"));
    if (!files.length) return;
    if (state.mode === "t2i") {
      setMode("i2i");
      showToast("已导入参考图并切换至「图生图」模式");
    }
    if (state.mode === "inpaint") {
      for (const f of files) {
        const d = await readFileAsDataUrl(f);
        await addInpaintImageFromDataUrl(d);
      }
      return;
    }
    if (asExtra && state.mode === "i2i") {
      for (const f of files) {
        const d = await readFileAsDataUrl(f);
        await addExtraRefFromDataUrl(d);
      }
      return;
    }
    const firstUrl = await readFileAsDataUrl(files[0]);
    await loadInitImageFromDataUrl(firstUrl);
    if (state.mode === "i2i" && files.length > 1) {
      for (const f of files.slice(1)) {
        const d = await readFileAsDataUrl(f);
        await addExtraRefFromDataUrl(d);
      }
    }
  }

  $("primaryRefBadge").addEventListener("click", () => insertPromptToken("@0"));
  $("uploadPlaceholder").addEventListener("click", () => $("imageFileInput").click());

  $("imageFileInput").addEventListener("change", async (e) => {
    await handleMultiFileSelection(e.target.files, false);
    e.target.value = "";
  });
  $("inpaintAddFileInput").addEventListener("change", async (e) => {
    await handleMultiFileSelection(e.target.files, false);
    e.target.value = "";
  });
  $("extraImageFileInput").addEventListener("change", async (e) => {
    await handleMultiFileSelection(e.target.files, true);
    e.target.value = "";
  });

  // Drag & Drop support on Reference Card (and Studio controls)
  const dropZone = $("imageInputCard");
  ["dragenter", "dragover"].forEach((evName) => {
    dropZone.addEventListener(evName, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((evName) => {
    dropZone.addEventListener(evName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });
  dropZone.addEventListener("drop", async (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length) {
      const appendExtra = state.mode === "i2i" && Boolean(state.initImageObj);
      await handleMultiFileSelection(files, appendExtra);
    }
  });

  // Global Ctrl+V Paste image upload in Studio
  window.addEventListener("paste", async (e) => {
    if (state.activeTab !== "studio") return;
    if (hasOpenModal()) return;
    const items = Array.from(e.clipboardData?.items || []);
    const imgFiles = items
      .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
      .map((it) => it.getAsFile())
      .filter(Boolean);
    if (!imgFiles.length) return;
    e.preventDefault();
    const appendExtra = state.mode === "i2i" && Boolean(state.initImageObj);
    await handleMultiFileSelection(imgFiles, appendExtra);
    showToast(appendExtra ? "已粘贴为次图参考" : "已从剪贴板粘贴参考图");
  });

  $("clearImageBtn").addEventListener("click", () => {
    state.initImageB64 = null;
    state.initImageObj = null;
    state.extraRefImages = [];
    state.inpaintItems = [];
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
    showToast("点击作品库任意图片即可送入当前模式");
    switchTab("gallery");
  });

  // Inpaint Modal Painter (Dual Width + Height Constrained for Accurate Pointer Mapping)
  async function openInpaintModal(idx) {
    const item = state.inpaintItems[idx];
    if (!item) return;
    state.editingInpaintIdx = idx;
    state.undoStack = [];
    $("inpaintModalTitle").textContent = `涂抹遮罩 · #${idx + 1} (${item.width}×${item.height})`;
    $("inpaintCanvasModal").classList.remove("hidden");
    const img = await loadImageObj(item.b64);
    requestAnimationFrame(() => setupInpaintCanvases(img, item.maskB64));
  }

  function closeInpaintModal(save) {
    const idx = state.editingInpaintIdx;
    if (save && idx >= 0 && state.inpaintItems[idx]) {
      const painted = hasNonEmptyMask();
      state.inpaintItems[idx].hasMask = painted;
      state.inpaintItems[idx].maskB64 = painted ? maskDataCanvas.toDataURL("image/png") : null;
      state.inpaintItems[idx].maskPreviewB64 = painted ? $("maskOverlayCanvas").toDataURL("image/png") : null;
      renderInpaintMultiGrid();
      showToast(painted ? `已保存 #${idx + 1} 遮罩` : `已清空 #${idx + 1} 遮罩`);
    }
    state.editingInpaintIdx = -1;
    $("inpaintCanvasModal").classList.add("hidden");
  }

  $("inpaintModalSaveBtn").addEventListener("click", () => closeInpaintModal(true));
  $("inpaintModalCancelBtn").addEventListener("click", () => closeInpaintModal(false));
  $("inpaintModalCloseBtn").addEventListener("click", () => closeInpaintModal(false));
  $("inpaintModalBackdrop").addEventListener("click", () => closeInpaintModal(false));

  function setupInpaintCanvases(img, existingMaskB64 = null) {
    const bgCanvas = $("bgCanvas");
    const overlayCanvas = $("maskOverlayCanvas");
    const bgCtx = bgCanvas.getContext("2d");
    const ovCtx = overlayCanvas.getContext("2d");

    const stage = $("canvasStage");
    const maxW = Math.max(240, (stage.clientWidth || 340) - 8);
    const maxH = Math.max(200, (stage.clientHeight || 380) - 8);
    const scale = Math.min(maxW / img.width, maxH / img.height, 1.5);
    const dispW = Math.max(120, Math.round(img.width * scale));
    const dispH = Math.max(120, Math.round(img.height * scale));

    bgCanvas.width = dispW;
    bgCanvas.height = dispH;
    bgCanvas.style.width = `${dispW}px`;
    bgCanvas.style.height = `${dispH}px`;

    overlayCanvas.width = dispW;
    overlayCanvas.height = dispH;
    overlayCanvas.style.width = `${dispW}px`;
    overlayCanvas.style.height = `${dispH}px`;

    bgCtx.clearRect(0, 0, dispW, dispH);
    bgCtx.drawImage(img, 0, 0, dispW, dispH);

    const maskScale = Math.min(1, 1280 / Math.max(img.width, img.height, 1));
    const mw = Math.max(64, Math.round(img.width * maskScale));
    const mh = Math.max(64, Math.round(img.height * maskScale));
    maskDataCanvas.width = mw;
    maskDataCanvas.height = mh;
    maskDataCtx.fillStyle = "#000000";
    maskDataCtx.fillRect(0, 0, mw, mh);
    ovCtx.clearRect(0, 0, dispW, dispH);

    if (existingMaskB64) {
      loadImageObj(existingMaskB64).then((mImg) => {
        maskDataCtx.drawImage(mImg, 0, 0, mw, mh);
        syncOverlayFromMaskData();
      }).catch(() => {});
    }
  }

  function syncOverlayFromMaskData() {
    const overlayCanvas = $("maskOverlayCanvas");
    const ovCtx = overlayCanvas.getContext("2d");
    ovCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

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
        d[i] = 37;
        d[i + 1] = 99;
        d[i + 2] = 235;
        d[i + 3] = Math.round((lum / 255) * 145);
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
    return {
      x: ((evt.clientX - rect.left) / Math.max(1, rect.width)) * overlay.width,
      y: ((evt.clientY - rect.top) / Math.max(1, rect.height)) * overlay.height,
    };
  }

  function drawStrokeSegment(p1, p2) {
    const overlay = $("maskOverlayCanvas");
    const ovCtx = overlay.getContext("2d");
    const scaleX = maskDataCanvas.width / Math.max(1, overlay.width);
    const scaleY = maskDataCanvas.height / Math.max(1, overlay.height);
    const isEraser = state.inpaintTool === "eraser";

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

    ovCtx.save();
    ovCtx.lineCap = "round";
    ovCtx.lineJoin = "round";
    ovCtx.lineWidth = state.brushSize;
    if (isEraser) {
      ovCtx.globalCompositeOperation = "destination-out";
      ovCtx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      ovCtx.globalCompositeOperation = "source-over";
      ovCtx.strokeStyle = "rgba(37, 99, 235, 0.55)";
    }
    ovCtx.beginPath();
    ovCtx.moveTo(p1.x, p1.y);
    ovCtx.lineTo(p2.x, p2.y);
    ovCtx.stroke();
    ovCtx.restore();
  }

  const overlayEl = $("maskOverlayCanvas");
  overlayEl.style.touchAction = "none";
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
    showToast(locked ? "已锁屏（禁止涂抹，可上下滑动）" : "已解锁（可涂抹遮罩）");
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
  $("promptInput").addEventListener("input", scheduleSaveStudioPrefs);
  $("negativePromptInput").addEventListener("input", scheduleSaveStudioPrefs);
  $("clearPromptBtn").addEventListener("click", () => {
    $("promptInput").value = "";
    scheduleSaveStudioPrefs();
  });

  document.querySelectorAll("#quickTags .tag-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const box = $("promptInput");
      const tag = chip.dataset.tag;
      box.value = box.value.trim() ? `${box.value.trim()}, ${tag}` : tag;
      scheduleSaveStudioPrefs();
    });
  });

  $("stepsSlider").addEventListener("input", (e) => {
    $("stepsVal").textContent = e.target.value;
    scheduleSaveStudioPrefs();
  });
  document.querySelectorAll(".mini-pill[data-steps]").forEach((p) => {
    p.addEventListener("click", () => {
      $("stepsSlider").value = p.dataset.steps;
      $("stepsVal").textContent = p.dataset.steps;
      scheduleSaveStudioPrefs();
    });
  });
  $("denoiseSlider").addEventListener("input", (e) => {
    $("denoiseVal").textContent = Number(e.target.value).toFixed(2);
    scheduleSaveStudioPrefs();
  });
  $("cfgSlider").addEventListener("input", (e) => {
    $("cfgVal").textContent = Number(e.target.value).toFixed(1);
    scheduleSaveStudioPrefs();
  });
  $("hiresScaleSlider").addEventListener("input", (e) => {
    $("hiresScaleVal").textContent = `${Number(e.target.value).toFixed(2)}x`;
    scheduleSaveStudioPrefs();
  });
  $("maskBlurSlider").addEventListener("input", (e) => {
    $("maskBlurVal").textContent = `${e.target.value}px`;
    scheduleSaveStudioPrefs();
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
    scheduleSaveStudioPrefs();
  });
  $("batchIncBtn").addEventListener("click", () => {
    state.batchSize = Math.min(8, state.batchSize + 1);
    $("batchVal").textContent = state.batchSize;
    scheduleSaveStudioPrefs();
  });
  $("resetParamsBtn").addEventListener("click", () => {
    const defSteps = state.modelDefaultSteps || 8;
    $("stepsSlider").value = String(defSteps);
    $("stepsVal").textContent = String(defSteps);
    $("denoiseSlider").value = "0.75";
    $("denoiseVal").textContent = "0.75";
    $("cfgSlider").value = "1.0";
    $("cfgVal").textContent = "1.0";
    $("hiresScaleSlider").value = "1.5";
    $("hiresScaleVal").textContent = "1.50x";
    $("maskBlurSlider").value = "8";
    $("maskBlurVal").textContent = "8px";
    $("seedInput").value = "-1";
    state.batchSize = 1;
    $("batchVal").textContent = "1";
    scheduleSaveStudioPrefs();
    showToast(`已恢复默认参数 (${defSteps} 步)`);
  });

  // -------------------------------------------------------------------------
  // Generation Flow & Serial Job Polling (Zero Overlapping Requests)
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
    state.generating = active;
    $("generateBtn").disabled = active;
    $("generateBtn").innerHTML = active
      ? '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>生成中</span>'
      : '<svg class="icon"><use href="#ic-play" /></svg><span>生成</span>';
    $("cancelGenBtn").classList.toggle("hidden", !active);
    $("cancelGenBtn").disabled = false;
    $("cancelGenBtn").innerHTML = '<svg class="icon"><use href="#ic-stop" /></svg><span>停止</span>';
    $("progressWrap").classList.toggle("hidden", !active);
    if ($("headerGenStatusText")) {
      $("headerGenStatusText").textContent = active ? "生成中" : "";
    }
    if (!active) {
      $("progressBarFill").style.width = "0%";
    }
    updateFabVisibility();
  }

  function formatShortToastError(msg) {
    const raw = String(msg || "未知错误").trim();
    if (/out of memory/i.test(raw)) return "生成失败: CUDA 显存不足，详情见诊断提示";
    return raw.length > 72 ? `生成失败: ${raw.slice(0, 70)}...` : `生成失败: ${raw}`;
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

  $("dismissErrorBannerBtn").addEventListener("click", () => {
    $("genErrorBanner").classList.add("hidden");
  });

  async function runSingleGenerationPayload(payload, appendResults = false, accumulatedResults = [], totalElapsedAcc = 0) {
    state.lastGenPayload = {
      ...payload,
      init_image_b64: Boolean(payload.init_image_b64),
      mask_image_b64: Boolean(payload.mask_image_b64),
      ref_images_b64: (payload.ref_images_b64 || []).length,
    };

    const resp = await apiPostJsonWithUploadProgress("/api/generate", payload, (ratio) => {
      const pct = Math.round(ratio * 100);
      if (pct > 0 && pct < 100) {
        $("progressMsg").textContent = "上传图片与参数...";
        $("progressPct").textContent = `${pct}%`;
        $("progressBarFill").classList.remove("indeterminate");
        $("progressBarFill").style.width = `${pct}%`;
      } else if (pct >= 100) {
        $("progressMsg").textContent = "已提交，等待排队...";
        $("progressPct").textContent = "准备中";
        $("progressBarFill").classList.add("indeterminate");
      }
    });
    state.activeJobId = resp.job_id;
    return await waitJobCompletion(resp.job_id, appendResults, accumulatedResults, totalElapsedAcc);
  }

  async function waitJobCompletion(jobId, appendResults = false, accumulatedResults = [], totalElapsedAcc = 0) {
    while (true) {
      const job = await apiFetch(`/api/jobs/${jobId}`);
      const isQueued = job.status === "queued";
      const pct = Math.round((job.progress || 0) * 100);
      const showIndeterminate = isQueued || (job.status === "loading_model" && pct <= 2);
      $("progressMsg").textContent = job.message || "生成中...";
      $("progressPct").textContent = showIndeterminate ? "准备中" : `${pct}%`;
      if ($("headerGenStatusText")) {
        $("headerGenStatusText").textContent = showIndeterminate ? "准备中" : `${pct}%`;
      }
      $("progressBarFill").classList.toggle("indeterminate", showIndeterminate);
      if (!showIndeterminate) {
        $("progressBarFill").style.width = `${pct}%`;
      }

      if (job.notice) {
        $("genNoticeBanner").textContent = job.notice;
        $("genNoticeBanner").classList.remove("hidden");
        if (state.notifiedJobId !== jobId) {
          state.notifiedJobId = jobId;
          showToast(job.notice, 4200);
        }
      }

      if (["completed", "cancelled", "error"].includes(job.status)) {
        state.activeJobId = null;
        if (job.status === "completed") {
          const allItems = appendResults ? [...accumulatedResults, ...(job.results || [])] : (job.results || []);
          const totTime = Number((totalElapsedAcc + (job.elapsed_sec || 0)).toFixed(2));
          renderLatestResults(allItems, totTime);
          if (job.results?.length) {
            state.lastSeed = job.results[job.results.length - 1].seed;
          }
          return { status: "completed", results: allItems, elapsed: totTime };
        }
        if (job.status === "cancelled") {
          const allItems = appendResults ? [...accumulatedResults, ...(job.results || [])] : (job.results || []);
          if (allItems.length) {
            renderLatestResults(allItems, Number((totalElapsedAcc + (job.elapsed_sec || 0)).toFixed(2)));
          }
          return { status: "cancelled", results: allItems };
        }
        throw new Error(job.error || "未知错误");
      }

      await new Promise((r) => setTimeout(r, 350));
    }
  }

  async function triggerGenerate() {
    if (state.generating) return;
    const prompt = $("promptInput").value.trim();
    if (!prompt) {
      showToast("请先输入画面描述");
      $("promptInput").focus();
      return;
    }

    recordPromptHistory(prompt);

    if (state.mode === "inpaint") {
      if (!state.inpaintItems.length) {
        showToast("请先上传需要局部重绘的图片");
        return;
      }
      const maskedItems = state.inpaintItems.filter((x) => x.hasMask && x.maskB64);
      if (!maskedItems.length) {
        showToast("请点击缩略图进入画布涂抹重绘区域");
        return;
      }
      const unmaskedRefs = state.inpaintItems.filter((x) => !x.hasMask).map((x) => x.b64);

      hideGenBanners();
      state.notifiedJobId = null;
      setGeneratingUI(true);
      $("progressMsg").textContent = "提交局部重绘任务...";
      $("progressPct").textContent = "0%";

      let accResults = [];
      let accTime = 0;
      try {
        for (let i = 0; i < maskedItems.length; i++) {
          const target = maskedItems[i];
          const otherRefs = state.inpaintItems.filter((x) => x.id !== target.id).map((x) => x.b64);
          const refsForJob = maskedItems.length === 1 ? otherRefs : unmaskedRefs;
          const payload = {
            mode: "inpaint",
            prompt,
            negative_prompt: $("negativePromptInput").value.trim(),
            width: align16($("widthInput").value),
            height: align16($("heightInput").value),
            steps: Number($("stepsSlider").value),
            denoise: Number($("denoiseSlider").value),
            true_cfg_scale: Number($("cfgSlider").value),
            seed: parseSeed(),
            batch_size: state.batchSize,
            hires_scale: Number($("hiresScaleSlider").value),
            mask_blur: Number($("maskBlurSlider").value),
            init_image_b64: target.b64,
            ref_images_b64: refsForJob.slice(0, 9),
            mask_image_b64: target.maskB64,
            lora_enabled: $("loraEnableToggle").checked,
            lora_name: $("loraSelect").value || null,
            lora_weight: Number($("loraWeightSlider").value),
            lora_trigger: $("loraTriggerInput").value.trim(),
            lora_action_mode: $("loraActionMode").value,
            lora_control_image_b64: $("loraActionMode").value === "control" ? state.loraControlB64 : null,
            lora_control_strength: Number($("loraCtrlStrengthSlider").value),
            inpaint_fast_crop: $("inpaintFastCropToggle").checked,
          };
          const out = await runSingleGenerationPayload(payload, i > 0, accResults, accTime);
          accResults = out.results || accResults;
          accTime = out.elapsed || accTime;
          if (out.status === "cancelled") {
            showToast("生成已停止");
            setGeneratingUI(false);
            return;
          }
        }
        setGeneratingUI(false);
        showToast(`局部重绘完成 (${accResults.length} 张 · ${accTime}s)`);
        refreshStats();
      } catch (err) {
        setGeneratingUI(false);
        showGenErrorBanner(err.message);
        showToast(formatShortToastError(err.message), 4200);
      }
      return;
    }

    if (state.mode !== "t2i" && !state.initImageB64) {
      showToast("当前模式需要先上传或粘贴参考底图");
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
      seed: parseSeed(),
      batch_size: state.batchSize,
      hires_scale: Number($("hiresScaleSlider").value),
      mask_blur: Number($("maskBlurSlider").value),
      init_image_b64: state.mode === "t2i" ? null : state.initImageB64,
      ref_images_b64: state.mode === "i2i" ? state.extraRefImages.map((x) => x.b64) : [],
      mask_image_b64: null,
      lora_enabled: $("loraEnableToggle").checked,
      lora_name: $("loraSelect").value || null,
      lora_weight: Number($("loraWeightSlider").value),
      lora_trigger: $("loraTriggerInput").value.trim(),
      lora_action_mode: $("loraActionMode").value,
      lora_control_image_b64: $("loraActionMode").value === "control" ? state.loraControlB64 : null,
      lora_control_strength: Number($("loraCtrlStrengthSlider").value),
      inpaint_fast_crop: $("inpaintFastCropToggle").checked,
    };

    try {
      setGeneratingUI(true);
      $("progressMsg").textContent = "提交生成任务...";
      $("progressPct").textContent = "0%";
      const out = await runSingleGenerationPayload(payload);
      setGeneratingUI(false);
      if (out.status === "cancelled") {
        showToast("生成已停止");
      } else {
        showToast(`生成完成 (${out.results.length} 张 · ${out.elapsed}s)`);
        refreshStats();
      }
    } catch (err) {
      setGeneratingUI(false);
      showGenErrorBanner(err.message);
      showToast(formatShortToastError(err.message), 4200);
    }
  }

  $("generateBtn").addEventListener("click", triggerGenerate);

  $("cancelGenBtn").addEventListener("click", async () => {
    if (!state.activeJobId) return;
    $("cancelGenBtn").disabled = true;
    $("cancelGenBtn").innerHTML = '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>停止中</span>';
    $("progressMsg").textContent = "正在中断生成...";
    try {
      await apiFetch(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST" });
      showToast("已发送中断指令");
    } catch (_) {}
  });

  function renderLatestResults(items, totalElapsed) {
    const grid = $("latestResultGrid");
    if (!items || !items.length) return;
    $("stageEmptyState").classList.add("hidden");
    grid.classList.remove("hidden");
    grid.classList.toggle("multi", items.length > 1);
    $("latestElapsedText").textContent = `${items.length} 张 · 耗时 ${totalElapsed}s`;
    grid.innerHTML = "";

    items.forEach((item) => {
      const el = document.createElement("div");
      el.className = "img-card";
      const hasCompare = Boolean(item.source_url);
      el.innerHTML = `
        <img src="${item.url}" alt="result" loading="lazy" />
        <div class="img-card-foot">
          <span>${MODE_LABELS[item.mode] || item.mode} · ${item.width}×${item.height}</span>
          <span>Seed ${item.seed}</span>
        </div>
        <div class="img-card-actions">
          ${hasCompare ? '<button class="mini-action" data-act="compare" type="button">前后对比</button>' : ""}
          <button class="mini-action" data-act="i2i" type="button">图生图</button>
          <button class="mini-action" data-act="hires" type="button">高清</button>
          <button class="mini-action" data-act="inpaint" type="button">重绘</button>
          <a class="mini-action" href="${item.url}" download="${item.filename}">保存</a>
        </div>
      `;
      el.querySelector("img").addEventListener("click", () => openLightbox(item, false));
      const cmpBtn = el.querySelector('[data-act="compare"]');
      if (cmpBtn) cmpBtn.addEventListener("click", () => openLightbox(item, true));
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
    showToast(`已导入至「${MODE_LABELS[targetMode]}」`);
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
    if ($("galleryCompareBtn")) {
      $("galleryCompareBtn").classList.toggle("hidden", !state.gallerySelecting || state.selectedIds.size !== 2);
    }

    state.galleryItems.forEach((item) => {
      const card = document.createElement("div");
      const isSelected = state.selectedIds.has(item.id);
      card.className = `img-card ${isSelected ? "selected" : ""}`;
      card.innerHTML = `
        ${state.gallerySelecting ? `<div class="select-badge">${isSelected ? '<svg class="icon"><use href="#ic-check" /></svg>' : ""}</div>` : ""}
        <img src="${item.url}" alt="${escapeHtml(item.filename)}" loading="lazy" />
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
          openLightbox(item, false);
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

  async function saveGalleryToGdrive(ids = []) {
    try {
      const res = await apiFetch(
        "/api/gallery/save-gdrive",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        },
        true
      );
      showToast(res.message || `已保存 ${res.saved_count} 张作品到 Google 云盘`, 4000);
    } catch (err) {
      showToast(`保存到云盘失败: ${err.message}`, 4500);
    }
  }

  if ($("gallerySaveGdriveBtn")) {
    $("gallerySaveGdriveBtn").addEventListener("click", () => {
      const ids = state.gallerySelecting && state.selectedIds.size ? Array.from(state.selectedIds) : [];
      saveGalleryToGdrive(ids);
    });
  }

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
        chunks.push(new Uint8Array(await res.arrayBuffer()));
      }
      fill.classList.remove("indeterminate");
      fill.style.width = "100%";
      pctEl.textContent = "100%";
      msg.textContent = "打包完成";

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
      showToast(`导出失败: ${err.message}`);
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

  if ($("galleryCompareBtn")) {
    $("galleryCompareBtn").addEventListener("click", () => {
      const ids = Array.from(state.selectedIds);
      if (ids.length !== 2) return showToast("请先勾选 2 张图片进行对比");
      const itemA = state.galleryItems.find((x) => x.id === ids[0]);
      const itemB = state.galleryItems.find((x) => x.id === ids[1]);
      if (!itemA || !itemB) return;
      openPairCompareLightbox(itemA, itemB);
    });
  }

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
  // Lightbox Modal + Before/After & Pair Comparison Slider + Google Drive
  // -------------------------------------------------------------------------
  function setComparePosition(pct) {
    const clamped = Math.max(0, Math.min(100, Number(pct)));
    const safePct = Number.isFinite(clamped) ? clamped : 50;
    $("compareBeforeClip").style.clipPath = `inset(0 ${100 - safePct}% 0 0)`;
    $("compareDivider").style.left = `${safePct}%`;
    $("compareRangeInput").value = String(Math.round(safePct));
  }

  function setLightboxView(viewMode) {
    const canCompare = Boolean(state.lightboxItem && (state.lightboxItem.source_url || state.lightboxItem._pairCompare));
    const useCompare = viewMode === "compare" && canCompare;
    $("lbSingleView").classList.toggle("hidden", useCompare);
    $("lbCompareView").classList.toggle("hidden", !useCompare);
    document.querySelectorAll("#lbViewToggle .seg-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.lbview === (useCompare ? "compare" : "result"));
    });
    if (useCompare) setComparePosition(50);
  }

  function openLightbox(item, startInCompare = false) {
    state.lightboxItem = item;
    $("lightboxTitle").textContent = `${MODE_LABELS[item.mode] || item.mode} · ${item.width}×${item.height}`;
    $("lightboxImg").src = item.url;
    $("lbDownloadBtn").href = item.url;
    $("lbDownloadBtn").download = item.filename;

    const hasSource = Boolean(item.source_url);
    $("lbViewToggle").classList.toggle("hidden", !hasSource);
    if ($("compareLeftLabel")) $("compareLeftLabel").textContent = "原图";
    if ($("compareRightLabel")) $("compareRightLabel").textContent = "生成图";
    if (hasSource) {
      $("compareBeforeImg").src = item.source_url;
      $("compareAfterImg").src = item.url;
    }
    setLightboxView(startInCompare && hasSource ? "compare" : "result");

    $("lightboxMeta").innerHTML = `
      <div><strong>Prompt:</strong> ${escapeHtml(item.prompt)}</div>
      ${item.negative_prompt ? `<div style="margin-top:3px"><strong>Negative:</strong> ${escapeHtml(item.negative_prompt)}</div>` : ""}
      <div style="margin-top:4px;color:var(--text-dim)">
        Steps ${item.steps} · Denoise ${item.denoise} · CFG ${item.true_cfg_scale} · Seed <code>${item.seed}</code> · ${item.elapsed_sec}s · ${escapeHtml(item.model_id || "")}
      </div>
    `;
    $("lightboxModal").classList.remove("hidden");
  }

  function openPairCompareLightbox(itemA, itemB) {
    state.lightboxItem = { ...itemB, _pairCompare: true };
    $("lightboxTitle").textContent = `双图对比 · ${itemA.width}×${itemA.height} vs ${itemB.width}×${itemB.height}`;
    $("lightboxImg").src = itemB.url;
    $("lbDownloadBtn").href = itemB.url;
    $("lbDownloadBtn").download = itemB.filename;
    $("lbViewToggle").classList.remove("hidden");
    if ($("compareLeftLabel")) $("compareLeftLabel").textContent = `图 A (${MODE_LABELS[itemA.mode] || itemA.mode})`;
    if ($("compareRightLabel")) $("compareRightLabel").textContent = `图 B (${MODE_LABELS[itemB.mode] || itemB.mode})`;
    $("compareBeforeImg").src = itemA.url;
    $("compareAfterImg").src = itemB.url;
    setLightboxView("compare");
    $("lightboxMeta").innerHTML = `
      <div><strong>图 A:</strong> ${escapeHtml(itemA.prompt)} (Seed <code>${itemA.seed}</code>)</div>
      <div style="margin-top:3px"><strong>图 B:</strong> ${escapeHtml(itemB.prompt)} (Seed <code>${itemB.seed}</code>)</div>
    `;
    $("lightboxModal").classList.remove("hidden");
  }

  function closeLightbox() {
    $("lightboxModal").classList.add("hidden");
    state.lightboxItem = null;
  }

  document.querySelectorAll("#lbViewToggle .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => setLightboxView(btn.dataset.lbview));
  });
  $("compareRangeInput").addEventListener("input", (e) => setComparePosition(e.target.value));
  $("compareRangeInput").addEventListener("change", (e) => setComparePosition(e.target.value));

  // Direct pointer/touch drag on comparison frame for reliable mobile & desktop scrubbing
  (function bindCompareFrameDrag() {
    const frame = $("compareFrame");
    if (!frame) return;
    let dragging = false;
    const updateFromClientX = (clientX) => {
      const rect = frame.getBoundingClientRect();
      if (!rect.width) return;
      const pct = ((clientX - rect.left) / rect.width) * 100;
      setComparePosition(pct);
    };
    frame.addEventListener("pointerdown", (e) => {
      dragging = true;
      try { frame.setPointerCapture(e.pointerId); } catch (_) {}
      updateFromClientX(e.clientX);
    });
    frame.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      e.preventDefault();
      updateFromClientX(e.clientX);
    });
    const stopDrag = () => { dragging = false; };
    frame.addEventListener("pointerup", stopDrag);
    frame.addEventListener("pointercancel", stopDrag);
  })();

  $("lightboxCloseBtn").addEventListener("click", closeLightbox);
  $("lightboxBackdrop").addEventListener("click", closeLightbox);

  if ($("lbSaveGdriveBtn")) {
    $("lbSaveGdriveBtn").addEventListener("click", () => {
      if (!state.lightboxItem) return;
      saveGalleryToGdrive([state.lightboxItem.id]);
    });
  }

  $("lbCopyPromptBtn").addEventListener("click", () => {
    if (!state.lightboxItem) return;
    copyToClipboard(state.lightboxItem.prompt || "", "已复制提示词");
  });

  $("lbCopyAllParamsBtn").addEventListener("click", () => {
    const it = state.lightboxItem;
    if (!it) return;
    const lines = [
      `Prompt: ${it.prompt || ""}`,
      it.negative_prompt ? `Negative: ${it.negative_prompt}` : null,
      `Mode: ${MODE_LABELS[it.mode] || it.mode} | Size: ${it.width}x${it.height}`,
      `Steps: ${it.steps} | Denoise: ${it.denoise} | CFG: ${it.true_cfg_scale} | Seed: ${it.seed}`,
      `Model: ${it.model_id || ""} (${it.quant_mode || ""})`,
    ].filter(Boolean);
    copyToClipboard(lines.join("\n"), "已复制全套生成参数");
  });

  $("lbReuseParamsBtn").addEventListener("click", () => {
    const item = state.lightboxItem;
    if (!item) return;
    setMode(item.mode || "t2i", true);
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
    scheduleSaveStudioPrefs();
    closeLightbox();
    switchTab("studio");
    showToast("已复用全套参数");
  });

  $("lbToI2IBtn").addEventListener("click", () => state.lightboxItem && sendItemToMode(state.lightboxItem, "i2i"));
  $("lbAddExtraRefBtn").addEventListener("click", async () => {
    if (!state.lightboxItem) return;
    const item = state.lightboxItem;
    setMode("i2i");
    switchTab("studio");
    const count = await addExtraRefFromUrl(item.url);
    closeLightbox();
    showToast(`已添加为次图 (@${count})`);
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
  // System Monitor & Maintenance & Shutdown
  // -------------------------------------------------------------------------
  function formatUptime(sec) {
    const s = Math.max(0, Math.floor(Number(sec) || 0));
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const r = String(s % 60).padStart(2, "0");
    return `${h}:${m}:${r}`;
  }

  function highlightActivePresetCard(modelId, quantMode) {
    document.querySelectorAll("#editionQuickBtns .preset-card").forEach((card) => {
      const match = card.dataset.model === modelId && card.dataset.quant === quantMode;
      card.classList.toggle("active", match);
    });
  }

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
          : '<span class="model-badge warn">按需待加载</span>';

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
            <div class="model-row-sub">强度: ${Number(loaded.active_lora_weight ?? 0.85).toFixed(2)}</div>
          </div>
          <div class="model-row-actions"><span class="model-badge ok">已挂载</span></div>
        </div>`;
    }
    loadedEl.innerHTML = loadedHtml;

    const hfList = (models && models.downloaded_hf) || [];
    if (!hfList.length) {
      hfEl.innerHTML = '<div class="model-row"><div class="model-row-sub">暂无本地 HF 缓存模型</div></div>';
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
            ${!m.is_current ? '<button class="btn-chip" data-act="use" type="button">切换</button>' : ""}
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
      lorasEl.innerHTML = '<div class="model-row"><div class="model-row-sub">暂无已下载的 LoRA 权重</div></div>';
    } else {
      lorasEl.innerHTML = "";
      loraList.forEach((item) => {
        const row = document.createElement("div");
        row.className = "model-row";
        const validBadge = item.valid === false
          ? `<span class="model-badge err" title="${escapeHtml(item.issue || "文件损坏")}">损坏</span>`
          : item.is_loaded
            ? '<span class="model-badge ok">已挂载</span>'
            : '<span class="model-badge info">可用</span>';
        row.innerHTML = `
          <div class="model-row-info">
            <div class="model-row-name">${escapeHtml(item.name)}</div>
            <div class="model-row-sub">${item.size_mb} MB${item.issue ? " · " + escapeHtml(item.issue) : ""}</div>
          </div>
          <div class="model-row-actions">
            ${validBadge}
            <button class="btn-chip danger" data-act="del" type="button">删除</button>
          </div>`;
        row.querySelector('[data-act="del"]').addEventListener("click", () => deleteLoraByName(item.name));
        lorasEl.appendChild(row);
      });
    }
  }

  async function refreshStats() {
    if (state.statsInFlight) return;
    state.statsInFlight = true;
    try {
      const s = await apiFetch("/api/stats");
      const gpu = s.gpu;
      const ram = s.ram;
      const disk = s.disk;
      const eng = s.engine;

      state.modelDefaultSteps = eng.default_steps || 8;

      let quantTag = "INT8";
      if (eng.quant_mode === "none") quantTag = `${eng.effective_dtype.toUpperCase()} 满血`;
      else if (eng.quant_mode === "gguf-q8-full" || eng.gguf_variant === "q8") quantTag = "GGUF Q8";
      else if (eng.quant_mode === "gguf-q4" || eng.gguf_variant === "q4") quantTag = "GGUF Q4";
      else if (eng.quant_mode === "fp8-int8") quantTag = "FP8+INT8";
      else if (eng.quant_mode === "fp8-all") quantTag = "FP8";
      $("headerQuantBadge").textContent = quantTag;

      const modelShort = (eng.model_id || "Qwen-Image-2.1").split("/").pop();
      const ramPct = Math.round(Number(ram.percent || 0));
      const diskPct = Math.round(Number(disk.percent || 0));
      $("engineStatusText").textContent = `${modelShort} · 内存 ${ramPct}% · 磁盘 ${diskPct}%`;

      highlightActivePresetCard(eng.model_id, eng.quant_mode);

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
        $("swapText").textContent = swap.supported ? "未开启" : "仅 Linux 支持";
        $("swapSubText").textContent = swap.supported ? "容器环境默认受限，按需尝试开启" : "本地非 Linux 环境";
        $("swapBar").style.width = "0%";
      }

      $("diskText").textContent = `${disk.used_gb} / ${disk.total_gb} GB (${disk.percent}%)`;
      $("diskFreeText").textContent = `剩余可用 ${disk.free_gb} GB`;
      $("diskBar").style.width = `${disk.percent}%`;

      $("outputsSizeText").textContent = `${s.storage.outputs_mb} MB`;
      $("hfCacheSizeText").textContent = `${s.storage.hf_cache_gb} GB`;
      $("modelLoadedText").textContent = eng.demo_mode ? "演示模式" : eng.loaded ? "已加载" : "待加载";
      $("uptimeText").textContent = formatUptime(eng.uptime_sec || 0);

      $("gpuProfileBadge").textContent = gpu.arch || "GPU 适配";
      $("gpuProfileTitle").textContent = gpu.profile_name || "未知 GPU";
      $("gpuProfileMeta").textContent =
        `精度: ${eng.effective_dtype.toUpperCase()} · 调度: ${eng.effective_offload} · VAE分块: ${eng.effective_vae_tiling ? "开" : "关"} · TF32: ${eng.effective_tf32 ? "开" : "关"}`;
      $("gpuProfileNotes").textContent = gpu.notes || "";

      if (s.models) renderMonitorModels(s.models);
      if (s.ai && $("aiModelBadge")) {
        $("aiModelBadge").textContent = s.ai.model || "qwen3.8-27b";
      }
      if (s.gdrive && $("galleryGdriveHint")) {
        $("galleryGdriveHint").textContent = s.gdrive.mounted
          ? `Google 云盘已挂载：${s.gdrive.target_dir}（可勾选作品或点击「保存云盘」备份）`
          : "未检测到 Google 云盘挂载（可在 Colab 启动器勾选 MOUNT_GDRIVE 后点击「保存云盘」备份）";
      }
    } catch (_) {
      $("engineStatusText").textContent = "连接已断开";
    } finally {
      state.statsInFlight = false;
    }
  }

  function scheduleNextStats() {
    clearTimeout(state.statsTimer);
    state.statsTimer = setTimeout(async () => {
      await refreshStats();
      scheduleNextStats();
    }, 3500);
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

  $("refreshStatsBtn").addEventListener("click", () => runOpsWithProgress("正在刷新资源状态...", refreshStats));
  $("refreshModelsBtn").addEventListener("click", () => runOpsWithProgress("正在刷新模型清单...", refreshStats));
  $("enableSwapBtn").addEventListener("click", () =>
    runOpsWithProgress("正在配置 8GB Linux Swap...", async () => {
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
    runOpsWithProgress("正在清理 CUDA 显存碎片...", async () => {
      await apiFetch("/api/system/free-vram", { method: "POST" });
      showToast("已释放 CUDA 显存碎片");
      await refreshStats();
    })
  );
  $("unloadModelBtn").addEventListener("click", () =>
    runOpsWithProgress("正在卸载当前模型...", async () => {
      await apiFetch("/api/system/free-vram?unload_model=true", { method: "POST" });
      showToast("已卸载模型并释放显存");
      await refreshStats();
    })
  );
  $("clearHfCacheBtn").addEventListener("click", () =>
    runOpsWithProgress("正在清理 HF 缓存...", async () => {
      await apiFetch("/api/system/clear-hf-cache", { method: "POST" });
      showToast("已清空 Hugging Face 磁盘缓存");
      await refreshStats();
    })
  );

  // -------------------------------------------------------------------------
  // Mobile Web Terminal & Server Logs (Serial Polling — No Duplicate Chunks)
  // -------------------------------------------------------------------------
  const termState = {
    mode: "shell",
    errorsOnly: false,
    activeCmdId: null,
    logsTimer: null,
    offset: 0,
    history: [],
    histIdx: -1,
    shellBuffer: "Web 终端就绪\n",
  };

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
      const url = `/api/logs?limit=200${termState.errorsOnly ? "&errors_only=true" : ""}`;
      const data = await apiFetch(url);
      const scr = $("terminalScreen");
      const wasNearBottom = scr.scrollHeight - scr.scrollTop - scr.clientHeight < 64;
      const emptyMsg = termState.errorsOnly ? "暂无报错日志记录" : "暂无日志输出...";
      scr.textContent = (data.lines || []).join("\n") || emptyMsg;
      if (forceBottom || wasNearBottom) scr.scrollTop = scr.scrollHeight;
    } catch (_) {}
  }

  function setTerminalViewMode(mode) {
    termState.mode = mode;
    $("termModeShellBtn").classList.toggle("active", mode === "shell");
    $("termModeLogsBtn").classList.toggle("active", mode === "logs");
    $("terminalInputBar").classList.toggle("hidden", mode !== "shell");
    $("termQuickCmds").classList.toggle("hidden", mode !== "shell");
    $("termLogsToolbar").classList.toggle("hidden", mode !== "logs");
    clearInterval(termState.logsTimer);

    if (mode === "logs") {
      fetchServerLogs(true);
      termState.logsTimer = setInterval(() => fetchServerLogs(false), 1200);
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
        body: JSON.stringify({ command: cmd, source: "user" }),
      });
      $("terminalCwdText").textContent = res.cwd || cwdNow;
      termState.activeCmdId = res.cmd_id;
      termState.offset = 0;
      setTermRunningUI(true, cmd);
      await pollTerminalTaskSerial(res.cmd_id);
    } catch (err) {
      appendTerminalText(`[请求失败] ${err.message}\n`);
    }
  }

  async function pollTerminalTaskSerial(cmdId) {
    try {
      while (termState.activeCmdId === cmdId) {
        const data = await apiFetch(`/api/terminal/poll/${cmdId}?offset=${termState.offset}`);
        if (data.chunk) appendTerminalText(data.chunk);
        termState.offset = data.offset;
        if (data.cwd) $("terminalCwdText").textContent = data.cwd;
        if (data.elapsed_sec !== undefined) {
          $("termProgressElapsed").textContent = `${Number(data.elapsed_sec).toFixed(1)}s`;
        }
        if (!data.running) {
          termState.activeCmdId = null;
          setTermRunningUI(false);
          if (data.exit_code !== 0 && data.exit_code !== null) {
            appendTerminalText(`[退出码 ${data.exit_code} · ${data.elapsed_sec}s]\n`);
          }
          return;
        }
        await new Promise((r) => setTimeout(r, 200));
      }
    } catch (err) {
      termState.activeCmdId = null;
      setTermRunningUI(false);
      appendTerminalText(`[轮询中断] ${err.message}\n`);
    }
  }

  $("termModeShellBtn").addEventListener("click", () => setTerminalViewMode("shell"));
  $("termModeLogsBtn").addEventListener("click", () => setTerminalViewMode("logs"));
  $("termErrorsOnlyToggle").addEventListener("change", (e) => {
    termState.errorsOnly = e.target.checked;
    fetchServerLogs(true);
  });
  $("termClearBtn").addEventListener("click", () => {
    if (termState.mode === "shell") {
      termState.shellBuffer = "";
      $("terminalScreen").textContent = "";
    } else {
      fetchServerLogs(true);
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

  // -------------------------------------------------------------------------
  // Settings Save (Decoupled Engine vs Lightweight Preferences vs AI Config)
  // -------------------------------------------------------------------------
  async function saveEngineConfig(recommendedSteps = null) {
    try {
      const res = await apiFetch(
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
            inpaint_fast_crop: $("inpaintFastCropToggle").checked,
          }),
        },
        true
      );
      if (recommendedSteps) {
        state.modelDefaultSteps = recommendedSteps;
        $("stepsSlider").value = String(recommendedSteps);
        $("stepsVal").textContent = String(recommendedSteps);
        scheduleSaveStudioPrefs();
      }
      if ($("hfTokenInput").value.trim()) {
        $("hfTokenInput").value = "";
        $("hfTokenInput").placeholder = "已配置 HF Token (留空保持不变)";
      }
      const qVal = $("quantSelect").value;
      if ($("comfyGgufSelect") && !$("comfyGgufSelect").disabled) {
        if (qVal === "gguf-q8-full") $("comfyGgufSelect").value = "q8";
        else if (qVal === "gguf-q4") $("comfyGgufSelect").value = "q4";
      }
      highlightActivePresetCard($("modelSelect").value, qVal);
      showToast("引擎配置已保存（下次生成生效）");
      if (res.stats) refreshStats();
    } catch (err) {
      showToast(`保存失败: ${err.message}`);
    }
  }

  async function savePreferenceOnly(patch, toastMsg = "偏好设置已保存") {
    try {
      await apiFetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (toastMsg) showToast(toastMsg);
    } catch (err) {
      showToast(`保存失败: ${err.message}`);
    }
  }

  function syncAiAutoRunToggles(enabled) {
    state.aiAutoRun = Boolean(enabled);
    if ($("aiAutoRunToggle")) $("aiAutoRunToggle").checked = state.aiAutoRun;
    if ($("aiModalAutoRunToggle")) $("aiModalAutoRunToggle").checked = state.aiAutoRun;
  }

  async function saveAiConfigOnly() {
    try {
      const res = await apiFetch(
        "/api/config",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ai_endpoint: $("aiEndpointInput").value.trim() || null,
            ai_model: $("aiModelInput").value.trim() || null,
            ai_api_key: $("aiApiKeyInput").value.trim() || null,
            ai_vision_enabled: $("aiVisionToggle").checked,
            ai_auto_run: $("aiAutoRunToggle").checked,
          }),
        },
        true
      );
      $("aiModelBadge").textContent = $("aiModelInput").value.trim() || "qwen3.8-27b";
      if ($("aiApiKeyInput").value.trim()) {
        const masked = res.stats?.ai?.api_key_masked || "已保存";
        $("aiApiKeyInput").value = "";
        $("aiApiKeyInput").placeholder = `已配置 (${masked})`;
      }
      syncAiAutoRunToggles($("aiAutoRunToggle").checked);
      showToast("AI 配置已保存");
    } catch (err) {
      showToast(`保存 AI 配置失败: ${err.message}`);
    }
  }

  $("inpaintFastCropToggle").addEventListener("change", () =>
    savePreferenceOnly({ inpaint_fast_crop: $("inpaintFastCropToggle").checked }, "已更新局部重绘加速设置")
  );
  $("aiVisionToggle").addEventListener("change", () =>
    savePreferenceOnly({ ai_vision_enabled: $("aiVisionToggle").checked }, "已更新 AI 识图开关")
  );
  $("aiAutoRunToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    syncAiAutoRunToggles(on);
    savePreferenceOnly({ ai_auto_run: on }, on ? "已开启 AI 自动运行代码（含安全护栏）" : "已关闭 AI 自动运行代码");
  });
  $("aiModalAutoRunToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    syncAiAutoRunToggles(on);
    savePreferenceOnly({ ai_auto_run: on }, on ? "已开启 AI 自动运行代码（含安全护栏）" : "已关闭 AI 自动运行代码");
  });

  document.querySelectorAll("#editionQuickBtns .preset-card").forEach((card) => {
    card.addEventListener("click", () => {
      $("modelSelect").value = card.dataset.model;
      $("quantSelect").value = card.dataset.quant;
      if (card.dataset.offload) $("offloadSelect").value = card.dataset.offload;
      saveEngineConfig(Number(card.dataset.steps || 8));
    });
  });

  async function runModelMaintenanceOp(endpoint, defaultLabel, confirmText = "") {
    if (confirmText && !window.confirm(confirmText)) return;
    const reloadBtn = $("reloadModelBtn");
    const reinstallBtn = $("reinstallModelBtn");
    const wrap = $("modelOpProgressWrap");
    const msgEl = $("modelOpProgressMsg");
    const pctEl = $("modelOpProgressPct");
    const fillEl = $("modelOpProgressFill");
    reloadBtn.disabled = true;
    reinstallBtn.disabled = true;
    wrap.classList.remove("hidden");
    msgEl.textContent = defaultLabel;
    pctEl.textContent = "2%";
    fillEl.style.width = "2%";

    try {
      const startRes = await apiFetch(endpoint, { method: "POST" }, true);
      const taskId = startRes.task_id;
      while (true) {
        const op = await apiFetch(`/api/model/op/${taskId}`);
        const pct = Math.max(2, Math.min(100, Math.round((Number(op.progress) || 0) * 100)));
        msgEl.textContent = op.message || defaultLabel;
        pctEl.textContent = `${pct}%`;
        fillEl.style.width = `${pct}%`;
        if (op.status === "completed") {
          showToast(op.message || "模型操作完成", 4500);
          await refreshStats();
          break;
        }
        if (op.status === "error") {
          throw new Error(op.error || op.message || "模型操作失败");
        }
        await new Promise((r) => setTimeout(r, 800));
      }
    } catch (err) {
      msgEl.textContent = `操作失败: ${err.message}`;
      showToast(`操作失败: ${err.message}`, 5500);
    } finally {
      reloadBtn.disabled = false;
      reinstallBtn.disabled = false;
      setTimeout(() => wrap.classList.add("hidden"), 1500);
    }
  }

  $("reloadModelBtn").addEventListener("click", () =>
    runModelMaintenanceOp("/api/model/reload", "正在重新加载现有模型...")
  );
  $("reinstallModelBtn").addEventListener("click", () =>
    runModelMaintenanceOp(
      "/api/model/reinstall",
      "正在删除所有模型并重新安装...",
      "确定要删除已安装的所有模型，并重新下载安装当前模型吗？"
    )
  );

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
  function openShutdownModal() { $("shutdownModal").classList.remove("hidden"); }
  function closeShutdownModal() { $("shutdownModal").classList.add("hidden"); }
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
      clearTimeout(state.statsTimer);
      $("engineStatusText").textContent = "服务已安全关闭";
      showToast(res.message || "服务已关闭", 6000);
    } catch (err) {
      showToast(`关闭请求异常: ${err.message}`);
    }
  });

  // -------------------------------------------------------------------------
  // LoRA Management & Download (Serial Polling)
  // -------------------------------------------------------------------------
  function updateSelectedLoraWarning() {
    const selName = $("loraSelect").value;
    const item = (state.lorasItems || []).find((x) => x.name === selName);
    $("loraDeleteBtn").classList.toggle("hidden", !selName);
    if (item && item.valid === false) {
      $("loraWarnText").textContent = `警告：${item.issue || "LoRA 文件无效或已损坏"}，请删除后重新下载。`;
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
        const badTag = item.valid === false ? " [已损坏]" : "";
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
    if (!cleanUrl) return showToast("请输入 LoRA 下载链接");
    ui.wrap.classList.remove("hidden");
    ui.fill.classList.add("indeterminate");
    ui.fill.style.width = "0%";
    ui.msg.textContent = "正在连接...";
    ui.pct.textContent = "准备中";
    if (ui.btn) ui.btn.disabled = true;

    try {
      const res = await apiFetch("/api/loras/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: cleanUrl, token: (token || "").trim() || null }),
      });
      const taskId = res.task_id;
      while (true) {
        const st = await apiFetch(`/api/loras/download/${taskId}`);
        const pct = Math.round((st.progress || 0) * 100);
        const indet = st.status === "downloading" && (!st.total_mb || pct === 0);
        ui.fill.classList.toggle("indeterminate", indet);
        if (!indet) ui.fill.style.width = `${pct}%`;
        ui.msg.textContent = st.message || "下载中...";
        ui.pct.textContent = indet ? `${st.downloaded_mb || 0} MB` : `${pct}%`;

        if (st.status === "completed" || st.status === "error") {
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
          break;
        }
        await new Promise((r) => setTimeout(r, 350));
      }
    } catch (err) {
      ui.wrap.classList.add("hidden");
      showToast(`下载失败: ${err.message}`);
    } finally {
      if (ui.btn) ui.btn.disabled = false;
    }
  }

  $("loraEnableToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("loraPanelBody").classList.toggle("hidden", !on);
    if (on) refreshLoraList();
    scheduleSaveStudioPrefs();
  });
  $("loraRefreshBtn").addEventListener("click", () => refreshLoraList());
  $("loraWeightSlider").addEventListener("input", (e) => {
    $("loraWeightVal").textContent = Number(e.target.value).toFixed(2);
    scheduleSaveStudioPrefs();
  });
  $("loraTriggerInput").addEventListener("input", scheduleSaveStudioPrefs);
  $("loraActionMode").addEventListener("change", (e) => {
    $("loraControlPanel").classList.toggle("hidden", e.target.value !== "control");
  });
  $("loraCtrlStrengthSlider").addEventListener("input", (e) => {
    $("loraCtrlStrengthVal").textContent = Number(e.target.value).toFixed(2);
  });
  $("loraCtrlFileInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const rawUrl = await readFileAsDataUrl(file);
    const dataUrl = await compressImageDataUrl(rawUrl, 1400, 0.88);
    state.loraControlB64 = dataUrl;
    $("loraCtrlPreviewImg").src = dataUrl;
    $("loraCtrlPlaceholder").classList.add("hidden");
    $("loraCtrlPreviewWrap").classList.remove("hidden");
    $("loraCtrlClearBtn").classList.remove("hidden");
    e.target.value = "";
  });
  $("loraCtrlClearBtn").addEventListener("click", () => {
    state.loraControlB64 = null;
    $("loraCtrlPreviewImg").src = "";
    $("loraCtrlPlaceholder").classList.remove("hidden");
    $("loraCtrlPreviewWrap").classList.add("hidden");
    $("loraCtrlClearBtn").classList.add("hidden");
  });

  function openLoraModal() { $("loraDownloadModal").classList.remove("hidden"); }
  function closeLoraModal() { $("loraDownloadModal").classList.add("hidden"); }
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
  // AI Assistant (Per-Scene History + Opt-in Safe Autonomous Code Execution)
  // -------------------------------------------------------------------------
  const aiState = {
    scene: "studio",
    histories: { studio: [], terminal: [], error: [] },
    busy: false,
    autoRunning: false,
    autoAbort: false,
    activeAutoCmdId: null,
  };

  function getActiveAiHistory() {
    return aiState.histories[aiState.scene] || (aiState.histories[aiState.scene] = []);
  }

  const AI_SCENE_META = {
    studio: {
      title: "AI 创作与调参助手",
      quick: [
        { label: "润色当前提示词", text: "帮我扩写并优化当前的提示词，提升光影、构图与细节描述，并返回可直接应用的 JSON。" },
        { label: "推荐高画质参数", text: "根据当前 GPU 与模型配置，推荐最适合的步数、CFG 和画幅比例并帮我设置好。" },
        { label: "生成中文海报方案", text: "帮我写一段高级感中文平面海报提示词，并设置适合竖屏海报的比例和参数。" },
      ],
    },
    terminal: {
      title: "AI 终端代码助手",
      quick: [
        { label: "全面体检 GPU 与环境", text: "请检查当前 GPU 显存、PyTorch CUDA、Diffusers 版本和磁盘可用空间，并在 JSON 中给出检查命令。" },
        { label: "分析最近日志", text: "帮我分析最近的终端输出或服务日志，如果有报错请给出一键修复命令。" },
        { label: "安全清理临时缓存", text: "帮我检查并安全清理 pip 缓存和无用临时文件。" },
      ],
    },
    error: {
      title: "AI 报错诊断与修复",
      quick: [
        { label: "定位根因并修复", text: "请分析刚才生图报错的根本原因，给出可直接应用的参数/设置修改或终端修复命令。" },
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
      seed: parseSeed(),
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
    if (actions.settings && typeof actions.settings === "object") parts.push("引擎设置");
    return parts.join(" · ");
  }

  async function applyAiActions(actions) {
    if (!actions || typeof actions !== "object") return;
    if (actions.mode && ["t2i", "i2i", "hires", "inpaint"].includes(actions.mode)) setMode(actions.mode);
    if (typeof actions.prompt === "string" && actions.prompt.trim()) $("promptInput").value = actions.prompt.trim();
    if (typeof actions.negative_prompt === "string") $("negativePromptInput").value = actions.negative_prompt;
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
    if (actions.width && actions.height) applyResolution(actions.width, actions.height);
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
    if (actions.seed !== undefined) $("seedInput").value = String(Math.round(Number(actions.seed)));
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
    if (typeof actions.lora_trigger === "string") $("loraTriggerInput").value = actions.lora_trigger;

    if (actions.settings && typeof actions.settings === "object") {
      const s = actions.settings;
      if (s.model_id) $("modelSelect").value = s.model_id;
      if (s.quant_mode) $("quantSelect").value = s.quant_mode;
      if (s.offload_mode) $("offloadSelect").value = s.offload_mode;
      if (s.dtype_mode) $("dtypeSelect").value = s.dtype_mode;
      if (s.vae_tiling_mode) $("vaeTilingSelect").value = s.vae_tiling_mode;
      await saveEngineConfig(actions.steps ? Number(actions.steps) : null);
    }

    scheduleSaveStudioPrefs();
    showToast("已应用 AI 建议的修改");
  }

  function setAutoRunBar(visible, text = "") {
    $("aiAutoRunBar").classList.toggle("hidden", !visible);
    if (text) $("aiAutoRunStatusText").textContent = text;
  }

  async function executeAiCommandOnce(cmd, allowDangerous = false, onUpdate = null) {
    const res = await apiFetch("/api/terminal/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd, source: "ai", allow_dangerous: allowDangerous }),
    });
    if (res.blocked) {
      return { blocked: true, reason: res.reason || "高危命令已拦截", output: "", exit_code: null };
    }
    const cmdId = res.cmd_id;
    aiState.activeAutoCmdId = cmdId;
    appendTerminalText(`\n[AI 自动执行]$ ${cmd}\n`);

    let offset = 0;
    let output = "";
    let exitCode = null;
    const deadline = Date.now() + 60_000;

    while (true) {
      if (aiState.autoAbort) {
        try {
          await apiFetch(`/api/terminal/kill/${cmdId}`, { method: "POST" });
        } catch (_) {}
        output += "\n[已手动停止自动执行]\n";
        exitCode = -1;
        break;
      }
      if (Date.now() > deadline) {
        try {
          await apiFetch(`/api/terminal/kill/${cmdId}`, { method: "POST" });
        } catch (_) {}
        output += "\n[命令超时 (60s) 已自动终止]\n";
        exitCode = -2;
        break;
      }
      const pr = await apiFetch(`/api/terminal/poll/${cmdId}?offset=${offset}`);
      if (pr.chunk) {
        output += pr.chunk;
        appendTerminalText(pr.chunk);
      }
      offset = pr.offset;
      if (pr.cwd) $("terminalCwdText").textContent = pr.cwd;
      if (onUpdate) onUpdate(output, pr.running, pr.exit_code);
      if (!pr.running) {
        exitCode = pr.exit_code;
        break;
      }
      await new Promise((r) => setTimeout(r, 220));
    }

    aiState.activeAutoCmdId = null;
    return { blocked: false, output: output.trim(), exit_code: exitCode };
  }

  function renderAiChat() {
    const list = $("aiChatList");
    const hist = getActiveAiHistory();
    list.innerHTML = "";
    if (!hist.length) {
      const modeHint = state.aiAutoRun
        ? "当前已开启「自动运行代码」，我给出的安全终端命令会自动执行并继续分析结果。"
        : "如需让我自动执行终端代码排查问题，可点击右上角开启「自动运行代码」。";
      list.innerHTML = `<div class="ai-msg assistant"><div class="ai-msg-body">你好！我是 Qwen-Image-2.1 助手（${escapeHtml($("aiModelInput").value || "qwen3.8-27b")}）。\n${modeHint}</div></div>`;
      return;
    }

    hist.forEach((msg) => {
      if (msg.type === "exec-log") {
        const box = document.createElement("div");
        box.className = "ai-msg exec-log";
        const statusStr = msg.running ? "执行中..." : `退出码 ${msg.exit_code ?? 0}`;
        box.innerHTML = `
          <div class="ai-exec-head">
            <span>$ ${escapeHtml(msg.command)}</span>
            <span>${escapeHtml(statusStr)}</span>
          </div>
          <pre class="ai-exec-pre">${escapeHtml(msg.output || "(无输出)")}</pre>
        `;
        list.appendChild(box);
        return;
      }

      if (msg.type === "guard") {
        const box = document.createElement("div");
        box.className = "ai-guard-box";
        box.innerHTML = `
          <div><strong>安全护栏已拦截高危命令：</strong>${escapeHtml(msg.reason || "")}</div>
          <div class="ai-cmd-row"><code>${escapeHtml(msg.command)}</code></div>
          <div class="ai-actions-btns">
            <button class="btn-chip danger" data-act="force" type="button" ${msg.resolved ? "disabled" : ""}>仍要手动执行</button>
            <button class="btn-chip" data-act="skip" type="button" ${msg.resolved ? "disabled" : ""}>跳过此命令</button>
          </div>
        `;
        const forceBtn = box.querySelector('[data-act="force"]');
        const skipBtn = box.querySelector('[data-act="skip"]');
        if (forceBtn && !msg.resolved) {
          forceBtn.addEventListener("click", async () => {
            msg.resolved = true;
            renderAiChat();
            const logItem = { role: "system", type: "exec-log", command: msg.command, output: "", running: true, exit_code: null };
            hist.push(logItem);
            renderAiChat();
            const res = await executeAiCommandOnce(msg.command, true, (out, running, code) => {
              logItem.output = out;
              logItem.running = running;
              logItem.exit_code = code;
              renderAiChat();
            });
            logItem.output = res.output;
            logItem.running = false;
            logItem.exit_code = res.exit_code;
            renderAiChat();
          });
        }
        if (skipBtn && !msg.resolved) {
          skipBtn.addEventListener("click", () => {
            msg.resolved = true;
            renderAiChat();
            showToast("已跳过该高危命令");
          });
        }
        list.appendChild(box);
        return;
      }

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
          cmdRow.innerHTML = `<code>${escapeHtml(cmd)}</code><button class="btn-chip primary" type="button">运行</button>`;
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

  function compressDataUrlForVision(dataUrl, maxEdge = 768) {
    return new Promise((resolve) => {
      if (!dataUrl) return resolve(null);
      loadImageObj(dataUrl).then((img) => {
        const scale = Math.min(1, maxEdge / Math.max(img.width, img.height, 1));
        const w = Math.max(64, Math.round(img.width * scale));
        const h = Math.max(64, Math.round(img.height * scale));
        const cvs = document.createElement("canvas");
        cvs.width = w;
        cvs.height = h;
        cvs.getContext("2d").drawImage(img, 0, 0, w, h);
        resolve(cvs.toDataURL("image/jpeg", 0.82));
      }).catch(() => resolve(null));
    });
  }

  async function collectStudioVisionImages() {
    if (!$("aiVisionToggle")?.checked || aiState.scene !== "studio") return [];
    const rawList = [];
    if (state.mode === "inpaint" && state.inpaintItems.length) {
      state.inpaintItems.slice(0, 4).forEach((x) => rawList.push(x.b64));
    } else {
      if (state.initImageB64) rawList.push(state.initImageB64);
      state.extraRefImages.slice(0, 3).forEach((x) => rawList.push(x.b64));
    }
    if (state.loraControlB64) rawList.push(state.loraControlB64);
    const out = [];
    for (const r of rawList.slice(0, 4)) {
      const compressed = await compressDataUrlForVision(r);
      if (compressed) out.push(compressed);
    }
    return out;
  }

  function updateAiVisionHint() {
    const hint = $("aiVisionHint");
    if (!hint) return;
    const visionOn = Boolean($("aiVisionToggle")?.checked) && aiState.scene === "studio";
    const imgCount = state.mode === "inpaint"
      ? state.inpaintItems.length
      : (state.initImageB64 ? 1 : 0) + state.extraRefImages.length + (state.loraControlB64 ? 1 : 0);
    if (visionOn && imgCount > 0) {
      hint.textContent = `识图已开启：将自动附带当前 ${Math.min(4, imgCount)} 张参考图发送给 AI`;
      hint.classList.remove("hidden");
    } else if (visionOn) {
      hint.textContent = "识图已开启（当前无参考图，仅发送文本）";
      hint.classList.remove("hidden");
    } else {
      hint.classList.add("hidden");
    }
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

    updateAiVisionHint();
    $("aiAssistantModal").classList.remove("hidden");
    renderAiChat();
    if (autoPrompt) sendAiMessage(autoPrompt);
    else $("aiChatInput").focus();
  }

  function closeAiModal() {
    $("aiAssistantModal").classList.add("hidden");
  }

  async function sendAiMessage(explicitText = null, autoRound = 0) {
    if (aiState.busy && autoRound === 0) return;
    const input = $("aiChatInput");
    const text = (explicitText !== null ? explicitText : input.value).trim();
    if (!text) return;
    if (explicitText === null) input.value = "";

    const hist = getActiveAiHistory();
    if (autoRound === 0) {
      hist.push({ role: "user", content: text });
      aiState.autoAbort = false;
      renderAiChat();
    }

    aiState.busy = true;
    $("aiChatSendBtn").disabled = true;
    $("aiChatSendBtn").innerHTML = '<svg class="icon icon-spin"><use href="#ic-loader" /></svg><span>思考中</span>';

    try {
      const visionImages = autoRound === 0 ? await collectStudioVisionImages() : [];
      const chatPayloadHistory = hist
        .filter((m) => m.role === "user" || m.role === "assistant")
        .slice(-10, autoRound === 0 ? -1 : undefined)
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await apiFetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene: aiState.scene,
          message: text,
          history: chatPayloadHistory,
          context: collectAiContext(aiState.scene),
          images: visionImages,
        }),
      }, true);

      hist.push({
        role: "assistant",
        content: res.reply || "",
        actions: res.actions || null,
      });
      renderAiChat();

      // Opt-in AI Autonomous Code Execution Loop (with Safety Guardrail & Round Cap)
      const cmds = Array.isArray(res.actions?.terminal_commands)
        ? res.actions.terminal_commands.filter((c) => typeof c === "string" && c.trim())
        : [];
      if (state.aiAutoRun && cmds.length > 0 && autoRound < MAX_AUTO_RUN_ROUNDS && !aiState.autoAbort) {
        aiState.autoRunning = true;
        setAutoRunBar(true, `AI 自动执行中（第 ${autoRound + 1}/${MAX_AUTO_RUN_ROUNDS} 轮 · 共 ${cmds.length} 条命令）`);
        const summaries = [];
        let blockedAny = false;

        for (const cmd of cmds) {
          if (aiState.autoAbort) break;
          const logItem = { role: "system", type: "exec-log", command: cmd, output: "", running: true, exit_code: null };
          hist.push(logItem);
          renderAiChat();

          const execRes = await executeAiCommandOnce(cmd, false, (out, running, code) => {
            logItem.output = out;
            logItem.running = running;
            logItem.exit_code = code;
            renderAiChat();
          });

          if (execRes.blocked) {
            hist.pop();
            hist.push({ role: "system", type: "guard", command: cmd, reason: execRes.reason, resolved: false });
            blockedAny = true;
            renderAiChat();
            break;
          }
          logItem.output = execRes.output;
          logItem.running = false;
          logItem.exit_code = execRes.exit_code;
          renderAiChat();
          summaries.push(`$ ${cmd}\n[exit_code=${execRes.exit_code}]\n${(execRes.output || "(无输出)").slice(-1500)}`);
        }

        aiState.autoRunning = false;
        setAutoRunBar(false);

        if (!blockedAny && !aiState.autoAbort && summaries.length > 0) {
          const feedback = `以下是刚才自动执行命令的实际终端输出：\n\n${summaries.join("\n\n")}\n\n请根据实际输出总结结果；若问题已解决请直接给结论（不要再输出 terminal_commands）；若仍需下一步排查或修复，请给出下一条安全命令。`;
          await sendAiMessage(feedback, autoRound + 1);
          return;
        }
      }
    } catch (err) {
      hist.push({
        role: "assistant",
        content: `请求 AI 失败：${err.message}\n请检查「设置 -> AI 助手配置」中的接口地址与 API Key。`,
      });
      renderAiChat();
    } finally {
      if (!aiState.autoRunning) {
        aiState.busy = false;
        setAutoRunBar(false);
        $("aiChatSendBtn").disabled = false;
        $("aiChatSendBtn").innerHTML = '<svg class="icon"><use href="#ic-sparkles" /></svg><span>发送</span>';
      }
    }
  }

  $("aiStopAutoRunBtn").addEventListener("click", async () => {
    aiState.autoAbort = true;
    if (aiState.activeAutoCmdId) {
      try {
        await apiFetch(`/api/terminal/kill/${aiState.activeAutoCmdId}`, { method: "POST" });
      } catch (_) {}
    }
    setAutoRunBar(false);
    showToast("已停止 AI 自动执行");
  });

  $("studioAiBtn").addEventListener("click", () => openAiModal("studio"));
  $("termAskAiBtn").addEventListener("click", () => openAiModal("terminal"));
  $("errorAskAiBtn").addEventListener("click", () => {
    const errText = state.lastGenError || "未知生图错误";
    openAiModal("error", `刚才生图报错了：${errText}\n请帮我分析原因并给出修复方案（如需改参数/设置/跑命令请直接给出 JSON）。`);
  });
  $("aiModalCloseBtn").addEventListener("click", closeAiModal);
  $("aiModalBackdrop").addEventListener("click", closeAiModal);
  $("aiClearChatBtn").addEventListener("click", () => {
    aiState.histories[aiState.scene] = [];
    renderAiChat();
  });
  $("aiChatSendBtn").addEventListener("click", () => sendAiMessage());
  $("aiChatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendAiMessage();
    }
  });

  // -------------------------------------------------------------------------
  // ComfyUI On-Demand Launcher (Serial Polling)
  // -------------------------------------------------------------------------
  function renderComfyState(st) {
    const badgeMap = { idle: "未启动", installing: "部署中", starting: "启动中", running: "运行中", error: "失败" };
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
      const pluginOk = st.gguf_plugin_installed ? "已安装 city96/ComfyUI-GGUF" : "自动安装 city96/ComfyUI-GGUF";
      const stagedCount = Array.isArray(st.staged_files) ? st.staged_files.length : 0;
      const wfCount = Array.isArray(st.workflows) ? st.workflows.length : 0;
      if (running || stagedCount > 0 || wfCount > 0) {
        $("comfyGgufMeta").textContent =
          `${pluginOk} · 档位 ${(st.gguf_variant || "q4").toUpperCase()}` +
          (stagedCount ? ` · 已挂载 ${stagedCount} 个模型文件` : "") +
          (wfCount ? ` · 已生成 ${wfCount} 份工作流` : "");
      } else {
        $("comfyGgufMeta").innerHTML = '按需部署 <a href="https://github.com/city96/ComfyUI-GGUF" target="_blank" class="text-link">city96/ComfyUI-GGUF</a> 扩展并自动生成工作流';
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

  async function pollComfyStatusLoop() {
    clearTimeout(state.comfyPollTimer);
    try {
      const st = await apiFetch("/api/comfyui/status");
      renderComfyState(st);
      if (st.status === "installing" || st.status === "starting") {
        state.comfyPollTimer = setTimeout(pollComfyStatusLoop, 900);
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
      state.comfyPollTimer = setTimeout(pollComfyStatusLoop, 900);
    } catch (err) {
      showToast(`ComfyUI 启动失败: ${err.message}`);
    }
  });

  $("comfyStopBtn").addEventListener("click", async () => {
    try {
      clearTimeout(state.comfyPollTimer);
      const st = await apiFetch("/api/comfyui/stop", { method: "POST" });
      renderComfyState(st);
      showToast("ComfyUI 已停止");
    } catch (_) {}
  });

  // -------------------------------------------------------------------------
  // Global Keyboard Shortcuts (Ctrl+Enter Generate, Esc Close Modals)
  // -------------------------------------------------------------------------
  function hasOpenModal() {
    return ["lightboxModal", "shutdownModal", "loraDownloadModal", "inpaintCanvasModal", "aiAssistantModal"]
      .some((id) => !$(id).classList.contains("hidden"));
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("lightboxModal").classList.contains("hidden")) closeLightbox();
      else if (!$("aiAssistantModal").classList.contains("hidden")) closeAiModal();
      else if (!$("inpaintCanvasModal").classList.contains("hidden")) closeInpaintModal(false);
      else if (!$("loraDownloadModal").classList.contains("hidden")) closeLoraModal();
      else if (!$("shutdownModal").classList.contains("hidden")) closeShutdownModal();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (state.activeTab === "studio" && !hasOpenModal()) {
        e.preventDefault();
        triggerGenerate();
      }
    }
  });

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------
  async function init() {
    const savedAccent = (() => {
      try { return localStorage.getItem(STORAGE_KEYS.accent); } catch (_) { return null; }
    })();
    if (savedAccent && ACCENT_THEMES[savedAccent]) {
      state.accentColor = savedAccent;
    }
    const savedTheme = (() => {
      try { return localStorage.getItem(STORAGE_KEYS.theme); } catch (_) { return null; }
    })();
    applyTheme(savedTheme || "light");

    setMode("t2i", true);
    applyResolution(1024, 1024, true);
    const hadLocalPrefs = restoreStudioPrefs();

    try {
      const stats = await apiFetch("/api/stats");
      const eng = stats.engine || {};
      state.modelDefaultSteps = eng.default_steps || 8;

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
      if (typeof eng.inpaint_fast_crop === "boolean") {
        $("inpaintFastCropToggle").checked = eng.inpaint_fast_crop;
      }
      if (!hadLocalPrefs && eng.default_steps) {
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
        if (stats.ai.has_api_key) {
          $("aiApiKeyInput").placeholder = `已配置 (${stats.ai.api_key_masked || "已保存"})`;
        }
        if (typeof stats.ai.vision_enabled === "boolean") {
          $("aiVisionToggle").checked = stats.ai.vision_enabled;
        }
        syncAiAutoRunToggles(Boolean(stats.ai.auto_run));
      }
    } catch (_) {}

    await refreshStats();
    refreshLoraList();
    pollComfyStatusLoop();
    scheduleNextStats();
  }

  init();
})();
