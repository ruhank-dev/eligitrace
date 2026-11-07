// Shared helpers: API calls with friendly errors, DOM builder, toasts, theme, health polling.

export class ApiError extends Error {
  constructor(status, code, detail, fields = []) { super(detail); this.status = status; this.code = code; this.fields = fields; }
}

export async function api(path, { method = "GET", body } = {}) {
  let res;
  try {
    res = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  } catch {
    throw new ApiError(0, "network", "Cannot reach the EligiTrace server. Is `task app` running?");
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 422 && Array.isArray(data.detail)) {
      const fields = data.detail.map((d) => ({ loc: d.loc.filter((x) => x !== "body"), msg: d.msg }));
      throw new ApiError(422, "validation", fields.map((f) => `${f.loc.join(" › ")}: ${f.msg}`).join("\n"), fields);
    }
    throw new ApiError(res.status, data.error || "error", data.detail || `HTTP ${res.status}`);
  }
  return data;
}

// Tiny element builder: text is always set via textContent (no HTML injection of user data).
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}

export function toast(title, msg = "", kind = "info", ms = 6000) {
  const t = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, h("b", { text: title }), msg);
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

export const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

export function initTheme() {
  const saved = store.get("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  document.getElementById("theme")?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    store.set("theme", document.documentElement.dataset.theme);
  });
}

export function pollHealth(onChange) {
  const pill = document.getElementById("status"), txt = document.getElementById("status-text");
  let last = null;
  async function tick() {
    let hstate;
    try { hstate = await api("/api/health"); } catch { hstate = { status: "error", detail: "server unreachable" }; }
    const s = hstate.status === "ready" && hstate.busy ? "busy" : hstate.status;
    pill.dataset.s = s;
    const where = hstate.gpu ? `${hstate.gpu.replace("NVIDIA GeForce ", "")} · ${hstate.dtype}` : hstate.device || "";
    txt.textContent = {
      ready: `Ready · ${hstate.adapter ? "fine-tuned" : "base model"} · ${where}`,
      busy: `Working… · ${where}`, loading: "Loading model…", idle: "Model not loaded",
      error: `Error: ${hstate.detail || "unknown"}`,
    }[s] || s;
    pill.title = hstate.detail || `${hstate.target_model || ""} ${hstate.adapter ? "+ " + hstate.adapter : ""} · VRAM ${hstate.vram_used_gb ?? "?"}/${hstate.vram_total_gb ?? "?"} GB`;
    if (JSON.stringify([s, hstate.detail]) !== last) { last = JSON.stringify([s, hstate.detail]); onChange?.(hstate); }
    setTimeout(tick, s === "loading" ? 1500 : 5000);
  }
  tick();
}

export const LABEL_CLASS = {
  "included": "ok", "not excluded": "ok", "not included": "bad", "excluded": "bad",
  "not enough information": "warn", "not applicable": "muted", "invalid": "muted",
};
export const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
