// Screen page: inputs, criteria editor, run + progress polling, history.
import { api, ApiError, h, initTheme, pollHealth, store, toast } from "./common.js";
import { renderResults } from "./results.js";

const $ = (id) => document.getElementById(id);
const state = { criteria: [], decoding: "kv_cached", explanation: "llm", ready: false, running: false, examples: [] };

initTheme();
pollHealth((hs) => { state.ready = hs.status === "ready"; syncRun(); if (hs.status === "ready" && hs.detail) toast("Note", hs.detail, "info", 9000); });

// ---------------- draft persistence (per browser, convenience only) ----------------
const draft = store.get("draft", {});
$("note").value = draft.note || "";
$("patient-label").value = draft.patient_label || "";
$("trial-title").value = draft.trial_title || "";
state.criteria = Array.isArray(draft.criteria) ? draft.criteria : [];
function saveDraft() {
  store.set("draft", { note: $("note").value, patient_label: $("patient-label").value, trial_title: $("trial-title").value, criteria: state.criteria });
}

// ---------------- note ----------------
function updateNoteCount() { $("note-count").textContent = `${$("note").value.length} / 12000`; }
$("note").addEventListener("input", () => { updateNoteCount(); clearErr("note"); saveDraft(); syncRun(); if (!$("preview").hidden) preview(); });
$("patient-label").addEventListener("input", saveDraft);
$("trial-title").addEventListener("input", saveDraft);
$("preview-toggle").addEventListener("click", (e) => { e.preventDefault(); $("preview").hidden = !$("preview").hidden; if (!$("preview").hidden) preview(); });
let previewTimer;
function preview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const text = $("note").value.trim();
    if (!text) { $("preview").replaceChildren(h("li", { text: "Nothing to preview yet." })); return; }
    try {
      const r = await api("/api/notes/segment", { method: "POST", body: { text } });
      $("preview").replaceChildren(...r.sentences.map((s) => h("li", {}, h("b", { text: `[${s.id}] ` }), s.text)));
    } catch (e) { $("preview").replaceChildren(h("li", { text: e.message })); }
  }, 250);
}

// ---------------- criteria editor ----------------
function renderCriteria() {
  const list = $("crit-list");
  list.replaceChildren(...state.criteria.map((c, i) => {
    const sel = h("select", { "aria-label": `Criterion ${i + 1} type`, onchange: (e) => { c.type = e.target.value; saveDraft(); } },
      h("option", { value: "inclusion", text: "Inclusion" }), h("option", { value: "exclusion", text: "Exclusion" }));
    sel.value = c.type;
    const ta = h("textarea", { "aria-label": `Criterion ${i + 1} text`, maxlength: 2000, oninput: (e) => { c.text = e.target.value; saveDraft(); syncRun(); } });
    ta.value = c.text;
    return h("div", { class: "crit-row" }, sel, ta,
      h("button", { class: "del", title: "Remove", "aria-label": `Remove criterion ${i + 1}`, onclick: () => { state.criteria.splice(i, 1); renderCriteria(); } }, "×"));
  }));
  if (!state.criteria.length) list.append(h("p", { class: "empty", text: "No criteria yet — paste text, fetch a trial, or add manually." }));
  $("crit-count").textContent = `${state.criteria.length} / 40`;
  saveDraft(); syncRun();
}
function setCriteria(items, title) {
  state.criteria = items.slice(0, 40).map((c) => ({ type: c.type, text: c.text }));
  if (title != null) $("trial-title").value = title || "";
  clearErr("crit"); renderCriteria();
  if (items.length > 40) toast("Trimmed", `Only the first 40 of ${items.length} criteria were kept.`, "info");
}
document.querySelectorAll("[data-add]").forEach((b) => b.addEventListener("click", () => {
  if (state.criteria.length >= 40) return toast("Limit reached", "At most 40 criteria per screening.", "error");
  state.criteria.push({ type: b.dataset.add, text: "" }); renderCriteria();
  $("crit-list").querySelector(".crit-row:last-child textarea")?.focus();
}));
$("clear-crit").addEventListener("click", () => setCriteria([], null));
document.querySelectorAll(".tabs button").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b === t));
  document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== t.dataset.tab; });
}));
$("parse-btn").addEventListener("click", async () => {
  const text = $("paste").value.trim();
  if (text.length < 3) return showErr("crit", "Paste some eligibility text first.");
  try {
    const r = await api("/api/criteria/parse", { method: "POST", body: { text } });
    if (!r.criteria.length) return showErr("crit", "No criteria found in the pasted text.");
    setCriteria(r.criteria, null); toast("Criteria loaded", `${r.criteria.length} criteria parsed — review the types before running.`, "ok");
  } catch (e) { showErr("crit", e.message); }
});
$("nct-btn").addEventListener("click", async () => {
  const id = $("nct").value.trim().toUpperCase();
  $("nct").classList.remove("invalid"); $("nct-err").textContent = "";
  if (!/^NCT\d{8}$/.test(id)) { $("nct").classList.add("invalid"); $("nct-err").textContent = "Use the format NCT followed by 8 digits, e.g. NCT01397994."; return; }
  $("nct-btn").disabled = true; $("nct-btn").textContent = "Fetching…";
  try {
    const r = await api(`/api/trials/${id}`);
    setCriteria(r.criteria, r.title); state.trialId = id;
    toast("Trial loaded", `${r.criteria.length} criteria from ClinicalTrials.gov.`, "ok");
  } catch (e) { $("nct-err").textContent = e.message; }
  finally { $("nct-btn").disabled = false; $("nct-btn").textContent = "Fetch from ClinicalTrials.gov"; }
});

// ---------------- examples (validation split only) ----------------
api("/api/examples").then((ex) => {
  state.examples = ex;
  $("example").append(...ex.slice(0, 30).map((e, i) => h("option", { value: i, text: `${e.patient_id} · ${e.trial_id} (${e.criteria.length} criteria)` })));
}).catch(() => {});
$("example").addEventListener("change", (e) => {
  const ex = state.examples[e.target.value];
  if (!ex) return;
  $("note").value = ex.note; $("patient-label").value = ex.patient_id; state.trialId = /^NCT\d{8}$/.test(ex.trial_id) ? ex.trial_id : null;
  setCriteria(ex.criteria, ex.trial_title); updateNoteCount(); clearErr("note");
});

// ---------------- options ----------------
function seg(id, key, after) {
  $(id).querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $(id).querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); state[key] = b.dataset.v; after?.();
  }));
}
seg("decoding", "decoding", () => { $("gamma-wrap").hidden = state.decoding !== "speculative"; });
seg("explain", "explanation");
$("gamma").addEventListener("input", (e) => { $("gamma-v").textContent = e.target.value; });

// ---------------- validation + run ----------------
function showErr(f, msg) { $(`${f}-err`).textContent = msg; if (f === "note") $("note").classList.add("invalid"); }
function clearErr(f) { $(`${f}-err`).textContent = ""; if (f === "note") $("note").classList.remove("invalid"); }
function syncRun() {
  const btn = $("run");
  btn.disabled = !state.ready || state.running;
  btn.title = !state.ready ? "Waiting for the model to load" : "";
}
function validate() {
  let ok = true;
  const note = $("note").value.trim();
  if (note.length < 20) { showErr("note", "Enter a clinical note of at least 20 characters."); ok = false; }
  const crit = state.criteria.filter((c) => c.text.trim().length >= 3);
  if (!crit.length) { showErr("crit", "Add at least one criterion (3+ characters)."); ok = false; }
  else if (crit.length < state.criteria.length) { showErr("crit", "Empty or very short criteria will be skipped."); }
  return ok ? crit : null;
}
$("run").addEventListener("click", async () => {
  clearErr("note"); clearErr("crit");
  const crit = validate();
  if (!crit) return;
  state.running = true; syncRun();
  const body = {
    patient_note: $("note").value.trim(), criteria: crit.map((c) => ({ type: c.type, text: c.text.trim() })),
    trial_title: $("trial-title").value.trim() || null, trial_id: state.trialId || null,
    patient_label: $("patient-label").value.trim() || null,
    options: { decoding: state.decoding, gamma: Number($("gamma").value), explanation: state.explanation },
  };
  try {
    const sc = await api("/api/screenings", { method: "POST", body });
    follow(sc.id);
  } catch (e) {
    state.running = false; syncRun();
    if (e instanceof ApiError && e.fields.length) {
      for (const f of e.fields) showErr(f.loc[0] === "patient_note" ? "note" : "crit", f.msg);
    } else toast(e.status === 503 ? "Model not ready" : "Could not start screening", e.message, "error", 9000);
  }
});

async function follow(id) {
  const box = $("results");
  box.hidden = false; box.scrollIntoView({ behavior: "smooth", block: "start" });
  let delay = 700;
  for (;;) {
    let sc;
    try { sc = await api(`/api/screenings/${id}`); } catch (e) { toast("Lost connection", e.message, "error"); break; }
    renderResults(box, sc);
    if (sc.status === "done" || sc.status === "error") {
      if (sc.status === "error") toast("Screening failed", sc.error || "", "error", 10000);
      break;
    }
    await new Promise((r) => setTimeout(r, delay)); delay = Math.min(delay * 1.2, 2000);
  }
  state.running = false; syncRun(); loadHistory();
}

// ---------------- history ----------------
async function loadHistory() {
  try {
    const rows = await api("/api/screenings?limit=15");
    if (!rows.length) return;
    $("history").replaceChildren(h("table", {},
      h("thead", {}, h("tr", {}, ...["When", "Patient", "Trial", "Criteria", "Outcome", ""].map((t) => h("th", { text: t })))),
      h("tbody", {}, ...rows.map((r) => h("tr", {},
        h("td", { text: new Date(r.created_at).toLocaleString() }), h("td", { text: r.patient_label || "–" }),
        h("td", { text: (r.trial_title || "–").slice(0, 60) }), h("td", { text: r.n_criteria }),
        h("td", { text: r.outcome || r.status }),
        h("td", {}, h("button", { class: "btn sm", onclick: () => follow(r.id) }, "Open")))))));
  } catch { /* history is optional */ }
}
$("hist-refresh").addEventListener("click", loadHistory);

updateNoteCount(); renderCriteria(); loadHistory();
