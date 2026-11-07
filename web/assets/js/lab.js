// Inference Lab: live decoding + attention comparisons, offline benchmark tables, run history.
import { api, fmt, h, initTheme, pollHealth, toast } from "./common.js";

const $ = (id) => document.getElementById(id);
let ready = false;
initTheme();
pollHealth((hs) => { ready = hs.status === "ready"; $("run-dec").disabled = !ready; $("run-dec").title = ready ? "" : "Waiting for the model"; });

const bind = (input, label) => $(input).addEventListener("input", (e) => { $(label).textContent = e.target.value; });
bind("ex", "ex-v"); bind("gamma", "g-v"); bind("maxnew", "m-v"); bind("seq", "s-v");
$("src").addEventListener("change", (e) => { $("custom-wrap").hidden = e.target.value !== "custom"; $("ex-wrap").hidden = e.target.value === "custom"; });

const MODE_NAME = { no_cache: "No cache (reference)", kv_cached: "KV-cached greedy", speculative: "Speculative greedy" };

function bars(rows, key, unit, digits = 1) {
  const max = Math.max(...rows.map((r) => r[key] || 0), 1e-9);
  return h("div", { class: "bars" }, ...rows.map((r) => h("div", { class: "bar" },
    h("span", { text: MODE_NAME[r.mode] || r.impl || r.mode }), h("div", { class: "track" }, h("div", { class: "fill", style: `width:${(100 * (r[key] || 0)) / max}%` })),
    h("span", { class: "v", text: `${fmt(r[key], digits)} ${unit}` }))));
}

function renderDecode(run) {
  const r = run.result, rows = r.modes;
  const kv = rows.find((m) => m.mode === "kv_cached"), sp = rows.find((m) => m.mode === "speculative"), nc = rows.find((m) => m.mode === "no_cache");
  const kpis = [["Prompt tokens", r.prompt_tokens], ["Output tokens", rows[0].output_tokens],
    ["All outputs identical", rows.every((m) => m.tokens_equal_reference) ? "Yes ✓" : "No ✕"]];
  if (nc && kv) kpis.push(["KV cache speed-up", `${fmt(nc.total_ms / kv.total_ms, 2)}×`]);
  if (sp) kpis.push(["Draft acceptance", `${fmt(sp.acceptance_rate * 100, 0)}%`], ["Speculative vs KV", `${fmt(sp.speedup_vs_kv_cached, 2)}×`]);
  $("dec-out").replaceChildren(
    h("div", { class: "kpis" }, ...kpis.map(([l, v]) => h("div", { class: "kpi" }, h("div", { class: "l", text: l }), h("div", { class: "v", text: v })))),
    h("div", { style: "overflow:auto" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ...["Mode", "Prefill ms", "Decode ms", "Total ms", "Tokens/s", "Peak +MiB", "Same tokens", "Acceptance"].map((t) => h("th", { text: t })))),
      h("tbody", {}, ...rows.map((m) => h("tr", {},
        h("td", { text: MODE_NAME[m.mode] + (m.gamma ? ` (γ=${m.gamma})` : "") }), h("td", { text: fmt(m.prefill_ms) }), h("td", { text: fmt(m.decode_ms) }),
        h("td", { text: fmt(m.total_ms) }), h("td", { text: fmt(m.tokens_per_s) }), h("td", { text: fmt(m.peak_extra_mib) }),
        h("td", { text: m.tokens_equal_reference ? "✓" : "✕" }), h("td", { text: m.acceptance_rate != null ? `${fmt(m.acceptance_rate * 100, 0)}%` : "–" })))))),
    h("p", { class: "hint", style: "margin:12px 0 0", text: "End-to-end latency (lower is better)" }), bars(rows, "total_ms", "ms", 0),
    h("p", { class: "hint", style: "margin:12px 0 4px", text: "Generated text (KV-cached reference)" }), h("div", { class: "out-text", text: r.output_text }));
  $("dec-when").textContent = new Date(run.created_at).toLocaleTimeString();
}

$("run-dec").addEventListener("click", async () => {
  $("dec-err").textContent = "";
  const modes = [...document.querySelectorAll(".checks input:checked")].map((c) => c.value);
  if (!modes.length) { $("dec-err").textContent = "Select at least one mode."; return; }
  const custom = $("src").value === "custom";
  if (custom && $("custom").value.trim().length < 3) { $("dec-err").textContent = "Enter a custom prompt."; return; }
  const btn = $("run-dec"); btn.disabled = true; btn.textContent = "Running on GPU…";
  $("dec-out").replaceChildren(h("div", { class: "skeleton", style: "width:60%" }), h("div", { class: "skeleton", style: "width:85%;margin-top:10px" }));
  try {
    const run = await api("/api/lab/decode", { method: "POST", body: {
      prompt_source: custom ? "custom" : "example", example_index: Number($("ex").value), custom_prompt: custom ? $("custom").value.trim() : null,
      modes, gamma: Number($("gamma").value), max_new_tokens: Number($("maxnew").value), ignore_eos: $("noeos").checked, repeats: Number($("repeats").value) } });
    renderDecode(run); loadRuns();
  } catch (e) { $("dec-out").replaceChildren(h("p", { class: "missing", text: e.message })); toast("Comparison failed", e.message, "error"); }
  finally { btn.disabled = !ready; btn.textContent = "Run comparison"; }
});

$("run-att").addEventListener("click", async () => {
  $("att-err").textContent = "";
  if (Number($("heads").value) % Number($("kvh").value)) { $("att-err").textContent = "Query heads must be a multiple of KV heads."; return; }
  const btn = $("run-att"); btn.disabled = true; btn.textContent = "Running…";
  try {
    const run = await api("/api/lab/attention", { method: "POST", body: {
      seq_len: Number($("seq").value), heads: Number($("heads").value), kv_heads: Number($("kvh").value), head_dim: Number($("hd").value),
      block: Number($("block").value), causal: $("causal").checked, dtype: $("adt").value } });
    const r = run.result;
    $("att-out").replaceChildren(
      h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["Implementation", "Median ms", "Max |err|", "Mean |err|", "Peak +MiB"].map((t) => h("th", { text: t })))),
        h("tbody", {}, ...r.rows.map((x) => h("tr", {}, h("td", { text: x.impl }), h("td", { text: fmt(x.ms, 2) }),
          h("td", { text: x.max_abs_err.toExponential(2) }), h("td", { text: x.mean_abs_err.toExponential(2) }), h("td", { text: fmt(x.peak_extra_mib) })))))),
      bars(r.rows.map((x) => ({ ...x, mode: x.impl })), "ms", "ms", 2),
      h("p", { class: "hint", style: "margin-top:10px", text: `Errors vs a float64 ground truth on ${r.device} (${r.dtype}). A materialised score matrix would need ${fmt(r.score_matrix_mib_if_materialised)} MiB in fp32; the tiled kernel only ever holds one tile.` }));
    loadRuns();
  } catch (e) { $("att-err").textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = "Run attention check"; }
});

function csvTable(csv, cols, labels) {
  const [head, ...lines] = csv.trim().split(/\r?\n/);
  const keys = head.split(",");
  const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [keys[i], v])));
  return h("div", { style: "overflow:auto" }, h("table", { class: "data" },
    h("thead", {}, h("tr", {}, ...labels.map((t) => h("th", { text: t })))),
    h("tbody", {}, ...rows.map((r) => h("tr", {}, ...cols.map((c) => { const v = r[c]; const n = Number(v); return h("td", { text: v === "" || v == null ? "–" : Number.isFinite(n) && v.includes(".") ? n.toFixed(2) : v }); }))))));
}

async function loadOffline() {
  try {
    const o = await api("/api/lab/offline");
    const parts = [];
    if (o.kvcache_csv) parts.push(h("h3", { class: "group-h", text: `KV cache (all greedy tokens equal: ${o.kvcache?.all_equal ? "yes" : "no"})` }),
      csvTable(o.kvcache_csv, ["mode", "input_len", "output_len", "prefill_ms", "decode_ms", "tokens_per_s", "peak_extra_mib"], ["Mode", "In", "Out", "Prefill ms", "Decode ms", "Tok/s", "Peak +MiB"]));
    if (o.speculative_csv) parts.push(h("h3", { class: "group-h", style: "margin-top:16px", text: "Speculative decoding" }),
      csvTable(o.speculative_csv, ["workload", "mode", "gamma", "latency_ms", "tokens_per_s", "acceptance_rate", "speedup", "all_equal"], ["Workload", "Mode", "γ", "Latency ms", "Tok/s", "Acceptance", "Speed-up", "Equal"]));
    if (o.attention_csv) parts.push(h("h3", { class: "group-h", style: "margin-top:16px", text: "Attention (causal, 12 heads / 2 KV heads, d=128)" }),
      csvTable(o.attention_csv, ["device", "dtype", "seq", "impl", "median_ms", "peak_extra_mib", "status"], ["Device", "Dtype", "Seq", "Impl", "Median ms", "Peak +MiB", "Status"]));
    $("offline").replaceChildren(...(parts.length ? parts : [h("p", { class: "empty", text: "No offline benchmark files yet (run task bench-kv / bench-spec / bench-attn)." })]));
  } catch (e) { $("offline").replaceChildren(h("p", { class: "empty", text: e.message })); }
}

async function loadRuns() {
  try {
    const runs = await api("/api/lab/runs?limit=10");
    if (!runs.length) return;
    $("runs").replaceChildren(h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["When", "Kind", "Settings", "Summary"].map((t) => h("th", { text: t })))),
      h("tbody", {}, ...runs.map((r) => h("tr", {}, h("td", { text: new Date(r.created_at).toLocaleTimeString() }), h("td", { text: r.kind }),
        h("td", { text: r.kind === "decode" ? `γ=${r.params.gamma} · ${r.params.max_new_tokens} tok · ${r.params.modes.join("/")}` : `seq ${r.params.seq_len} · tile ${r.params.block} · ${r.params.dtype}` }),
        h("td", { text: r.kind === "decode" ? r.result.modes.map((m) => `${m.mode} ${fmt(m.total_ms, 0)}ms`).join(" · ") : r.result.rows.map((x) => `${x.impl} ${fmt(x.ms, 1)}ms`).join(" · ") }))))));
  } catch { /* optional */ }
}
$("runs-refresh").addEventListener("click", loadRuns);
loadOffline(); loadRuns();
