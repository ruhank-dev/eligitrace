// Renders a screening: overall rule-based outcome, numbered note with highlights, criterion cards.
import { fmt, h, LABEL_CLASS } from "./common.js";

const GLYPH = { "Likely eligible": "✓", "Likely ineligible": "✕", "Needs more information": "?", "Incomplete": "…" };

function highlight(note, ids, focus = false) {
  note.querySelectorAll("li").forEach((li) => li.classList.remove(focus ? "focus" : "cited"));
  ids.forEach((id) => note.querySelector(`li[data-id="${id}"]`)?.classList.add(focus ? "focus" : "cited"));
}

function criterionCard(r, note) {
  const lab = r.label || (r.status === "pending" ? "pending" : r.status);
  const ids = r.evidence.map((e) => e.id);
  const prefix = r.criterion_type === "inclusion" ? "INC" : "EXC";
  const card = h("article", { class: "card crit-card", tabindex: 0, "aria-label": `${prefix}-${r.position + 1}: ${lab}` },
    h("div", { class: "top" },
      h("div", {}, h("div", { class: "cid", text: `${prefix}-${r.position + 1}` }), h("div", { class: "ctext", text: r.criterion_text })),
      r.status === "pending" ? h("span", { class: "badge muted" }, "screening…")
        : h("span", { class: `badge ${LABEL_CLASS[lab] || "muted"}` }, lab)));
  if (r.status === "pending") { card.append(h("div", { class: "skeleton", style: "margin-top:10px;width:70%" })); return card; }
  if (r.status === "error") { card.append(h("p", { class: "missing", text: r.error || "This criterion failed; try again." })); return card; }
  if (ids.length) {
    card.append(h("div", { class: "chips" }, h("span", { class: "hint", text: "Evidence:" }),
      ...ids.map((id) => h("button", { class: "chip", title: "Show in note", onclick: () => {
        highlight(note, [id], true); note.querySelector(`li[data-id="${id}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
      } }, `#${id}`))));
    card.append(h("ul", { class: "evq" }, ...r.evidence.map((e) => h("li", {}, h("b", { text: `[${e.id}] ` }), e.text))));
  } else {
    card.append(h("div", { class: "chips" }, h("span", { class: "hint", text: "No sentence cited." })));
  }
  if (r.explanation) {
    card.append(h("p", { class: "explain" }, r.explanation, " ",
      h("span", { class: "src", text: r.explanation_source === "llm-base-grounded" ? "· grounded LLM rationale" : "· evidence template" })));
  }
  if (r.missing_info) card.append(h("div", { class: "missing", text: "Missing information — ask for the facts this criterion needs; absence is not treated as proof." }));
  if (r.parse_ok === false) card.append(h("div", { class: "missing", text: "The model output could not be parsed; this criterion is treated as unscreened." }));
  if (r.unsupported_ids?.length) card.append(h("div", { class: "missing", text: `Model cited non-existent sentence IDs ${r.unsupported_ids.join(", ")} — removed.` }));
  const st = r.decoding_stats || {};
  card.append(h("div", { class: "meta" },
    h("span", { text: `${r.decoding === "speculative" ? `speculative γ=${st.gamma}` : "KV-cached greedy"}` }),
    st.acceptance_rate != null ? h("span", { text: `acceptance ${fmt(st.acceptance_rate * 100, 0)}%` }) : null,
    h("span", { text: `generate ${fmt(r.generation_ms, 0)} ms` }), h("span", { text: `explain ${fmt(r.explanation_ms, 0)} ms` })));
  card.addEventListener("mouseenter", () => highlight(note, ids));
  card.addEventListener("focus", () => highlight(note, ids));
  return card;
}

export function renderResults(box, sc) {
  const pct = sc.progress.total ? (100 * sc.progress.done) / sc.progress.total : 0;
  const o = sc.overall;
  const head = h("div", { class: "page-head" },
    h("div", {}, h("h1", { text: "Screening results" }),
      h("p", { text: `${sc.trial_title || "Untitled trial"}${sc.trial_id ? " · " + sc.trial_id : ""}${sc.patient_label ? " · " + sc.patient_label : ""} · ${sc.progress.done}/${sc.progress.total} criteria` })),
    sc.status === "done" ? h("div", { class: "export" },
      h("a", { class: "btn", href: `/api/screenings/${sc.id}/report.html`, target: "_blank", rel: "noopener" }, "Open printable report"),
      h("a", { class: "btn", href: `/api/screenings/${sc.id}/report.md`, download: "" }, "Markdown"),
      h("a", { class: "btn", href: `/api/screenings/${sc.id}/report.json`, download: "" }, "JSON")) : null);
  const outcome = o
    ? h("div", { class: "card outcome", "data-o": o.outcome },
        h("div", { class: "glyph", "aria-hidden": "true", text: GLYPH[o.outcome] || "•" }),
        h("div", {}, h("div", { class: "hint", text: "Preliminary outcome · rule-based aggregation (not a ground-truth label)" }),
          h("div", { class: "big", text: o.outcome }), h("ul", {}, ...o.trace.slice(0, 6).map((t) => h("li", { text: t })))),
        h("div", { class: "tally" }, ...Object.entries(o.counts).map(([k, v]) => h("span", { class: `badge ${LABEL_CLASS[k] || "muted"}`, text: `${v} × ${k}` }))))
    : h("div", { class: "card", style: "padding:14px 16px" }, h("b", { text: sc.status === "error" ? "Screening failed" : "Screening in progress…" }),
        h("div", { class: "progress" }, h("div", { style: `width:${pct}%` })));
  const note = h("ol", {}, ...sc.sentences.map((s) => h("li", { "data-id": s.id }, h("span", { class: "sid", text: s.id }), h("span", { text: s.text }))));
  const noteCard = h("aside", { class: "card note-view" }, h("div", { class: "card-h" }, h("h2", { text: "Patient note" }), h("span", { class: "hint", text: "hover a decision to see its evidence" })),
    h("div", { class: "card-b" }, note));
  const groups = ["inclusion", "exclusion"].map((t) => {
    const rows = sc.results.filter((r) => r.criterion_type === t);
    return rows.length ? h("div", {}, h("div", { class: "group-h", text: `${t} criteria (${rows.length})` }), ...rows.map((r) => criterionCard(r, note))) : null;
  });
  box.replaceChildren(head, outcome, h("div", { class: "res-grid" }, noteCard, h("div", {}, ...groups)),
    h("p", { class: "hint", style: "margin-top:12px", text: sc.disclaimer }));
}
