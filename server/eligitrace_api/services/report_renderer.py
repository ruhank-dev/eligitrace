"""Readable, self-contained screening reports (HTML for print/PDF, Markdown, JSON) with provenance."""

import html
import json

BADGE = {"included": "#1e7d4f", "not excluded": "#1e7d4f", "not included": "#b3261e", "excluded": "#b3261e",
         "not enough information": "#9a6700", "not applicable": "#5f6b7a", "invalid": "#5f6b7a"}


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _when(ts) -> str:
    """DB timestamps are naive UTC; show them unambiguously, without microseconds."""
    s = str(ts or "")
    return s[:19].replace("T", " ") + " UTC" if s else ""


def to_json(out: dict) -> str:
    return json.dumps(out, indent=2, default=str)


def to_markdown(out: dict) -> str:
    o = out.get("overall") or {}
    L = [f"# EligiTrace screening report `{out['id']}`", "", f"> **{out['disclaimer']}**", "",
         f"- **Trial:** {out.get('trial_title') or '-'} {('(' + out['trial_id'] + ')') if out.get('trial_id') else ''}",
         f"- **Patient:** {out.get('patient_label') or '-'}",
         f"- **Preliminary outcome (rule-based):** **{o.get('outcome', '-')}**",
         f"- **Rule:** {o.get('rule', '-')}", ""]
    L += [f"  - {t}" for t in o.get("trace", [])]
    L += ["", "## Patient note (numbered sentences)", ""] + [f"{s['id']}. {s['text']}" for s in out["sentences"]]
    for typ in ("inclusion", "exclusion"):
        rows = [r for r in out["results"] if r["criterion_type"] == typ]
        if not rows:
            continue
        L += ["", f"## {typ.capitalize()} criteria", ""]
        for r in rows:
            ev = "; ".join(f"[{e['id']}] {e['text']}" for e in r["evidence"]) or "none cited"
            L += [f"### {typ[:3].upper()}-{r['position'] + 1}: {r['criterion_text']}",
                  f"- **Decision:** {r.get('label') or r['status']}",
                  f"- **Evidence:** {ev}", f"- **Explanation:** {r.get('explanation') or '-'}"]
            if r.get("missing_info"):
                L.append("- **Missing information:** the note does not contain the facts this criterion needs.")
    p = out.get("provenance") or {}
    L += ["", "## Provenance", "", "```json", json.dumps(p, indent=2), "```",
          f"Generated: {_when(out['created_at'])}"]
    return "\n".join(L)


def to_html(out: dict) -> str:
    o = out.get("overall") or {}
    sent = "".join(f"<li value='{s['id']}' id='s{s['id']}'>{_e(s['text'])}</li>" for s in out["sentences"])
    blocks = []
    for typ in ("inclusion", "exclusion"):
        rows = [r for r in out["results"] if r["criterion_type"] == typ]
        if not rows:
            continue
        cards = []
        for r in rows:
            lab = r.get("label") or r["status"]
            ev = "".join(f"<li><b>[{e['id']}]</b> {_e(e['text'])}</li>" for e in r["evidence"]) or "<li><i>No sentence cited</i></li>"
            miss = "<p class='miss'>Missing information: the note does not state what this criterion requires.</p>" if r.get("missing_info") else ""
            cards.append(f"<div class='card'><div class='hd'><span class='id'>{typ[:3].upper()}-{r['position'] + 1}</span>"
                         f"<span class='badge' style='background:{BADGE.get(lab, '#5f6b7a')}'>{_e(lab)}</span></div>"
                         f"<p class='crit'>{_e(r['criterion_text'])}</p><ul class='ev'>{ev}</ul>"
                         f"<p class='ex'>{_e(r.get('explanation'))}</p>{miss}</div>")
        blocks.append(f"<h2>{typ.capitalize()} criteria</h2>{''.join(cards)}")
    trace = "".join(f"<li>{_e(t)}</li>" for t in o.get("trace", []))
    prov = _e(json.dumps(out.get("provenance") or {}, indent=2))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EligiTrace report {_e(out['id'])}</title>
<style>
body{{font:14px/1.6 "Manrope",system-ui,-apple-system,"Segoe UI",sans-serif;color:#17140f;max-width:880px;margin:32px auto;padding:0 20px;background:#f5f1e8}}
h1{{font-family:"Cormorant Garamond",Georgia,serif;font-size:38px;font-weight:600;margin:0;letter-spacing:-.005em}}
h2{{font-family:"Cormorant Garamond",Georgia,serif;font-size:24px;font-weight:600;margin:34px 0 12px;border-bottom:1px solid #d6cdb6;padding-bottom:6px;color:#0f3b33}}
.warn{{background:#f3e9cf;border:1px solid #d9c08a;color:#6d5018;padding:11px 14px;border-radius:12px;margin:16px 0;font-size:13px}}
.out{{border:1px solid #e6dfcd;border-left:4px solid #a9853f;border-radius:16px;padding:16px 20px;background:#fffdf8}} .out b{{font-family:"Cormorant Garamond",Georgia,serif;font-size:26px}}
.card{{border:1px solid #e6dfcd;border-radius:16px;padding:14px 18px;margin:12px 0;break-inside:avoid;background:#fffdf8}}
.hd{{display:flex;justify-content:space-between;align-items:center}} .id{{font-weight:700;color:#a9853f;letter-spacing:.12em;font-size:12px}}
.badge{{color:#fff;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}}
.crit{{font-family:"Cormorant Garamond",Georgia,serif;font-size:20px;font-weight:600;margin:8px 0 4px}} .ev{{margin:6px 0;padding-left:18px;color:#4d463a;font-style:italic}} .ex{{color:#4d463a}}
.miss{{color:#8a6420;font-weight:600}} ol{{padding-left:28px}} pre{{background:#f1ebdd;padding:12px;border-radius:12px;overflow:auto;font-size:12px}}
@media print{{body{{margin:0;background:#fff}} .card{{page-break-inside:avoid}}}}
</style></head><body>
<h1>EligiTrace screening report</h1><p>ID <code>{_e(out['id'])}</code> &middot; {_e(_when(out['created_at']))}</p>
<div class="warn"><b>{_e(out['disclaimer'])}</b></div>
<p><b>Trial:</b> {_e(out.get('trial_title') or '-')} {_e(out.get('trial_id') or '')} &nbsp; <b>Patient:</b> {_e(out.get('patient_label') or '-')}</p>
<div class="out">Preliminary outcome (rule-based aggregation): <b>{_e(o.get('outcome', '-'))}</b><ul>{trace}</ul>
<small>Rule: {_e(o.get('rule', ''))}</small></div>
<h2>Patient note (numbered sentences)</h2><ol start="0">{sent}</ol>
{''.join(blocks)}
<h2>Provenance</h2><pre>{prov}</pre></body></html>"""
