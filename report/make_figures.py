"""Generate every report table (.tex) and figure (.png) from files in results/ (no hand-typed numbers).

Run: `task figures`  (missing inputs are skipped with a message, so it can run at any stage).
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
C = {"base": "#8a94a6", "ft": "#2f5bd3", "kv": "#2f5bd3", "nc": "#c0392b", "spec": "#18794e", "tiled": "#d4860b"}


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def rows_csv(p):
    p = R / p
    return list(csv.DictReader(p.open())) if p.exists() else None


def write(name, body):
    (TAB / name).write_text(body, encoding="utf-8")
    print("table", name)


def table(label, caption, cols, header, rows, star=False, size="\\footnotesize"):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "--"


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# ----------------------------------------------------------------------------- data / model
def t_split():
    m = json.loads((ROOT / "data/splits/split_manifest.json").read_text())
    labs = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
    short = ["incl", "not incl", "excl", "not excl", "NEI", "N/A"]
    rows = []
    for s in ("train", "val", "test"):
        x = m["splits"][s]
        rows.append([s, x["n_patients"], x["n_annotations"], x["type_counts"].get("inclusion", 0),
                     x["type_counts"].get("exclusion", 0)] + [x["label_counts"].get(lab, 0) for lab in labs])
    write("tab_split.tex", table("tab:split", "Patient-grouped split (seed 42) and expert label counts.",
                                 "lrrrr" + "r" * 6, ["Split", "Pat.", "Ann.", "Inc.", "Exc."] + short, rows, star=True))
    shutil.copy(R / "data/token_hist.png", FIG / "token_hist.png")


def t_candidates():
    d = load("model_selection/candidates.json")
    rows = []
    for c in d["candidates"]:
        rows.append([esc(c["model"].split("/")[1]), f(c["params"] / 1e9, 2), c["layers"],
                     f"{c['q_heads']}/{c['kv_heads']}", c["head_dim"], f"{c['vocab_size'] // 1000}k",
                     f"{c['context_window'] // 1024}k", f(c["weights_gb_fp16"], 1), f(c["kv_cache_kib_per_token_fp16"], 0),
                     esc(c["license"].replace(" Community License", " CL").replace("Terms of Use", "ToU"))])
    write("tab_candidates.tex", table(
        "tab:candidates", "Candidate decoders (from config.json; exact parameter counts; 16-bit memory). "
        "KV = KiB of cache per token. All use RoPE, RMSNorm/LayerNorm variants and GQA.",
        "lrrcrrrrrl", ["Model", "Params (B)", "L", "Hq/Hkv", "d", "Vocab", "Ctx", "GiB", "KV", "Licence"], rows, star=True))


def t_membench():
    d = load("training/memory_bench.json")
    rows = []
    for r in d["rows"]:
        st = r["status"]
        st = "OOM" if st == "OOM" else ("incompatible" if "meta" in st else st)
        rows.append([esc(r["strategy"]), "yes" if r["grad_ckpt"] else "no", r["micro_batch"],
                     f(r.get("weights_gb"), 2) if r.get("weights_gb") else "--",
                     f(r.get("peak_gb"), 2) if r.get("peak_gb") else "--",
                     f(r.get("step_s"), 2) if r.get("step_s") else "--", st])
    write("tab_membench.tex", table(
        "tab:membench", f"Measured training configurations on {esc(d['gpu'])} ({d['gpu_total_gb']} GB): "
        "LoRA r=16, longest training example (623 tokens), 4 optimizer steps, first step excluded from timing.",
        "lcrrrrl", ["Strategy", "Ckpt", "MB", "Weights GiB", "Peak GiB", "s/step", "Status"], rows))


def t_hparams():
    import yaml
    c = yaml.safe_load((ROOT / "finetune/configs/lora_qwen15b.yaml").read_text())
    tr = c["train"]
    rows = [["Base model", esc(c["base_model"])], ["Precision", "bf16 base, fp32 LoRA, bf16 autocast"],
            ["LoRA r / $\\alpha$ / dropout", f"{c['lora']['r']} / {c['lora']['alpha']} / {c['lora']['dropout']}"],
            ["Target modules", "q,k,v,o,gate,up,down"], ["Optimizer", "AdamW, wd " + str(tr["weight_decay"])],
            ["LR / schedule", f"{tr['lr']} / cosine, {int(tr['warmup_ratio'] * 100)}\\% warm-up"],
            ["Batch", f"{tr['micro_batch']} $\\times$ {tr['grad_accum']} accumulation"],
            ["Epochs / clip / seed", f"{tr['epochs']} / {tr['max_grad_norm']} / {c['seed']}"],
            ["Max length", f"{c['max_seq_len']} (no truncation)"], ["Grad. checkpointing", str(c["grad_ckpt"])],
            ["Selection", "best val macro-F1 over epochs"]]
    write("tab_hparams.tex", table("tab:hparams", "Fine-tuning configuration.", "ll", ["Setting", "Value"], rows))


# ----------------------------------------------------------------------------- training
def fig_training():
    lh = load("training/lora_r16_bf16/log_history.json")
    vm = load("training/lora_r16_bf16/val_metrics_by_epoch.json")
    if not lh:
        return print("skip training curves")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    st = lh["steps"]
    ax[0].plot([s["epoch"] for s in st], [s["train_loss"] for s in st], color=C["ft"], label="train (target tokens)")
    ax[0].plot([e["epoch"] for e in lh["epochs"]], [e["val_loss"] for e in lh["epochs"]], "o-", color=C["nc"], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].legend(frameon=False)
    if vm:
        ep = [e["epoch"] for e in vm["epochs"]]
        ax[1].plot(ep, [e["macro_f1"] for e in vm["epochs"]], "o-", color=C["ft"], label="val macro-F1")
        ax[1].plot(ep, [e["accuracy"] for e in vm["epochs"]], "s-", color=C["spec"], label="val accuracy")
        ax[1].plot(ep, [e["evidence_micro_f1"] for e in vm["epochs"]], "^-", color=C["tiled"], label="val evidence micro-F1")
        ax[1].axvline(vm["best"]["epoch"], color="#999", ls=":", lw=1)
        ax[1].set_xticks(ep)
        ax[1].set_xlabel("epoch")
        ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)
    print("fig training_curves")


# ----------------------------------------------------------------------------- held-out results
def _m(tag):
    return load(f"final_test/{tag}/metrics.json")


def t_results():
    b, ft = _m("base"), _m("finetuned")
    vb, vf = load("base/val/metrics.json"), load("training/lora_r16_bf16/val_best_metrics.json")
    if not (b and ft):
        return print("skip final test tables")
    rows = []
    for name, m in (("Base (zero-shot)", vb), ("Fine-tuned", vf)):
        if m:
            rows.append([f"{name}, val", f(m["all"]["accuracy"]), f(m["all"]["macro_f1"]), f(m["all"]["macro_f1_present"]),
                         f(m["inclusion"]["macro_f1"]), f(m["exclusion"]["macro_f1"]), f(m["evidence"]["micro"]["f1"]),
                         f(m["evidence"]["macro"]["f1"]), f(m["parse_fail_rate"] * 100, 1)])
    for name, m in (("Base (zero-shot)", b), ("Fine-tuned", ft)):
        rows.append([f"\\textbf{{{name}, test}}", f(m["all"]["accuracy"]), f(m["all"]["macro_f1"]), f(m["all"]["macro_f1_present"]),
                     f(m["inclusion"]["macro_f1"]), f(m["exclusion"]["macro_f1"]), f(m["evidence"]["micro"]["f1"]),
                     f(m["evidence"]["macro"]["f1"]), f(m["parse_fail_rate"] * 100, 1)])
    write("tab_main.tex", table(
        "tab:main", "Same model before and after fine-tuning on identical examples (greedy, KV-cached decoder, bf16, "
        "max 64 new tokens). Macro-F1 over all 6 labels; MF1$_p$ over labels present in gold; Inc./Exc. = 4-label macro-F1 "
        "per criterion type; Ev = evidence F1; Inv = \\% unparseable outputs.",
        "lrrrrrrrr", ["Model / split", "Acc", "MF1", "MF1$_p$", "Inc.", "Exc.", "Ev$_\\mu$", "Ev$_M$", "Inv"], rows, star=True))
    # per class (test)
    rows = []
    for sub in ("inclusion", "exclusion"):
        for lab, v in ft[sub]["per_class"].items():
            vb_ = b[sub]["per_class"][lab]
            rows.append([sub[:3] + ".", esc(lab), v["support"], f(vb_["f1"]), f(v["precision"]), f(v["recall"]), f(v["f1"])])
    write("tab_perclass.tex", table("tab:perclass", "Per-class results on the held-out test split (fine-tuned; base F1 for "
                                    "reference). Classes with zero support cannot be scored.",
                                    "llrrrrr", ["Type", "Label", "n", "F1$_{base}$", "P", "R", "F1"], rows))
    rows = []
    for name, m in (("Base", b), ("Fine-tuned", ft)):
        e = m["evidence"]
        rows.append([name, f(e["micro"]["precision"]), f(e["micro"]["recall"]), f(e["micro"]["f1"]),
                     f(e["macro"]["precision"]), f(e["macro"]["recall"]), f(e["macro"]["f1"]),
                     f(e["nonempty_gold"]["micro"]["f1"]), f"{e['empty_gold']['correct_abstention']}/{e['empty_gold']['n']}",
                     e["unsupported_citations"]])
    write("tab_evidence.tex", table(
        "tab:evidence", "Evidence-sentence retrieval on the test split. Micro pools TP/FP/FN; macro averages per record "
        "(both-empty = 1, see Sec.~\\ref{sec:metrics}). NE = non-empty-gold subset; Abst. = correct empty predictions "
        "for empty gold; Uns. = cited IDs not in the note.",
        "lrrrrrrrrr", ["Model", "P$_\\mu$", "R$_\\mu$", "F1$_\\mu$", "P$_M$", "R$_M$", "F1$_M$", "F1$_{NE}$", "Abst.", "Uns."], rows, star=True))
    for tag in ("base", "finetuned"):
        for sub in ("all", "inclusion", "exclusion"):
            src = R / "final_test" / tag / f"confusion_{sub}.png"
            if src.exists():
                shutil.copy(src, FIG / f"cm_{tag}_{sub}.png")
    ci = ft.get("bootstrap", {})
    (TAB / "ci.tex").write_text(
        f"\\newcommand{{\\ciacc}}{{[{f(ci.get('accuracy_95ci', [0, 0])[0])}, {f(ci.get('accuracy_95ci', [0, 0])[1])}]}}\n"
        f"\\newcommand{{\\cimf}}{{[{f(ci.get('macro_f1_present_95ci', [0, 0])[0])}, {f(ci.get('macro_f1_present_95ci', [0, 0])[1])}]}}\n")
    err = load("final_test/finetuned/error_analysis.json")
    if err:
        rows = [[esc(k), v] for k, v in sorted(err["category_counts"].items(), key=lambda x: -x[1])]
        write("tab_errors.tex", table("tab:errors", f"Rule-assigned error categories, fine-tuned model, test split "
                                      f"({err['n_wrong_label']} wrong labels of {err['n']}; one error may have several).",
                                      "lr", ["Category", "Count"], rows))


# ----------------------------------------------------------------------------- inference
def kv():
    rows, eq = rows_csv("kvcache/finetuned/benchmark.csv"), load("kvcache/finetuned/equality.json")
    if not rows:
        return print("skip kv")
    idx = {(r["mode"], r["input_len"], r["output_len"]): r for r in rows}
    out = []
    for (mode, i, o), r in idx.items():
        if mode != "kv_cached":
            continue
        n = idx[("no_cache", i, o)]
        e = next(c for c in eq["cases"] if str(c["input_len"]) == i and str(c["output_len"]) == o)
        out.append([i, o, f(r["prefill_ms"], 0), f(r["decode_ms"], 0), f(r["tokens_per_s"], 1), f(n["decode_ms"], 0),
                    f(n["tokens_per_s"], 1), f(float(n["decode_ms"]) / float(r["decode_ms"]), 1),
                    f(r["peak_extra_mib"], 0), f(n["peak_extra_mib"], 0), "yes" if e["equal"] else f"no@{e['n_match_prefix']}"])
    write("tab_kv.tex", table(
        "tab:kv", "KV-cached vs.\\ no-cache greedy decoding (fine-tuned target, bf16, batch 1, EOS disabled; mean of 3 "
        "synchronised runs after 1 warm-up). Peak = extra MiB over resident weights. Eq. = token-identical output.",
        "rrrrrrrrrrc", ["In", "Out", "Pre ms", "Dec ms", "tok/s", "Dec ms$_{nc}$", "tok/s$_{nc}$", "Speed-up", "Peak", "Peak$_{nc}$", "Eq."],
        out, star=True))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
    for i, ls in zip(sorted({int(r["input_len"]) for r in rows}), ["-", "--", ":"]):
        for mode, col in (("kv_cached", C["kv"]), ("no_cache", C["nc"])):
            pts = sorted((int(r["output_len"]), float(r["tokens_per_s"])) for r in rows if r["mode"] == mode and int(r["input_len"]) == i)
            ax[0].plot(*zip(*pts), ls, marker="o", ms=3, color=col, label=f"{'KV' if mode == 'kv_cached' else 'no cache'}, in={i}")
            pk = sorted((int(r["output_len"]), float(r["peak_extra_mib"])) for r in rows if r["mode"] == mode and int(r["input_len"]) == i)
            ax[1].plot(*zip(*pk), ls, marker="o", ms=3, color=col)
    ax[0].set_xlabel("output tokens")
    ax[0].set_ylabel("tokens / s")
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=6, ncol=2)
    ax[1].set_xlabel("output tokens")
    ax[1].set_ylabel("peak extra memory (MiB)")
    fig.tight_layout()
    fig.savefig(FIG / "kv_benchmark.png")
    plt.close(fig)
    print("fig kv_benchmark")


def attention():
    s, rows = load("attention/summary.json"), rows_csv("attention/benchmark.csv")
    if not s:
        return print("skip attention")
    vr = []
    for k, v in sorted(s["validation_summary"].items()):
        impl, dt = k.split("|")
        vr.append([esc(impl), dt, v["cases"], f"{v['max_abs_err']:.2e}", f"{v['mean_abs_err_avg']:.2e}"])
    write("tab_attn_err.tex", table(
        "tab:attnerr", "Attention agreement with a float64 reference over the validation grid (3 head layouts incl.\\ "
        "GQA 12:2, 8 $(S_q,S_k)$ shapes incl.\\ decode $S_q{=}1$, $d\\in\\{32,64,128\\}$, causal on/off; tiled also over "
        "tile sizes 16/64/128).", "llrrr", ["Impl.", "dtype", "Cases", "max $|\\Delta|$", "mean $|\\Delta|$"], vr))
    gpu = [r for r in rows if r["device"] == "cuda"]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
    names = {"reference": ("reference", C["nc"]), "tiled_b64": ("tiled 64", C["tiled"]), "tiled_b128": ("tiled 128", "#a0522d"),
             "torch_math": ("torch math", C["base"]), "torch_efficient": ("torch mem-efficient", C["kv"])}
    for impl, (lab, col) in names.items():
        pts = [(int(r["seq"]), float(r["median_ms"]), float(r["peak_extra_mib"])) for r in gpu if r["impl"] == impl and r["status"] == "ok"]
        if pts:
            ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=col, label=lab)
            ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", ms=3, color=col, label=lab)
    for a, yl in zip(ax, ("median ms (causal, H=12, d=128, bf16)", "peak extra MiB")):
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xlabel("sequence length")
        a.set_ylabel(yl)
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "attention_benchmark.png")
    plt.close(fig)
    print("fig attention_benchmark")


def speculative():
    rows, s = rows_csv("speculative/finetuned/benchmark.csv"), load("speculative/finetuned/summary.json")
    if not rows:
        return print("skip speculative")
    out = [[esc(r["workload"]), r["gamma"] if r["mode"] == "speculative" else "greedy", f(r["latency_ms"], 0), f(r["tokens_per_s"], 1),
            f(r["acceptance_rate"], 2) if r["acceptance_rate"] else "--", f(r["mean_accepted_per_round"], 2) if r["mean_accepted_per_round"] else "--",
            f(r["target_forwards_per_token"], 2), f(r["speedup"], 2), "yes" if r["all_equal"] == "True" else "no"] for r in rows]
    write("tab_spec.tex", table(
        "tab:spec", f"Greedy speculative decoding (target: fine-tuned Qwen2.5-1.5B, draft: Qwen2.5-0.5B, bf16) vs.\\ "
        f"KV-cached greedy on the same {s['n_prompts']} validation prompts. Acc.\\ = accepted/proposed; "
        "$\\bar n$ = accepted per round; Fwd/tok = target forwards per output token.",
        "lrrrrrrrc", ["Workload", "$\\gamma$", "Lat. ms", "tok/s", "Acc.", "$\\bar n$", "Fwd/tok", "Speed-up", "Eq."], out, star=True))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
    for wl, col in zip(sorted({r["workload"] for r in rows}), (C["ft"], C["spec"])):
        pts = [(int(r["gamma"]), float(r["speedup"]), float(r["acceptance_rate"])) for r in rows if r["workload"] == wl and r["mode"] == "speculative"]
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=col, label=wl.replace("_", " "))
        ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", ms=3, color=col, label=wl.replace("_", " "))
    ax[0].axhline(1, color="#999", ls=":", lw=1)
    ax[0].set_ylabel("speed-up vs KV greedy")
    ax[1].set_ylabel("draft acceptance rate")
    for a in ax:
        a.set_xlabel("$\\gamma$ (draft tokens per round)")
        a.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "spec_benchmark.png")
    plt.close(fig)
    print("fig spec_benchmark")


if __name__ == "__main__":
    for fn in (t_split, t_candidates, t_membench, t_hparams, fig_training, t_results, kv, attention, speculative):
        try:
            fn()
        except Exception as e:  # keep going; report what is missing
            print(f"!! {fn.__name__}: {type(e).__name__}: {e}")
