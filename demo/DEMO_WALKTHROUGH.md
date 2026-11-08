# EligiTrace: demo video walkthrough (5-7 minutes)

You record and narrate this yourself. Everything below was tested on the project machine
(GTX 1660 SUPER). Nothing here modifies the experimental evidence in `results/`.

## 0. Before you press record (2 minutes, off camera)

1. Close games and overlays (VALORANT held about 1.9 GB of VRAM when measured).
2. In a terminal at the repo root:
   ```
   uv run python -m demo.preflight
   ```
   Every line should say `PASS` (the "app is not running" line is a `WARN` until the next step).
3. Start the app and wait for the green status pill (about 1-2 minutes):
   ```
   task app
   ```
   Open <http://127.0.0.1:8000>. The pill reads `Ready · fine-tuned · GTX 1660 SUPER · bfloat16`.
4. Keep two files open to copy from: `demo/sample_note.txt` and `demo/sample_criteria.txt`.
5. Do **not** run `task train`, `task eval-final` or the `bench-*` tasks before or during the demo:
   they overwrite committed results, and the final test evaluation refuses to run twice.

## 1. Opening (0:00-0:40) screen: README on GitHub

Say: "I am Muhammad Ruhan Kamran, 23i-0062, CS-A. EligiTrace takes a patient note and one clinical-trial
criterion, and returns a label from the expert categories plus the note sentences that justify it. It uses
Qwen2.5-1.5B fine-tuned with LoRA on the TrialGPT criterion annotations, runs locally on a 6 GB GPU, and
includes my own KV-cache decoder, tiled online-softmax attention and speculative decoding. It is an
educational prototype, not a clinical decision tool."

## 2. Results at a glance (0:40-1:30) screen: README "Results" table, then `report/EligiTrace_Report.pdf`

Highlight (all from `results/final_test/`, evaluated once and frozen in `FROZEN.json`):
- Same model before vs after fine-tuning, identical inputs: accuracy 0.160 -> 0.733; evidence micro-F1 0.169 -> 0.561;
  0 % unparseable output.
- Split is by patient (37/8/8, seed 42) and the GPT-4 columns are never used (a test proves it).
- Mention the honest limits: 8 test patients, rare labels are weak, no "excluded" case in the test set.

## 3. Live screening of an unseen case (1:30-3:30) screen: app, Screen tab

1. Paste `demo/sample_note.txt` into the note box; paste `demo/sample_criteria.txt` into the criteria box and click
   **Split into criteria**; fill any title; click **Run screening**.
2. Expected: 7 criterion cards appear one by one; the overall banner ends as **Needs more information**.
3. Point at the numbered note. Hover a card: its evidence sentences highlight. Show a `not enough information` card
   and say: "missing facts are never treated as proof".
4. Say: "the overall outcome is an explicit three-step rule shown with its trace, not a prediction."
5. Click the report buttons (HTML / Markdown / JSON) and show the **Provenance** block: model, adapter SHA-256,
   decoding settings, commit.

## 4. Invalid input recovery (3:30-3:50)

Clear the note and click **Run**: field-level messages appear and nothing typed is lost. Type `NCT12` in
**Fetch by NCT ID** to show the 422 message.

## 5. Inference Lab (3:50-5:30) screen: Inference Lab tab

1. Validation example 0, gamma = 4, click **Run comparison**. Expected: three rows (no cache, KV cache, speculative),
   all with token equality OK; KV cache about 2.5-2.7x faster than no cache; speculative acceptance 100 % on this
   prompt and faster than KV-cached (the measured spread over repeated runs is in `results/lab/live_runs.json`).
2. Click **Run attention check** (float32, 512 tokens): tiled attention error around 1e-6 against a float64 reference,
   far less memory than the full score matrix, but slower than PyTorch's fused call.
3. Scroll to **Offline full benchmarks** and say: "over the full grid the KV cache reaches 18-20x at 1,024 tokens. Speculative
   decoding is slower on average here (0.39-0.69x) because the 0.5B draft step costs 79 % of a target step; it only
   wins when almost every draft token is accepted."

## 6. Tests and evidence (5:30-6:20) screen: terminal

```
task test                       # CPU suite: all pass in about 30 s
uv run pytest -m gpu            # real models in bf16: passes (ties are checked, see below)
```
Say: "In bf16 a few outputs differ between decoders at logit ties within one rounding step. The test measures the
size of the tie at the first divergence and only accepts differences it explains; in float32 on the CPU the same
real models are token-identical." Show `results/logs/test_gpu_after_fix.log` and `results/kvcache/fp32_check.json`.

## 7. Experiments tour (6:20-6:50) screen: report section "Additional validation experiments"

One sentence each: seeds (mean and spread), oversampling, QLoRA, hyper-parameters, fine-tuned draft model. All on the
validation split only, with the test set untouched.

## 8. Rubric map and closing (6:50-7:00)

Show `README.md` project structure and the Hugging Face adapter page. Say: "the adapter is on Hugging Face and
`task download-weights` rebuilds the setup. Thank you."

## If something goes wrong on camera

- App stuck on Loading for more than 3 minutes: stop it (Ctrl+C), check no other program is using the GPU, rerun `task app`.
- A Lab timing looks different from the report: these are live measurements on a shared GPU; the repeated-run
  spread is reported in `results/lab/live_runs.json`.
