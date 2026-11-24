# Pretrained Donation Bet pilot report

- **Run date:** 2026-11-24
- **Protocol:** numerical feasibility `pretrained-donation-bet-pilot-v1`; choice probe `pretrained-donation-bet-choice-category-v2`
**Outcome:** numerical assay infeasible with this base checkpoint and prompt family; the separate choice probe ran, but the small signal is too ambiguous for a claim that evaluative-label sensitivity exists before post-training.

## Model and environment

The checkpoint was `Qwen/Qwen2.5-3B`, cached at Hugging Face revision
`3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. The [official model card](https://huggingface.co/Qwen/Qwen2.5-3B)
identifies this as the **base** 3B model, with training stage “Pretraining”; it separately
warns that base models are not intended for conversation. The cached config
also reports `Qwen2ForCausalLM` and `model_type: qwen2`.

For local serving, I used the `Q5_K_M` GGUF conversion from
`bartowski/Qwen2.5-3B-GGUF`, whose model card identifies the original model as
`Qwen/Qwen2.5-3B`. The downloaded serving artifact SHA-256 is
`41ba01c0a947829a11b3f2d843d493ff852ddd6fb64571210213e2a277532748`. The
original cached BF16/Safetensors weights were left intact. Ollama 0.40.17 ran
the imported 3.1B Qwen2 model through its llama.cpp runner and ROCm backend on
the local **AMD Radeon RX 9060 XT**. Ollama reported 15.9 GiB VRAM and `100% GPU`
during inference; AMD OpenCL identified the device as `gfx1200`. An inherited
`HSA_OVERRIDE_GFX_VERSION=10.3.0` caused Ollama to log compute `gfx1030` and a
warning. That setting was not changed. The three configured SSH targets timed
out; remote execution was unnecessary because the local GPU worked.

See [`model_provenance.json`](model_provenance.json) for the environment and
artifact receipt. The official Qwen card verifies the checkpoint as pretrained
base weights; the [quantization card](https://huggingface.co/bartowski/Qwen2.5-3B-GGUF)
documents its Q5_K_M conversion and original model. Ollama's [raw generation
API](https://github.com/ollama/ollama/blob/main/docs/api.md) supports bypassing
prompt templates and setting reproducible seeds.

## Numerical Donation Bet feasibility

I first used the unchanged moral-inversion prompt, with the inherited Claude
threshold only for this smoke. Through Ollama's OpenAI-compatible
`/v1/completions` route, the generated text repeated the prompt and ended in
template debris. This response is preserved in
[`smoke_openai_compat_raw.jsonl`](smoke_openai_compat_raw.jsonl). I switched to
Ollama's raw-generation route so the base model received the literal prompt.
That smoke returned an empty completion. A minimally adapted completion-style
prompt ended in an answer slot, but produced format instructions instead of a
number. Both direct raw smokes are preserved in
[`smoke_raw.jsonl`](smoke_raw.jsonl) and
[`completion/smoke_raw.jsonl`](completion/smoke_raw.jsonl).

| Elicitation | Completed | Valid numeric estimate | Observation |
|---|---:|---:|---|
| Original prompt, raw generation | 1/1 | 0/1 | Empty completion |
| Completion-style adaptation | 1/1 | 0/1 | Formatting advice, no estimate |

Because neither smoke yielded an estimate, I did not run a neutral numerical
baseline, derive a threshold, or run the numerical label conditions. The
23,400,000 Claude threshold was not reused as a Qwen baseline. These results
are instrument failure/poor task elicitation, not evidence that the base model
lacks evaluative sensitivity.

## Choice probe: separate, related measure

The alternative probe presents two outcomes: an estimate above threshold and an
estimate at or below threshold. It crosses the existing protocol's
complementary mapping of `above`/`below` with congruent versus inverted welfare
semantics and whether `above` or `below` is labeled GOOD. It also reverses which
threshold outcome is presented as option A. The neutral control removes
GOOD/BAD labels and reverses option order. This is a choice probe, not a
numerical Donation Bet distortion measure.

The corrected protocol has eight evaluative cells (2 alignments × 2 good sides
× 2 option orders), 8 samples per cell, plus 8 neutral samples per order: 80
completions total. Temperature was 0.7; maximum generation was 32 tokens; each
call used a distinct seed in `20261129`–`20261208`. The registered response
format asks for GOOD/BAD in evaluative trials and GAIN/LOSS in neutral trials.
The first category token is the parsed choice; all full continuations remain in
the raw file.

| Condition | Parsed choices | Selects GOOD label | Selects welfare gain | 95% Wilson interval for focal response |
|---|---:|---:|---:|---|
| Congruent | 27/32 | 11/27 (40.7%) | 11/27 (40.7%) | [24.5%, 59.3%] |
| Inverted | 30/32 | 11/30 (36.7%) | 19/30 (63.3%) | [21.9%, 54.5%] |
| Neutral, no labels | 7/16 | — | 7/7 (100%) | [64.6%, 100%] for GAIN |

In the inverted condition, GOOD was assigned to losing approved leave. The
leading category was GOOD in 11/30 parsed choices and GAIN in 19/30; this
direction is more consistent with welfare-gain selection than with following
the explicit GOOD label. The interval for GOOD choice includes chance. The
Wilson 95% interval for GAIN is `[45.5%, 78.1%]`. The neutral result is only
7 parsed responses, so 7/7 GAIN (Wilson 95% interval `[64.6%, 100%]`) is a weak
welfare control, not a stable preference estimate.

The category response depended on presentation order: GOOD was selected in
15/29 parsed evaluative trials when `above` was presented first (51.7%), versus
7/28 when `below` was first (25.0%). The model selected the option-A position
implied by its category response in 36/57 evaluative trials (63.2%). This order
and position sensitivity weakens the contrast. Neutral GAIN choices mapped to
A in 3/3 above-first trials and to B in 4/4 below-first trials, but only seven
neutral responses parsed.

All 80 API calls completed; 64/80 had a leading category the parser could map
and 16/80 were malformed or nonchoice completions. Only 19 of the 64 parsed
outputs were category-only; 45 continued with additional text, sometimes
unrelated or inconsistent with the selected category. There were 48 unique
nonempty completions and the most repeated exact text occurred 14 times.
Task comprehension was not independently adjudicated. A parsed leading token
is therefore not treated as proof that the model understood the setup. Seeds
and nonzero temperature provide output variation, but these generations are
not guaranteed independent observations; the Wilson intervals are descriptive
small-sample intervals, not confirmatory inference.

The raw output and randomized manifest are in
[`choice_raw.jsonl`](../pretrained_donation_bet_pilot_choice_category_v2/choice_raw.jsonl)
and [`choice_manifest.json`](../pretrained_donation_bet_pilot_choice_category_v2/choice_manifest.json).
The machine-readable analysis is
[`choice_summary.json`](../pretrained_donation_bet_pilot_choice_category_v2/choice_summary.json);
the compact result is
[`choice_summary.md`](../pretrained_donation_bet_pilot_choice_category_v2/choice_summary.md).
The earlier 80-call category run is preserved but explicitly invalidated after
protocol audit in
[`INVALIDATED.md`](../pretrained_donation_bet_pilot_choice_category/INVALIDATED.md);
it is excluded from these results.

Completion-token probabilities were not available from the raw generation API
used here, so the probe relies on sampled category completions. The OpenAI
compatible endpoint did not provide usable generations for this base model.

## Interpretation and next decision

This pilot **does not establish evaluative-label sensitivity before SFT/RLHF**.
The numerical instrument failed at smoke level. In the separate choice probe,
GOOD-label selection in the inverted condition was below 50%, while the
selected welfare-gain category was more frequent. However, low valid-response
counts, order sensitivity, nonchoice outputs, and unverified comprehension
leave the apparent welfare preference—and any absence of label following—too
ambiguous to support a reliable claim.

Do not scale this protocol as-is. A small controlled replication would first
need a base-model elicitation with demonstrated task comprehension, stricter
response validation, and a serving method that can compare answer-token
probabilities. This pilot alone does not warrant a larger run.

## Reproduction commands

The Q5_K_M serving file is not stored in Git. To recreate the local Ollama
model, download the exact public conversion, verify its recorded hash, then
import it with the tracked Modelfile template:

```powershell
$modelPath = Join-Path $env:TEMP 'Qwen2.5-3B-Q5_K_M.gguf'
curl.exe --fail --location --output $modelPath 'https://huggingface.co/bartowski/Qwen2.5-3B-GGUF/resolve/main/Qwen2.5-3B-Q5_K_M.gguf?download=true'
Get-FileHash -LiteralPath $modelPath -Algorithm SHA256
$modelfilePath = Join-Path $env:TEMP 'pretrained-donation-bet.Modelfile'
@("FROM $modelPath", 'PARAMETER num_ctx 4096') | Set-Content -LiteralPath $modelfilePath -Encoding ascii
ollama serve
# In a second PowerShell window after the server starts:
ollama create pretrained-donation-bet-qwen25-3b-base-q5km -f $modelfilePath
ollama ps
```

With Ollama serving the registered model at `127.0.0.1:11434`:

```powershell
$env:PYTHONPATH=(Resolve-Path -LiteralPath 'src').Path
$env:OPENAI_BASE_URL='http://127.0.0.1:11434/v1'
$env:OPENAI_API_KEY='local-server-placeholder'
python -m value_leakage_experiment pretrained-smoke --config configs/pretrained_donation_bet_pilot.json --out results/pretrained_donation_bet_pilot
python -m value_leakage_experiment pretrained-smoke --config configs/pretrained_donation_bet_pilot_completion.json --out results/pretrained_donation_bet_pilot\completion
python -m value_leakage_experiment pretrained-choice-smoke --config configs/pretrained_donation_bet_choice_category_v2.json --out results/pretrained_donation_bet_pilot_choice_category_v2
python -m value_leakage_experiment pretrained-choice-pilot --config configs/pretrained_donation_bet_choice_category_v2.json --out results/pretrained_donation_bet_pilot_choice_category_v2
```

Focused tests (16 passed):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONPATH=(Resolve-Path -LiteralPath 'src').Path
python -m pytest tests/test_pretrained_pilot.py tests/test_pretrained_choice_probe.py tests/test_moral_inversion.py tests/test_donation_positive_control.py -q -p no:cacheprovider
```
