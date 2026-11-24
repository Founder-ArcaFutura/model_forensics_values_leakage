# Donation Bet pretrained base-model probability pilot

- **Protocol:** `pretrained-donation-bet-full-likelihood-v1`
- **Date:** 2026-11-24
- **Branch:** `pretrained-donation-bet-pilot`

## Finding

The instrument now measures exact conditional likelihoods for the defined answer strings. The results provide a **weak, mixed indication** of evaluative-label following before SFT/RLHF, not a robust demonstration of it. In the A/B outcome-choice arm, the inverted-label GOOD-minus-BAD margin was slightly positive for both checkpoints: **+0.116** for Qwen2.5-3B and **+0.102** for Qwen2.5-14B. Since GOOD denoted loss in this arm, the corresponding gain-minus-loss margin was slightly negative. The direction appeared in 4/8 and 5/8 prompt variants, respectively. These margins were small, prompt-variable, and vulnerable to response-position effects.

Other controls caution against a simple moral-preference reading. In the direct GOOD/BAD response arm, both models strongly favored the token `BAD` under both welfare mappings. The 3B model selected A as the higher-likelihood A/B answer on all 20 choice prompts, including reversed option orders. The 14B model had a smaller A preference. A separate four-item comprehension check was 3/4 for 3B and 4/4 for 14B. Thus, the pilot shows a candidate-level signal worth checking again, while label-token, answer-position and task-format biases remain material.

The numerical Donation Bet assay did not become usable. The unchanged 14B prompt produced an empty completion. A completion-style smoke emitted an initial `10,000,000,000` estimate, then continued with contradictory task text and a purported “correct answer” of `31,200,000`; the inherited parser extracted that later number. No neutral baseline or model-specific threshold was run. The 3B numerical failures remain recorded in the earlier pilot.

## Models and execution

| Model | Checkpoint stage | Serving quantization | Result records | Exact candidate pairs |
|---|---|---|---:|---:|
| Qwen2.5-3B | Base pretrained, not Instruct; Qwen base revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | Q5_K_M | 44 | 40/40 |
| Qwen2.5-14B | Base pretrained, not Instruct; official base repo revision checked as `97e1e76335b7017d8f67c08a19d103c0504298c9` | Q4_K_M | 44 | 40/40 |

The 14B file came from `mradermacher/Qwen2.5-14B-GGUF` at repository revision `d0a2788fa6dc8e83ce2f0aae07882eeecb06ef1f`; it is a static quant of the Qwen2.5-14B base. Its SHA-256 is `597e04a092b2ca80a66741a0dd4b5f7bc291e2077ede3fb312cb51b48b53c9a2`. The quant card does not pin the exact upstream weight commit used by its quantizer; the independently checked official base revision is therefore recorded separately. The official Qwen card distinguishes the pretrained base from the Instruct model, and the quant card describes the source as Qwen/Qwen2.5-14B. ([Qwen base card](https://huggingface.co/Qwen/Qwen2.5-14B), [GGUF quant card](https://huggingface.co/mradermacher/Qwen2.5-14B-GGUF))

The local GPU was an **AMD Radeon RX 9060 XT, 16,304 MiB**. Ollama/ROCm reported 100% GPU for both locally stored models during discovery. The standalone llama.cpp ROCm binary reported no available devices, so I used the official Windows Vulkan build, which identified `Vulkan0: AMD Radeon RX 9060 XT`. Both native server logs assigned model layers to Vulkan0; there were no CPU model-layer assignments. The scoring build was llama.cpp `b12606`, archive SHA-256 `76e3c7df27b892bff158bf46f30c3a01aff7ff20a05830d6f492721f9ebc136b`.

## Conditions and scoring

Each checkpoint received the same 44 distinct prompts with seed `20261124`:

- **A/B outcome choice:** 16 evaluative prompts (congruent/inverted × GOOD side above/below × presentation order × two wording conditions) plus four neutral prompts.
- **GOOD/BAD label response:** 16 evaluative prompts under a separate answer format.
- **GAIN/LOSS neutral response:** four neutral prompts. `GAIN` tokenizes into two pieces and its full continuation probability is the product of both conditional token probabilities.
- **Task-comprehension check:** four separate A/B items covering welfare semantics and label-to-threshold mapping with reversed option orders.

Probabilities came from llama.cpp native `/completion`, with `n_probs=5000`, `temperature=-1`, `post_sampling_probs=false`, `top_k=0`, `top_p=1`, `min_p=0`, `typical_p=1`, and neutral repetition/frequency/presence penalties. The server's documented negative-temperature path samples greedily while returning a softmax of raw logits without sampler modifications; `post_sampling_probs` was false. ([llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md))

A separate smoke requested the full **151,936-token** distribution. All alternatives were returned, and the sum of exponentiated log probabilities was `1.000254` (small accumulated floating-point/logprob rounding). Candidate token IDs and exact token pieces are preserved. Every target token in the 80 main candidate pairs was present in the initial top 5,000, so no pair was truncated or approximated and no full-vocabulary fallback was needed. The raw candidate likelihoods, log-probability margins, probability mass for each defined pair, top-20 head, exact prompts and request settings are in each model's `raw_probability_results.jsonl`.

The earlier Ollama-only scoring attempts are preserved separately. Ollama's 20-alternative limit left 23/36 3B and 21/36 14B answer pairs incomplete in that staging run; its reported values also changed with temperature. The native llama.cpp run is the primary scoring dataset.

## Probability results

Margins are mean log probability of the first semantic candidate minus the second across distinct prompts. For mapped A/B rows, the summary also re-expresses the margin as GOOD-minus-BAD and welfare GAIN-minus-LOSS. There are eight distinct prompts per evaluative cell and four per neutral cell. These are prompt-level descriptive results, **not independent repeated draws**; no inferential p-values or confidence intervals are claimed.

| Model | Direct GOOD−BAD response: congruent / inverted | A/B implied GOOD−BAD: congruent / inverted | A/B welfare GAIN−LOSS: congruent / inverted / neutral |
|---|---:|---:|---:|
| Qwen2.5-3B Q5_K_M | −0.489 / −0.604 | +0.077 / +0.116 | +0.077 / −0.116 / +0.044 |
| Qwen2.5-14B Q4_K_M | −0.329 / −0.840 | +0.528 / +0.102 | +0.528 / −0.102 / +0.988 |

The direct label response favored `BAD` in 15/16 3B prompts and 16/16 14B prompts, even though the GOOD/BAD-to-welfare mapping was counterbalanced. That is a large response-token/format tendency and should not be called moral preference. In the separate neutral GAIN/LOSS response arm, GAIN had the higher full-continuation likelihood in all four prompts for both models (+2.557 3B; +2.242 14B), but this compares different category strings and is only a four-prompt control. The neutral A/B semantic choice was much smaller for 3B (+0.044) and more positive for 14B (+0.988).

The inverted A/B label margin was positive in both wording conditions and both answer orders when averaged within each stratum, but each stratum contains only two prompts. Individual prompt signs were mixed (4/8 positive for 3B, 5/8 for 14B). Under inversion, GOOD explicitly named the welfare-loss outcome; the resulting mean welfare margin was correspondingly negative. For 14B, the aligned congruent margin (+0.528) was much larger than the inverted label margin (+0.102), consistent with welfare semantics changing or attenuating the response. The 3B contrast was small (+0.077 congruent, +0.116 inverted) and its A/B choices were dominated by A preference.

## Validity, generated choices, and position

All **80/80** main candidate pairs were exactly scored. The tokenizer assigned one token each to leading-space `A` (362), `B` (425), `GOOD` (21722) and `BAD` (47718); leading-space `GAIN` was two tokens (479, 6836), and the scorer summed both conditional log probabilities. Tokenization of no-space variants is also in `tokenizer_probe.json`.

Argmax over each explicitly defined answer pair was A on 20/20 3B letter-choice prompts, versus A on 14/20 and B on 6/20 for 14B. Direct label-token argmax was BAD on 15/16 3B and 16/16 14B prompts; neutral GAIN/LOSS argmax was GAIN on 4/4 prompts for both. These are argmax counts across prompt conditions, not stochastic sample frequencies. Unconstrained one-token greedy output was sometimes empty or unrelated (6/20 empty for 14B letter prompts, for example), so it is retained as a generation-quality diagnostic and not used as the preference estimate.

Mean raw A-minus-B margins for the letter-answer arm were **+0.976 / +1.042** for 3B (above-first / below-first) and **+0.372 / +0.018** for 14B. This persistent A preference, especially for 3B, is a material position/answer-convention effect. The task-comprehension argmax was correct on **3/4** 3B checks and **4/4** 14B checks. That is a small independent instruction-parsing diagnostic, not evidence of moral preference or general capability.

The corrected mapping was audited in code and regression-tested: under inversion, GOOD denotes leave loss and BAD denotes leave gain for either `good_side`; reversing option order changes the A/B mapping without changing the underlying outcome mapping. The earlier invalidated choice run and its correction remain intact in the previous pilot directory.

## Numerical assay and earlier pilot

The 14B unchanged-prompt smoke is at [`numerical_original_smoke`](../pretrained_donation_bet_probability_14b/qwen25_14b/numerical_original_smoke/): empty completion, 0/1 parser-valid estimates. The separately named completion-style smoke is at [`numerical_completion_smoke`](../pretrained_donation_bet_probability_14b/qwen25_14b/numerical_completion_smoke/): the parser extracted 31.2 million from a later contradictory “correct answer,” while the response's initial stated estimate was 10 billion. I treat this as malformed and not task-comprehending, so no neutral baseline or Qwen threshold was calibrated.

The previous Qwen2.5-3B generation pilot remains at [`pretrained_donation_bet_pilot`](../pretrained_donation_bet_pilot/REPORT.md) and [`pretrained_donation_bet_pilot_choice_category_v2`](../pretrained_donation_bet_pilot_choice_category_v2/). Its numerical assay failed and its 80-response categorical pilot was ambiguous and order-sensitive; those results are not pooled here.

## Interpretation and next decision

Because the models are pretrained base checkpoints, the weak inverted-label margin is **consistent with some label sensitivity being elicitable before SFT/RLHF**. It is not strong evidence that this behaviour is reliably present in pretrained models: one model has strong A-position bias, both strongly favor the `BAD` response token in the label-category format, neutral controls vary by answer format, and individual inverted prompts split in direction. The 3B Q5_K_M versus 14B Q4_K_M comparison also mixes parameter count and quantization; the 14B diagnostic advantage (4/4 vs 3/4) is too small to establish a general capacity difference.

**Decision:** run a focused, counterbalanced replication of the inverted A/B effect before making a claim to the research collaborators. Keep direct GOOD/BAD token outputs separate, add another neutral wording family, and vary answer symbols to test the A convention. The pilot is useful feasibility evidence, but not yet a clean result to present as establishing a pretraining/post-training distinction.

## Artifacts

- [`model_provenance.json`](model_provenance.json): base/quant revisions, hashes, tokenizer, GPU and backend.
- [`side_by_side_summary.json`](side_by_side_summary.json): machine-readable model comparison.
- [`condition_analysis.csv`](condition_analysis.csv): trial-level analysis table for both models.
- [`qwen25_3b/manifest.json`](qwen25_3b/manifest.json), [`qwen25_14b/manifest.json`](qwen25_14b/manifest.json): exact randomized prompts and mappings.
- [`qwen25_3b/raw_probability_results.jsonl`](qwen25_3b/raw_probability_results.jsonl), [`qwen25_14b/raw_probability_results.jsonl`](qwen25_14b/raw_probability_results.jsonl): exact candidate scores and prompt/response provenance.
- [`backend_validation.json`](backend_validation.json): full-vocabulary smoke and raw probability normalization.
- [`commands.md`](commands.md): local setup, run and focused test commands.
