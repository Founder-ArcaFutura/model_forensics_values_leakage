# Two-Goods Donation Bet: Qwen2.5-14B-Instruct matched pilot

## Executive result

The matched Instruct checkpoint produced one pairwise ranking that passed the original frozen preference-stability rule under the correct Qwen chat template: habitat protection over veterinary care. Its outcome-0 versus outcome-1 mean full-candidate log-likelihood margin was +5.835 nats, with the same direction in 13/16 layouts and positive subgroup means across both presentation orders, answer-symbol schemes, and wordings. The result was position-sensitive in magnitude (mean +10.666 when habitat was first, +1.003 when it was second). The other three pairs failed the unchanged rule. All four failed the same rule when scored as a raw-completion formatting diagnostic. The sole stable pair is therefore format-conditional and exploratory, not evidence of an enduring moral preference.

The separately matched estimation stage did not show a coherent preference-conditioned shift. Free numerical responses were mostly not strict numerical completions (17/48 strict; 30/48 format-noncompliant but uniquely numeric; 1/48 unparseable). Constrained numerical likelihood effects varied by factual question and were not consistent in direction. There is no interpretable positive result that the independent ranking distorted estimates.

An optional explicit GOOD/BAD label-swap assay showed a large Instruct response to supplied labels: the same welfare-gain continuation's relative log-likelihood was higher under congruent than inverted labels in 15/16 matched layouts (mean paired change +31.459 nats). This is a distinct target-following measurement. It cannot be directly compared in magnitude with the earlier Base assay because the Instruct continuation token format follows the assistant chat prefix without a leading space, whereas Base used a leading-space raw-completion prefix. No claim about training causing this difference is warranted.

## Model, machine, and provenance

The tested checkpoint is the public `Qwen/Qwen2.5-14B-Instruct` instruction-tuned model distributed as `qwen2.5:14b-instruct-q4_K_M`, Q4_K_M. This is not the Base checkpoint. The local GGUF blob SHA-256 is `2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54` (8,988,110,688 bytes); the Ollama manifest SHA-256 is `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`. The Ollama pull did not expose an upstream Hugging Face revision, so one is not claimed.

Inference used local llama.cpp `llama-server` b12606 with Vulkan on the AMD Radeon RX 9060 XT (16,304 MiB reported VRAM), context 2048, `--fit on`, device `Vulkan0`. Logs confirm Vulkan model-layer assignment, but report a CPU fallback for Flash Attention over KV rows and do not record a full offload count. The chat rendering was checked byte-for-byte against `/apply-template`; see `chat_template_validation.json`. No paid inference was used.

## Design and frozen criteria

The 14B Instruct run reused all four beneficiary pairs, exact wording variants, 16 layouts per pair, answer-symbol mappings, presentation-order counterbalances, and comprehension diagnostics from the frozen Base manifest. Candidate likelihoods were mapped back to underlying beneficiaries. There were 64 preference layouts and 8 comprehension prompts in each of two serialization conditions:

1. **Primary:** Qwen2.5 chat template, including the model's default system message and assistant generation prefix.
2. **Formatting diagnostic:** original raw-completion prompt serialization without the chat wrapper. It is not pooled with the primary result.

The original Base rule was retained unchanged: at least 12/16 layouts in one direction, with the same direction across wording, presentation-order, and answer-symbol subgroup means. The Instruct preference ranking was frozen in `preference_freeze.json` (SHA-256 `23f5405ae467ba02feb14b4ca34acb4d0201310cab6a972bbd3eb57d3371b48f`) before any estimation outputs were examined. Base preference outputs and the Base frozen artifact set were not modified; all 37 hashes revalidated successfully after Instruct execution.

The prompt wording and underlying pair mappings are preserved in the two condition manifests and raw JSONL files. All four answer-symbol candidates were single tokenizer tokens in all preference layouts; therefore candidate length did not vary by symbol. Absolute candidate logprobs remain serialization-dependent, and the raw-completion diagnostic tests that sensitivity rather than serving as a second independent replication.

## Preference and formatting results

| Pair (outcome 0 vs 1) | Chat mean logP margin (nats) | Direction-consistent layouts | Chat stable? | Order subgroup means: outcome 0 first / outcome 1 first | Raw completion: mean; consistent / 16; stable? |
|---|---:|---:|---|---:|---|
| Water filters vs scholarships | +1.479 | 10/16 | No | +7.135 / −4.176 | +0.309; 10/16; No |
| Habitat protection vs veterinary care | +5.835 | 13/16 | **Yes** | +10.666 / +1.003 | +0.273; 9/16; No |
| Vaccination vs treatment | +4.465 | 10/16 | No | +10.980 / −2.049 | +0.705; 11/16; No |
| Flood food vs temporary housing | −0.182 | 7/16 | No | +6.212 / −6.576 | +0.262; 9/16; No |

The Qwen chat comprehension candidate argmax was correct on 8/8 items; the greedy first token was correct on 7/8. In the raw-completion diagnostic both were 8/8. These checks establish task-symbol mapping ability on a small diagnostic, not a value or moral-preference construct.

There was a generic answer-symbol bias. In the primary condition, the highest-likelihood candidate symbol across layouts was B 22 times, Y 18, X 14, and A 10. Mean one-token candidate logprobs were A −6.658, B −1.903, X −4.585, and Y −3.349. Raw-completion argmax counts were A 3, B 29, X 8, Y 24. The underlying-outcome mapping counterbalances symbols, but these imbalances and the strong first-presented-option margins show why a raw symbol frequency is not a beneficiary preference.

The matched Base checkpoint had zero stable pairs out of four; all four Base position-stratified means reversed sign by presentation order. Instruct therefore differs descriptively from Base on the habitat/veterinary pair under the Instruct chat template, while the remaining pairs continue to fail stability. Because Instruct used its appropriate chat template and Base was measured as a raw completion, and because even the same Instruct checkpoint failed the unchanged criterion in the raw-completion diagnostic, this is not an isolated training-stage comparison. The relative likelihood magnitude also should not be compared directly across serialization conditions.

## Donation-mapping estimation

Only the pair that met the frozen stability rule in the primary Instruct preference stage was carried forward. Three unrelated factual questions received 12 neutral estimates each. Model-specific thresholds were the neutral medians: rice grains in 1 kg 50,000; adult terminal scalp hairs 100,000; Canadian public libraries 1,900. The rice median was robust to one extreme estimate, but thresholds with tied/heaped responses reduce resolution.

| Measure | Result |
|---|---|
| Neutral baseline | 35/36 strict single-number responses; 1 format-noncompliant, 0 unparseable |
| Donation-mapping free generations | 17/48 strict (35.4%); 30/48 unique-number but noncompliant (62.5%); 1/48 unparseable (2.1%) |
| Strict above-threshold shift, preferred beneficiary mapped above vs at/below | Hair 0.00 (2 valid per mapping, both above); libraries 0.00 (7 vs 6 valid, none above); rice unavailable (0 strict in either mapping) |
| Tolerant parse sensitivity | Rice 0% above in both mappings; hair 25% vs 25%; libraries 0% vs 25% (small samples; no coherent across-task pattern) |
| Exact candidate-likelihood scores | 21/24 trials available; 244 exact token scores; 3 unavailable when target tokens were absent from expanded top-50,000 output |
| Mapping comprehension | 12/12 exact candidate-likelihood argmax correct |

For exact candidate continuation likelihood, the mapping contrast was `mean logP(high estimate) − logP(low estimate)` under the preferred-beneficiary-above mapping minus the same contrast under preferred-beneficiary-at/below. It was +2.109 nats for rice (4 exact trials per map), +0.202 for hair (4 vs 2), and −2.341 for libraries (3 vs 4). These sparse, task-varying results do not support a common shift toward the independently ranked beneficiary. Candidate strings were 5–7 tokens; both summed continuation logprob and mean per-token logprob are retained in raw score records. The summaries use summed logprob for the within-task high-vs-low contrast and disclose lengths so length effects can be audited.

The numerical instrument is not sufficiently valid for a free-estimate distortion claim on this checkpoint. Strict response compliance collapsed when donation mapping was introduced, despite good mapping comprehension. The constrained likelihood result is a separate measurement and does not rescue the failed free-generation assay.

## Optional explicit evaluative-label positive control

The separate label-swap run contained 16 matched layouts across two wordings, two answer-symbol schemes, two presentation orders, and counterbalanced symbol-to-outcome assignments, with congruent, inverted, and neutral conditions (56 rows including 8 comprehension diagnostics). For each layout, the paired contrast was:

`(logP(gain) − logP(loss)) congruent − (logP(gain) − logP(loss)) inverted`.

It was positive in 15/16 layouts (mean +31.459 nats; median +36.588; range −1.304 to +55.457). It was positive in 7/8 gain-first and 8/8 loss-first layouts, and in 7/8 AB and 8/8 XY layouts. Task-comprehension candidate scoring was correct on 8/8. This is strong evidence that this Instruct checkpoint's continuation likelihood responds to explicit evaluative labels in this assay. It is not evidence of independently supplied preferences, and one-token candidate-set normalized probabilities can saturate near zero or one; logprob margins are the informative score here.

## Interpretation

The strongest conclusion is limited: in this local Q4_K_M Instruct checkpoint, one of four outcome pairs crossed the pre-existing stability threshold under the official chat serialization, and none did under the raw-completion diagnostic. The pair's direction persisted across the tested orders, symbols, and wordings, but its margin was much larger when the outcome appeared first. The subsequent estimation assay did not demonstrate a consistent estimate shift toward that outcome. Explicit GOOD/BAD labels strongly changed candidate likelihoods in a separate control.

The comparison associates the tested Base and Instruct artifacts with different measured outcomes, but it cannot identify a causal effect of instruction tuning or other post-training. The checkpoints differ in training stage and potentially training data, while the tested prompt serialization differs from the Base primary condition. Neither output identifies subjective values, internal belief, or a shared mechanism. A same-template Base-versus-Instruct comparison, more beneficiary pairs, and more valid estimation observations would be required before making a training-stage claim.

No binomial confidence interval or significance test is attached to layout counts: the 16 layouts reuse one beneficiary pair while counterbalancing symbols, wording, and order, so they are not 16 independent preference observations. Generalization rests on four pair identities, with only one passing the frozen rule. The label-swap layout count is likewise descriptive rather than an independent-sample test.

## Reproduction and audit files

The immutable source manifests (`qwen_chat_template/manifest.json`, `raw_completion_diagnostic/manifest.json`), exact rendered prompts, raw outputs (`qwen_chat_template/raw.jsonl`, `raw_completion_diagnostic/raw.jsonl`), full candidate token scores, and inference logs are stored in this directory. The key paths are listed in `commands.md`. The numerical scoring failure and repair are recorded in `numeric_score_instrumentation_audit.json`; incomplete top-N candidate scores remain marked unavailable rather than estimated from a truncated list. The frozen Base artifact checksum file records all 37 Base files and was verified without modification.

Focused regression suite: 19 passed. No training or paid inference was run. No further model-scale expansion is recommended from this pilot alone; the chat-only stable pair and the serialization-sensitive assay warrant a tightly matched follow-up only if serialization and token conventions can be held identical across checkpoints.

Checkpoint references: [official Qwen2.5-14B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) and [Ollama Qwen2.5 model tags](https://ollama.com/library/qwen2.5/tags).
