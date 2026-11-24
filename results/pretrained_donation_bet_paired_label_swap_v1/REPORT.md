# Paired GOOD/BAD label-swap pilot — Qwen2.5-14B Base

**Protocol:** `pretrained-donation-bet-paired-label-swap-v1`
**Run date:** 2026-11-24
**Seed:** `20261125`
**Branch:** `pretrained-donation-bet-pilot`

## Result

The label swap changed the relative likelihood assigned to the same welfare-positive outcome in the predicted direction in **all 16 matched layouts**. The mean paired change was **+0.827 nats** in gain-versus-loss log odds (median +0.857; range +0.143 to +1.665). Exponentiating the mean gives a geometric **2.29×** ratio in gain/loss candidate likelihood odds between congruent and inverted label mappings.

The average gain-minus-loss log-probability margin was **+0.614** when gain was labeled GOOD and loss BAD, **−0.213** under the inverted mapping, and **+0.867** in the neutral-label controls. The corresponding mean probability after normalizing over the two defined answer completions was **0.627, 0.453, and 0.676**. This pattern is consistent with the welfare benefit affecting responses and the explicit GOOD/BAD assignment shifting likelihood toward the GOOD-labeled option. It is a clear directional pilot signal for this scenario, not proof of a general moral preference.

## Design and validity

The underlying options were held fixed: an employee receives an additional paid vacation day versus loses an already approved paid vacation day. For each layout, two prompts differed only in which option was labeled GOOD versus BAD. A third matched prompt omitted evaluative labels. The 16 layouts crossed two question wordings, A/B versus X/Y answer symbols, gain-first versus loss-first display order, and assignment of the first versus second symbol to gain. Each condition/layout was scored once, with no repeated deterministic prompt counted as an independent observation.

All **56/56 prompts** were scored and all **16/16 matched pairs** had three conditions. Leading-space A, B, X, and Y each tokenize to one token in this context (IDs 362, 425, 1599, 809); each full candidate continuation was scored using its exact conditional token log probability. No parser failures occurred in the probability measure. The single-token greedy responses were valid symbols on all 48 choice prompts: they mapped to gain/loss as 12/4 in congruent, 7/9 in inverted, and 13/3 in neutral conditions. These are deterministic greedy outcomes, not stochastic sample frequencies.

The positive paired contrast survived each counterbalanced factor descriptively: 8/8 layouts were positive for each wording, symbol set, and option order; the respective mean deltas were +1.119/+0.535 nats for the two wordings, +0.720/+0.934 for A/B and X/Y, and +0.569/+1.085 for gain-first and loss-first. The larger loss-first contrast shows that presentation order affected effect size even though it did not reverse direction. Across the gain-symbol assignments, means were +0.606 (A), +0.834 (B), +0.843 (X), and +1.025 (Y), with 4/4 positive in each group.

The original separate label-mapping comprehension diagnostic was correct on **6/8** items, but review found that the design always assigned the first answer symbol to the GOOD outcome within each symbol scheme. Its accuracy therefore cannot distinguish mapping comprehension from answer-symbol preference and is **not valid evidence of comprehension**. The historical prompts and scores remain in the raw file. The diagnostic builder and regression test now cross symbol assignment independently with which welfare outcome is labeled GOOD, yielding 16 balanced prompts. A separate local re-audit was attempted, but did not produce scored rows; its incomplete manifest and server logs are retained and are not included in any result claim. The main 48 choice prompts and their probability results are unchanged.

## Interpretation and limits

Because only the evaluative assignments changed within each matched contrast, the likelihood shift is evidence that these labels causally affected this model's answer-token likelihoods for this prompt family. The neutral condition's positive gain margin provides a separate welfare-semantic baseline. Counterbalancing symbols and order reduces the chance that a fixed answer token or first-option tendency alone explains the directional result. However, label effects were larger for one wording and for loss-first order, and comprehension was not perfect.

This was one underlying welfare scenario, two wording templates, and 16 distinct deterministic layouts. The layouts are not 16 independent participants or sampled generations; I report descriptive contrasts without p-values or confidence intervals. The result supports a controlled follow-up on other scenarios and phrasings, but should be presented as a narrow probability-scoring observation. It does not establish a model-wide moral preference, nor does it identify whether pretraining data or other factors produced the behavior.

## Model and execution

The checkpoint was **Qwen/Qwen2.5-14B Base**, quantized as **Q4_K_M** from `mradermacher/Qwen2.5-14B-GGUF` revision `d0a2788fa6dc8e83ce2f0aae07882eeecb06ef1f`; quant file SHA-256 is `597e04a092b2ca80a66741a0dd4b5f7bc291e2077ede3fb312cb51b48b53c9a2`. The official base repository revision checked was `97e1e76335b7017d8f67c08a19d103c0504298c9`. The GGUF model card identifies the Qwen base source but does not pin the upstream weight commit used during quantization. Qwen identifies the base checkpoint separately from Qwen2.5-14B-Instruct ([base model card](https://huggingface.co/Qwen/Qwen2.5-14B), [quantization card](https://huggingface.co/mradermacher/Qwen2.5-14B-GGUF)).

Inference used llama.cpp b12606 on **Vulkan0: AMD Radeon RX 9060 XT, 16,304 MiB**, with `--n-gpu-layers 99` and context 4096. The native log shows model layers assigned to Vulkan0. It also warns that Flash Attention over K/V rows is unsupported on this Vulkan path and assigned to CPU; therefore the run used GPU model-layer inference with that operation falling back to CPU. The server used greedy selection (`temperature=-1`) and raw-logit probabilities (`post_sampling_probs=false`), with top-k/top-p/min-p/typical-p filters disabled. See [`llama_server_stderr.log`](llama_server_stderr.log) for the recorded load and device warnings.

## Artifacts

- [`manifest.json`](manifest.json): exact randomized prompt and mapping manifest, seed and manifest hash.
- [`raw_paired_probability_results.jsonl`](raw_paired_probability_results.jsonl): exact prompts, tokenizations, token log probabilities, full candidate likelihoods, raw completion distributions and greedy text.
- [`summary.json`](summary.json) and [`paired_effects.csv`](paired_effects.csv): paired contrasts and condition summaries.
- [`tokenizer_probe.json`](tokenizer_probe.json), [`model_provenance.json`](model_provenance.json), [`commands.md`](commands.md), [`run_local_pilot.py`](run_local_pilot.py).
- [`focused_tests.log`](focused_tests.log): regression and scorer test results.
- [`comprehension_reaudit_manifest.json`](comprehension_reaudit_manifest.json) and [`run_comprehension_reaudit.py`](run_comprehension_reaudit.py): corrected diagnostic plan and isolated rerun command. The attempt did not complete; see [`comprehension_reaudit_status.json`](comprehension_reaudit_status.json).
