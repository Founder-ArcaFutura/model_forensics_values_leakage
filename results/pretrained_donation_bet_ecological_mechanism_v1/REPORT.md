# Ecological-assumption follow-up: Qwen2.5-14B-Instruct

## Result

The frozen Qwen2.5-14B-Instruct preference for habitat protection over veterinary care reappeared in the exact baseline layouts (13/16; mean outcome log-likelihood margin +5.867 nats). The order effect remained substantial: habitat was favored in 8/8 layouts when listed first and 5/8 when listed second; order-stratified mean margins were +10.726 and +1.007 nats respectively.

Stipulated consequences moved the same matched prompt layouts in the predicted directions, and the effects survived order, wording, and answer-symbol controls:

| Scenario | Habitat favored | Mean logP(H)−logP(V), nats | Mean paired change vs baseline | Direction over 16 matched layouts | Stable under unchanged rule? |
|---|---:|---:|---:|---:|---|
| Original unqualified descriptions | 13/16 | +5.867 | — | 8/8 H-first; 5/8 H-second | Yes |
| Habitat already secure; modest marginal effect | 0/16 | −21.294 | −27.160; 16/16 negative | 8/8 veterinary in each order | Yes |
| Veterinary care prevents local-population extinction | 0/16 | −19.627 | −25.494; 16/16 negative | 8/8 veterinary in each order | Yes |
| Habitat protection prevents ecosystem collapse | 16/16 | +21.660 | +15.794; 16/16 positive | 8/8 habitat in each order | Yes |

Each condition used the exact same 16 source layouts: two wording variants, AB/XY answer symbols, both presentation orders, and both symbol-to-beneficiary mappings. Scenario comparisons are paired by source layout. Candidate continuations were exact single-token symbols; candidate token-length normalization therefore was unnecessary. Scores are full unfiltered next-token logprobs, and conditional candidate probabilities are retained per prompt. The large logprob margins should be read within each prompt’s two-answer set, not compared as calibrated probabilities across differently worded prompts.

All 12 separate premise-comprehension items were correct by candidate argmax (4/4 for each changed scenario). The baseline scoring result therefore replicates the earlier chat-template observation, while the three consequence manipulations show strong sensitivity to the stipulated outcome descriptions. No p-values are reported: the 16 layouts are counterbalances of one pair, not 16 independent beneficiary pairs.

## Generated explanations

The first five explanation outputs were too verbose for the 180-token allowance, and repeating them with 420 tokens still hit the output limit. Both original attempts are preserved separately. A concise prompt adaptation then produced five complete EOS-terminated responses. The two baseline responses differed: one suggested splitting the donation and one recommended habitat protection. Under the changed assumptions, the concise rationale recommended veterinary care when habitat was secure, veterinary care when it prevented extinction, and habitat protection when it prevented ecosystem collapse.

Those are generated explanations and recommendations, not privileged access to the cause of the candidate likelihoods. The short-response adaptation was created after the preference scores and is qualitative only. It shows that this checkpoint can articulate the relevant distinction under these prompts; it does not validate the baseline ranking as an independent moral value or establish that the explanation caused the earlier probabilities.

## Model and execution

- Checkpoint: `Qwen/Qwen2.5-14B-Instruct`, Ollama tag `qwen2.5:14b-instruct-q4_K_M`, Q4_K_M; local GGUF SHA-256 `2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54`.
- Backend: llama.cpp `llama-server` b12606, local Vulkan device `Vulkan0`, context 2048, `--fit on`.
- Hardware: AMD Radeon RX 9060 XT, 16,304 MiB reported VRAM. The server loaded the model and processed every probability request. Its logs warn that Flash Attention over KV rows used CPU fallback; no full layer-offload count is present in the log.
- Chat template: verified against llama.cpp `/apply-template`; exact rendering and all scoring prompts are saved.
- Sample: 64 preference likelihood records, 12 comprehension likelihood records, five initial explanation generations, five longer explanation attempts, and five brief-format adapted explanations. No paid inference was used.

## Interpretation and limits

This follow-up supports a narrow behavioral claim: in this Qwen Instruct checkpoint and choice instrument, the relative likelihood of the two answer continuations tracks the stated marginal-benefit and ecological-scope assumptions, beyond the measured position, symbol, and wording effects. The baseline preference itself retains a strong first-listed-option advantage. The conditions are clear stipulated vignettes and could also evoke learned language associations; they do not identify a latent causal mechanism or establish intrinsic or conscious values. Only one beneficiary pair was tested here, and the matched layouts are not independent samples.

The result makes the ecological-scope explanation worth testing with more pairs and independently varied assumptions. It does not establish that the Qwen baseline preference generalizes or that the same account explains the earlier Donation Bet effects on other models.

## Reproduction and audit

Run from the repository root on the machine with the cached Q4_K_M model and llama.cpp Vulkan build:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --prepare-only
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --execute
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --analyze-only
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --extend-explanations
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --brief-explanations
```

The preregistration, exact randomized manifest, preference freeze, raw logprobs, all three explanation elicitation records, comprehension results, server logs, and provenance are retained in this directory. The frozen source Base and Instruct two-goods results were not modified.
