# Claude Opus 4.8 endogenous-preference Donation Bet: execution status

## Status

The experiment is designed and its preference and neutral-baseline manifests are frozen, but **Claude inference did not run**. The single preference-format smoke request reached Anthropic and was rejected with `authentication_error: invalid x-api-key` before a model response. No preference batch, neutral baseline batch, or intervention batch was submitted. There are no successful token usage records. The exact error is saved in `auth_blocker.json`; no credential value is stored.

The local preference, baseline, and intervention runners are ready to resume after the credential is refreshed or an AWS Bedrock route is explicitly selected and configured. The current runner targets Anthropic's Messages Batch API; no Bedrock call or paid inference was made.

## Frozen design

The independent-preference stage contains 64 prompts: all four preselected beneficial-outcome pairs, with 16 unique layouts each. Each pair counterbalances two phrasings, AB/XY symbols, presentation order, and symbol-to-outcome assignment. A preference passes only if at least 12/16 layouts choose the same outcome and all wording, order, and symbol subgroup majorities agree. Every pair meeting this rule will be carried forward; the procedure will not select the pair with the largest effect.

After ranking freeze, the design has 40 neutral baseline estimates for each of four factual questions (giraffe spots, rice grains, adult scalp hairs, and Canadian public libraries). Each question's threshold is its own median valid neutral estimate, with a minimum of 36/40 valid baselines. If a ranking is stable and the thresholds are eligible, the original generic GOOD/BAD control has 40 `above_good` and 40 `below_good` responses per question. Each stable unlabeled-beneficiary pair receives the same 40-by-40 reversed mapping on all four questions. Thus the number of intervention requests is determined mechanically by the number of stable preference pairs.

The preference rule, counts, prompt wording, parse behavior, and thresholds are specified in `protocol.json`, `preference_manifest.json`, and `neutral_baseline_manifest.json`. The preference output is frozen before any numerical stage. Full returned model identifiers, prompt/response pairs, estimates, parsing outcomes, token usage, and batch receipts will be saved if the run resumes.

## Cost estimate and why it is uncertain

The plan contains `544 + 320 × K` total requests, where `K` is the number of independently stable pairs (0–4). That includes 64 preference prompts, 160 neutral baselines, 320 explicit-label controls, and 320 two-goods requests per stable pair. The recent prior Opus positive control used 23,840 input and 115,981 output tokens for 120 calls and recorded a batch estimate of $1.5094. Scaling that observed average gives a rough estimate of about **$6.84 + $4.03 × K**: about $10.87 if one pair is stable, up to $22.95 if all four pass.

This is a historical-usage projection, not a quote or a spending limit. It may overstate the new run because preference responses use a small output setting, and it may understate it if new responses use more extended-thinking tokens. No spend occurred in the rejected smoke request; it returned no usage record. Anthropic's [Opus 4.8 batch pricing list](https://www-cdn.anthropic.com/files/4zrzovbb/website/3684c2faafb97418665782cea0001f439f74b1d2.pdf) lists $2.50 per million input tokens and $12.50 per million output tokens for global batch processing, effective 2026-05-27; actual current billing and account credits should be checked at execution time.

## Interpretation boundary

There are no Claude findings from this run. Earlier Claude Opus 4.8 Donation Bet results remain the reference positive control, not a result from this new comparison. The experiment must not be interpreted until the independently elicited rankings and both estimate conditions have actually completed under a verified model route.
