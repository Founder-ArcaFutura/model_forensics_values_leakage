# Claude Opus 4.8 low-effort label-masking pilot

The runner is `src/value_leakage_experiment/low_effort_claude_pilot.py`. It never submits a request during preparation or validation. Each paid submission requires its own explicit `submit-*` command. Do not invoke the earlier `run_claude_pilot.py` for this pilot; that is the separate 40-per-cell, high-effort design.

## Recovering the completed preference batch

The batch `msgbatch_01UiaYn7gu5wDhpNo7stv7yq` is already ended. The local `provider_results.jsonl` created by the earlier retrieval path fails JSON parsing because it contains four unescaped `0x1A` bytes. Its error at line 1, column 821 points to byte offset 820. The root `claude_preferences_results.jsonl` was validated locally as 64 LF-delimited UTF-8 JSON records, with exact manifest IDs and 64 succeeded results. It is the selected recovery source; both files are preserved and checksummed.

The earlier PowerShell path converted `.Content` to a PowerShell string, wrote it through console text output, then captured it with Python text decoding. The resulting local artifact contains invalid control bytes. Response IDs, model, usage, and stop-reason metadata match the manual file in all 64 rows after diagnostic escaping, while 42 complete message objects differ; the true raw HTTP entity bytes were not captured, so the manual file is retained as the validated source and the transport artifact is not overwritten.

For this batch, recover and freeze using local files only:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot poll-preferences
python -m value_leakage_experiment.low_effort_claude_pilot recover-manual-preferences
python -m value_leakage_experiment.low_effort_claude_pilot freeze-preferences
```

The poll command returns the saved ended status locally. `recover-manual-preferences` validates `claude_preferences_results.jsonl`, verifies all 64 `custom_id` mappings against the frozen request manifest, preserves the original file bytes and usage metadata, and makes no API calls. `transport_diagnostic.json` records the byte comparison and parser evidence. This batch's predeclared habitat/veterinary pair had zero strict symbol choices among 16 layouts, so the freeze marks it unstable; do not prepare a baseline or interventions from this preference result.

The JSONL parser now splits on LF only. Python's broader `str.splitlines()` behavior could also split legal U+0085/U+2028/U+2029 characters inside JSON strings; that latent issue was not the cause of this file's observed error, since both artifacts have 64 LF-delimited rows. Future result downloads use a byte-returning PowerShell stream path instead of converting response bodies through console text encoding.

## Preregistered pilot design

- Model: `claude-opus-4-8`.
- Numerical estimates all use adaptive thinking with `output_config.effort = low` and `max_tokens = 2048`. Preference choices use 256 output tokens and no extended thinking.
- Stage 1: 64 balanced preference prompts, 16 for each of four frozen pairs. The preregistered primary pair is `habitat_vs_veterinary_care`; it is not selected after observing Claude's choices. Preference-conditioned arms are prohibited unless this pair passes the existing 12-of-16 plus subgroup-majority stability rule.
- Stage 2: 12 neutral giraffe-spots estimates. The threshold is their median; all 12 must be complete numeric estimates to freeze it.
- Stage 3: 72 estimates: three arms, two mappings per arm, 12 requests in each arm-by-mapping cell. Arm A is the generic GOOD/BAD positive control. Arm B uses the same named beneficiaries without GOOD/BAD language. Arm C uses the exact same beneficiaries and mappings as B with only the GOOD/BAD outcome labels added.

These are pilot sample sizes, not a spending cap. The runner records input/output token usage from every returned message, plus provider, prompt, manifest, and normalized-response checksums. It does not set a hard spending limit.

## Failed preference instrument and revised format probe

The completed first 64-request preference batch is **not interpretable preference evidence**. In its preregistered primary habitat-versus-veterinary pair, all 16 answers reached `max_tokens` at 256 output tokens; the completions spent those tokens explaining why the forced one-symbol answer was underdetermined. Across all 64 requests, only one response was a strict symbol choice. The saved response review and freeze therefore mark this instrument unstable. Do not rerun or reinterpret those responses as evidence that Claude has no preference.

The comparison of the two saved result copies is now classified. All 64 custom IDs, response IDs, returned models, stop reasons, and usage fields agree after diagnostic-only escaping of the four raw SUB bytes. The 42 differing message objects differ in text: the PowerShell-transcoded copy changed 74 U+2014 em dashes to ASCII hyphens and changed four U+2192 right arrows to unescaped U+001A SUB bytes. Those SUB bytes caused the invalid JSON at byte 820 / line 1, column 821. The original raw HTTP entity bytes were not saved, so the manual UTF-8 JSONL remains the recovery artifact; the converted artifact remains preserved as transport evidence.

The revised instrument uses Anthropic structured JSON output with a per-prompt schema whose `choice` property is an enum containing only the symbols mapped in that prompt. This is an explicit **forced conditional choice between two options**, not spontaneous preference, and does not measure the model's willingness to reject the choice. The schema can guarantee the response shape when the request completes, but it cannot make an underspecified preference meaningful. The four-request format probe uses the predeclared primary pair and balances wording, symbol scheme, and presentation order. Review its raw completions, stop reasons, and token usage before deciding whether to freeze the instrument. The relevant provider format is documented in Anthropic's [Structured outputs guide](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

The probe runner is local-only until `submit-probe`; manifest generation and validation never call the API. Run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = 'NUL'
python -m value_leakage_experiment.claude_preference_format_probe prepare-probe
python -m value_leakage_experiment.claude_preference_format_probe validate-probe
```

Inspect `results/claude_opus_4_8_endogenous_preference_v1/low_effort_pilot/structured_choice_preference_v1/format_probe/manifest.json` and `requests.json`. Only this explicit command submits the four paid probe requests:

```powershell
python -m value_leakage_experiment.claude_preference_format_probe submit-probe
```

Save the returned batch ID in the local state automatically written by the runner. Resume with these separate commands; repeat polling only while the provider status is not `ended`:

```powershell
python -m value_leakage_experiment.claude_preference_format_probe poll-probe
python -m value_leakage_experiment.claude_preference_format_probe retrieve-probe
python -m value_leakage_experiment.claude_preference_format_probe freeze-format
```

The raw provider JSONL, normalized response records, usage receipt, and format freeze are retained in the probe directory and its parent. Do not prepare or submit the 64-request preference stage until the four probe results have been inspected and the format freeze says `format_passed: true`. A successful probe validates only the constrained format and response budget, not the stability of the underlying preference. If approved after reviewing the probe, generate and validate the 64-request stage locally, then submit it only with the separate explicit command:

```powershell
python -m value_leakage_experiment.claude_preference_format_probe prepare-preferences
python -m value_leakage_experiment.claude_preference_format_probe validate-preferences
python -m value_leakage_experiment.claude_preference_format_probe submit-preferences
python -m value_leakage_experiment.claude_preference_format_probe poll-preferences
python -m value_leakage_experiment.claude_preference_format_probe retrieve-preferences
python -m value_leakage_experiment.claude_preference_format_probe freeze-preferences
```

The revised 64-request stage uses the same predeclared pairs and stability rule. Preference-conditioned Fermi arms remain blocked unless the predeclared habitat-versus-veterinary pair passes that rule. The Fermi baseline and interventions are separate stages and are not launched by this runner.

## PowerShell workflow

Run from the repository root. Set `PYTHONPATH` once; no API key is needed for preparation or validation. The API helper reads `ANTHROPIC_API_KEY` from the process environment or the repository's `.env.local`/`.env` when an explicit submission or retrieval command runs.

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = 'NUL'
```

### Stage 1: preference choice

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot prepare-preferences
python -m value_leakage_experiment.low_effort_claude_pilot validate-preferences
```

Inspect `results/claude_opus_4_8_endogenous_preference_v1/low_effort_pilot/stage1_preferences/manifest.json` and `requests.json`. Preparation and validation are local only. Submit only when you explicitly choose to start inference:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot submit-preferences
```

Poll once per command until the returned status is `ended`; polling does not submit another batch. Then retrieve and freeze the decision:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot poll-preferences
python -m value_leakage_experiment.low_effort_claude_pilot retrieve-preferences
python -m value_leakage_experiment.low_effort_claude_pilot freeze-preferences
```

Repeat only `poll-preferences` while the provider reports the batch is processing. The preference summary and frozen primary-pair decision are saved at the pilot output root.

### Stage 2: neutral numerical baseline

This stage requires a frozen stage 1 decision. A failed preference decision still permits this neutral calibration stage, but it will block stage 3.

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot prepare-baseline
python -m value_leakage_experiment.low_effort_claude_pilot validate-baseline
```

Inspect the 12 prompts and low-effort request parameters in `stage2_baseline/manifest.json` and `requests.json`. Submit explicitly, poll until ended, retrieve, and freeze the median threshold:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot submit-baseline
python -m value_leakage_experiment.low_effort_claude_pilot poll-baseline
python -m value_leakage_experiment.low_effort_claude_pilot retrieve-baseline
python -m value_leakage_experiment.low_effort_claude_pilot freeze-threshold
```

Repeat only `poll-baseline` as needed. `freeze-threshold` refuses to proceed unless all 12 responses are complete and parse as numeric estimates.

### Stage 3: interventions

Preparation fails closed if the predeclared pair was unstable, the threshold was not frozen, or either freeze no longer matches the manifest chain. Inspect the 72 prompts and confirm B/C differ only by the GOOD/BAD labels before validating and explicitly submitting:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot prepare-interventions
python -m value_leakage_experiment.low_effort_claude_pilot validate-interventions
python -m value_leakage_experiment.low_effort_claude_pilot submit-interventions
python -m value_leakage_experiment.low_effort_claude_pilot poll-interventions
python -m value_leakage_experiment.low_effort_claude_pilot retrieve-interventions
python -m value_leakage_experiment.low_effort_claude_pilot analyze
```

Repeat only `poll-interventions` until `ended`. Analysis outputs `analysis.json` and `analysis.md`; raw provider JSONL and normalized prompt/response records remain in the stage directory.

## Resume and duplicate-submission protection

Each stage records its manifest, exact batch request body, batch ID, status, raw provider result JSONL, normalized raw records, receipt, hashes, and token totals. Poll and retrieval are separate so they can be resumed independently. Retrieval reuses saved results once complete.

The runner persists a `submitting` state before the POST. If the connection fails before the ID is saved, it marks the state `submission_unknown` and refuses to submit again. Reconcile against Anthropic's batch list/console, then attach the real ID without resubmitting:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot adopt-preferences msgbatch_...
# or adopt-baseline / adopt-interventions for the relevant stage
```

Then resume with that stage's `poll-*` command. Never substitute a placeholder ID. A known submitted batch also cannot be submitted twice by this runner.

## API parameter reference

Numerical request parameters are generated per request as:

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 2048,
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "low"},
  "messages": [{"role": "user", "content": "<frozen prompt>"}]
}
```

The batch API accepts `{"requests": [{"custom_id": "...", "params": { ... }}]}`. The runner sends this body only from `submit-*`. See Anthropic's [effort settings](https://platform.claude.com/docs/en/build-with-claude/effort) and [Create a Message Batch reference](https://platform.claude.com/docs/en/api/messages/batches/create).
