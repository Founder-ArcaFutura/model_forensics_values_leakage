# Continuation from the structured preference v2 freeze

This is a separate workspace for the neutral giraffe-spots baseline and its later intervention stage. It does not replace or modify `low_effort_pilot/preference_freeze.json` from the failed free-text experiment, nor does it modify the structured v2 source artifacts.

## Validated preference handoff

The handoff verifies the structured v2 freeze's own checksum and its links to the format freeze, preference manifest freeze, manifest, request body, provider receipt, batch state/status, provider JSONL, normalized responses, analysis, and summary. It also checks all 64 custom IDs and response-to-beneficiary mappings. The source's original frozen JSON is copied byte-for-byte to `preference_freeze_structured_source.json`; `preference_handoff.json` records the source hashes and validation outcome.

- Batch: `msgbatch_01V2jrFeFLFMj98LzwgmvDZr`
- Primary pair: `habitat_vs_veterinary_care`
- Result: 13/16 valid forced choices favor habitat; preregistered stability passed.
- Measurement meaning: a forced comparative choice between the two displayed beneficiaries, not spontaneous preference.
- Compatibility freeze used by the existing runner: `preference_freeze.json`.

The structured result remains at `../structured_choice_preference_v2/`, including the original provider JSONL and normalized responses. The compatibility freeze is separately hashed and ties back to the exact source freeze and its provenance.

## Neutral baseline

The 12-request `stage2_baseline` manifest has been generated and validated locally. Every request uses `claude-opus-4-8`, adaptive thinking, `output_config.effort = low`, and `max_tokens = 2048`. This is the numerical-estimate configuration; it is explicit in `stage2_baseline/manifest.json` and `requests.json`.

From the repository root, the following commands operate only in this continuation workspace:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = 'NUL'
$continuation = '.\results\claude_opus_4_8_endogenous_preference_v1\low_effort_pilot\structured_preference_v2_continuation'

# Local validation only.
python -m value_leakage_experiment.claude_preference_handoff validate-handoff --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot validate-baseline --out-dir $continuation

# Explicit paid-inference action. This command submits only the 12 baseline requests.
python -m value_leakage_experiment.low_effort_claude_pilot submit-baseline --out-dir $continuation

# Repeat polling while processing_status is not "ended".
python -m value_leakage_experiment.low_effort_claude_pilot poll-baseline --out-dir $continuation

# Retrieve provider results and freeze the median threshold after all 12 estimates validate.
python -m value_leakage_experiment.low_effort_claude_pilot retrieve-baseline --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot freeze-threshold --out-dir $continuation
```

The runner persists batch IDs and response artifacts, records token usage and hashes, and refuses duplicate or uncertain submissions. `freeze-threshold` requires all 12 responses to be complete numeric estimates. No spending cap is built into the runner.

## Interventions after reviewing the baseline

After the baseline threshold is frozen and reviewed, local generation creates 72 requests: 24 per arm, with 12 each for the preferred beneficiary mapped above and below the threshold. Arm A is the generic GOOD/BAD positive control. Arm B uses named habitat and veterinary beneficiaries without evaluative labels. Arm C preserves exactly the same recipient-to-threshold mappings as B and adds only GOOD/BAD labels tied to the frozen preferred and other beneficiaries.

The intervention preparation command verifies the structured preference handoff, threshold-freeze hash chain, and stable primary pair before it can generate requests. To prepare and validate locally after the baseline freeze:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot prepare-interventions --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot validate-interventions --out-dir $continuation
```

Only submit interventions with the separate explicit command after inspecting the frozen preference and baseline results:

```powershell
python -m value_leakage_experiment.low_effort_claude_pilot submit-interventions --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot poll-interventions --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot retrieve-interventions --out-dir $continuation
python -m value_leakage_experiment.low_effort_claude_pilot analyze --out-dir $continuation
```

No baseline or intervention inference was submitted while creating this handoff. Generating manifests is local-only; only the explicit `submit-*` commands make paid provider requests.
