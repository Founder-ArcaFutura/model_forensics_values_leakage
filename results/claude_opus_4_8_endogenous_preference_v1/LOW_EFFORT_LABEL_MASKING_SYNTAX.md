# Low-effort label-masking pilot: API and shell syntax

This is a **separate exploratory scaffold**. It does not change the existing `run_claude_pilot.py`, whose generated estimation manifests use adaptive thinking with high effort and whose full design is much larger. Do not use that runner's `all` command for this pilot. This note documents request syntax; it does **not** generate `pilot_requests.json`. That reviewed request file still has to be built before the submission commands can work.

No inference is launched by these examples. `prepare`/manifest generation is local work; the POST to `/v1/messages/batches` is the first paid model operation.

## Frozen pilot shape

- Model: `claude-opus-4-8`.
- Keep adaptive thinking on and set `output_config.effort` to `low` in every estimation arm.
- Use one predeclared Fermi question (the giraffe-spots item) and one predeclared beneficiary pair. Choose the pair in the protocol before seeing the preference answers; do not choose whichever pair looks strongest afterward.
- 64 counterbalanced preference prompts (16 for each of the existing four pairs); 12 neutral Fermi baselines; 24 estimates in each of three arms (12 above-threshold mapping, 12 reversed mapping): **148 requests total**.
- This is a pilot for format, direction, and observed token usage. With 12 neutral baselines the threshold is their median; it is an exploratory estimate, not the 40-repetition threshold in the earlier protocol. The 24 estimates per arm are split across two mappings, not 24 per mapping.
- Arm A: original generic GOOD/BAD Donation Bet positive control.
- Arm B: named beneficiaries, no evaluative GOOD/BAD language; the independently preferred beneficiary is fixed before estimates.
- Arm C: same named beneficiaries and exact mappings as B, with only the GOOD/BAD outcome labels added. Keep the underlying mappings identical cell by cell so B versus C isolates the labels.

Do not proceed to B/C unless the predeclared pair meets the already specified preference stability rule. Do not silently replace it with another pair. Any additional pair or larger sample is a separately specified extension, not part of this 148-request pilot.

## Anthropic Messages API settings

Use these generation parameters for all estimate requests (neutral, A, B, and C):

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 2048,
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "low"},
  "messages": [
    {"role": "user", "content": "<one frozen prompt from the manifest>"}
  ]
}
```

Preference choices do not need adaptive thinking. Use a small output allowance and omit both `thinking` and `output_config` for those requests:

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 256,
  "messages": [
    {"role": "user", "content": "<one frozen counterbalanced preference prompt>"}
  ]
}
```

Each batch item wraps the corresponding Messages parameters:

```json
{
  "custom_id": "estimate-arm-b-giraffe-above-001",
  "params": {
    "model": "claude-opus-4-8",
    "max_tokens": 2048,
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "low"},
    "messages": [
      {"role": "user", "content": "<frozen prompt>"}
    ]
  }
}
```

The manifest submitted as the batch body is `{"requests": [ ... ]}`. Build and inspect the complete JSON locally first. Preserve every `custom_id`, exact prompt, condition, mapping, threshold, and manifest hash with the results.

## PowerShell request syntax

From the repository root, after the API key has been deliberately refreshed in the process environment (do not paste it into a command, file, or transcript):

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
# Set securely in this PowerShell process, without echoing the value:
$env:ANTHROPIC_API_KEY = Read-Host 'Anthropic API key' -AsSecureString | `
  ForEach-Object { [System.Net.NetworkCredential]::new('', $_).Password }

$headers = @{
  'x-api-key' = $env:ANTHROPIC_API_KEY
  'anthropic-version' = '2023-06-01'
  'content-type' = 'application/json'
}

# First check that the locally generated request file exists. This note does
# not generate it. Do not continue until it contains the reviewed request body.
$requestPath = '.\pilot_requests.json'
if (-not (Test-Path -LiteralPath $requestPath -PathType Leaf)) {
  throw "Missing $requestPath. The request manifest has not been generated."
}
$body = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $requestPath))
$parsedBody = $body | ConvertFrom-Json
if (-not $parsedBody.requests -or $parsedBody.requests.Count -ne 148) {
  throw 'Expected a reviewed JSON body with exactly 148 requests.'
}

# First paid action: submits the prepared batch. The raw JSON string is passed
# through unchanged, avoiding PowerShell-version-specific hashtable options.
$batch = Invoke-RestMethod -Method Post `
  -Uri 'https://api.anthropic.com/v1/messages/batches' `
  -Headers $headers -ContentType 'application/json' -Body $body
if (-not $batch.id) { throw 'No batch ID returned; do not poll.' }
$batch.id

# Later: poll status using the returned ID, then fetch results when ended.
$batchId = $batch.id
Invoke-RestMethod -Method Get `
  -Uri "https://api.anthropic.com/v1/messages/batches/$batchId" `
  -Headers $headers
Invoke-WebRequest -Method Get `
  -Uri "https://api.anthropic.com/v1/messages/batches/$batchId/results" `
  -Headers $headers | Select-Object -ExpandProperty Content
```

For a single non-batch format smoke call, the endpoint is `POST https://api.anthropic.com/v1/messages` and the body is one Messages request above. A smoke response is not part of the registered sample; record it separately. The old runner's `smoke` phase submits a request and therefore is not a dry run.

## Stop-and-review points

1. Generate the 148 request objects locally and inspect the 64 preference prompts and the complete A/B/C wording before submission.
2. Run only the preference batch first. Freeze the preregistered pair decision and preference results.
3. Generate the neutral and estimate batches only if that pair passed the frozen rule. Confirm that B and C prompts differ only by the GOOD/BAD labels.
4. Record returned model IDs, stop reasons, input/output token usage, API receipts, and actual billed batch cost. The prior cost range was an estimate, not a guarantee.

There is no hard spending cap encoded in this scaffold. A 148-request sample size is an experimental allocation; it is not an API billing ceiling.

## Official API references

- [Anthropic adaptive thinking and effort settings](https://docs.anthropic.com/en/docs/build-with-claude/adaptive-thinking)
- [Anthropic Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches)
