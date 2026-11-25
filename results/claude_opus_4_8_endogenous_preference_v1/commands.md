# Claude Opus 4.8 pilot commands

The current runner uses the direct Anthropic Messages API. Do not run it with the rejected credential. After refreshing the key, run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py prepare
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py smoke
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py preferences
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py baseline
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py interventions
python results/claude_opus_4_8_endogenous_preference_v1/run_claude_pilot.py analyze
```

Or use `run_claude_pilot.py all` to perform those phases in order. The phase runner persists request payloads, batch IDs, statuses, provider JSONL, normalized raw results, and receipts so completed stages are reused on resume. The current implementation does not route through Bedrock; a Bedrock adapter requires explicit route configuration before use.
