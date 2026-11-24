# Qwen ecological probe reproduction

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -p no:cacheprovider --basetemp results/pretrained_donation_bet_ecological_mechanism_v1/pytest_tmp tests/test_ecological_preference_probe.py -q
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --prepare-only
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --execute
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --analyze-only
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --extend-explanations
python results/pretrained_donation_bet_ecological_mechanism_v1/run_ecological_probe.py --brief-explanations
```

`manifest.json` and `preference_freeze.json` hash the preregistered conditions before inference. Raw candidate scores and complete prompts are in `preference_raw.jsonl`; premise diagnostics are in `comprehension_raw.jsonl`. The initial and extended explanation generations are separate from the later concise-format adaptation. The runner expects the local cached model blob recorded in `model_provenance.json` and the Vulkan llama.cpp executable path in the script.
