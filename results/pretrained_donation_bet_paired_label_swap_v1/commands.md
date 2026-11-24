# Reproduce the paired label-swap pilot

Run from the repository root on branch `pretrained-donation-bet-pilot`. The model file and llama.cpp binary are local, untracked artifacts; see `model_provenance.json` for hashes.

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python results/pretrained_donation_bet_paired_label_swap_v1/run_local_pilot.py
```

`run_local_pilot.py` starts the local llama.cpp b12606 Vulkan server against the Qwen2.5-14B Q4_K_M GGUF, waits for its local health endpoint, runs the probability-scoring module, saves logs and raw outputs, then stops the server. The model path and server executable are set near the top of that script and should be updated if the local files are moved.

To score against an already-running compatible server on `127.0.0.1:8081`:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python -m value_leakage_experiment.paired_label_swap `
  --model-path 'C:\Users\Main PC 2\AppData\Local\Temp\Qwen2.5-14B.Q4_K_M.gguf' `
  --model-name Qwen2.5-14B-Base-Q4_K_M `
  --checkpoint 'Qwen/Qwen2.5-14B@97e1e76335b7017d8f67c08a19d103c0504298c9' `
  --quantization Q4_K_M `
  --output results/pretrained_donation_bet_paired_label_swap_v1 `
  --seed 20261125
```

Focused regression/scorer tests:

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests/test_paired_label_swap.py tests/test_native_probability_scoring.py -q
```

The corrected, isolated label-mapping diagnostic can be run without overwriting
the frozen 56-record probability pilot:

```powershell
$env:SSLKEYLOGFILE = ''
python results/pretrained_donation_bet_paired_label_swap_v1/run_comprehension_reaudit.py
```

The first attempted re-audit in this revision loaded the model but produced no
scored rows; see `comprehension_reaudit_status.json`. Its output is not used in
the report's behavioral findings.
