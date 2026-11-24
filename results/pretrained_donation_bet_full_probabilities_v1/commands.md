# Reproduction commands

Commands were run in PowerShell from the repository root on branch `pretrained-donation-bet-pilot`. The quantized files stay outside Git. `model_provenance.json` records their SHA-256 hashes and the pinned source locations.

## Start the native GPU scorer

The tested llama.cpp b12606 Windows Vulkan archive is extracted under `%TEMP%\llama-b12606-vulkan`. Keep the server in a foreground PowerShell window while running the scoring command in another one. The 14B quant file is at `%TEMP%\Qwen2.5-14B.Q4_K_M.gguf`.

```powershell
$taskServer = "$env:TEMP\llama-b12606-vulkan\llama-server.exe"
$taskModel = "$env:TEMP\Qwen2.5-14B.Q4_K_M.gguf"
& $taskServer -m $taskModel --host 127.0.0.1 --port 8081 --ctx-size 4096 --n-gpu-layers 99 --device Vulkan0
```

For the retained 3B GGUF, stop the first server and start the same executable with this model path instead:

```powershell
$taskModel = "$env:USERPROFILE\.ollama\models\blobs\sha256-41ba01c0a947829a11b3f2d843d493ff852ddd6fb64571210213e2a277532748"
& $taskServer -m $taskModel --host 127.0.0.1 --port 8081 --ctx-size 4096 --n-gpu-layers 99 --device Vulkan0
```

The server must report `Vulkan0: AMD Radeon RX 9060 XT`. The ROCm llama.cpp binary tested in this run reported no devices; do not remove `--device Vulkan0` or `--n-gpu-layers 99` and treat a CPU run as comparable.

## Run the matched experiments

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''

python -m value_leakage_experiment.native_probability_scoring `
  --model-path "$env:USERPROFILE\.ollama\models\blobs\sha256-41ba01c0a947829a11b3f2d843d493ff852ddd6fb64571210213e2a277532748" `
  --model-name Qwen2.5-3B-Base-Q5_K_M `
  --checkpoint Qwen/Qwen2.5-3B@3aab1f1954e9cc14eb9509a215f9e5ca08227a9b `
  --quantization Q5_K_M `
  --output results/pretrained_donation_bet_full_probabilities_v1/qwen25_3b `
  --seed 20261124
```

```powershell
python -m value_leakage_experiment.native_probability_scoring `
  --model-path "$env:TEMP\Qwen2.5-14B.Q4_K_M.gguf" `
  --model-name Qwen2.5-14B-Base-Q4_K_M `
  --checkpoint 'Qwen/Qwen2.5-14B@97e1e76335b7017d8f67c08a19d103c0504298c9 (official base repo revision checked; quant card does not pin its upstream weight revision)' `
  --quantization Q4_K_M `
  --output results/pretrained_donation_bet_full_probabilities_v1/qwen25_14b `
  --seed 20261124
```

Analyze the raw files:

```powershell
python -m value_leakage_experiment.analyze_native_probability `
  --root results/pretrained_donation_bet_full_probabilities_v1
```

## Re-download the 14B quant if needed

This downloads the same Q4_K_M file from the pinned GGUF repository revision; verify the hash against `model_provenance.json` before importing it.

```powershell
$taskUrl = 'https://huggingface.co/mradermacher/Qwen2.5-14B-GGUF/resolve/d0a2788fa6dc8e83ce2f0aae07882eeecb06ef1f/Qwen2.5-14B.Q4_K_M.gguf'
curl.exe -L --fail --continue-at - --output "$env:TEMP\Qwen2.5-14B.Q4_K_M.gguf" $taskUrl
Get-FileHash "$env:TEMP\Qwen2.5-14B.Q4_K_M.gguf" -Algorithm SHA256
```

## Focused tests

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python -m pytest `
  tests/test_native_probability_scoring.py `
  tests/test_pretrained_probability_probe.py `
  tests/test_pretrained_choice_probe.py `
  tests/test_pretrained_pilot.py -q
```
