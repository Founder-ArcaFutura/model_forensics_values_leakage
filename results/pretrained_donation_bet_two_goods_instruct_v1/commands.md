# Reproduction commands and artifact map

Run from the repository root in PowerShell. The local Windows paths in the runner refer to the tested model blob and llama.cpp build on the execution machine.

```powershell
$env:PYTHONPATH = 'src'
$env:SSLKEYLOGFILE = ''
python results/pretrained_donation_bet_two_goods_instruct_v1/run_instruct_pilot.py --smoke
python results/pretrained_donation_bet_two_goods_instruct_v1/run_instruct_pilot.py
python results/pretrained_donation_bet_two_goods_instruct_v1/run_instruct_estimation.py --smoke
python results/pretrained_donation_bet_two_goods_instruct_v1/run_instruct_estimation.py
python results/pretrained_donation_bet_two_goods_instruct_v1/score_instruct_estimation_probabilities.py
python results/pretrained_donation_bet_two_goods_instruct_v1/run_instruct_label_swap.py
python results/pretrained_donation_bet_two_goods_instruct_v1/analyze_instruct_estimation.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -p no:cacheprovider --basetemp results/pretrained_donation_bet_two_goods_instruct_v1/pytest_tmp tests/test_two_goods_exact_expansion.py tests/test_two_goods_instruct_runner.py tests/test_two_goods_pilot.py tests/test_native_probability_scoring.py tests/test_paired_label_swap.py -q
```

The preference run manifests are `qwen_chat_template/manifest.json` and `raw_completion_diagnostic/manifest.json`; raw scores and exact prompts are in each condition's `raw.jsonl`. The frozen primary preference decision is `preference_freeze.json`. Numerical outputs and score records are `estimation_manifest.json`, `neutral_baseline_raw.jsonl`, `donation_estimation_raw.jsonl`, `numeric_candidate_likelihood_raw.jsonl`, and `mapping_comprehension_raw.jsonl`. Explicit label-swap material is under `explicit_label_swap_chat_template/`.

Key validation files: `chat_template_validation.json`, `numeric_score_instrumentation_audit.json`, `estimation_summary.json`, `summary.json`, and `model_provenance.json`. `artifact_checksums.json` hashes the Instruct evidence set. The Base data remains in `results/pretrained_donation_bet_two_goods_v1/`; its frozen checksums are in that directory's `frozen_artifact_checksums.json`.
