import json

from value_leakage_experiment.opaque_clause_control import create_opaque_manifest, opaque_prompt


def test_prompt_counterbalances_mapping_and_order():
    above_first = opaque_prompt(100, True, True, "ZX-41", "QM-72")
    below_first = opaque_prompt(100, True, False, "ZX-41", "QM-72")
    assert "exceeds the threshold, outcome identifier ZX-41" in above_first
    assert "does not exceed the threshold, outcome identifier QM-72" in above_first
    assert above_first.index("exceeds the threshold") < above_first.index("does not exceed the threshold")
    assert below_first.index("does not exceed the threshold") < below_first.index("exceeds the threshold")
    assert "arbitrary and carry no stated evaluation or meaning" in above_first


def test_manifest_has_balanced_independent_cells(tmp_path):
    config = {
        "protocol_version": "opaque-test", "threshold": 100,
        "threshold_provenance": {"method": "test"}, "opaque_identifiers": ["ZX-41", "QM-72"],
        "trials_per_cell": 3, "seed": 7,
    }
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = create_opaque_manifest(config_path, output_path)
    assert len(manifest["trials"]) == 12
    assert set(manifest["cell_counts"].values()) == {3}
    assert len({trial["prompt_sha256"] for trial in manifest["trials"]}) == 4
    assert max(len(trial["custom_id"]) for trial in manifest["trials"]) <= 64
