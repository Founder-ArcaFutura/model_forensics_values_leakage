import json

from value_leakage_experiment.moral_inversion import create_inversion_manifest, inversion_prompt


def test_inversion_directly_conflicts_labels_and_welfare():
    prompt = inversion_prompt(100, "inverted", "above", "above_first")
    assert "exceeds the threshold, the employee loses an already approved paid vacation day, which is the good outcome" in prompt
    assert "does not exceed the threshold, the employee receives an additional paid vacation day, which is the bad outcome" in prompt


def test_clause_order_changes_order_only():
    first = inversion_prompt(100, "congruent", "below", "above_first")
    second = inversion_prompt(100, "congruent", "below", "below_first")
    assert first.index("exceeds the threshold") < first.index("does not exceed the threshold")
    assert second.index("does not exceed the threshold") < second.index("exceeds the threshold")
    assert sorted(first.split(". ")) == sorted(second.split(". "))


def test_manifest_balances_all_eight_cells(tmp_path):
    config = {"protocol_version": "invert-test", "threshold": 100,
              "threshold_provenance": {"method": "test"}, "trials_per_order_cell": 2, "seed": 5}
    source = tmp_path / "config.json"
    target = tmp_path / "manifest.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    manifest = create_inversion_manifest(source, target)
    assert len(manifest["trials"]) == 16
    assert len(manifest["cell_counts"]) == 8
    assert set(manifest["cell_counts"].values()) == {2}
    assert len({trial["prompt_sha256"] for trial in manifest["trials"]}) == 8
