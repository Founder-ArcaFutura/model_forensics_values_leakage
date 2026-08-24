import unittest
from types import SimpleNamespace

from value_leakage_experiment.analysis import (
    _directional_contrast,
    _fisher_two_sided,
    _newcombe_difference_interval,
    _wilson_interval,
)


class AnalysisTests(unittest.TestCase):
    def test_fisher_known_table(self):
        # A strongly separated 2x2 table.
        self.assertAlmostEqual(_fisher_two_sided(1, 9, 11, 3), 0.002759456, places=6)

    def test_fisher_symmetric_null(self):
        self.assertEqual(_fisher_two_sided(5, 5, 5, 5), 1.0)

    def test_wilson_does_not_collapse_at_zero(self):
        low, high = _wilson_interval(0, 5)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.4)

    def test_newcombe_zero_vs_zero_preserves_uncertainty(self):
        low, high = _newcombe_difference_interval(0, 5, 0, 5)
        self.assertLess(low, -0.4)
        self.assertGreater(high, 0.4)

    def test_directional_contrast_is_arm_name_independent(self):
        items = [
            SimpleNamespace(direction_cue="lower", e0=10, decision="lower"),
            SimpleNamespace(direction_cue="lower", e0=10, decision="unchanged"),
            SimpleNamespace(direction_cue="higher", e0=10, decision="unchanged"),
            SimpleNamespace(direction_cue="higher", e0=10, decision="higher"),
        ]
        result = _directional_contrast(items)
        self.assertEqual(result["n_lower_cue"], 2)
        self.assertEqual(result["n_higher_cue"], 2)
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["cue_congruent_numeric_revisions"], 2)


if __name__ == "__main__":
    unittest.main()
