import unittest

from value_leakage_experiment.donation_positive_control import BASELINE_PROMPT, donation_prompt


class DonationPositiveControlTests(unittest.TestCase):
    def test_baseline_has_no_value_condition(self):
        self.assertNotIn("good cause", BASELINE_PROMPT)
        self.assertNotIn("bad cause", BASELINE_PROMPT)
        self.assertNotIn("threshold", BASELINE_PROMPT.lower())

    def test_above_and_below_swap_only_cause_order(self):
        above = donation_prompt(24_000_000, "above")
        below = donation_prompt(24_000_000, "below")
        self.assertIn("exceeds the threshold, we donate to some good cause", above)
        self.assertIn("does not exceed the threshold, we donate to some good cause", below)
        self.assertIn("24,000,000", above)
        self.assertIn("24,000,000", below)

    def test_invalid_good_side_fails_closed(self):
        with self.assertRaises(ValueError):
            donation_prompt(10, "sideways")


if __name__ == "__main__":
    unittest.main()
