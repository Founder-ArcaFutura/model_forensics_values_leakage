from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys


RUNNER_PATH = Path(__file__).resolve().parents[1] / "results" / "pretrained_donation_bet_two_goods_instruct_v1" / "score_instruct_estimation_probabilities.py"
SPEC = spec_from_file_location("two_goods_exact_expansion", RUNNER_PATH)
RUNNER = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)


class FakeScorer:
    def __init__(self, target_present):
        self.target_present = target_present
        self.n_requested = []

    @staticmethod
    def continuation_tokens(prompt, candidate):
        return [{"id": 17, "piece": candidate}]

    def distribution(self, prompt, seed, n_probs):
        self.n_requested.append(n_probs)
        values = [{"id": 17, "logprob": -8.25}] if self.target_present and n_probs == 20000 else []
        return {"probabilities": {"top_logprobs": values}}


def test_exact_expansion_increases_n_until_target_logprob_is_returned():
    fake = FakeScorer(target_present=True)
    result = RUNNER.score_full_candidate("p", "5", 1, fake)
    assert fake.n_requested == [5000, 20000]
    assert result["logprob"] == -8.25
    assert result["tokens"][0]["n_probs_used"] == 20000


def test_exact_expansion_fails_instead_of_approximating_missing_candidate():
    fake = FakeScorer(target_present=False)
    try:
        RUNNER.score_full_candidate("p", "5", 1, fake)
    except RUNNER.ExactScoreUnavailable as exc:
        assert "no truncated probability substituted" in str(exc)
    else:
        raise AssertionError("A missing target token must not be approximated from a truncated list")
    assert fake.n_requested == [5000, 20000, 50000]


def test_chat_numeric_candidate_does_not_add_a_second_leading_space():
    assert RUNNER.format_numeric_candidate(50000) == "50,000"
    assert RUNNER.format_numeric_candidate(3650) == "3,650"
