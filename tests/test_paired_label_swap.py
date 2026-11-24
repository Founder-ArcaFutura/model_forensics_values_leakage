import re
import math

from value_leakage_experiment import paired_label_swap
from value_leakage_experiment.paired_label_swap import build_trials


def test_label_swap_pairs_hold_welfare_options_and_answer_symbols_fixed():
    trials = build_trials(4242)
    choices = [trial for trial in trials if trial["condition"] in {"congruent", "inverted", "neutral"}]
    by_pair = {}
    for trial in choices:
        by_pair.setdefault(trial["pair_id"], {})[trial["condition"]] = trial

    assert len(by_pair) == 16
    for conditions in by_pair.values():
        assert set(conditions) == {"congruent", "inverted", "neutral"}
        congruent = conditions["congruent"]
        inverted = conditions["inverted"]
        neutral = conditions["neutral"]

        # This regression guard prevents the old outcome/label mapping defect:
        # the symbols remain attached to the same outcomes as the labels swap.
        assert congruent["outcome_symbols"] == inverted["outcome_symbols"] == neutral["outcome_symbols"]
        assert congruent["gain_symbol"] == inverted["gain_symbol"] == neutral["gain_symbol"]
        assert congruent["outcome_labels"] == {"gain": "GOOD", "loss": "BAD"}
        assert inverted["outcome_labels"] == {"gain": "BAD", "loss": "GOOD"}
        assert neutral["outcome_labels"] is None

        canonical = lambda prompt: re.sub(r"labeled (GOOD|BAD)", "labeled LABEL", prompt)
        assert canonical(congruent["prompt"]) == canonical(inverted["prompt"])
        assert neutral["prompt"] not in {congruent["prompt"], inverted["prompt"]}


def test_order_and_answer_symbol_assignment_are_fully_crossed():
    trials = build_trials(151)
    choices = [trial for trial in trials if trial["condition"] in {"congruent", "inverted", "neutral"}]
    for wording in {trial["wording_id"] for trial in choices}:
        for scheme in {"AB", "XY"}:
            layouts = {
                (trial["option_order"], trial["gain_symbol"])
                for trial in choices
                if trial["wording_id"] == wording and trial["symbol_scheme"] == scheme
                and trial["condition"] == "congruent"
            }
            symbols = {trial["gain_symbol"] for trial in choices if trial["symbol_scheme"] == scheme}
            assert layouts == {(order, symbol) for order in ("gain_first", "loss_first") for symbol in symbols}


def test_comprehension_diagnostics_are_separate_and_correctly_mapped():
    trials = build_trials(8)
    diagnostics = [trial for trial in trials if trial.get("elicitation") == "task_comprehension_diagnostic"]
    assert len(diagnostics) == 8
    for trial in diagnostics:
        assert trial["expected_symbol"] == next(
            symbol for outcome, symbol in trial["outcome_symbols"].items()
            if trial["outcome_labels"][outcome] == "GOOD"
        )


def test_scored_outcome_margin_uses_mapping_not_answer_symbol(monkeypatch):
    trial = {
        "pair_id": "mapping-regression",
        "condition": "inverted",
        "elicitation": "choice",
        "prompt": "test prompt",
        "seed": 3,
        "outcome_symbols": {"gain": "B", "loss": "A"},
        "outcome_labels": {"gain": "BAD", "loss": "GOOD"},
    }
    monkeypatch.setattr(
        paired_label_swap,
        "continuation_tokens",
        lambda prompt, text: [{"id": text.strip(), "piece": text}],
    )

    def fake_score_sequence(prompt, tokens, seed, cache):
        symbol = tokens[0]["id"]
        if not cache:
            cache[(prompt, 5000)] = {
                "body": {"content": f" {symbol}"},
                "probabilities": {"top_logprobs": []},
            }
        logprob = {"A": -1.2, "B": -0.2}[symbol]
        return {"logprob": logprob, "token_count": 1, "tokens": tokens}

    monkeypatch.setattr(paired_label_swap, "_score_sequence", fake_score_sequence)
    result = paired_label_swap.score_trial(trial)
    assert result["outcome_logprobs"] == {"gain": -0.2, "loss": -1.2}
    assert math.isclose(result["logprob_gain_minus_loss"], 1.0)
    assert math.isclose(result["gain_candidate_set_probability"], 1 / (1 + math.exp(-1)))
