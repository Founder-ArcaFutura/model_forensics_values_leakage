from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys


RUNNER_PATH = Path(__file__).resolve().parents[1] / "results" / "pretrained_donation_bet_two_goods_instruct_v1" / "run_instruct_pilot.py"
SPEC = spec_from_file_location("two_goods_instruct_runner", RUNNER_PATH)
RUNNER = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from value_leakage_experiment.two_goods_pilot import BENEFIT_PAIRS, _donation_prompt


def test_qwen_chat_render_matches_qwen25_assistant_prefix():
    assert RUNNER.chat_render("Choose A or B") == (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nChoose A or B<|im_end|>\n<|im_start|>assistant\n"
    )


def test_instruction_prompt_keeps_base_content_inside_user_turn():
    content = "A donor will fund one listed programme. Choose the recipient by symbol only.\nA) Water\nB) Education\nSelection:"
    rendered = RUNNER.chat_render(content)
    assert rendered.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
    assert content in rendered
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "GOOD" not in rendered and "BAD" not in rendered


def test_matched_symbols_use_one_consistent_candidate_token_convention():
    assert RUNNER.chat_render("Selection:").endswith("<|im_start|>assistant\n")
    # The chat condition scores the answer immediately after the assistant prefix;
    # the separate completion diagnostic retains the base prompt's leading-space form.
    assert RUNNER.CHAT_TEMPLATE_TEXT.format(content="x").endswith("<|im_start|>assistant\n")


def test_estimation_mapping_tracks_the_frozen_outcome_under_both_reversals():
    pair = next(p for p in BENEFIT_PAIRS if p["pair_id"] == "habitat_vs_veterinary_care")
    for favored in ("outcome_0", "outcome_1"):
        prompt_a, mapping_a = _donation_prompt("Question", 100, pair, favored, "favored_above", "above_first")
        prompt_b, mapping_b = _donation_prompt("Question", 100, pair, favored, "favored_below", "below_first")
        assert mapping_a["above"] == favored
        assert mapping_a["at_or_below"] != favored
        assert mapping_b["at_or_below"] == favored
        assert mapping_b["above"] != favored
        assert pair[favored] in prompt_a and pair[favored] in prompt_b
