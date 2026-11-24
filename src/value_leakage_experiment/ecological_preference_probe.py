"""Matched ecological-assumption probes for the Qwen two-goods result."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = "pretrained-donation-bet-ecological-mechanism-v1"
SEED = 20261127
HABITAT = "Native habitat is protected across 100 acres."
VETERINARY = "Veterinary care is provided to 100 wild animals."

SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Original unqualified pair",
        "habitat_text": HABITAT,
        "veterinary_text": VETERINARY,
        "expected_shift_vs_baseline": 0,
        "changes": "Exact source outcome descriptions; this is a matched repeat of the frozen Instruct chat condition.",
    },
    "habitat_secure": {
        "label": "Habitat already secure; modest marginal effect",
        "habitat_text": "The 100 acres of native habitat are already intact and legally protected; added protection would prevent only low-probability future damage.",
        "veterinary_text": VETERINARY,
        "expected_shift_vs_baseline": -1,
        "changes": "Changes only the habitat description to stipulate secure status and low marginal benefit.",
        "comprehension_true": "The habitat is already intact and legally protected.",
        "comprehension_false": "The habitat is threatened with imminent ecosystem collapse.",
    },
    "vet_prevents_extinction": {
        "label": "Veterinary care prevents local-population extinction",
        "habitat_text": HABITAT,
        "veterinary_text": "Veterinary care is provided to 100 wild animals in a critically endangered local population; the care is expected to prevent that population's extinction.",
        "expected_shift_vs_baseline": -1,
        "changes": "Changes only the veterinary description to stipulate a population-level extinction-prevention effect.",
        "comprehension_true": "The veterinary care is expected to prevent the local population's extinction.",
        "comprehension_false": "The veterinary care improves health but has no population-level survival effect.",
    },
    "ecosystem_wide": {
        "label": "Habitat protection prevents ecosystem collapse",
        "habitat_text": "The 100 acres of native habitat are protected, preventing collapse of a threatened ecosystem and sustaining many species for decades.",
        "veterinary_text": VETERINARY,
        "expected_shift_vs_baseline": 1,
        "changes": "Changes only the habitat description to stipulate ecosystem-wide, long-duration benefits.",
        "comprehension_true": "Habitat protection prevents ecosystem collapse and sustains many species for decades.",
        "comprehension_false": "Habitat protection adds only a modest buffer to a secure ecosystem.",
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def load_habitat_source_trials(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in manifest["trials"]
            if row.get("pair_id") == "habitat_vs_veterinary_care"
            and row.get("stage") == "independent_preference"]
    if len(rows) != 16:
        raise ValueError(f"Expected 16 frozen habitat preference layouts, found {len(rows)}")
    return manifest, rows


def contextualize_prompt(prompt: str, scenario_id: str) -> str:
    scenario = SCENARIOS[scenario_id]
    if prompt.count(HABITAT) != 1 or prompt.count(VETERINARY) != 1:
        raise ValueError("Source prompt does not contain each expected outcome exactly once")
    return prompt.replace(HABITAT, scenario["habitat_text"]).replace(VETERINARY, scenario["veterinary_text"])


def build_preference_trials(source_rows: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    scenario_names = list(SCENARIOS)
    for sidx, scenario_id in enumerate(scenario_names):
        for lidx, source in enumerate(source_rows):
            prompt = contextualize_prompt(source["prompt"], scenario_id)
            symbol_to_outcome = {symbol: outcome for outcome, symbol in source["outcome_symbols"].items()}
            # In the frozen source pair, outcome_0 is habitat and outcome_1 is veterinary care.
            if symbol_to_outcome.get(source["outcome_symbols"].get("outcome_0")) != "outcome_0":
                raise ValueError("Outcome-to-symbol map is malformed")
            lines = [line for line in prompt.splitlines() if re.match(r"^[ABXY]\)\s", line)]
            if len(lines) != 2:
                raise ValueError("Expected exactly two symbol-labelled options")
            label_order = [line[0] for line in lines]
            outcome_for_symbol = dict(source["outcome_symbols"])
            habitat_symbol = outcome_for_symbol["outcome_0"]
            veterinary_symbol = outcome_for_symbol["outcome_1"]
            trial = {
                "protocol": PROTOCOL,
                "stage": "contextual_preference_likelihood",
                "scenario_id": scenario_id,
                "scenario_label": SCENARIOS[scenario_id]["label"],
                "source_trial_id": source["trial_id"],
                "layout_id": source["trial_id"],
                "trial_id": f"{scenario_id}::{source['trial_id']}",
                "wording_id": source["wording_id"],
                "symbol_scheme": source["symbol_scheme"],
                "presentation_order": source["presentation_order"],
                "option_symbols_in_presentation_order": label_order,
                "habitat_symbol": habitat_symbol,
                "veterinary_symbol": veterinary_symbol,
                "outcome_symbols": {"habitat": habitat_symbol, "veterinary": veterinary_symbol},
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "seed": seed + sidx * len(source_rows) + lidx,
                "answer_format": "bare one-token option symbol after Qwen assistant prefix",
                "expected_direction_vs_baseline": SCENARIOS[scenario_id]["expected_shift_vs_baseline"],
            }
            trials.append(trial)
    return trials


def build_comprehension_trials(seed: int = SEED + 10000) -> list[dict[str, Any]]:
    trials = []
    rng = random.Random(seed)
    idx = 0
    for scenario_id in ("habitat_secure", "vet_prevents_extinction", "ecosystem_wide"):
        scenario = SCENARIOS[scenario_id]
        for statement_order in ("true_first", "false_first"):
            for correct_symbol in ("A", "B"):
                correct = scenario["comprehension_true"]
                incorrect = scenario["comprehension_false"]
                options = {correct_symbol: correct, ("B" if correct_symbol == "A" else "A"): incorrect}
                if statement_order == "false_first":
                    options = {k: v for k, v in reversed(list(options.items()))}
                option_text = "\n".join(f"{symbol}) {text}" for symbol, text in options.items())
                description = f"Habitat program: {scenario['habitat_text']}\nVeterinary program: {scenario['veterinary_text']}"
                prompt = (f"{description}\n\nWhich statement is explicitly supported by the scenario?\n"
                          f"{option_text}\nReply with the option symbol only.\nAnswer:")
                trials.append({
                    "protocol": PROTOCOL,
                    "stage": "assumption_comprehension",
                    "scenario_id": scenario_id,
                    "trial_id": f"comprehension::{scenario_id}::{statement_order}::{correct_symbol}",
                    "statement_order": statement_order,
                    "expected_symbol": correct_symbol,
                    "symbol_scheme": "AB",
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "seed": seed + idx,
                    "candidate_text": options,
                })
                idx += 1
    return trials


def build_explanation_prompts(seed: int = SEED + 20000) -> list[dict[str, Any]]:
    rows = []
    # First elicit the original comparison as requested, with presentation order counterbalanced.
    for idx, reverse in enumerate((False, True)):
        outcomes = [("Native habitat protection across 100 acres", "Veterinary care for 100 wild animals")]
        if reverse:
            outcomes[0] = tuple(reversed(outcomes[0]))
        prompt = (
            "A charitable donor must choose between " + outcomes[0][0] + " and " + outcomes[0][1] + ". "
            "What considerations would determine which intervention produces the greater overall benefit? "
            "Briefly explain the trade-offs, then make a recommendation. Do not assume either option is preferred."
        )
        rows.append({"stage": "baseline_explanation", "scenario_id": "baseline", "order_reversed": reverse,
                     "trial_id": f"explanation::baseline::{idx}", "prompt": prompt,
                     "prompt_sha256": sha256_text(prompt), "seed": seed + idx,
                     "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 180}})
    # Then ask for a separate explanation of each assumption scenario. These are qualitative diagnostics.
    for idx, scenario_id in enumerate(("habitat_secure", "vet_prevents_extinction", "ecosystem_wide"), start=2):
        scenario = SCENARIOS[scenario_id]
        prompt = (
            f"Consider two charitable programmes. Programme H: {scenario['habitat_text']} "
            f"Programme V: {scenario['veterinary_text']} "
            "What considerations determine which intervention produces greater overall benefit? "
            "Briefly explain the trade-offs, then make a recommendation. Do not assume either option is preferred."
        )
        rows.append({"stage": "assumption_explanation", "scenario_id": scenario_id, "order_reversed": False,
                     "trial_id": f"explanation::{scenario_id}", "prompt": prompt,
                     "prompt_sha256": sha256_text(prompt), "seed": seed + idx,
                     "generation": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_tokens": 180}})
    return rows


def make_protocol_manifest(base_manifest: dict[str, Any], source_rows: list[dict[str, Any]],
                           preregistration_sha256: str) -> dict[str, Any]:
    pref = build_preference_trials(source_rows)
    comp = build_comprehension_trials()
    exp = build_explanation_prompts()
    core = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "base_checkpoint_protocol": "Qwen2.5-14B-Instruct Q4_K_M; frozen habitat-veterinary 16-layout source; same chat-template continuation scoring",
        "source_preference_manifest_core_sha256": base_manifest["manifest_core_sha256"],
        "source_freeze_sha256": json.loads((Path(__file__).resolve().parents[2] / "results" /
                                              "pretrained_donation_bet_two_goods_instruct_v1" /
                                              "preference_freeze.json").read_text(encoding="utf-8"))["freeze_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "scenario_specs": SCENARIOS,
        "trials": pref,
        "comprehension_trials": comp,
        "explanation_prompts": exp,
        "scoring": {"target": "logP(habitat continuation) - logP(veterinary continuation)",
                    "candidate_continuations": "bare single-token answer symbol after verified Qwen assistant generation prefix",
                    "distribution": "native unfiltered raw-logit softmax, temperature -1, top-k 0, top-p 1, post_sampling_probs false",
                    "candidate_set_probability": "conditional softmax over the two full candidate continuation log likelihoods; not vocabulary-wide probability"},
        "stability_rule": "unchanged: for a scenario, at least 12 of the 16 matched layouts favor the same beneficiary and the mean margin agrees across both wordings, both presentation orders, and both answer-symbol schemes",
        "primary_contrasts": ["habitat_secure minus baseline margin < 0",
                              "vet_prevents_extinction minus baseline margin < 0",
                              "ecosystem_wide minus baseline margin > 0"],
        "interpretation_guardrails": ["explanations are generated outputs, not privileged causal reports",
                                      "layout repetitions are counterbalances, not independent model samples",
                                      "a scenario-dependent shift can support sensitivity to stipulated consequences, not intrinsic moral values"],
    }
    return {**core, "manifest_sha256": sha256_json(core)}


def summarize_preferences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = {}
    preference_rows = [r for r in rows if r.get("stage") == "contextual_preference_likelihood"]
    baseline_by_layout = {r["layout_id"]: r for r in preference_rows if r["scenario_id"] == "baseline"}
    for scenario_id in SCENARIOS:
        group = [r for r in preference_rows if r["scenario_id"] == scenario_id]
        margins = [float(r["logprob_habitat_minus_veterinary"]) for r in group]
        mean = sum(margins) / len(margins)
        favored_sign = 1 if mean >= 0 else -1
        subgroups = {}
        for field, levels in (("wording_id", sorted({r["wording_id"] for r in group})),
                              ("presentation_order", ("zero_first", "one_first")),
                              ("symbol_scheme", ("AB", "XY"))):
            subgroups[field] = {}
            for level in levels:
                vals = [float(r["logprob_habitat_minus_veterinary"]) for r in group if r[field] == level]
                subgroups[field][level] = sum(vals) / len(vals) if vals else None
        consistent = sum(m * favored_sign > 0 for m in margins)
        subgroup_agree = all(value is not None and value * favored_sign > 0
                             for levels in subgroups.values() for value in levels.values())
        paired = []
        for row in group:
            if scenario_id == "baseline":
                continue
            base = baseline_by_layout[row["layout_id"]]
            paired.append(float(row["logprob_habitat_minus_veterinary"]) -
                          float(base["logprob_habitat_minus_veterinary"]))
        scenarios[scenario_id] = {
            "n_layouts": len(group),
            "mean_logprob_habitat_minus_veterinary": mean,
            "median_logprob_habitat_minus_veterinary": statistics.median(margins),
            "habitat_favored_layouts": sum(m > 0 for m in margins),
            "veterinary_favored_layouts": sum(m < 0 for m in margins),
            "favored_beneficiary_by_mean": "habitat" if mean > 0 else "veterinary" if mean < 0 else "tie",
            "direction_consistency_n_of_16": consistent,
            "subgroup_mean_margins": subgroups,
            "stable_by_unchanged_rule": len(group) == 16 and consistent >= 12 and subgroup_agree,
            "matched_delta_vs_baseline": {
                "n_layout_pairs": len(paired),
                "mean": sum(paired) / len(paired) if paired else None,
                "median": statistics.median(paired) if paired else None,
                "positive_n": sum(x > 0 for x in paired),
                "negative_n": sum(x < 0 for x in paired),
            } if paired else None,
            "greedy_argmax_symbol_counts": {symbol: sum(r.get("candidate_argmax") == symbol for r in group)
                                             for symbol in ("A", "B", "X", "Y")},
        }
    comp = [r for r in rows if r["stage"] == "assumption_comprehension"]
    comp_by_scenario = {}
    for scenario_id in ("habitat_secure", "vet_prevents_extinction", "ecosystem_wide"):
        group = [r for r in comp if r["scenario_id"] == scenario_id]
        comp_by_scenario[scenario_id] = {"n": len(group), "correct": sum(r["correct_by_argmax"] for r in group),
                                         "accuracy": sum(r["correct_by_argmax"] for r in group) / len(group) if group else None}
    return {"protocol": PROTOCOL, "preference_scenarios": scenarios,
            "comprehension": {"n": len(comp), "correct": sum(r["correct_by_argmax"] for r in comp),
                              "accuracy": sum(r["correct_by_argmax"] for r in comp) / len(comp) if comp else None,
                              "by_scenario": comp_by_scenario},
            "candidate_lengths": {"preference_token_counts": sorted({n for r in rows if r["stage"] == "contextual_preference_likelihood"
                                                                         for n in r["candidate_token_counts"].values()}),
                                  "comprehension_token_counts": sorted({n for r in comp for n in r["candidate_token_counts"].values()})},
            "independent_unit_note": "Scenario-layout rows are matched prompt counterbalances, not independent model samples. The scenario is one beneficiary pair; no p-values are computed."}

