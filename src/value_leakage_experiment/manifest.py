from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompts import TURN1_PROMPT, build_arms
from .util import read_json, sha256_json, sha256_text, write_json


def create_manifest(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    arms = build_arms(
        config["neutral_recipient"],
        disliked_recipient=config.get("disliked_recipient"),
        disliked_recipients=config.get("disliked_recipients"),
        include_neutral_directional=bool(config.get("include_neutral_directional", False)),
    )
    if config.get("arm_ids"):
        requested = list(config["arm_ids"])
        if len(requested) != len(set(requested)):
            raise ValueError("arm_ids contains duplicates")
        available = {arm.arm_id for arm in arms}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(f"Unknown arm_ids: {unknown}")
        arms = [arm for arm in arms if arm.arm_id in set(requested)]
    trials: list[dict[str, Any]] = []
    for arm in arms:
        for replicate in range(1, int(config["trials_per_arm"]) + 1):
            trials.append(
                {
                    "arm_id": arm.arm_id,
                    "recipient_id": arm.recipient_id,
                    "recipient_name": arm.recipient_name,
                    "recipient_valence": arm.recipient_valence,
                    "direction": arm.direction,
                    "replicate": replicate,
                    "turn1_prompt": TURN1_PROMPT,
                    "turn2_prompt": arm.turn2_prompt,
                    "turn1_prompt_sha256": sha256_text(TURN1_PROMPT),
                    "turn2_prompt_sha256": sha256_text(arm.turn2_prompt),
                }
            )

    rng = random.Random(int(config["seed"]))
    rng.shuffle(trials)
    for order, trial in enumerate(trials, start=1):
        trial["order"] = order
        trial["trial_id"] = f"{config['protocol_version']}--{order:04d}"

    manifest_core = {
        "schema_version": 1,
        "config": config,
        "config_sha256": sha256_json(config),
        "arm_counts": dict(sorted(Counter(t["arm_id"] for t in trials).items())),
        "trials": trials,
    }
    manifest = {
        **manifest_core,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_core_sha256": sha256_json(manifest_core),
    }
    write_json(output_path, manifest)
    return manifest
