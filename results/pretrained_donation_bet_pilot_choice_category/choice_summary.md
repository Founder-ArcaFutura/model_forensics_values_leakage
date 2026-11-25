# Pretrained Donation Bet choice probe

**Invalidated after protocol audit:** the outcome mapping did not preserve the
existing moral-inversion contrast. See `INVALIDATED.md`; do not interpret the
frequencies below as label-versus-welfare evidence.

Protocol: `pretrained-donation-bet-choice-category-v1`; checkpoint: `Qwen/Qwen2.5-3B` at `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
This is a choice measure, separate from numerical Donation Bet distortion. Valid choices: 67/80.

| Condition | Valid | Chooses GOOD label | Chooses welfare gain | Chooses A |
|---|---:|---:|---:|---:|
| congruent | 30 | 0.500 | 0.467 | 0.767 |
| inverted | 30 | 0.400 | 0.633 | 0.600 |
| neutral (no labels) | 7 | NA | 1.000 | 0.286 |

Rates are exploratory raw choice frequencies; Wilson intervals and order strata are in `choice_summary.json`. Only 22 of the 67 leading category parses were category-only completions; 45 continued into additional text, sometimes unrelated or inconsistent. A leading token does not establish task comprehension.
