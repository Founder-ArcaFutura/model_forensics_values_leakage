# Pretrained Donation Bet choice probe

Protocol: `pretrained-donation-bet-choice-category-v2`; checkpoint: `Qwen/Qwen2.5-3B` at `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
This is a choice measure, separate from numerical Donation Bet distortion. Valid choices: 64/80.

| Condition | Valid | Chooses GOOD label | Chooses welfare gain | Implied option A |
|---|---:|---:|---:|---:|
| congruent | 27 | 0.407 | 0.407 | 0.630 |
| inverted | 30 | 0.367 | 0.633 | 0.633 |
| neutral (no labels) | 7 | NA | 1.000 | 0.429 |

Rates are exploratory raw choice frequencies; Wilson intervals and order strata are in `choice_summary.json`.
