# Decision rules and recorded verdicts

Each rule below was fixed before the corresponding result existed. The encoder rule was fixed when only the first seed of the resolution-preserving model had a result, before any result of its other seeds or its intervention existed. The thresholds for the held-out season were set after the 2024/25 results were known and before the single run on the 2025/26 matches. The same rules are given in Section S2 of the Supplementary Material. The table scripts in `scripts/paper/` read the verdicts and the recorded values from the last two sections of this file.

## Rules

1. **Main encoder.** The resolution-preserving encoder becomes the main model only if (a) its destination top-1 exceeds 15.8% in all three seeds with a mean of at least 19%, (b) under the destination intervention its destination gate falls by at least 0.20 and to one of the two lowest ranks, and (c) its top-three gate tasks overlap in at least two tasks across seeds and its destination rank varies by at most two.
2. **Enriched scalars in the unified model, gate question.** Established if in all three seeds the mean gate of the seven outcome tasks falls by at least 0.10, at least five of seven tasks fall, and the destination gate changes by less than 0.10 and stays first. Not supported if all three drops are below 0.05 or the destination gate falls more than the outcome mean. Otherwise inconclusive.
3. **Enriched scalars in the unified model, accuracy question.** Matches enriched XGBoost if at least six of the seven per-task differences (unified mean over three seeds minus XGBoost mean over five seeds) are no lower than -0.003.
4. **Two-channel spatial branch.** Established if the Spearman correlation between the task-level gate mean and the test-time loss from shuffling the player channels is at least 0.70 in all three seeds and pressure ranks in the top four by gate in all three. Not supported if all correlations are below 0.40 or pressure ranks in the bottom three in all seeds. Otherwise inconclusive.
5. **Concatenation control under the destination intervention.** Established if the concatenation model loses at least 0.005 more AUC than the gated model, averaged over the seven other tasks, in all three seeds. Not supported if the absolute difference is within 0.002 in all three.
6. **Soft gate penalty.** The penalty lambda is chosen on one seed as the largest value whose validation score is within 0.002 of the unpenalised run, from the grid 0.003, 0.01, 0.03 and 0.1 (0.003 and 0.1 were not run). The confirmation on three further seeds requires, for the enriched setting, an outcome-task gate mean of at most 0.10 with the destination gate at least 0.50 and first, at least seven of eight tasks within tolerance of the unpenalised accuracy, and a test-time loss from shuffling the whole tensor of at most 0.010 AUC averaged over outcome tasks, in every seed. For the thin setting it requires a Spearman correlation of at least 0.70 between the gate order and the single-task CNN gain over the eight tasks and at least 0.60 over the seven AUC tasks, with at least seven of eight tasks within tolerance, in every seed; not supported if the eight-task correlation is below 0.40 in every seed.
7. **Hard gate.** Same selection of lambda as for the soft penalty. No lambda qualified, so no confirmation run was made; the confirmation criteria a qualifying lambda would have faced are in scripts/eval/summarize_u1_clean.py.
8. **Player passing preference.** Supported for a group if the 95% bootstrap interval of the pooled player-level log-loss difference lies entirely below zero and both folds have a negative point estimate. Not supported if the pooled estimate is non-negative or the lower bound is above -0.001. Otherwise inconclusive. A supported verdict is lowered to inconclusive if the pooled estimate of any of the three seeds is not negative.
9. **Held-out season.** Differences are seed means; the judgements use pass, ball recovery and pressure among the outcome tasks, and the destination task.

| Claim | Confirmed when | Not confirmed when |
|---|---|---|
| C1 Destination | The resolution-preserving variant exceeds the strongest rule by at least 0.030 and the enriched scalar model by at least 0.060 in top-1, every seed above the rule | It exceeds the rule by at most 0.010 or the scalar model by at most 0.020 |
| C2 Substitutability | Three-task mean of enriched XGBoost minus thin CNN at least -0.003; three-task mean of the enriched CNN gain at most half the thin gain and at most 0.010; pressure thin gain at least 0.010 and enriched gain at most 0.010 | Three-task mean of enriched XGBoost minus thin CNN at most -0.008, or enriched gain at least the thin gain, or pressure enriched gain at least half its thin gain |
| C3 Unified model | Absolute difference to enriched XGBoost at most 0.006 on pass and pressure, at most 0.015 on ball recovery, three-task mean within +-0.006 | Above 0.010 on pass, 0.012 on pressure, 0.020 on ball recovery, or three-task mean outside +-0.010 |
| C4 Gate pattern | Baseline destination gate first among six tasks and at least 0.60 in every seed; intervened destination gate last and at most 0.35 in every seed (holds) | Any intervened seed ranks fourth or higher, or any baseline seed is not first (does not hold) |

## Recorded values

Rule 1, destination top-1 of the resolution-preserving variant (percent).

| key | value |
|---|---|
| main_encoder/top1_seed0 | 21.5 |
| main_encoder/top1_seed1 | 21.7 |
| main_encoder/top1_seed2 | 22.1 |

Rule 3, enriched unified model (three seeds) minus enriched XGBoost (five seeds), AUC on the 202 test matches of 2024/25.

| task | unified (thin scalars) | unified (enriched scalars) | enriched XGBoost | enriched unified minus enriched XGBoost |
|---|---|---|---|---|
| Pass | 0.9155 | 0.9152 | 0.9194 | -0.0042 |
| Shot | 0.7444 | 0.7422 | 0.7461 | -0.0039 |
| Interception | 0.6441 | 0.6416 | 0.6410 | 0.0006 |
| Ball recovery | 0.7020 | 0.7022 | 0.7030 | -0.0008 |
| Pressure | 0.6106 | 0.6083 | 0.6047 | 0.0036 |
| Dribble | 0.6563 | 0.6587 | 0.6606 | -0.0019 |
| Tackle | 0.6160 | 0.6150 | 0.6157 | -0.0007 |

## Recorded verdicts

| key | verdict |
|---|---|
| main_encoder | Not met; the pooled encoder is the main model. Conditions (a) and (c) hold, (b) does not (destination gate 0.49 to 0.40, rank 4 to 5, one seed) |
| enriched_gate | Inconclusive; the expected effect was not seen (outcome gate drops of 0.018, 0.062 and 0.015, and -0.013 in a fourth seed) |
| enriched_accuracy | Not matched (five of seven tasks at or above -0.003; unchanged with a fourth seed) |
| two_channel | Inconclusive (0.643, 0.476 and 0.310, pressure fifth in every seed) |
| concat_control | Not supported (differences 0.0016, 0.0013 and -0.0003) |
| soft_penalty_enriched | Inconclusive (gate means 0.093, 0.080 and 0.112; shuffle losses 0.064, 0.045 and 0.054) |
| soft_penalty_thin | Not supported (eight-task correlations 0.33, 0.10 and 0.10) |
| hard_gate_enriched | Not established; stopped at the lambda selection step (validation score fell by 0.0037 and 0.0046 at lambda 0.01 and 0.03) |
| hard_gate_thin | Not established; stopped at the lambda selection step (validation score fell by 0.0031 and 0.0037 at lambda 0.01 and 0.03) |
| preference_all_players | Supported |
| preference_changed_team | Inconclusive |
| holdout_C1 | Confirmed |
| holdout_C2 | Confirmed, Pressure near the bound (pressure enriched gain 0.0091 against the bound of 0.010) |
| holdout_C3 | Inconclusive |
| holdout_C4 | Holds |
