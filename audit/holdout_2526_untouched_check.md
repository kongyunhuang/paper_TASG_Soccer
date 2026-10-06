# 留出数据确认，25/26 是否被今天的任何模型评估过（只读检查）

由 `scripts/eval/holdout_2526_untouched_check.py` 生成，2026-10-02 23:11。
25/26 比赛 146 场（{'11_318_2025_26_part': 79, '2_318_2025_26_part': 67}），事件 502,561 个。

## 零  按 event_id 扫（空间张量索引和单任务预测只有 event_id）

带 event 字段的文件 87 个，有交集的 9 个，其中结果或中间文件 **1** 个（L1_events_v3.parquet 本身是输入，列在这里只为完整）。

| 文件 | 类别 | 字段 | 命中事件数 | 文件内事件数 |
|---|---|---|---:|---:|
| L1_events_v3.parquet | 结果或中间文件 | event_id | 502,561 | 8,450,369 |
| action_soccermaps_ball_receipt_idx.parquet | 输入索引 | event_id | 112,398 | 1,993,372 |
| action_soccermaps_ball_recovery_idx.parquet | 输入索引 | event_id | 11,997 | 206,412 |
| action_soccermaps_dribble_idx.parquet | 输入索引 | event_id | 3,540 | 61,609 |
| action_soccermaps_duel_idx.parquet | 输入索引 | event_id | 8,638 | 149,138 |
| action_soccermaps_interception_idx.parquet | 输入索引 | event_id | 1,775 | 41,067 |
| action_soccermaps_pass_idx.parquet | 输入索引 | event_id | 125,130 | 2,063,498 |
| action_soccermaps_pressure_idx.parquet | 输入索引 | event_id | 47,803 | 743,376 |
| action_soccermaps_shot_idx.parquet | 输入索引 | event_id | 3,420 | 51,730 |

带 event 字段但命中为 0 的文件

| 文件 | 字段 | 文件内事件数 |
|---|---|---:|
| gradcam_dribble_examples.npz | event_ids | 4 |
| multiseed/_mirrored_frames_runs/ball_receipt_test_meta.npz | event_id | 96,194 |
| multiseed/_mirrored_frames_runs/dribble_test_meta.npz | event_id | 19,695 |
| multiseed/preds/ball_recovery_test_meta.npz | event_id | 64,011 |
| multiseed/preds/interception_test_meta.npz | event_id | 11,295 |
| multiseed/preds/pass_test_meta.npz | event_id | 99,216 |
| multiseed/preds/pressure_test_meta.npz | event_id | 107,803 |
| multiseed/preds/shot_test_meta.npz | event_id | 10,414 |
| multiseed_fixed/_first_attempt_event_loc_reference/dribble_test_meta.npz | event_id | 19,695 |
| multiseed_fixed/_second_attempt_flag_swap/dribble_test_meta.npz | event_id | 19,695 |
| multiseed_fixed/_second_attempt_flag_swap/tackle_test_meta.npz | event_id | 23,982 |
| multiseed_fixed/_third_attempt_no_translation/dribble_test_meta.npz | event_id | 19,695 |
| multiseed_fixed/preds/dribble_test_meta.npz | event_id | 19,695 |
| multiseed_fixed/preds/tackle_test_meta.npz | event_id | 23,982 |
| perspective_fixed/_second_attempt_flag_swap/soccermaps_dribble_idx.parquet | event_id | 58,069 |
| perspective_fixed/_second_attempt_flag_swap/soccermaps_duel_idx.parquet | event_id | 140,492 |
| perspective_fixed/_third_attempt_no_translation/soccermaps_dribble_idx.parquet | event_id | 58,069 |
| perspective_fixed/_third_attempt_no_translation/soccermaps_duel_idx.parquet | event_id | 140,492 |
| perspective_fixed/soccermaps_dribble_idx.parquet | event_id | 58,069 |
| perspective_fixed/soccermaps_duel_idx.parquet | event_id | 140,492 |
| predictions/dribble_G1_Gating_2ch_relaxed.npz | event_id | 3,943 |
| predictions/dribble_M4_CNN_2ch_relaxed.npz | event_id | 3,943 |
| predictions/duel_G1_Gating_2ch_relaxed.npz | event_id | 9,083 |
| predictions/duel_M4_CNN_2ch_relaxed.npz | event_id | 9,083 |
| predictions/pass_M4_CNN_2ch_strict_with_ids.npz | event_id | 99,216 |
| predictions/u1_per_task.npz | event_id | 220,435 |
| single_clean/preds/ball_recovery_test_meta.npz | event_id | 18,995 |
| single_clean/preds/dest_test_meta.npz | event_id | 30,000 |
| single_clean/preds/dribble_test_meta.npz | event_id | 5,725 |
| single_clean/preds/interception_test_meta.npz | event_id | 3,002 |
| single_clean/preds/pass_test_meta.npz | event_id | 30,000 |
| single_clean/preds/pressure_test_meta.npz | event_id | 30,000 |
| single_clean/preds/shot_test_meta.npz | event_id | 5,154 |
| single_clean/preds/tackle_test_meta.npz | event_id | 6,922 |
| single_clean_rich/preds/ball_recovery_test_meta.npz | event_id | 18,995 |
| single_clean_rich/preds/dest_test_meta.npz | event_id | 30,000 |
| single_clean_rich/preds/dribble_test_meta.npz | event_id | 5,725 |
| single_clean_rich/preds/interception_test_meta.npz | event_id | 3,002 |
| single_clean_rich/preds/pass_test_meta.npz | event_id | 30,000 |
| single_clean_rich/preds/pressure_test_meta.npz | event_id | 30,000 |
| single_clean_rich/preds/shot_test_meta.npz | event_id | 5,154 |
| single_clean_rich/preds/tackle_test_meta.npz | event_id | 6,922 |
| u1_clean/shuf_dest_s0.npz | event_id | 125,151 |
| u1_clean/shuf_dest_s1.npz | event_id | 125,151 |
| u1_clean/shuf_dest_s2.npz | event_id | 125,151 |
| u1_clean/shuf_pressure_s0.npz | event_id | 125,151 |
| u1_clean/shuf_tackle_s0.npz | event_id | 125,151 |
| u1_clean/shuf_tackle_s1.npz | event_id | 125,151 |
| u1_clean/shuf_tackle_s2.npz | event_id | 125,151 |
| u1_clean/u1_2ch_s0.npz | event_id | 125,151 |
| u1_clean/u1_2ch_s1.npz | event_id | 125,151 |
| u1_clean/u1_2ch_s2.npz | event_id | 125,151 |
| u1_clean/u1_s0.npz | event_id | 125,151 |
| u1_clean/u1_s1.npz | event_id | 125,151 |
| u1_clean/u1_s2.npz | event_id | 125,151 |
| u1_clean/u1_s3.npz | event_id | 125,151 |
| u1_clean/u1chk_s0.npz | event_id | 125,151 |
| u1_clean/u1concat_s0.npz | event_id | 125,151 |
| u1_clean/u1concat_s1.npz | event_id | 125,151 |
| u1_clean/u1concat_s2.npz | event_id | 125,151 |
| u1_clean/u1concat_shuf_dest_s0.npz | event_id | 125,151 |
| u1_clean/u1concat_shuf_dest_s1.npz | event_id | 125,151 |
| u1_clean/u1concat_shuf_dest_s2.npz | event_id | 125,151 |
| u1_clean/u1fcn_s0.npz | event_id | 125,151 |
| u1_clean/u1fcn_s1.npz | event_id | 125,151 |
| u1_clean/u1fcn_s2.npz | event_id | 125,151 |
| u1_clean/u1fcn_shuf_dest_s0.npz | event_id | 125,151 |
| u1_clean/u1hard010_s0.npz | event_id | 125,151 |
| u1_clean/u1hard030_s0.npz | event_id | 125,151 |
| u1_clean/u1pen010_s0.npz | event_id | 125,151 |
| u1_clean/u1pen030_s0.npz | event_id | 125,151 |
| u1_clean/u1rich_s0.npz | event_id | 125,151 |
| u1_clean/u1rich_s1.npz | event_id | 125,151 |
| u1_clean/u1rich_s2.npz | event_id | 125,151 |
| u1_clean/u1richhard010_s0.npz | event_id | 125,151 |
| u1_clean/u1richhard030_s0.npz | event_id | 125,151 |
| u1_clean/u1richpen010_s0.npz | event_id | 125,151 |
| u1_clean/u1richpen030_s0.npz | event_id | 125,151 |

## 一  data/cache/ 下带 match 字段的文件和 146 场的交集

扫了 812 个文件。有交集的 1 个，其中结果或中间文件 **1** 个，输入索引 0 个。

**结果或中间文件里命中 25/26 的（必须逐个解释）**

| 文件 | 字段 | 命中场次 | 文件内场次数 |
|---|---|---:|---:|
| L1_events_v3.parquet | match_id | 146 | 2426 |

输入索引里含 25/26 的（空间张量缓存本来就覆盖 25/26，不是模型成绩）

| 文件 | 字段 | 命中场次 | 文件内场次数 |
|---|---|---:|---:|

带 match 字段但命中为 0 的文件

| 文件 | 字段 | 文件内场次数 |
|---|---|---:|
| multiseed/_mirrored_frames_runs/ball_receipt.jsonl | json 内所有整数 | 2 |
| multiseed/_mirrored_frames_runs/ball_receipt_test_meta.npz | match_id | 760 |
| multiseed/_mirrored_frames_runs/dribble.jsonl | json 内所有整数 | 4 |
| multiseed/_mirrored_frames_runs/dribble_test_meta.npz | match_id | 760 |
| multiseed/ball_recovery.jsonl | json 内所有整数 | 6 |
| multiseed/interception.jsonl | json 内所有整数 | 6 |
| multiseed/pass.jsonl | json 内所有整数 | 6 |
| multiseed/preds/ball_recovery_test_meta.npz | match_id | 760 |
| multiseed/preds/interception_test_meta.npz | match_id | 760 |
| multiseed/preds/pass_test_meta.npz | match_id | 760 |
| multiseed/preds/pressure_test_meta.npz | match_id | 760 |
| multiseed/preds/shot_test_meta.npz | match_id | 760 |
| multiseed/pressure.jsonl | json 内所有整数 | 6 |
| multiseed/shot.jsonl | json 内所有整数 | 6 |
| multiseed_fixed/_first_attempt_event_loc_reference/dribble.jsonl | json 内所有整数 | 2 |
| multiseed_fixed/_first_attempt_event_loc_reference/dribble_test_meta.npz | match_id | 760 |
| multiseed_fixed/_second_attempt_flag_swap/dribble.jsonl | json 内所有整数 | 6 |
| multiseed_fixed/_second_attempt_flag_swap/dribble_test_meta.npz | match_id | 760 |
| multiseed_fixed/_second_attempt_flag_swap/tackle.jsonl | json 内所有整数 | 3 |
| multiseed_fixed/_second_attempt_flag_swap/tackle_test_meta.npz | match_id | 760 |
| multiseed_fixed/_third_attempt_no_translation/dribble.jsonl | json 内所有整数 | 3 |
| multiseed_fixed/_third_attempt_no_translation/dribble_test_meta.npz | match_id | 760 |
| multiseed_fixed/dribble.jsonl | json 内所有整数 | 6 |
| multiseed_fixed/preds/dribble_test_meta.npz | match_id | 760 |
| multiseed_fixed/preds/tackle_test_meta.npz | match_id | 760 |
| multiseed_fixed/tackle.jsonl | json 内所有整数 | 6 |
| perspective_fixed/_second_attempt_flag_swap/soccermaps_dribble_idx.parquet | match_id,match_date | 2280 |
| perspective_fixed/_second_attempt_flag_swap/soccermaps_duel_idx.parquet | match_id,match_date | 2280 |
| perspective_fixed/_third_attempt_no_translation/soccermaps_dribble_idx.parquet | match_id,match_date | 2280 |
| perspective_fixed/_third_attempt_no_translation/soccermaps_duel_idx.parquet | match_id,match_date | 2280 |
| perspective_fixed/soccermaps_dribble_idx.parquet | match_id,match_date | 2280 |
| perspective_fixed/soccermaps_duel_idx.parquet | match_id,match_date | 2280 |
| pilot_third_view/contest_skill_results.json | json 内所有整数 | 7 |
| pilot_third_view/pass_selection_results.json | json 内所有整数 | 3 |
| pilot_third_view/player_decision_style_b_results.json | json 内所有整数 | 4 |
| pilot_third_view/player_decision_style_c_results.json | json 内所有整数 | 2 |
| pilot_third_view/player_decision_style_results.json | json 内所有整数 | 6 |
| pilot_third_view/pressure_time_label_results.json | json 内所有整数 | 2 |
| pilot_third_view/pseudo_dynamics_results.json | json 内所有整数 | 2 |
| pilot_third_view/rich_scalar_baseline_results.json | json 内所有整数 | 12 |
| player_pref/fold1_s0.json | json 内所有整数 | 8 |
| player_pref/fold1_s1.json | json 内所有整数 | 7 |
| player_pref/fold1_s2.json | json 内所有整数 | 8 |
| player_pref/fold2_s0.json | json 内所有整数 | 8 |
| player_pref/fold2_s1.json | json 内所有整数 | 8 |
| player_pref/fold2_s2.json | json 内所有整数 | 7 |
| player_pref_smoke/fold1_s0.json | json 内所有整数 | 8 |
| player_pref_smoke/fold2_s0.json | json 内所有整数 | 8 |
| predictions/dribble_G1_Gating_2ch_relaxed.npz | match_id | 152 |
| predictions/dribble_M4_CNN_2ch_relaxed.npz | match_id | 152 |
| predictions/duel_G1_Gating_2ch_relaxed.npz | match_id | 152 |
| predictions/duel_M4_CNN_2ch_relaxed.npz | match_id | 152 |
| predictions/pass_M4_CNN_2ch_strict_with_ids.npz | match_id | 760 |
| predictions/u1_per_task.npz | match_id | 760 |
| results_all.json | json 内所有整数 | 1 |
| results_cross_league_8tasks_7ch.json | json 内所有整数 | 1 |
| results_dribble_ranking.json | json 内所有整数 | 18 |
| results_failure_analysis.json | json 内所有整数 | 9 |
| results_season_shift.json | json 内所有整数 | 10 |
| results_team_dribble_quality.json | json 内所有整数 | 38 |
| results_u1_player_profiles.json | json 内所有整数 | 43 |
| results_within_2425.json | json 内所有整数 | 7 |
| results_xg_pearson.json | json 内所有整数 | 1 |
| results_xpass_vs_sb.json | json 内所有整数 | 1 |
| single_clean/ball_recovery.jsonl | json 内所有整数 | 5 |
| single_clean/dest.jsonl | json 内所有整数 | 5 |
| single_clean/dribble.jsonl | json 内所有整数 | 5 |
| single_clean/interception.jsonl | json 内所有整数 | 5 |
| single_clean/pass.jsonl | json 内所有整数 | 5 |
| single_clean/preds/ball_recovery_test_meta.npz | match_id | 229 |
| single_clean/preds/dest_test_meta.npz | match_id | 229 |
| single_clean/preds/dribble_test_meta.npz | match_id | 229 |
| single_clean/preds/interception_test_meta.npz | match_id | 229 |
| single_clean/preds/pass_test_meta.npz | match_id | 229 |
| single_clean/preds/pressure_test_meta.npz | match_id | 229 |
| single_clean/preds/shot_test_meta.npz | match_id | 229 |
| single_clean/preds/tackle_test_meta.npz | match_id | 229 |
| single_clean/pressure.jsonl | json 内所有整数 | 5 |
| single_clean/shot.jsonl | json 内所有整数 | 5 |
| single_clean/tackle.jsonl | json 内所有整数 | 5 |
| single_clean_rich/ball_recovery.jsonl | json 内所有整数 | 5 |
| single_clean_rich/dest.jsonl | json 内所有整数 | 3 |
| single_clean_rich/dribble.jsonl | json 内所有整数 | 5 |
| single_clean_rich/interception.jsonl | json 内所有整数 | 5 |
| single_clean_rich/pass.jsonl | json 内所有整数 | 5 |
| single_clean_rich/preds/ball_recovery_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/dest_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/dribble_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/interception_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/pass_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/pressure_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/shot_test_meta.npz | match_id | 229 |
| single_clean_rich/preds/tackle_test_meta.npz | match_id | 229 |
| single_clean_rich/pressure.jsonl | json 内所有整数 | 5 |
| single_clean_rich/shot.jsonl | json 内所有整数 | 5 |
| single_clean_rich/tackle.jsonl | json 内所有整数 | 5 |
| u1_clean/shuf_dest_s0.json | json 内所有整数 | 32 |
| u1_clean/shuf_dest_s0.npz | match_id | 229 |
| u1_clean/shuf_dest_s1.json | json 内所有整数 | 31 |
| u1_clean/shuf_dest_s1.npz | match_id | 229 |
| u1_clean/shuf_dest_s2.json | json 内所有整数 | 31 |
| u1_clean/shuf_dest_s2.npz | match_id | 229 |
| u1_clean/shuf_pressure_s0.json | json 内所有整数 | 32 |
| u1_clean/shuf_pressure_s0.npz | match_id | 229 |
| u1_clean/shuf_tackle_s0.json | json 内所有整数 | 32 |
| u1_clean/shuf_tackle_s0.npz | match_id | 229 |
| u1_clean/shuf_tackle_s1.json | json 内所有整数 | 31 |
| u1_clean/shuf_tackle_s1.npz | match_id | 229 |
| u1_clean/shuf_tackle_s2.json | json 内所有整数 | 31 |
| u1_clean/shuf_tackle_s2.npz | match_id | 229 |
| u1_clean/u1_2ch_s0.json | json 内所有整数 | 27 |
| u1_clean/u1_2ch_s0.npz | match_id | 229 |
| u1_clean/u1_2ch_s1.json | json 内所有整数 | 22 |
| u1_clean/u1_2ch_s1.npz | match_id | 229 |
| u1_clean/u1_2ch_s2.json | json 内所有整数 | 30 |
| u1_clean/u1_2ch_s2.npz | match_id | 229 |
| u1_clean/u1_s0.json | json 内所有整数 | 32 |
| u1_clean/u1_s0.npz | match_id | 229 |
| u1_clean/u1_s1.json | json 内所有整数 | 31 |
| u1_clean/u1_s1.npz | match_id | 229 |
| u1_clean/u1_s2.json | json 内所有整数 | 31 |
| u1_clean/u1_s2.npz | match_id | 229 |
| u1_clean/u1_s3.json | json 内所有整数 | 31 |
| u1_clean/u1_s3.npz | match_id | 229 |
| u1_clean/u1chk_s0.json | json 内所有整数 | 32 |
| u1_clean/u1chk_s0.npz | match_id | 229 |
| u1_clean/u1concat_s0.json | json 内所有整数 | 32 |
| u1_clean/u1concat_s0.npz | match_id | 229 |
| u1_clean/u1concat_s1.json | json 内所有整数 | 30 |
| u1_clean/u1concat_s1.npz | match_id | 229 |
| u1_clean/u1concat_s2.json | json 内所有整数 | 31 |
| u1_clean/u1concat_s2.npz | match_id | 229 |
| u1_clean/u1concat_shuf_dest_s0.json | json 内所有整数 | 32 |
| u1_clean/u1concat_shuf_dest_s0.npz | match_id | 229 |
| u1_clean/u1concat_shuf_dest_s1.json | json 内所有整数 | 31 |
| u1_clean/u1concat_shuf_dest_s1.npz | match_id | 229 |
| u1_clean/u1concat_shuf_dest_s2.json | json 内所有整数 | 29 |
| u1_clean/u1concat_shuf_dest_s2.npz | match_id | 229 |
| u1_clean/u1fcn_s0.json | json 内所有整数 | 31 |
| u1_clean/u1fcn_s0.npz | match_id | 229 |
| u1_clean/u1fcn_s1.json | json 内所有整数 | 31 |
| u1_clean/u1fcn_s1.npz | match_id | 229 |
| u1_clean/u1fcn_s2.json | json 内所有整数 | 31 |
| u1_clean/u1fcn_s2.npz | match_id | 229 |
| u1_clean/u1fcn_shuf_dest_s0.json | json 内所有整数 | 29 |
| u1_clean/u1fcn_shuf_dest_s0.npz | match_id | 229 |
| u1_clean/u1hard010_s0.json | json 内所有整数 | 32 |
| u1_clean/u1hard010_s0.npz | match_id | 229 |
| u1_clean/u1hard030_s0.json | json 内所有整数 | 32 |
| u1_clean/u1hard030_s0.npz | match_id | 229 |
| u1_clean/u1pen010_s0.json | json 内所有整数 | 32 |
| u1_clean/u1pen010_s0.npz | match_id | 229 |
| u1_clean/u1pen030_s0.json | json 内所有整数 | 32 |
| u1_clean/u1pen030_s0.npz | match_id | 229 |
| u1_clean/u1rich_s0.json | json 内所有整数 | 32 |
| u1_clean/u1rich_s0.npz | match_id | 229 |
| u1_clean/u1rich_s1.json | json 内所有整数 | 31 |
| u1_clean/u1rich_s1.npz | match_id | 229 |
| u1_clean/u1rich_s2.json | json 内所有整数 | 31 |
| u1_clean/u1rich_s2.npz | match_id | 229 |
| u1_clean/u1richhard010_s0.json | json 内所有整数 | 32 |
| u1_clean/u1richhard010_s0.npz | match_id | 229 |
| u1_clean/u1richhard030_s0.json | json 内所有整数 | 32 |
| u1_clean/u1richhard030_s0.npz | match_id | 229 |
| u1_clean/u1richpen010_s0.json | json 内所有整数 | 32 |
| u1_clean/u1richpen010_s0.npz | match_id | 229 |
| u1_clean/u1richpen030_s0.json | json 内所有整数 | 32 |
| u1_clean/u1richpen030_s0.npz | match_id | 229 |
| u1_epoch_history.json | json 内所有整数 | 5 |

没扫的文件（二进制张量、权重、日志等，共 63 个）

- action_soccermaps_ball_receipt.npy  未扫描（npy）
- action_soccermaps_ball_recovery.npy  未扫描（npy）
- action_soccermaps_dribble.npy  未扫描（npy）
- action_soccermaps_duel.npy  未扫描（npy）
- action_soccermaps_interception.npy  未扫描（npy）
- action_soccermaps_pass.npy  未扫描（npy）
- action_soccermaps_pressure.npy  未扫描（npy）
- action_soccermaps_shot.npy  未扫描（npy）
- dribble_cnn_weights.pt  未扫描（pt）
- perspective_fixed/_second_attempt_flag_swap/perspective_fix_validation.md  未扫描（md）
- perspective_fixed/_second_attempt_flag_swap/soccermaps_dribble.npy  未扫描（npy）
- perspective_fixed/_second_attempt_flag_swap/soccermaps_duel.npy  未扫描（npy）
- perspective_fixed/_third_attempt_no_translation/soccermaps_dribble.npy  未扫描（npy）
- perspective_fixed/_third_attempt_no_translation/soccermaps_duel.npy  未扫描（npy）
- perspective_fixed/soccermaps_dribble.npy  未扫描（npy）
- perspective_fixed/soccermaps_duel.npy  未扫描（npy）
- pilot_third_view/contest_skill.log  未扫描（log）
- pilot_third_view/global_selection_fcn.pt  未扫描（pt）
- pilot_third_view/pass_selection.log  未扫描（log）
- pilot_third_view/player_decision_style.log  未扫描（log）
- pilot_third_view/player_decision_style_b.log  未扫描（log）
- pilot_third_view/player_decision_style_c.log  未扫描（log）
- pilot_third_view/pressure_time_label.log  未扫描（log）
- pilot_third_view/pseudo_dynamics.log  未扫描（log）
- pilot_third_view/rich_scalar_baseline.log  未扫描（log）
- pilot_third_view/tasg_selection_gate.log  未扫描（log）
- player_pref_smoke/global_fold1_s0.pt  未扫描（pt）
- player_pref_smoke/global_fold2_s0.pt  未扫描（pt）
- player_pref_smoke/smoke_summary.md  未扫描（md）
- u1_clean/shuf_dest_s0.pt  未扫描（pt）
- u1_clean/shuf_dest_s1.pt  未扫描（pt）
- u1_clean/shuf_dest_s2.pt  未扫描（pt）
- u1_clean/shuf_pressure_s0.pt  未扫描（pt）
- u1_clean/shuf_tackle_s0.pt  未扫描（pt）
- u1_clean/shuf_tackle_s1.pt  未扫描（pt）
- u1_clean/shuf_tackle_s2.pt  未扫描（pt）
- u1_clean/u1_2ch_s0.pt  未扫描（pt）
- u1_clean/u1_2ch_s1.pt  未扫描（pt）
- u1_clean/u1_2ch_s2.pt  未扫描（pt）
- u1_clean/u1_s3.pt  未扫描（pt）
- u1_clean/u1chk_s0.pt  未扫描（pt）
- u1_clean/u1concat_s0.pt  未扫描（pt）
- u1_clean/u1concat_s1.pt  未扫描（pt）
- u1_clean/u1concat_s2.pt  未扫描（pt）
- u1_clean/u1concat_shuf_dest_s0.pt  未扫描（pt）
- u1_clean/u1concat_shuf_dest_s1.pt  未扫描（pt）
- u1_clean/u1concat_shuf_dest_s2.pt  未扫描（pt）
- u1_clean/u1fcn_s0.pt  未扫描（pt）
- u1_clean/u1fcn_s1.pt  未扫描（pt）
- u1_clean/u1fcn_s2.pt  未扫描（pt）
- u1_clean/u1fcn_shuf_dest_s0.pt  未扫描（pt）
- u1_clean/u1hard010_s0.pt  未扫描（pt）
- u1_clean/u1hard030_s0.pt  未扫描（pt）
- u1_clean/u1pen010_s0.pt  未扫描（pt）
- u1_clean/u1pen030_s0.pt  未扫描（pt）
- u1_clean/u1rich_s0.pt  未扫描（pt）
- u1_clean/u1rich_s1.pt  未扫描（pt）
- u1_clean/u1rich_s2.pt  未扫描（pt）
- u1_clean/u1richhard010_s0.pt  未扫描（pt）
- u1_clean/u1richhard030_s0.pt  未扫描（pt）
- u1_clean/u1richpen010_s0.pt  未扫描（pt）
- u1_clean/u1richpen030_s0.pt  未扫描（pt）
- u1_weights.pt  未扫描（pt）

## 二  scripts/ 下读 L1_events_v3.parquet 的脚本及其赛季过滤

| 脚本 | 出现的过滤标识 | 导入 build_dataset | 文本里提到 25/26 |
|---|---|---|---|
| scripts/ablation/ablation_channels.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/ablation/ablation_cross_league.py | FULL_SEASONS season_dir | False | False |
| scripts/ablation/ablation_cross_league_7ch.py | FULL_SEASONS season_dir | False | False |
| scripts/audit/analyze_raw_360.py | 2025_26 318 season_dir | False | True |
| scripts/audit/audit_frame_perspective.py | FULL_SEASONS season_dir | False | False |
| scripts/audit/audit_label_leakage.py | FULL_SEASONS TRAIN_SEASONS season_dir | False | False |
| scripts/audit/audit_split_integrity.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/cache/cache_action_soccermaps.py | season_dir | False | False |
| scripts/cache/prepare_l1_events.py | season_dir | False | False |
| scripts/diagnostic/diagnose_season_pairs.py | FULL_SEASONS season_dir | False | False |
| scripts/diagnostic/diagnose_season_shift.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/diagnostic/gradcam_dribble.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/diagnostic/relaxed_holdout_dribble_duel.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/eval/compute_xg_pearson.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/eval/eval_calibration_sweep.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/eval/fit_temperature.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/eval/holdout_2526_untouched_check.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS build_dataset 2025_26 318 season_dir | False | True |
| scripts/eval/player_decision_preference.py | season_dir | False | False |
| scripts/eval/save_pass_predictions.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/pilot_third_view/pilot_pass_selection.py | season_dir | False | False |
| scripts/pilot_third_view/pilot_player_decision_style.py | season_dir | False | False |
| scripts/pilot_third_view/pilot_player_decision_style_b.py | season_dir | False | False |
| scripts/pilot_third_view/pilot_player_decision_style_c.py | season_dir | False | False |
| scripts/pilot_third_view/pilot_tasg_selection_gate.py | season_dir | False | False |
| scripts/plotting/compare_xpass_vs_sb.py | season_dir | False | False |
| scripts/plotting/player_dribble_ranking.py | season_dir | False | False |
| scripts/plotting/plot_method_overview.py |  | False | False |
| scripts/plotting/plot_saliency_grid.py |  | False | False |
| scripts/plotting/plot_u1_radar.py |  | False | False |
| scripts/supplementary_plots/plot_failure_analysis.py | season_dir | False | False |
| scripts/supplementary_plots/plot_player_case_study.py | season_dir | False | False |
| scripts/supplementary_plots/plot_team_dribble_dist.py | season_dir | False | False |
| scripts/supplementary_plots/plot_what_if_simulation.py | season_dir | False | False |
| scripts/training/train_all.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/training/train_multiseed.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/training/train_multiseed_fixed.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/training/train_unified.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS season_dir | False | False |
| scripts/training/train_unified_clean.py | FULL_SEASONS TRAIN_SEASONS TEST_SEASONS build_dataset season_dir | False | False |

读法。导入 build_dataset 的脚本一律先按 FULL_SEASONS 过滤，拿不到 25/26。其余脚本要逐个看它的过滤逻辑（见方案文件里的逐个说明）。
