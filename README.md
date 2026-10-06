# Task-Adaptive Spatial Gating (TASG)

Code and result summaries for the paper

**Task-Adaptive Spatial Gating for Predicting the Outcome and Destination of Soccer Actions**
Kongyun Huang, Yingzhe Song, Miguel-Ángel Gómez-Ruano, Javier M. Buldú
Submitted to Knowledge-Based Systems, October 2026.

This repository holds the code that produced every number, table and figure in the manuscript and its supplementary material, together with the result summaries the manuscript was written from. The raw StatsBomb data are not included (see Data below). The earlier May 2026 version of this repository is kept in `legacy_2026-05/` and is superseded (see Legacy below).

## Layout

```
paper_TASG_Soccer/
  README.md  LICENSE  requirements.txt
  scripts/
    cache/        raw StatsBomb JSON to the event table and the spatial tensors
    audit/        audit of the raw 360 frames (coordinate orientation, teammate flags, per-match state)
    training/     unified gating model and the single-task models
    eval/         result summaries, test-time ablations, rule baselines, player preference,
                  the 2025/26 hold-out confirmation and the 2024/25 paired bootstrap
    paper_v2/     every generated table and figure, and a number-by-number provenance check
    mini_runs/    the shell queues that ran the training jobs (macOS, CPU only)
  audit/          result summaries (Markdown and JSON) that the manuscript numbers come from
  memory_logs/    decision criteria written down before each experiment was run
  logs/           training log of the unified model, seed 0 (read by tab_tasks.py)
  data/cache/holdout_2526/   the two JSON summaries of the hold-out confirmation
  legacy_2026-05/ the superseded May 2026 release
```

All commands below run from the repository root with `PYTHONPATH=.` set, inside the Python environment described under Environment.

## Data

The event and 360 freeze-frame data (StatsBomb, English Premier League and La Liga, seasons 2022/23 to 2024/25 for training and testing, and 146 matches of 2025/26 for the hold-out confirmation) were obtained under an academic licence and cannot be redistributed. Every cache derived from them (`data/cache/`, `audit/raw_scan/`) is excluded for the same reason. The summaries under `audit/` are included so that every number in the paper can be traced to its source without rerunning anything.

The scripts read the raw files from the directory named by the environment variable `STATSBOMB_DATA_DIR` (default `data/statsbomb` under the repository root), with one sub-directory per season named `<competition_id>_<season_id>_<season>`. Competition 2 is the Premier League and 11 is La Liga.

```
$STATSBOMB_DATA_DIR/
  2_235_2022_23/
    00_info_all_matches_<anything>.json    the season's match list from the StatsBomb matches endpoint
    <match_id>_events.json                 events endpoint, one file per match
    <match_id>_360.json                    360 frames endpoint, one file per match
  2_281_2023_24/   2_317_2024_25/   2_318_2025_26_part/
  11_235_2022_23/  11_281_2023_24/  11_317_2024_25/  11_318_2025_26_part/
```

The two `*_2025_26_part` directories hold the 2025/26 matches used for the hold-out confirmation and are read only by the hold-out scripts.

## Environment

Python 3.10. Install the pinned packages with

```
pip install -r requirements.txt
```

The versions are the ones the reported runs used. Training ran on CPU only (the number of torch threads is read from the environment variable `TASG_THREADS`, default 4). The shell queues in `scripts/mini_runs/` use `caffeinate`, which exists on macOS only; on other systems run the Python commands they contain directly.

## Reproduction

Steps 0 to 4 need the raw data. Step 5 needs the per-event prediction files written by steps 2 to 4; three of its outputs (Table S6, Table S9, Figure S2) can be regenerated from the shipped summaries alone.

### Step 0. Audit the raw 360 frames and derive the per-match state

```
python -u scripts/audit/scan_raw_360.py --workers 3
python -u scripts/audit/analyze_raw_360.py
python -u scripts/audit/audit_frame_perspective.py
python -u scripts/audit/plot_frame_pairs.py
```

Writes `audit/raw_scan/` (match inventory, per-event frame diagnostics, and `match_state_2425.parquet`, which every later script uses to select the 202 clean test matches of 2024/25), `audit/raw_360_check_tables.md`, `audit/frame_perspective_report.md` and `audit/figs/frame_pair_examples.png`.

### Step 1. Build the caches

```
python -u scripts/cache/prepare_l1_events.py
python -u scripts/cache/cache_action_soccermaps.py
python -u scripts/cache/cache_soccermaps_perspective_fixed.py
```

Writes `data/cache/L1_events_v3.parquet`, `data/cache/action_soccermaps_<task>.npy` with the matching `_idx.parquet`, `data/cache/perspective_fixed/` (the re-oriented tensors for Dribble and Duel) and `audit/perspective_fix_validation.md`.

### Step 2. Train

Unified gating model, one run per seed and tag. The tags that appear in the paper are listed; the exact queues, order and thread counts are in `scripts/mini_runs/run_chain_*.sh`.

```
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1                              # seeds 0 to 3
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1rich --rich_scalars           # seeds 0 to 3
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1fcn --encoder fcn             # seeds 0 to 2
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1_2ch --channels 2             # seeds 0 to 2
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1concat --fusion concat        # seeds 0 to 2
python -u scripts/training/train_unified_clean.py --seed 0 --tag shuf_dest --shuffle_spatial dest      # seeds 0 to 2, likewise shuf_tackle (0 to 2) and shuf_pressure (0)
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1pen010 --gate_penalty 0.01    # soft gate penalty sweep, see run_chain_P.sh, run_chain_Q.sh, run_chain_S.sh and run_chain_T.sh
python -u scripts/training/train_unified_clean.py --seed 0 --tag u1hard010 --hard_gate --gate_penalty 0.01
```

Single-task models on exactly the same per-task data.

```
python -u scripts/training/train_single_clean.py --seeds 0,1,2,3,4
python -u scripts/training/train_single_clean.py --rich_scalars --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/training/train_single_clean.py --rich_scalars --models M2_MLP_360,M4_CNN_Full,G1_Gating --seeds 0,1,2
```

Analyses that use the trained models.

```
python -u scripts/eval/ablate_u1_saved.py --tag u1 --seeds 3                    # test-time ablation of saved weights, likewise u1rich, u1_2ch and u1fcn
python -u scripts/eval/dest_rule_baselines.py                                    # audit/dest_rule_baselines.md
python -u scripts/eval/player_decision_preference.py --folds 1,2 --seeds 0,1,2   # data/cache/player_pref/
python -u scripts/eval/player_decision_preference.py --summarize                 # audit/player_decision_preference_summary.md
```

### Step 3. Summarise

```
python -u scripts/eval/summarize_u1_clean.py                    # audit/u1_clean_summary.md
python -u scripts/eval/paired_bootstrap_2425.py --n_boot 2000   # audit/2026-10-06_paired_bootstrap_2425.md and .json
```

### Step 4. Hold-out confirmation on 2025/26

The plan and the judgement thresholds were frozen before the run in `memory_logs/2026-10-02_留出数据确认_冻结方案和判定标准.md` (snapshot in `audit/holdout_2526_frozen_plan_snapshot.md`). The inference step was run once.

```
python -u scripts/eval/holdout_2526_untouched_check.py       # audit/holdout_2526_untouched_check.md
python -u scripts/eval/holdout_2526_structure_check.py       # audit/holdout_2526_structure_check.md and data/cache/holdout_2526/clean_matches_2526.parquet
python -u scripts/eval/holdout_2526_dataset.py --verify      # audit/holdout_2526_dataset_verify.md
python -u scripts/eval/holdout_2526_flag_check.py            # audit/holdout_2526_flag_check.md
bash scripts/mini_runs/run_holdout_U.sh U1                   # retrain the unified model, seeds 1 and 2
bash scripts/mini_runs/run_holdout_U.sh U2                   # retrain the single-task models for the hold-out
python -u scripts/eval/holdout_2526_summarize.py --check_2425   # audit/holdout_2526_summary_check2425.md
zsh scripts/eval/holdout_2526_run_phase2.sh                  # one-shot inference and judgement
```

The last command writes `audit/holdout_2526_summary.md`, `data/cache/holdout_2526/summary.json` and `data/cache/holdout_2526/dest_rules_holdout.json`.

### Step 5. Tables and figures

```
zsh scripts/paper_v2/make_all.sh
```

Writes `outputs/tables/*.tex`, `outputs/figures/v2/*.pdf` with `.png` copies, and `outputs/tables/数据来源_2026-10-04.md`, the number-by-number provenance report. Behind it is `scripts/paper_v2/out/`, one `_sources.json` per table or figure that records, for every number, the file, line and column it was read from, and one `_data.csv` per figure with the plotted values. The copies of `scripts/paper_v2/out/` in this repository are the ones behind the submitted manuscript. The chain stops at the first failing step.

| Item in the paper | Output file | Script | Reads |
|---|---|---|---|
| Table 2 (tasks) | tab_tasks.tex | scripts/paper_v2/tab_tasks.py | audit/u1_clean_summary.md, audit/holdout_2526_summary.md, logs/u1_clean_u1_s0_2026-10-02.log, per-event predictions |
| Table 6 (destination) | tab_destination.tex | scripts/paper_v2/tab_destination.py | audit/u1_clean_summary.md, audit/dest_rule_baselines.md, the hold-out JSONs, per-event predictions |
| Table 7 (outcome tasks) | tab_outcome_main.tex | scripts/paper_v2/tab_outcome_main.py | audit/u1_clean_summary.md, per-event predictions |
| Table 8 (interventions) | tab_interventions.tex | scripts/paper_v2/tab_interventions.py | audit/u1_clean_summary.md, per-event predictions |
| Table 9 (hold-out) | tab_holdout.tex | scripts/paper_v2/tab_holdout.py | audit/holdout_2526_summary.md, the hold-out JSONs, the frozen plan, per-event predictions |
| Figure 1 (overview) | drawn by hand, not generated | scripts/paper_v2/fig_overview.py is an unused vector redraw, kept because verify_all.py reads the model's layer sizes through it | |
| Figure 2 | fig_destination.pdf | scripts/paper_v2/fig_destination.py | as Table 6 |
| Figure 3 | fig_outcome_vs_scalars.pdf | scripts/paper_v2/fig_outcome_vs_scalars.py | as Table 7 |
| Figure 4 | fig_gate.pdf | scripts/paper_v2/fig_gate.py | audit/u1_clean_summary.md, per-event predictions |
| Figure 5 | fig_holdout.pdf | scripts/paper_v2/fig_holdout.py | as Table 9 |
| Table S6 (gate variants) | tab_gate_variants_supp.tex | scripts/paper_v2/tab_gate_variants_supp.py | audit/u1_clean_summary.md and the criteria in memory_logs/ (regenerates from the shipped files) |
| Table S9 (player preference) | tab_application_supp.tex | scripts/paper_v2/tab_application_supp.py | audit/player_decision_preference_summary.md and its criteria (regenerates from the shipped files) |
| Figure S1 | figS1_gate_vs_shuffle.pdf | scripts/paper_v2/figS1_gate_vs_shuffle.py | audit/u1_clean_summary.md, the ablation JSONs, per-event predictions |
| Figure S2 | figS2_soft_penalty.pdf | scripts/paper_v2/figS2_soft_penalty.py | audit/u1_clean_summary.md (regenerates from the shipped files) |
| Figure S3 | figS3_player_preference.pdf | scripts/paper_v2/figS3_player_preference.py | audit/player_decision_preference_summary.md, data/cache/player_pref/ |

Tables 1, 3, 4 and 5 of the main text and the supplementary tables S1 to S5, S7, S8, S10 and S11 are typed in the manuscript source. Their numbers come from `audit/u1_clean_summary.md`, `audit/holdout_2526_summary.md` and `audit/2026-10-06_paired_bootstrap_2425.md`.

### Checking a number without rerunning anything

Each `scripts/paper_v2/out/<name>_sources.json` lists every number of that table or figure with its source. For a value read from a Markdown summary the entry gives the file, the line number and the column, so the number can be found by opening the file at that line. `audit/u1_clean_summary.md` is the central summary; its section headings name the run tag and the quantity.

## Legacy

`legacy_2026-05/` is the repository as released in May 2026. It was built before the audit of the raw 360 frames, on data in which the freeze frames of failed dribbles and lost duels were recorded from the opponent's viewpoint (documented in `audit/frame_perspective_report.md` and `audit/raw_360_check_tables.md`). Its Dribble, Ball Receipt and Duel results are therefore not valid, and none of its code, numbers or figures is used in the submitted manuscript. It is kept for the record only and should not be run.

## Licence

The code is released under the MIT licence (see `LICENSE`). The StatsBomb data and every cache derived from them are excluded from this licence and from the repository.

## Contact

Questions and reproducibility issues can be filed through the repository issue tracker.
