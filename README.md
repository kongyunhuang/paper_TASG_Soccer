# Task-Adaptive Spatial Gating (TASG)

Code and result summaries for the paper

**Task-Adaptive Spatial Gating for Predicting the Outcome and Destination of Soccer Actions**
Kongyun Huang, Yingzhe Song, Miguel-Ángel Gómez-Ruano, Javier M. Buldú

The paper asks for which soccer action-prediction tasks an explicit spatial encoding of the StatsBomb 360 freeze-frame adds information beyond scalar summaries of the same frame, and whether one gated model can learn that difference. This repository contains the code that produced every number, table and figure in the paper and its supplementary material, and the result summaries the tables are built from.

## Layout

```
README.md  LICENSE  requirements.txt
scripts/
  audit/      audit of the raw 360 frames (frame perspective, teammate flags, per-match state)
  cache/      raw StatsBomb JSON to the event table and the 7 x 8 x 12 spatial tensors
  training/   the unified gating model and the single-task models
  eval/       test-time ablations, destination rules, player passing preference, result summaries,
              bootstrap intervals and the held-out season
  paper/      the tables and figures of the paper
  run_experiments.sh   every training and evaluation run, in order
results/      result summaries (Markdown and JSON) and the data behind each figure
logs/         training log of the unified model, seed 0
```

All commands below run from the repository root with `PYTHONPATH=.` set.

## Data

The event and 360 freeze-frame data (StatsBomb, English Premier League and La Liga, seasons 2022/23 to 2024/25 for training and testing, and the 146 matches of 2025/26 played between 15 August and 5 October 2025 for the held-out confirmation) were obtained from StatsBomb under a licence and are not included in this repository, nor is any cache derived from them; the data availability statement of the paper describes access. The summaries in `results/` are included so that the numbers in the tables and figures can be traced to their source without rerunning anything.

The scripts read the raw files from the directory named by the environment variable `STATSBOMB_DATA_DIR` (default `data/statsbomb`), with one sub-directory per competition and season named `<competition_id>_<season_id>_<season>`. Competition 2 is the Premier League and 11 is La Liga.

```
$STATSBOMB_DATA_DIR/
  2_235_2022_23/
    00_info_all_matches_<anything>.json    the season's match list (StatsBomb matches endpoint)
    <match_id>_events.json                 events endpoint, one file per match
    <match_id>_360.json                    360 frames endpoint, one file per match
  2_281_2023_24/   2_317_2024_25/   2_318_2025_26_part/
  11_235_2022_23/  11_281_2023_24/  11_317_2024_25/  11_318_2025_26_part/
```

The two `*_2025_26_part` directories hold the held-out matches and are read only by the held-out scripts.

## Environment

Python 3.10, CPU only. Install the pinned packages with

```
pip install -r requirements.txt
```

The number of torch threads is read from the environment variable `TASG_THREADS` (default 4).

## Reproducing the results

**Frame audit and caches.** The audit derives the per-match state used to select the 202 test matches of 2024/25 (`data/raw_scan/match_state_2425.parquet`); the caches hold the event table and the spatial tensors, with the dribble and tackle frames re-oriented to the acting player.

```
python -u scripts/audit/scan_raw_360.py --workers 3
python -u scripts/audit/analyze_raw_360.py
python -u scripts/audit/audit_frame_perspective.py
python -u scripts/cache/prepare_l1_events.py
python -u scripts/cache/cache_action_soccermaps.py
python -u scripts/cache/cache_soccermaps_perspective_fixed.py
```

**Training and evaluation.** `scripts/run_experiments.sh` lists every run in the paper with its tag, seeds and options, followed by the test-time ablations, the destination rules, the player passing preference, the result summaries and the held-out season. A unified run takes about half an hour on CPU (32 minutes for seed 0, see `logs/u1_s0_training.log`).

```
bash scripts/run_experiments.sh
```

**Tables and figures.** The seven data tables and seven data figures are generated from the summaries in `results/` and from the per-event predictions in `data/cache/`.

```
zsh scripts/paper/make_all.sh
```

| Paper | Script |
|---|---|
| Table 2 | `scripts/paper/tab_tasks.py` |
| Table 6, Figure 2 | `scripts/paper/tab_destination.py`, `fig_destination.py` |
| Table 7, Figure 3 | `scripts/paper/tab_outcome_main.py`, `fig_outcome_vs_scalars.py` |
| Table 8, Figure 4 | `scripts/paper/tab_interventions.py`, `fig_gate.py` |
| Table 9, Figure 5 | `scripts/paper/tab_holdout.py`, `fig_holdout.py` |
| Table S6, Figures S1 and S2 | `scripts/paper/tab_gate_variants_supp.py`, `figS1_gate_vs_shuffle.py`, `figS2_soft_penalty.py` |
| Table S9, Figure S3 | `scripts/paper/tab_application_supp.py`, `figS3_player_preference.py` |

## Results

| File | Content |
|---|---|
| `results/u1_clean_summary.md` | every unified and single-task run on the 202 test matches of 2024/25, gates, ablations and interventions |
| `results/dest_rule_baselines.md` | the occupancy rules for pass destination |
| `results/player_decision_preference_summary.md` | the player passing preference application |
| `results/paired_bootstrap_2425.md`, `.json` | bootstrap intervals over matches for the outcome tasks in 2024/25 |
| `results/holdout_2526_summary.md`, `.json` | the single run on the held-out 2025/26 matches, claim by claim |
| `results/holdout_2526_summary_check2425.md` | the same summary computed on the 2024/25 test matches |
| `results/holdout_2526_dest_rules.json` | the destination rules on the held-out matches |
| `results/decision_rules.md` | the decision rules fixed before each experiment and the recorded verdicts |
| `results/figure_data/` | the values plotted in each figure |

## Run tags and model names

The result summaries and the scripts name the models by short tags.

| Tag | Model |
|---|---|
| `u1` | unified gating model with the thin scalars and the pooled spatial encoder, the main unified model of the paper (seeds 0 to 3) |
| `u1chk` | the configuration of `u1` seed 0, run again to store the weights used on the held-out season |
| `u1rich` | unified model with the enriched scalars |
| `u1fcn` | resolution-preserving variant of the unified model (`--encoder fcn`) |
| `u1_2ch` | unified model whose spatial branch receives only the two player channels |
| `u1concat` | unified model with concatenation in place of the gate |
| `shuf_dest`, `shuf_tackle`, `shuf_pressure` | unified model retrained with the spatial tensors of one task shuffled (intervention) |
| `u1concat_shuf_dest`, `u1fcn_shuf_dest` | the destination intervention applied to `u1concat` and to `u1fcn` |
| `u1pen010`, `u1pen030`, `u1richpen010`, `u1richpen030` | soft gate with 0.01 or 0.03 times the gate mean added to the loss, with thin or enriched scalars |
| `u1hard010`, `u1hard030`, `u1richhard010`, `u1richhard030` | hard gate with the same penalties |
| `B2_XGB_360` | single-task gradient-boosted trees (XGBoost) on the scalar features |
| `M2_MLP_360` | single-task scalar network |
| `M4_CNN_Full` | single-task concatenation network of the spatial encoder and the scalar network |
| `G1_Gating` | single-task gated model |
| `xgbrich`, `cnnrich` | `B2_XGB_360` and `M4_CNN_Full` with enriched scalars (in `results/paired_bootstrap_2425.md`) |

The thin scalars are the scalar features shared by all models; the enriched scalars add 15 hand-crafted summaries of the spatial tensor (Table S1 of the Supplementary Material). `dest` is the pass destination task. In the code and in file names such as `u1_clean_summary.md`, the 202 test matches of 2024/25 whose freeze frames passed the consistency check are called the clean matches.

## Licence

MIT, see `LICENSE`.
