#!/bin/bash
# Run every experiment reported in the paper, in order
# ===================================================
# Trains the unified models and the single-task models on the 2022/23 and 2023/24 seasons, evaluates them on the
# 202 test matches of 2024/25, runs the destination rules and the player-preference application, writes the result
# summaries to results/, and finally runs the single confirmation on the held-out 2025/26 matches.
# Training is CPU only. Each unified run takes about half an hour; the full list takes more than a day of computing,
# so in practice the blocks below are run separately.
#
# Inputs   the caches written by scripts/cache/ (data/cache/) and the frame audit tables (data/raw_scan/)
# Outputs  data/cache/u1_clean/, data/cache/single_clean{,_rich}/, data/cache/player_pref/, data/cache/holdout_2526/, results/
#
# Usage
#   bash scripts/run_experiments.sh        (from the repository root, inside the environment of requirements.txt)
#
# Last modified 2026-10-28
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONUNBUFFERED=1
export TASG_THREADS="${TASG_THREADS:-4}"
unified() { tag=$1; shift; for s in $SEEDS; do python -u scripts/training/train_unified_clean.py --seed "$s" --tag "$tag" "$@"; done; }

# Unified gating model and its variants (tag, seeds, options)
SEEDS="0 1 2 3" unified u1
SEEDS="0"       unified u1chk                                   # same configuration as u1 seed 0, stores the weights used on the held-out season
SEEDS="0 1 2 3" unified u1rich --rich_scalars
SEEDS="0 1 2"   unified u1fcn --encoder fcn                     # resolution-preserving variant
SEEDS="0 1 2"   unified u1_2ch --channels 2
SEEDS="0 1 2"   unified u1concat --fusion concat
SEEDS="0 1 2"   unified shuf_dest --shuffle_spatial dest
SEEDS="0 1 2"   unified shuf_tackle --shuffle_spatial tackle
SEEDS="0"       unified shuf_pressure --shuffle_spatial pressure
SEEDS="0 1 2"   unified u1concat_shuf_dest --fusion concat --shuffle_spatial dest
SEEDS="0"       unified u1fcn_shuf_dest --encoder fcn --shuffle_spatial dest
SEEDS="0 1 2 3" unified u1pen010 --gate_penalty 0.01
SEEDS="0"       unified u1pen030 --gate_penalty 0.03
SEEDS="0 1 2 3" unified u1richpen010 --rich_scalars --gate_penalty 0.01
SEEDS="0"       unified u1richpen030 --rich_scalars --gate_penalty 0.03
SEEDS="0"       unified u1hard010 --hard_gate --gate_penalty 0.01
SEEDS="0"       unified u1hard030 --hard_gate --gate_penalty 0.03
SEEDS="0"       unified u1richhard010 --rich_scalars --hard_gate --gate_penalty 0.01
SEEDS="0"       unified u1richhard030 --rich_scalars --hard_gate --gate_penalty 0.03

# Single-task models on the same per-task data (thin scalars, then enriched scalars)
python -u scripts/training/train_single_clean.py --seeds 0,1,2,3,4
python -u scripts/training/train_single_clean.py --rich_scalars --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/training/train_single_clean.py --rich_scalars --models M2_MLP_360,M4_CNN_Full,G1_Gating --seeds 0,1,2

# Test-time ablations of saved unified models
for run in u1:3 u1rich:0,1,2,3 u1fcn:0,1,2 u1_2ch:0,1,2 u1pen010:0,1,2,3 u1richpen010:0,1,2,3 u1richpen030:0 u1richhard010:0 u1richhard030:0; do
  python -u scripts/eval/ablate_u1_saved.py --tag "${run%%:*}" --seeds "${run##*:}"
done

# Destination rules, player passing preference, and the result summaries
python -u scripts/eval/dest_rule_baselines.py
python -u scripts/eval/player_decision_preference.py --folds 1,2 --seeds 0,1,2
python -u scripts/eval/player_decision_preference.py --summarize
python -u scripts/eval/summarize_u1_clean.py
python -u scripts/paper/tab_outcome_main.py                     # the bootstrap reads the sources registry of this table
python -u scripts/eval/paired_bootstrap_2425.py --n_boot 2000

# Held-out season 2025/26 (data checks, retraining where weights were not stored, then one inference run)
python -u scripts/eval/holdout_2526_untouched_check.py
python -u scripts/eval/holdout_2526_structure_check.py
python -u scripts/eval/holdout_2526_dataset.py --verify
python -u scripts/eval/holdout_2526_flag_check.py
python -u scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify
python -u scripts/eval/holdout_2526_retrain_unified.py --seed 2 --verify
python -u scripts/eval/holdout_2526_single_task.py --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --rich_scalars --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --rich_scalars --seeds 0,1,2
python -u scripts/eval/holdout_2526_summarize.py --check_2425
zsh scripts/eval/holdout_2526_run_phase2.sh

echo "done"
