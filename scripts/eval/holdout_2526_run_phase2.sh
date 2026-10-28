#!/bin/zsh
# holdout_2526_run_phase2.sh, one-shot phase 2 inference of the hold-out confirmation
# ===================================================================================
# Runs the held-out season inference steps in order (rule 9 of results/decision_rules.md gives the judgements); all outputs go to
# data/cache/holdout_2526/. Run only once, after the decision rules are fixed and the retrained weights are in place and checked (MD5).
# Order  unified-model inference (saved and retrained weights), single-task inference from the weights (base and rich
#        scalars), destination rule baselines, summary and judgement.
# Any failing step stops the run (set -e), and the log shows which step failed.
#
# Inputs   data/cache/u1_clean/*.pt, data/cache/holdout_2526/retrain/*.pt, data/cache/holdout_2526/single{_rich}/weights/*
# Outputs  data/cache/holdout_2526/unified/*_holdout.npz, single{_rich}/holdout_preds/*, dest_rules_holdout.*, summary.json, results/holdout_2526_summary.md
#
# Usage
#   cd <repository root> and activate the Python environment of requirements.txt
#   zsh scripts/eval/holdout_2526_run_phase2.sh 2>&1 | tee logs/holdout_2526_phase2_$(date +%Y-%m-%d_%H%M).log
#
# Last modified 2026-10-28
set -e
export PYTHONPATH=.
export TASG_THREADS=4
echo "[phase2] start $(date '+%Y-%m-%d %H:%M:%S')"

echo "[phase2] 1/4 unified-model inference"
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1chk --seeds 0 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 3 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 1,2 --weights_dir data/cache/holdout_2526/retrain --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1fcn --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag shuf_dest --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1concat --seeds 0,1,2 --holdout
# u1chk s0 serves as u1 s0 and is copied to u1_s0_holdout.npz (the reproduction check file keeps its original name)
cp data/cache/holdout_2526/unified/u1chk_s0_holdout.npz data/cache/holdout_2526/unified/u1_s0_holdout.npz

echo "[phase2] 2/4 single-task inference from the weights"
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --models B2_XGB_360,M2_MLP_360,M4_CNN_Full --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --rich_scalars --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --rich_scalars --models M2_MLP_360,M4_CNN_Full --seeds 0,1,2

echo "[phase2] 3/4 destination rule baselines"
python -u scripts/eval/holdout_2526_dest_rules.py --holdout

echo "[phase2] 4/4 summary and judgement"
python -u scripts/eval/holdout_2526_summarize.py --n_boot 2000

echo "[phase2] done $(date '+%Y-%m-%d %H:%M:%S')"
