#!/bin/zsh
# 留出数据确认，第二阶段一次性推理  按冻结方案第五节的顺序在 MacBook 上跑一遍，全部输出到 data/cache/holdout_2526/
# ============================
# 只在方案定稿、主会话放行、Mac mini 的重训结果同步回来并核对 MD5 之后运行，且只运行一次（见方案第五节“只跑一次的规矩”）。
# 顺序  统一模型推理（已有权重和重训权重）、单任务从权重推理（薄、加厚）、落点规则基线、汇总判定。
# 任何一步出错就停（set -e），错在哪一步日志里看得到。
#
# 输入  data/cache/u1_clean/*.pt，data/cache/holdout_2526/retrain/*.pt，data/cache/holdout_2526/single{_rich}/weights/*
# 输出  data/cache/holdout_2526/unified/*_holdout.npz，single{_rich}/holdout_preds/*，dest_rules_holdout.*，summary.json，audit/holdout_2526_summary.md
#
# Usage
#   cd <repository root> and activate the Python environment of requirements.txt
#   zsh scripts/eval/holdout_2526_run_phase2.sh 2>&1 | tee logs/holdout_2526_phase2_$(date +%Y-%m-%d_%H%M).log
#
# Last modified 2026-10-06 (public release: usage comment only; logic unchanged since 2026-10-03)
set -e
export PYTHONPATH=.
export TASG_THREADS=4
echo "[phase2] start $(date '+%Y-%m-%d %H:%M:%S')  host $(hostname)"

echo "[phase2] 1/4 统一模型推理"
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1chk --seeds 0 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 3 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 1,2 --weights_dir data/cache/holdout_2526/retrain --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1fcn --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag shuf_dest --seeds 0,1,2 --holdout
python -u scripts/eval/holdout_2526_predict_unified.py --tag u1concat --seeds 0,1,2 --holdout
# u1chk s0 当 u1 s0 用，改名成 u1_s0_holdout.npz（复现核对文件保留原名）
cp data/cache/holdout_2526/unified/u1chk_s0_holdout.npz data/cache/holdout_2526/unified/u1_s0_holdout.npz

echo "[phase2] 2/4 单任务从权重推理"
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --models B2_XGB_360,M2_MLP_360,M4_CNN_Full --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --rich_scalars --models B2_XGB_360 --seeds 0,1,2,3,4
python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --rich_scalars --models M2_MLP_360,M4_CNN_Full --seeds 0,1,2

echo "[phase2] 3/4 落点规则基线"
python -u scripts/eval/holdout_2526_dest_rules.py --holdout

echo "[phase2] 4/4 汇总判定"
python -u scripts/eval/holdout_2526_summarize.py --n_boot 2000

echo "[phase2] done $(date '+%Y-%m-%d %H:%M:%S')"
