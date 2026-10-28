#!/bin/zsh
# make_all.sh, regenerates every table and figure of the paper
# ============================================================
# Runs the seven table scripts, the four main data figures and the three supplementary figures in turn and stops at the
# first failing step. Each step prints its start time and duration; the whole chain takes one to two minutes.
#
# Inputs   result files under results/, data/cache/ and logs/ (read only)
# Outputs  outputs/tables/tab_*.tex, outputs/figures/fig*.{pdf,png}, results/figure_data/*_data.csv and *_labels.csv,
#          outputs/provenance/*_sources.json
#
# Usage
#   zsh scripts/paper/make_all.sh        (from the repository root, inside the Python environment of requirements.txt)
#
# Last modified 2026-10-28
set -e
set -o pipefail   # a failing python step must stop the chain even though its output is piped through grep
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
for s in tab_tasks tab_outcome_main tab_destination tab_interventions tab_holdout tab_gate_variants_supp tab_application_supp \
         fig_outcome_vs_scalars fig_destination fig_gate fig_holdout \
         figS1_gate_vs_shuffle figS2_soft_penalty figS3_player_preference; do
  t0=$(date +%s)
  echo "[make_all] $(date +%H:%M:%S) start $s"
  python -u scripts/paper/$s.py 2>&1 | grep -v -E "NOT subset|post.stringData|timestamp seems very low"
  echo "[make_all] $s took $(( $(date +%s) - t0 )) s"
done
echo "[make_all] all done"
