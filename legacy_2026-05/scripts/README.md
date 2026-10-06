# `scripts/`: active scripts (43 files in 8 functional subdirs)

All Python scripts run from the project root, e.g. `python scripts/<subdir>/<script>.py`. Plotting scripts also support direct IDE-Run via an internal `sys.path` insert.

---

## `training/`: model training (2 scripts)

| Script | Purpose | Output |
|---|---|---|
| `train_all.py` | Core library: defines `SoccerMapEncoder`, `CNNMLPModel`, `SpatialGatingModel`, `ActionDataset`, `train_nn`, `TASK_CONFIG`, `compute_pressure_labels`. Also runs all 10 single-task models on each of the 9 tasks under the strict holdout. | `data/results_all.json`, `data/predictions/*.npz`, per-task model weights |
| `train_unified.py` | Trains the U1 multi-task model with shared backbone, task embedding $\mathbf{E} \in \mathbb{R}^{9 \times 16}$ and per-task gate vector | `weights/u1_weights.pt`, `data/u1_per_task.npz` (proprietary, not in release), `data/u1_epoch_history.json` |

**Imported by**: `ablation/*`, `diagnostic/*`, `eval/fit_temperature.py`, `eval/save_pass_predictions.py`, `eval/eval_calibration_sweep.py`, `diagnostic/relaxed_holdout_dribble_duel.py`, `supplementary_plots/plot_what_if_simulation.py`.

---

## `eval/`: calibration, metrics and predictions (3 scripts)

| Script | Purpose | Output |
|---|---|---|
| `eval_calibration_sweep.py` | ECE, Brier and reliability bins across the four (task, model) combinations | `data/results_calibration.json`, `data/predictions/*_test_calib.npz` |
| `fit_temperature.py` | Temperature scaling fit and post-$T^{\star}$ recalibration, used to test the "CNN overconfidence" hypothesis ($T^{\star} \approx 1.0$ rules out uniform overconfidence) | `data/results_temperature.json` |
| `save_pass_predictions.py` | Saves Pass M4 CNN 2-channel strict predictions with event-id join keys for the xPass-vs-StatsBomb comparison (Fig. 7) | `data/predictions/pass_M4_CNN_2ch_strict_with_ids.npz` |

---

## `audit/`: pipeline integrity audits (2 scripts)

| Script | Purpose | Output |
|---|---|---|
| `audit_label_leakage.py` | Single-feature max-direction AUC scan to detect label leakage. All eight binary tasks pass (max AUC < 0.79 against the leakage threshold of 0.90). | Static report (markdown) |
| `audit_split_integrity.py` | Eight structural integrity checks: match-id disjointness, season composition, event-id uniqueness, temporal ordering, seed determinism. | Static report (markdown) |

The two audits are also exposed as runnable notebooks in `notebooks/label_leakage_audit.ipynb` and `notebooks/split_integrity_audit.ipynb`.

---

## `cache/`: data pipeline (2 scripts)

| Script | Purpose | Output |
|---|---|---|
| `prepare_l1_events.py` | StatsBomb raw JSON to L1 events parquet (64 columns: 21 base + per-type specific + 7 SB-360 fields + 8 derived). Reads from the path in `STATSBOMB_DATA_DIR`. | `data/L1_events_v3.parquet` (~794 MB, excluded from release) |
| `cache_action_soccermaps.py` | L1 events plus 360 JSON to SoccerMap 7-channel 8×12 tensors per task. Reads from `STATSBOMB_DATA_DIR`. | `data/action_soccermaps_{task}.npy` $\times$ 8 (~13 GB total, excluded from release) |

Set `STATSBOMB_DATA_DIR` to the local path of the StatsBomb 360 JSON tree before running either script. See `../data/DATA_AVAILABILITY.md`.

---

## `ablation/`: channel and cross-league ablations (3 scripts)

| Script | Purpose | Output |
|---|---|---|
| `ablation_channels.py` | 2-channel (position only) versus 7-channel (full SoccerMap) AUC ablation; three CNN variants $\times$ eight tasks | `data/results_channel_ablation.json` (feeds Table 8 and Fig. 4b) |
| `ablation_cross_league.py` | EPL → La Liga and La Liga → EPL cross-league AUC (2-channel arm plus L1/L2/B2 baselines, Table 7) | `data/results_cross_league_8tasks.json` |
| `ablation_cross_league_7ch.py` | Cross-league 7-channel arm (M4 and G1, Table 7 main columns and Fig. 4c) | `data/results_cross_league_8tasks_7ch.json` |

The two cross-league scripts are complementary (different channel arms); neither supersedes the other.

---

## `diagnostic/`: covariate shift, relaxed holdout and Grad-CAM (4 scripts)

Three-phase diagnostic chain (strictly sequential):

| Script | Purpose | Output |
|---|---|---|
| `diagnose_season_shift.py` | Phase 1: KS test and $\Delta\mu/\sigma$ per-feature shift ranking (train versus test). Defines `soccermap_stats_per_event`. | `data/results_season_shift.json` |
| `diagnose_season_pairs.py` | Phase 2: pairwise season comparison (22/23 ↔ 23/24, 23/24 ↔ 24/25, 22/23 ↔ 24/25), localises the shift to the 23/24 → 24/25 boundary. Feeds Fig. 4 panel (e). | `data/results_season_pairs.json` |
| `relaxed_holdout_dribble_duel.py` | Adds 80% of 24/25 to train, retrains four (task, model) combinations. Defines `build_relaxed_split` used by Grad-CAM and What-If. | `data/predictions/{dribble,duel}_{M4,G1}_2ch_relaxed.npz` (proprietary, not in release) and `data/results_relaxed_holdout.json` |
| `gradcam_dribble.py` | Extracts Grad-CAM activations from the M4 CNN 2ch Conv2 layer for four representative Dribble events (HP_TruePos / HP_FalsePos / LP_FalseNeg / LP_TrueNeg) used in Fig. 6. | `data/gradcam_dribble_examples.npz` (proprietary, not in release) |

---

## `plotting/`: main paper figures (9 scripts, Figs. 1-8 plus shared lib)

| Script | Output | Paper reference |
|---|---|---|
| `plot_utils.py` | Shared library (no figure output): pitch drawing, heatmap overlay, colour constants, paper style | imported by every plotting script |
| `plot_method_overview.py` | `figures/fig1.{png,pdf}` | Fig. 1, TASG framework schematic |
| `plot_main_heatmap.py` | `figures/fig2.{png,pdf}` | Fig. 2, main results AUC trajectory and best per task |
| `plot_gate_vs_delta_auc.py` | `figures/fig3.{png,pdf}` | Fig. 3, U1 gate value versus CNN marginal gain |
| `plot_fig5_diagnostic.py` | `figures/fig4.{png,pdf}` | Fig. 4, covariate-shift six-panel diagnostic |
| `player_dribble_ranking.py` | `figures/fig5.{png,pdf}` | Fig. 5, M4 vs G1 Dribble quality scatter (32 players) |
| `plot_saliency_grid.py` | `figures/fig6.{png,pdf}` | Fig. 6, Grad-CAM four-panel saliency |
| `compare_xpass_vs_sb.py` | `figures/fig7.{png,pdf}` and `data/results_xpass_vs_sb.json` | Fig. 7, xPass vs StatsBomb reliability curve |
| `plot_u1_radar.py` | `figures/fig8.{png,pdf}` and `data/results_u1_player_profiles.json` | Fig. 8, U1 multi-task player profile radar (six panels) |

Plotting scripts use `from scripts.plotting.plot_utils import ...`. Most prepend `sys.path.insert(0, str(_PROJECT_ROOT))` to support direct IDE "Run" without a `PYTHONPATH` env var.

---

## `supplementary_plots/`: supplementary figures (8 scripts)

| Script | Output | Supplementary reference |
|---|---|---|
| `plot_failure_analysis.py` | `figures/figS1.{png,pdf}` | Section S1, covariate-shift diagnostic |
| `plot_player_case_study.py` | `figures/figS2.{png,pdf}` | Section S2, extended case studies |
| `plot_reliability_diagrams.py` | `figures/figS3.{png,pdf}` | Section S1, calibration (4$\times$2 grid) |
| `plot_strict_vs_relaxed.py` | `figures/figS4.{png,pdf}` | Section S1, strict vs relaxed ECE comparison |
| `plot_team_dribble_dist.py` | `figures/figS5.{png,pdf}` | Section S2, per-team dribble quality |
| `plot_u1_loss_curve.py` | `figures/figS6.{png,pdf}` | Companion to `notebooks/training_convergence.png` (training-convergence diagnostic) |
| `plot_what_if_simulation.py` | `figures/figS7.{png,pdf}` | Section S2, what-if perturbation case |
| `analyze_within_2425_shift.py` | `data/results_within_2425.json` | Section S1, within-24/25 residual analysis |

---

## Imports

```python
# Plotting scripts use:
from scripts.plotting.plot_utils import draw_pitch, overlay_heatmap, set_paper_style

# Scripts that need core model classes use:
from scripts.training.train_all import CNNMLPModel, SpatialGatingModel, TASK_CONFIG

# Diagnostic chain uses (sequential):
from scripts.diagnostic.diagnose_season_shift import soccermap_stats_per_event
from scripts.diagnostic.relaxed_holdout_dribble_duel import build_relaxed_split

# Eval cross-imports:
from scripts.eval.eval_calibration_sweep import expected_calibration_error
```

Run from the project root: `python scripts/<subdir>/<script>.py`. The plotting scripts that contain `sys.path` blocks also support the IDE Run button (`_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` accounts for the three-level nesting).
