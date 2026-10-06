# When Does Spatial Structure Matter? A Task-Adaptive Spatial Gating Framework for Soccer Action Outcome Prediction

This repository hosts the code, derived metrics and trained model weights for a systematic nine-task benchmark comparing scalar 360 summaries against full seven-channel SoccerMap CNN representations across 2,280 EPL and La Liga matches. The Task-Adaptive Spatial Gating (TASG) framework learns per-task when the CNN spatial structure adds predictive value: gains concentrate on Dribble and Ball Receipt, while Pass and Ball Recovery reach near-parity with the scalar baselines. A multi-task unified model (U1) with task embedding recovers per-task gate means that align independently with the CNN marginal AUC gains, and a five-step diagnostic protocol attributes the observed calibration drift on Dribble and Duel to a covariate shift at the 23/24-24/25 StatsBomb pipeline boundary.

## Project structure

```
paper_TASG_Soccer/
├── README.md                       This file
├── LICENSE                         MIT licence (with StatsBomb-data note)
├── environment.yml                 Conda environment (Python 3.10, PyTorch, sklearn, xgboost)
├── .gitignore                      Excludes raw StatsBomb data and ID-bearing predictions
├── scripts/                           33 active Python scripts (8 sub-directories)
│   ├── cache/                      Raw StatsBomb JSON → event parquet → SoccerMap tensors
│   ├── training/                   Single-task (10 models) + U1 multi-task training
│   ├── eval/                       Calibration scan, temperature scaling, xG R²+Pearson
│   ├── ablation/                   Channel ablation (7-ch vs 2-ch), cross-league
│   ├── diagnostic/                 5-step covariate shift protocol + Grad-CAM
│   ├── plotting/                   8 main paper figures + plot_utils helper
│   ├── supplementary_plots/        7 supplementary figures
│   └── audit/                      Label-leakage and split-integrity audits
├── data/                           15 aggregated results JSON + 48 clean prediction NPZ
│   ├── DATA_AVAILABILITY.md        What is included and what requires StatsBomb licence
│   ├── results_*.json              Aggregated metrics powering all paper tables
│   ├── u1_epoch_history.json       Training-time diagnostic for U1
│   └── predictions/                48 NPZ files (y_true, y_prob) with no event IDs
├── figures/                        8 main figures (fig1-8) + 7 supplementary (figS1-7), PDF + PNG
├── weights/                        Trained model state dictionaries
│   ├── dribble_cnn_weights.pt      M4 CNN 2-channel (Dribble task; ~170 KB)
│   └── u1_weights.pt               U1 multi-task unified (~225 KB)
└── notebooks/
    ├── reproduce.ipynb                Main paper-results reproduction (Tables 5/6/7/9 + S1/S5/S7/S9, U1 training diagnostics, val-vs-test gap, model-weights demo)
    ├── label_leakage_audit.ipynb      Single-feature AUC scan over the 22 tabular features (requires StatsBomb licence to execute)
    └── split_integrity_audit.ipynb    Match-id, season, event-id and seed checks on the U1 train/val/test split (requires StatsBomb licence)
```

## Quick start

```bash
# 1. Set up the environment
conda env create -f environment.yml
conda activate tasg-soccer

# 2. Open the reproduction notebook
jupyter notebook notebooks/reproduce.ipynb
```

## Reproducing paper results without raw data

The aggregated metric tables (`data/results_*.json`) and 48 clean prediction arrays (`data/predictions/*.npz`, `y_true` and `y_prob` only) are sufficient to verify every numerical claim in the paper. A subset of the figures can also be regenerated from the public data alone:

| Figure | Script | Status from public data |
|---|---|---|
| Fig. 2 main results heatmap | `plotting/plot_main_heatmap.py` | regenerable |
| Fig. 3 gate vs $\Delta$AUC | `plotting/plot_gate_vs_delta_auc.py` | regenerable |
| Supp. reliability diagrams | `supplementary_plots/plot_reliability_diagrams.py` | regenerable |
| Supp. U1 loss curve | `supplementary_plots/plot_u1_loss_curve.py` | regenerable |
| Supp. within-24/25 residual | `supplementary_plots/analyze_within_2425_shift.py` | regenerable |
| Figs. 1, 4, 5, 6, 7, 8 and remaining supplementary plots | other scripts | require predictions or raw data that fall under the StatsBomb licence; see `data/DATA_AVAILABILITY.md` |

For example, to regenerate Figure 2:

```bash
PYTHONPATH=. python scripts/plotting/plot_main_heatmap.py
# Output: figures/fig2.pdf and figures/fig2.png
```

## Reproducing from raw StatsBomb data

Requires an active StatsBomb academic licence. See `data/DATA_AVAILABILITY.md` for details. The full pipeline is:

```
cache/prepare_l1_events.py
    → cache/cache_action_soccermaps.py
        → training/train_all.py
        → training/train_unified.py
            → eval/* and ablation/* scripts
                → plotting/* scripts
```

## Trained model inference

Both weight files can be loaded for inference on new events with matching feature schema. The `scripts/diagnostic/gradcam_dribble.py` and `scripts/supplementary_plots/plot_what_if_simulation.py` scripts show how to load `dribble_cnn_weights.pt`; `scripts/training/train_unified.py` defines the U1 architecture that matches `u1_weights.pt`.

## Licence

Code is released under the MIT licence (see `LICENSE`). StatsBomb-derived raw data are excluded; see `LICENSE` and `data/DATA_AVAILABILITY.md` for the data-redistribution policy.

## Contact

Questions, bug reports and reproducibility issues can be filed via the repository issue tracker.
