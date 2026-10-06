# Data Availability

## Summary

This repository **does not** include the raw event data or freeze-frame data underlying the experiments. The raw data are proprietary to **StatsBomb Services Ltd.** and were obtained under a research-use academic licence. To reproduce the full pipeline from raw data, a separate StatsBomb academic licence is required.

What is included:

| Category | Files | Provenance |
| --- | --- | --- |
| Aggregated metric tables (15 JSON files, ~80 KB) | `results_*.json` | Derived statistics from our trained models; reproduce the figures and tables in the paper. |
| Training diagnostic | `u1_epoch_history.json` | Per-epoch validation AUC trajectory for U1 multi-task. |
| Clean prediction arrays (48 NPZ files, ~13 MB) | `predictions/{task}_{model}.npz` | Each file contains only `y_true` and `y_prob` / `logits` arrays, with no event identifiers. The row ordering does not allow joining back to StatsBomb event records. |
| Trained model weights (~400 KB total) | `../weights/dribble_cnn_weights.pt`, `../weights/u1_weights.pt` | PyTorch state dictionaries for inference. |
| Final paper figures (8 main + 7 supplementary) | `../figures/fig{1-8}.pdf,png`, `figS{1-7}.pdf,png` | Reproduction outputs of the plotting scripts. |

## What is excluded and why

- **`L1_events_v3.parquet`** (~800 MB): the raw event table extracted from StatsBomb 360 data, with full event metadata, freeze-frame coordinates and StatsBomb-internal model outputs (`shot_statsbomb_xg`, `pass_success_probability`). Distribution would violate the StatsBomb licence.
- **`action_soccermaps_*.npy`** (~14 GB total across 8 tasks): the rasterised 7-channel × 8 × 12 SoccerMap tensors. These are a direct derivative of the freeze-frame coordinate data and therefore subject to the same licence.
- **Prediction NPZ files containing event identifiers** (6 files): `*_relaxed.npz`, `*_with_ids.npz`, `u1_per_task.npz`. These carry `event_id`, `match_id` and `season_dir` columns that allow joining the predictions back to specific StatsBomb event records.
- **Selected case-study artefacts**: `gradcam_dribble_examples.npz`, `results_failure_analysis.json`, `results_what_if_simulation.json`. These contain raw pitch coordinates and event identifiers for individual events, with the same redistribution restriction.

## How to reproduce from raw

If you have an active StatsBomb academic licence:

1. Install dependencies (see `../environment.yml`).
2. Place the StatsBomb JSON event tree under a local directory and export its path as the `STATSBOMB_DATA_DIR` environment variable, e.g.

   ```bash
   export STATSBOMB_DATA_DIR=/path/to/statsbomb/datos
   ```

   `prepare_l1_events.py` and `cache_action_soccermaps.py` both read this variable; running them without it set will raise a clear error.
3. Run the pipeline scripts in order: `prepare_l1_events.py` → `cache_action_soccermaps.py` → `training/train_all.py` → `training/train_unified.py` → individual evaluation and plotting scripts.

If you do not have a licence:

1. Use the included aggregated metric tables (`results_*.json`) and the clean prediction arrays (`predictions/*.npz`) to verify the numbers reported in the paper and to regenerate the figures via the plotting scripts.
2. Use the included trained model weights (`weights/*.pt`) for inference on your own data.
