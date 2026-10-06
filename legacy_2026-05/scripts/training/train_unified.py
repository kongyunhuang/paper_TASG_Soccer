#!/usr/bin/env python3
"""
Unified model: Multi-Task Spatial Gating
========================================
A single model trained on data from all tasks. The action_type embedding lets
the gating mechanism automatically learn each task's reliance on spatial
information.

Difference from per-task G1:
  - G1: train a separate gating model per task
  - U1: train one model on all tasks combined, with action_type as input

Architecture (based on GMU, Arevalo et al. 2017):
  h_event   = EventMLP(concat(x_event, action_type_embed))
  h_spatial = SharedCNN(x_soccermap)
  g         = Gate(h_event, h_spatial)  <- gate input includes action_type information
  h_fused   = g * h_spatial + (1-g) * h_event
  y_hat     = PredHead(h_fused)

Input: data/L1_events_v3.parquet + action_soccermaps_*.npy (7ch)
Output: results appended to data/results_all.json

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/training/train_unified.py

"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = Path("data")

FULL_SEASONS = [
    "2_235_2022_23", "2_281_2023_24", "2_317_2024_25",
    "11_235_2022_23", "11_281_2023_24", "11_317_2024_25",
]
TRAIN_SEASONS = ["2_235_2022_23", "2_281_2023_24", "11_235_2022_23", "11_281_2023_24"]
TEST_SEASONS = ["2_317_2024_25", "11_317_2024_25"]

# 8 binary classification tasks + 1 regression task
TASK_DEFS = {
    "pass": {"type_name": "Pass", "label_fn": lambda df: df["pass_outcome_name"].isna().astype(int).values, "max_samples": 100000, "task_type": "binary"},
    "dribble": {"type_name": "Dribble", "label_fn": lambda df: (df["dribble_outcome_name"] == "Complete").astype(int).values, "max_samples": None, "task_type": "binary"},
    "ball_receipt": {"type_name": "Ball Receipt*", "label_fn": lambda df: df["ball_receipt_outcome_name"].isna().astype(int).values, "max_samples": 100000, "task_type": "binary"},
    "shot": {"type_name": "Shot", "label_fn": lambda df: (df["shot_outcome_name"] == "Goal").astype(int).values, "filter_fn": lambda df: df[df["shot_type_name"] != "Penalty"], "max_samples": None, "task_type": "binary"},
    "duel": {"type_name": "Duel", "label_fn": lambda df: df["duel_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values, "max_samples": None, "task_type": "binary"},
    "interception": {"type_name": "Interception", "label_fn": lambda df: df["interception_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values, "max_samples": None, "task_type": "binary"},
    "ball_recovery": {"type_name": "Ball Recovery", "label_fn": lambda df: (df["ball_recovery_failure"] != True).astype(int).values, "max_samples": 100000, "task_type": "binary"},
    "pressure": {"type_name": "Pressure", "label_fn": None, "max_samples": 100000, "task_type": "binary"},
    "xg": {"type_name": "Shot", "label_fn": lambda df: df["shot_statsbomb_xg"].fillna(0).values.astype(np.float32), "filter_fn": lambda df: df[df["shot_type_name"] != "Penalty"], "max_samples": None, "task_type": "regression"},
}

# Unified feature set (union across all tasks)
UNIFIED_EVENT_FEATS = [
    "location_x", "location_y", "dist_to_goal", "angle_to_goal", "angle_to_goal_center",
    "under_pressure", "duration",
    "pass_length", "pass_angle",
    "pass_end_dist_to_goal", "pass_end_angle_to_goal",
    "pass_is_progressive", "pass_lateral_displacement",
    "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back",
    "dribble_overrun",
]

SB_360 = [
    "sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side",
    "sb_visible_teammates", "sb_visible_opponents",
]

# T8 on-ball types
ON_BALL_TYPES = {
    "Pass", "Dribble", "Shot", "Ball Receipt*", "Carry",
    "Clearance", "Interception", "Ball Recovery", "Block",
    "Foul Committed", "Foul Won", "Duel", "Dribbled Past",
    "Dispossessed", "Miscontrol", "Goal Keeper", "Pressure",
}

TASK_NAMES = list(TASK_DEFS.keys())
N_TASKS = len(TASK_NAMES)
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}

# xG and Shot share the same SoccerMap cache
SMAP_TAG_OVERRIDE = {"xg": "shot"}


# -- T8 Pressure labels -----------------------------------------------

def compute_pressure_labels(df):
    df = df.reset_index(drop=True)
    pressure_indices = df.index[df["type_name"] == "Pressure"].values
    all_types = df["type_name"].values
    all_poss = df["possession_team_id"].values
    all_match = df["match_id"].values
    is_onball = np.array([t in ON_BALL_TYPES for t in all_types])
    labels = np.zeros(len(pressure_indices), dtype=int)
    for i, pidx in enumerate(pressure_indices):
        mid = all_match[pidx]
        team = all_poss[pidx]
        onball_count = 0
        for offset in range(1, 20):
            ni = pidx + offset
            if ni >= len(df): break
            if all_match[ni] != mid: break
            if not is_onball[ni]: continue
            onball_count += 1
            if all_poss[ni] != team:
                labels[i] = 1
                break
            if onball_count >= 2: break
    return labels


# -- Unified model ----------------------------------------------------

class UnifiedGatingModel(nn.Module):
    """Multi-Task Spatial Gating.

    The action_type embedding tells the gate which action is currently being
    processed, so the gate can automatically adjust the weight of the spatial
    features.
    """

    def __init__(self, event_dim, n_tasks=8, task_embed_dim=16,
                 cnn_channels=7, embed_dim=64, dropout=0.3):
        super().__init__()

        # Action type embedding
        self.task_embed = nn.Embedding(n_tasks, task_embed_dim)

        # Event encoder: event_feats + task_embedding → embed_dim
        self.event_enc = nn.Sequential(
            nn.Linear(event_dim + task_embed_dim, embed_dim),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

        # Shared CNN encoder
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(256, embed_dim), nn.Tanh(),
        )

        # Gate (input contains action_type information via h_event)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # Shared prediction head
        self.pred = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_event, x_spatial, task_ids):
        t_embed = self.task_embed(task_ids)  # (B, task_embed_dim)
        x_combined = torch.cat([x_event, t_embed], dim=-1)

        h_e = self.event_enc(x_combined)
        h_s = self.spatial_enc(x_spatial)
        g = self.gate(torch.cat([h_e, h_s], dim=-1))
        h = g * h_s + (1 - g) * h_e
        return self.pred(h).squeeze(-1)

    def get_gate_values(self, x_event, x_spatial, task_ids):
        with torch.no_grad():
            t_embed = self.task_embed(task_ids)
            x_combined = torch.cat([x_event, t_embed], dim=-1)
            h_e = self.event_enc(x_combined)
            h_s = self.spatial_enc(x_spatial)
            return self.gate(torch.cat([h_e, h_s], dim=-1))


class UnifiedDataset(Dataset):
    def __init__(self, X_tab, y, smaps, task_ids, is_regression):
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.smaps = smaps
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)
        self.is_regression = torch.tensor(is_regression, dtype=torch.bool)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_tab[idx],
                torch.tensor(self.smaps[idx].copy(), dtype=torch.float32),
                self.task_ids[idx],
                self.y[idx],
                self.is_regression[idx])


# -- Main logic -------------------------------------------------------

def main():
    t0_all = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading data ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)

    # Preprocessing
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)
    for col in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[col] = df[col].fillna(False).astype(int)

    # T8 labels
    print("Computing T8 Pressure labels ...")
    pressure_labels = compute_pressure_labels(df)

    # Load all SoccerMap caches
    smap_lookups = {}
    smaps_all = {}
    _loaded_tags = set()
    for task_name in TASK_NAMES:
        smap_tag = SMAP_TAG_OVERRIDE.get(task_name, task_name)
        if smap_tag in _loaded_tags:
            # xg and shot share the same cache
            smap_lookups[task_name] = smap_lookups[smap_tag]
            smaps_all[task_name] = smaps_all[smap_tag]
            print(f"  {task_name}: reusing {smap_tag} cache")
            continue
        idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
        npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
        if idx_path.exists() and npy_path.exists():
            df_idx = pd.read_parquet(idx_path)
            smaps_all[task_name] = np.load(npy_path, mmap_mode="r")
            smap_lookups[task_name] = {eid: i for i, eid in enumerate(df_idx["event_id"].values)}
            _loaded_tags.add(smap_tag)
            print(f"  {task_name}: {smaps_all[task_name].shape}")

    # Build the unified dataset
    print("\nBuilding unified dataset ...")
    all_X = []
    all_y = []
    all_smaps_list = []
    all_task_ids = []
    all_is_regression = []
    all_seasons = []
    all_match_ids = []
    all_event_ids = []

    all_feats = UNIFIED_EVENT_FEATS + SB_360
    feat_cols = [f for f in all_feats if f in df.columns]

    for task_name, cfg in TASK_DEFS.items():
        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if "filter_fn" in cfg:
            df_task = cfg["filter_fn"](df_task)

        if task_name == "pressure":
            labels = pressure_labels
        else:
            labels = cfg["label_fn"](df_task)

        # Keep only samples that have a SoccerMap
        if task_name not in smap_lookups:
            print(f"  SKIP {task_name}: no SoccerMap cache")
            continue
        lookup = smap_lookups[task_name]
        has_smap = df_task["event_id"].isin(lookup).values
        df_task = df_task[has_smap].reset_index(drop=True)
        labels = labels[has_smap]

        # Sub-sampling
        max_s = cfg.get("max_samples")
        if max_s and len(df_task) > max_s:
            idx_sub = np.random.choice(len(df_task), max_s, replace=False)
            idx_sub.sort()
            df_task = df_task.iloc[idx_sub].reset_index(drop=True)
            labels = labels[idx_sub]

        # Features
        X = df_task[feat_cols].fillna(0).values.astype(np.float32)

        # SoccerMap indices
        eids = df_task["event_id"].values
        smap_indices = np.array([lookup[eid] for eid in eids])
        task_smaps = smaps_all[task_name][smap_indices]

        task_id = TASK_TO_ID[task_name]

        is_reg = 1 if cfg.get("task_type") == "regression" else 0
        all_X.append(X)
        all_y.append(labels)
        all_smaps_list.append(task_smaps)
        all_task_ids.append(np.full(len(labels), task_id, dtype=int))
        all_is_regression.append(np.full(len(labels), is_reg, dtype=int))
        all_seasons.append(df_task["season_dir"].values)
        all_match_ids.append(df_task["match_id"].values)
        all_event_ids.append(eids)

        print(f"  {task_name}: {len(labels):,} samples, pos_rate={labels.mean():.3f}")

    # Merge
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y).astype(np.float32)
    smaps_concat = np.concatenate(all_smaps_list)
    task_ids_all = np.concatenate(all_task_ids)
    is_reg_all = np.concatenate(all_is_regression)
    seasons_all = np.concatenate(all_seasons)
    match_ids_all = np.concatenate(all_match_ids)
    event_ids_all = np.concatenate(all_event_ids)

    print(f"\n  Total unified: {len(y_all):,} samples")
    print(f"  Features: {len(feat_cols)} event + task_embed")

    # Temporal split
    is_train_season = np.isin(seasons_all, TRAIN_SEASONS)
    is_test_season = np.isin(seasons_all, TEST_SEASONS)

    train_matches = np.unique(match_ids_all[is_train_season])
    np.random.shuffle(train_matches)
    n_t = int(0.8 * len(train_matches))
    train_mask = np.isin(match_ids_all, train_matches[:n_t])
    val_mask = np.isin(match_ids_all, train_matches[n_t:])
    test_mask = is_test_season

    print(f"  Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")

    # Scale
    scaler = StandardScaler().fit(X_all[train_mask])
    X_tr = scaler.transform(X_all[train_mask])
    X_va = scaler.transform(X_all[val_mask])
    X_te = scaler.transform(X_all[test_mask])

    y_tr, y_va, y_te = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    tid_tr, tid_va, tid_te = task_ids_all[train_mask], task_ids_all[val_mask], task_ids_all[test_mask]
    isreg_tr, isreg_va, isreg_te = is_reg_all[train_mask], is_reg_all[val_mask], is_reg_all[test_mask]
    sm_tr = smaps_concat[train_mask]
    sm_va = torch.tensor(smaps_concat[val_mask].copy(), dtype=torch.float32)
    sm_te = torch.tensor(smaps_concat[test_mask].copy(), dtype=torch.float32)

    # -- Training --
    print("\nTraining Unified Gating Model ...")
    model = UnifiedGatingModel(
        event_dim=len(feat_cols),
        n_tasks=N_TASKS,
        task_embed_dim=16,
        cnn_channels=7,
        embed_dim=64,
        dropout=0.3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    ds = UnifiedDataset(X_tr, y_tr, sm_tr, tid_tr, isreg_tr)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    tid_va_t = torch.tensor(tid_va, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    tid_te_t = torch.tensor(tid_te, dtype=torch.long)

    # Compute AUC on the validation set using only binary classification samples
    val_cls_mask = isreg_va == 0

    best_val_auc = 0
    best_state = None

    # epoch-wise history for loss curve plotting (Stage 10 figure)
    epoch_history = {
        "n_epochs": 30,
        "early_stop_criterion": "monitor val_auc_combined; track best, no actual stop (full 30 epochs run)",
        "n_train_batches_per_epoch": None,  # filled after epoch 1
        "n_train_samples": int(len(X_tr)),
        "n_val_samples": int(len(X_va)),
        "train_loss_per_epoch": [],
        "val_auc_combined_per_epoch": [],
        "val_auc_per_task_per_epoch": {tn: [] for tn in TASK_DEFS},
        "val_mae_xg_per_epoch": [],
        "best_epoch": None,
        "best_val_auc_combined": None,
    }

    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0
        n_batch = 0
        for xb, sb, tb, yb, is_reg_b in dl:
            pred = model(xb, sb, tb)
            # multi-task loss: BCE for binary classification, MSE for regression
            cls_mask = ~is_reg_b  # binary classification samples
            reg_mask = is_reg_b   # regression samples
            loss = torch.tensor(0.0)
            if cls_mask.any():
                loss = loss + bce_fn(pred[cls_mask], yb[cls_mask]).mean()
            if reg_mask.any():
                # Regression: pred is a raw logit; map through sigmoid to [0,1] before computing MSE
                loss = loss + mse_fn(torch.sigmoid(pred[reg_mask]), yb[reg_mask]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t, sm_va, tid_va_t).numpy()
        # Compute val AUC using only binary classification samples
        val_prob = 1 / (1 + np.exp(-val_pred[val_cls_mask]))
        val_auc = roc_auc_score(y_va[val_cls_mask], val_prob)

        # NEW: per-task val AUC + xG MAE (for loss curve plot)
        for tn, _cfg in TASK_DEFS.items():
            t_id = TASK_TO_ID[tn]
            t_mask_cls = (tid_va == t_id) & (isreg_va == 0)
            t_mask_reg = (tid_va == t_id) & (isreg_va == 1)
            if t_mask_cls.sum() > 0:
                t_prob = 1 / (1 + np.exp(-val_pred[t_mask_cls]))
                # Need at least 2 unique labels for AUC
                if len(np.unique(y_va[t_mask_cls])) > 1:
                    t_auc = roc_auc_score(y_va[t_mask_cls], t_prob)
                else:
                    t_auc = float('nan')
                epoch_history["val_auc_per_task_per_epoch"][tn].append(float(t_auc))
            elif t_mask_reg.sum() > 0:
                # xG regression task: record MAE (separate list)
                t_pred_reg = 1 / (1 + np.exp(-val_pred[t_mask_reg]))
                t_mae = float(np.abs(t_pred_reg - y_va[t_mask_reg]).mean())
                epoch_history["val_mae_xg_per_epoch"].append(t_mae)
                epoch_history["val_auc_per_task_per_epoch"][tn].append(float('nan'))  # placeholder
            else:
                epoch_history["val_auc_per_task_per_epoch"][tn].append(float('nan'))

        # Append epoch-wise scalars
        epoch_history["train_loss_per_epoch"].append(float(epoch_loss / n_batch))
        epoch_history["val_auc_combined_per_epoch"].append(float(val_auc))
        if epoch == 1:
            epoch_history["n_train_batches_per_epoch"] = int(n_batch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epoch_history["best_epoch"] = int(epoch)

        if epoch <= 3 or epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={epoch_loss/n_batch:.4f}, val_auc={val_auc:.4f}")

    epoch_history["best_val_auc_combined"] = float(best_val_auc)
    # Persist epoch history for loss-curve plotting (Stage 10)
    history_path = CACHE_DIR / "u1_epoch_history.json"
    with open(history_path, "w") as f:
        json.dump(epoch_history, f, indent=2)
    print(f"\n[epoch history] saved {history_path} (best_epoch={epoch_history['best_epoch']}, best_val_auc={best_val_auc:.4f})")

    if best_state:
        model.load_state_dict(best_state)

    # -- Evaluation: compute test AUC per task --
    print(f"\n{'='*60}")
    print("UNIFIED MODEL: PER-TASK TEST RESULTS")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        test_pred = model(X_te_t, sm_te, tid_te_t).numpy()
    test_prob = 1 / (1 + np.exp(-test_pred))

    # Gate values
    with torch.no_grad():
        gate_vals = model.get_gate_values(X_te_t, sm_te, tid_te_t).numpy()

    results = []
    print(f"\n{'Task':<15} {'Metric':>12} {'Gate Mean':>10} {'Gate Std':>10} {'N':>8}")
    print("-" * 60)

    for task_name, cfg in TASK_DEFS.items():
        tid = TASK_TO_ID[task_name]
        mask = tid_te == tid
        if mask.sum() == 0:
            continue
        g_mean = gate_vals[mask].mean()
        g_std = gate_vals[mask].std()

        task_type = cfg.get("task_type", "binary")
        result = {
            "task": task_name,
            "model": "U1_Unified",
            "task_type": task_type,
            "gate_mean": float(g_mean),
            "gate_std": float(g_std),
            "time_s": 0,
        }

        if task_type == "binary":
            auc = roc_auc_score(y_te[mask], test_prob[mask])
            result["test_auc"] = float(auc)
            print(f"{task_name:<15} {'AUC='+f'{auc:.4f}':>12} {g_mean:>10.4f} {g_std:>10.4f} {mask.sum():>8,}")
        else:
            # xG regression: sigmoid(pred) maps to [0,1]
            pred_xg = 1 / (1 + np.exp(-test_pred[mask]))
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(y_te[mask], pred_xg)
            r2 = r2_score(y_te[mask], pred_xg)
            result["test_mae"] = float(mae)
            result["test_r2"] = float(r2)
            print(f"{task_name:<15} {'MAE='+f'{mae:.4f}':>12} {g_mean:>10.4f} {g_std:>10.4f} {mask.sum():>8,}")

        results.append(result)

    # Save (drop any previous U1 results before appending)
    results_path = CACHE_DIR / "results_all.json"
    with open(results_path) as f:
        all_results = json.load(f)
    all_results = [r for r in all_results if r.get("model") != "U1_Unified"]
    all_results.extend(results)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # -- Case 4 (Session B): save weights + per-event predictions --
    weights_path = CACHE_DIR / "u1_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\n  Saved weights → {weights_path}")

    pred_dir = CACHE_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "u1_per_task.npz"
    np.savez_compressed(
        pred_path,
        event_id=event_ids_all[test_mask],
        task_id=tid_te.astype(np.int32),
        is_regression=isreg_te.astype(np.int32),
        y_true=y_te.astype(np.float32),
        y_prob=test_prob.astype(np.float32),
        gate_mean=gate_vals.mean(axis=1).astype(np.float32),
        season_dir=seasons_all[test_mask],
        match_id=match_ids_all[test_mask],
    )
    print(f"  Saved per-event predictions → {pred_path} ({len(y_te):,} events)")

    total_time = time.time() - t0_all
    print(f"\nDone: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
