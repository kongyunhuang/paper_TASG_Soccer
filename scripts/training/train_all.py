#!/usr/bin/env python3
"""
Phase 2: unified training script: all models x all tasks
=========================================================
Following TECHNICAL_SPEC.md, runs 9 model configurations x 9 tasks (8 binary + 1 regression).

Models: L1(LR Event), L2(LR+360), B1(XGB Event), B2(XGB+360),
        M1(MLP Event), M2(MLP+360), M3(CNN+MLP), M4(CNN+MLP Full), G1(Gating)
Tasks:  T1-T8 binary + T9 xG regression

Data split: time-series holdout (TECHNICAL_SPEC 4.2)
  Train: 22/23 + 23/24 (EPL + La Liga), match-level 80%
  Val:   22/23 + 23/24, match-level 20%
  Test:  24/25 (EPL + La Liga)

Inputs:
  data/L1_events_v3.parquet
  data/action_soccermaps_{task}.npy + _idx.parquet (7ch)

Outputs:
  data/results_all.json

Note: the main experiment defaults to cnn_channels=7 (full A-G channels).
      Subsequent derivative experiments (ablation_*/eval_*/relaxed_*/save_pass_predictions/gradcam)
      switch to cnn_channels=2 by taking the first 2 channels from the 7ch npy (smap[:, :2]).
      Ablation A1 found 2ch to be superior on Dribble/Duel.
      The paper must state explicitly which version each reported number comes from.

Usage:
  PYTHONUNBUFFERED=1 python scripts/training/train_all.py

"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier, XGBRegressor

CACHE_DIR = Path("data")

FULL_SEASONS = [
    "2_235_2022_23", "2_281_2023_24", "2_317_2024_25",
    "11_235_2022_23", "11_281_2023_24", "11_317_2024_25",
]
TRAIN_SEASONS = ["2_235_2022_23", "2_281_2023_24", "11_235_2022_23", "11_281_2023_24"]
TEST_SEASONS = ["2_317_2024_25", "11_317_2024_25"]

# T8 on-ball event types (aligned with exPress, Table 2, p.6)
ON_BALL_TYPES = {
    "Pass", "Dribble", "Shot", "Ball Receipt*", "Carry",
    "Clearance", "Interception", "Ball Recovery", "Block",
    "Foul Committed", "Foul Won", "Duel", "Dribbled Past",
    "Dispossessed", "Miscontrol", "Goal Keeper", "Pressure",
}

# ── Feature definitions (TECHNICAL_SPEC Section 2.3) ─────────────────────────────

BASE_EVENT = ["location_x", "location_y", "dist_to_goal", "angle_to_goal", "angle_to_goal_center"]

SB_360_COMMON = ["sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side",
                 "sb_visible_teammates", "sb_visible_opponents"]

TASK_CONFIG = {
    "pass": {
        "type_name": "Pass",
        "label_fn": lambda df: (df["pass_outcome_name"].isna()).astype(int).values,
        "event_feats": BASE_EVENT + [
            "under_pressure", "pass_length", "pass_angle",
            "pass_end_location_x", "pass_end_location_y",
            "pass_end_dist_to_goal", "pass_end_angle_to_goal",
            "pass_is_progressive", "pass_lateral_displacement",
            "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back",
        ],
        "event_onehot": {"pass_height_name": ["Ground Pass", "Low Pass", "High Pass"],
                         "pass_body_part_name": ["Right Foot", "Left Foot", "Head"]},
        "extra_360": ["sb_line_breaking_pass", "ux_dist_defender_end", "ux_nb_opp_in_path"],
        "smap_tag": "pass",
        "task_type": "binary",
        "max_samples": 300000,
    },
    "dribble": {
        "type_name": "Dribble",
        "label_fn": lambda df: (df["dribble_outcome_name"] == "Complete").astype(int).values,
        "event_feats": BASE_EVENT + ["dribble_overrun"],
        "event_onehot": {},
        "extra_360": [],
        "smap_tag": "dribble",
        "task_type": "binary",
        "max_samples": None,
    },
    "ball_receipt": {
        "type_name": "Ball Receipt*",
        "label_fn": lambda df: (df["ball_receipt_outcome_name"].isna()).astype(int).values,
        "event_feats": BASE_EVENT + ["under_pressure"],
        "event_onehot": {},
        "extra_360": ["sb_ball_receipt_in_space", "sb_ball_receipt_exceeds_distance"],
        "smap_tag": "ball_receipt",
        "task_type": "binary",
        "max_samples": 300000,
    },
    "shot": {
        "type_name": "Shot",
        "label_fn": lambda df: (df["shot_outcome_name"] == "Goal").astype(int).values,
        "filter_fn": lambda df: df[df["shot_type_name"] != "Penalty"],
        "event_feats": BASE_EVENT + ["under_pressure"],
        "event_onehot": {"shot_body_part_name": ["Right Foot", "Left Foot", "Head"]},
        "extra_360": ["ux_dist_defender_end", "ux_nb_opp_in_path"],
        "smap_tag": "shot",
        "task_type": "binary",
        "max_samples": None,
    },
    "duel": {
        "type_name": "Duel",
        "label_fn": lambda df: df["duel_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values,
        "event_feats": BASE_EVENT + ["under_pressure", "duration"],
        "event_onehot": {},
        "extra_360": [],
        "smap_tag": "duel",
        "task_type": "binary",
        "max_samples": None,
    },
    "interception": {
        "type_name": "Interception",
        "label_fn": lambda df: df["interception_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values,
        "event_feats": BASE_EVENT + ["under_pressure", "duration"],
        "event_onehot": {},
        "extra_360": [],
        "smap_tag": "interception",
        "task_type": "binary",
        "max_samples": None,
    },
    "ball_recovery": {
        "type_name": "Ball Recovery",
        "label_fn": lambda df: (df["ball_recovery_failure"] != True).astype(int).values,
        "event_feats": BASE_EVENT + ["under_pressure", "duration"],
        "event_onehot": {},
        "extra_360": [],
        "smap_tag": "ball_recovery",
        "task_type": "binary",
        "max_samples": 300000,
    },
    "pressure": {
        "type_name": "Pressure",
        "label_fn": None,  # special handling: compute_pressure_labels
        "event_feats": BASE_EVENT + ["duration"],
        "event_onehot": {},
        "extra_360": [],
        "smap_tag": "pressure",
        "task_type": "binary",
        "max_samples": 300000,
    },
    "xg": {
        "type_name": "Shot",
        "label_fn": lambda df: df["shot_statsbomb_xg"].fillna(0).values,
        "filter_fn": lambda df: df[df["shot_type_name"] != "Penalty"],
        "event_feats": BASE_EVENT + ["under_pressure"],
        "event_onehot": {"shot_body_part_name": ["Right Foot", "Left Foot", "Head"]},
        "extra_360": ["ux_dist_defender_end", "ux_nb_opp_in_path"],
        "smap_tag": "shot",
        "task_type": "regression",
        "max_samples": None,
    },
}


# ── Model definitions ──────────────────────────────────────────────────────────

class SoccerMapEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(256, output_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


class CNNMLPModel(nn.Module):
    def __init__(self, tab_dim, cnn_channels=7, embed_dim=64, dropout=0.3):
        super().__init__()
        self.event_mlp = nn.Sequential(
            nn.Linear(tab_dim, embed_dim), nn.ELU(), nn.Dropout(dropout),
        )
        self.cnn = SoccerMapEncoder(cnn_channels, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x_tab, x_smap):
        h_tab = self.event_mlp(x_tab)
        h_cnn = self.cnn(x_smap)
        return self.head(torch.cat([h_tab, h_cnn], -1)).squeeze(-1)


class SpatialGatingModel(nn.Module):
    """GMU Gating (Arevalo et al., ICLR Workshop 2017, p.5)"""

    def __init__(self, tab_dim, cnn_channels=7, embed_dim=64, dropout=0.3):
        super().__init__()
        self.event_enc = nn.Sequential(
            nn.Linear(tab_dim, embed_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
        )
        self.spatial_enc = SoccerMapEncoder(cnn_channels, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.pred = nn.Sequential(nn.Linear(embed_dim, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x_tab, x_smap):
        h_e = self.event_enc(x_tab)
        h_s = self.spatial_enc(x_smap)
        g = self.gate(torch.cat([h_e, h_s], -1))
        h = g * h_s + (1 - g) * h_e
        return self.pred(h).squeeze(-1)

    def get_gate_values(self, x_tab, x_smap):
        with torch.no_grad():
            h_e = self.event_enc(x_tab)
            h_s = self.spatial_enc(x_smap)
            return self.gate(torch.cat([h_e, h_s], -1))


class MLPModel(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActionDataset(Dataset):
    def __init__(self, X, y, smaps=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.smaps = smaps

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.smaps is not None:
            return self.X[idx], torch.tensor(self.smaps[idx].copy(), dtype=torch.float32), self.y[idx]
        return self.X[idx], self.y[idx]


# ── Training functions ──────────────────────────────────────────────────────────

def train_nn(model, train_loader, X_val, y_val, val_smaps,
             use_cnn, task_type, epochs=30, lr=1e-3, patience=7):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if task_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        _base_mse = nn.MSELoss()
        loss_fn = lambda p, y: _base_mse(torch.sigmoid(p), y)

    best_metric = -1e9
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if use_cnn:
                tab, smap, label = batch
                pred = model(tab, smap)
            else:
                tab, label = batch
                pred = model(tab)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if use_cnn and val_smaps is not None:
                val_pred = model(X_val, val_smaps).numpy()
            else:
                val_pred = model(X_val).numpy()

        if task_type == "binary":
            val_prob = 1 / (1 + np.exp(-val_pred))
            metric = roc_auc_score(y_val, val_prob)
        else:
            val_xg = 1 / (1 + np.exp(-val_pred))
            metric = -mean_absolute_error(y_val, val_xg)

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_metric


def evaluate_model(model, X_test, y_test, test_smaps, use_cnn, task_type):
    model.eval()
    with torch.no_grad():
        if use_cnn and test_smaps is not None:
            pred = model(X_test, test_smaps).numpy()
        else:
            pred = model(X_test).numpy()

    if task_type == "binary":
        prob = 1 / (1 + np.exp(-pred))
        return {"test_auc": float(roc_auc_score(y_test, prob))}
    else:
        xg_pred = 1 / (1 + np.exp(-pred))
        return {
            "test_mae": float(mean_absolute_error(y_test, xg_pred)),
            "test_r2": float(r2_score(y_test, xg_pred)),
        }


# ── T8 Pressure labels ─────────────────────────────────────────────────

def compute_pressure_labels(df):
    df = df.reset_index(drop=True)
    pressure_mask = df["type_name"] == "Pressure"
    pressure_indices = df.index[pressure_mask].values
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
            if ni >= len(df):
                break
            if all_match[ni] != mid:
                break
            if not is_onball[ni]:
                continue
            onball_count += 1
            if all_poss[ni] != team:
                labels[i] = 1
                break
            if onball_count >= 2:
                break
    return labels


# ── Main logic ────────────────────────────────────────────────────────────

def main():
    t0_all = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading data ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)
    df["pass_is_progressive"] = df["pass_is_progressive"].fillna(False).astype(int)
    for col in ["pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[col] = df[col].fillna(False).astype(int)
    print(f"  Events: {len(df):,}")

    # T8 label pre-computation
    print("Computing T8 Pressure labels ...")
    pressure_labels = compute_pressure_labels(df)
    print(f"  Pressure turnover rate: {pressure_labels.mean():.4f}")

    only_tasks = os.environ.get("ONLY_TASKS")
    only_tasks_set = set(only_tasks.split(",")) if only_tasks else None
    only_models = os.environ.get("ONLY_MODELS")
    only_models_set = set(only_models.split(",")) if only_models else None
    merge_results = os.environ.get("MERGE_RESULTS", "0") == "1"

    all_results = []

    for task_name, cfg in TASK_CONFIG.items():
        if only_tasks_set and task_name not in only_tasks_set:
            continue
        print(f"\n{'='*60}")
        print(f"TASK: {task_name.upper()}")
        print(f"{'='*60}")

        # Filter events
        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if "filter_fn" in cfg:
            df_task = cfg["filter_fn"](df_task)

        # Labels
        if task_name == "pressure":
            df_task = df_task.reset_index(drop=True)
            labels = pressure_labels
        else:
            labels = cfg["label_fn"](df_task)

        # One-hot encoding
        for col, vals in cfg.get("event_onehot", {}).items():
            for v in vals:
                df_task[f"{col}_{v}"] = (df_task[col] == v).astype(int)

        # Feature columns
        event_feats = list(cfg["event_feats"])
        for col, vals in cfg.get("event_onehot", {}).items():
            event_feats += [f"{col}_{v}" for v in vals]

        all_360 = SB_360_COMMON + cfg.get("extra_360", [])

        # Keep only columns that actually exist
        event_feats = [f for f in event_feats if f in df_task.columns]
        all_360 = [f for f in all_360 if f in df_task.columns]

        # SoccerMap
        smap_tag = cfg.get("smap_tag")
        smap_lookup = {}
        smaps_all = None
        if smap_tag:
            idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
            npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
            if idx_path.exists() and npy_path.exists():
                df_idx = pd.read_parquet(idx_path)
                smaps_all = np.load(npy_path, mmap_mode="r")
                smap_lookup = {eid: i for i, eid in enumerate(df_idx["event_id"].values)}

        # Keep only events that have a SoccerMap (for tasks that use the CNN)
        if smap_lookup:
            has_smap = df_task["event_id"].isin(smap_lookup).values
            df_task = df_task[has_smap].reset_index(drop=True)
            labels = labels[has_smap]

        # Sub-sampling
        max_s = cfg.get("max_samples")
        if max_s and len(df_task) > max_s:
            idx_sub = np.random.choice(len(df_task), max_s, replace=False)
            idx_sub.sort()
            df_task = df_task.iloc[idx_sub].reset_index(drop=True)
            labels = labels[idx_sub]

        print(f"  Samples: {len(df_task):,}, Label mean: {labels.mean():.4f}")
        print(f"  Event feats: {len(event_feats)}, 360 feats: {len(all_360)}")

        # Time-series split
        is_train_season = df_task["season_dir"].isin(TRAIN_SEASONS).values
        is_test_season = df_task["season_dir"].isin(TEST_SEASONS).values

        train_matches = df_task.loc[is_train_season, "match_id"].unique()
        np.random.shuffle(train_matches)
        n_t = int(0.8 * len(train_matches))
        train_mask = df_task["match_id"].isin(set(train_matches[:n_t])).values
        val_mask = df_task["match_id"].isin(set(train_matches[n_t:])).values
        test_mask = is_test_season

        y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]
        print(f"  Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")

        # Feature matrices
        X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
        X_360 = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
        X_event_360 = np.hstack([X_event, X_360])

        # Scale
        scaler_e = StandardScaler().fit(X_event[train_mask])
        scaler_e360 = StandardScaler().fit(X_event_360[train_mask])

        Xe_tr, Xe_va, Xe_te = scaler_e.transform(X_event[train_mask]), scaler_e.transform(X_event[val_mask]), scaler_e.transform(X_event[test_mask])
        Xe360_tr, Xe360_va, Xe360_te = scaler_e360.transform(X_event_360[train_mask]), scaler_e360.transform(X_event_360[val_mask]), scaler_e360.transform(X_event_360[test_mask])

        # SoccerMap tensors
        sm_train = sm_val = sm_test = None
        if smap_lookup and smaps_all is not None:
            eids = df_task["event_id"].values
            smap_indices = np.array([smap_lookup[eid] for eid in eids])
            sm_train = smaps_all[smap_indices[train_mask]]
            sm_val = torch.tensor(smaps_all[smap_indices[val_mask]].copy(), dtype=torch.float32)
            sm_test = torch.tensor(smaps_all[smap_indices[test_mask]].copy(), dtype=torch.float32)

        task_type = cfg["task_type"]

        # ── Run models ──
        model_configs = [
            ("L1_LR_Event", "lr", Xe_tr, Xe_va, Xe_te, False),
            ("L2_LR_360", "lr", Xe360_tr, Xe360_va, Xe360_te, False),
            ("B1_XGB_Event", "xgb", Xe_tr, Xe_va, Xe_te, False),
            ("B2_XGB_360", "xgb", Xe360_tr, Xe360_va, Xe360_te, False),
            ("M1_MLP_Event", "mlp", Xe_tr, Xe_va, Xe_te, False),
            ("M2_MLP_360", "mlp", Xe360_tr, Xe360_va, Xe360_te, False),
        ]
        if smap_lookup and smaps_all is not None:
            model_configs += [
                ("M3_CNN_Event", "cnn", Xe_tr, Xe_va, Xe_te, True),
                ("M4_CNN_Full", "cnn", Xe360_tr, Xe360_va, Xe360_te, True),
                ("G1_Gating", "gating", Xe360_tr, Xe360_va, Xe360_te, True),
            ]

        for model_name, model_type, X_tr, X_va, X_te, use_cnn in model_configs:
            if only_models_set and model_name not in only_models_set:
                continue
            t0 = time.time()
            print(f"\n  {model_name} ...", end=" ", flush=True)

            result = {"task": task_name, "model": model_name, "task_type": task_type}

            if model_type == "lr":
                if task_type == "binary":
                    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
                    pipe.fit(X_tr, y_train)
                    result["val_auc"] = float(roc_auc_score(y_val, pipe.predict_proba(X_va)[:, 1]))
                    result["test_auc"] = float(roc_auc_score(y_test, pipe.predict_proba(X_te)[:, 1]))
                else:
                    pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
                    pipe.fit(X_tr, y_train)
                    pred_te = pipe.predict(X_te)
                    result["test_mae"] = float(mean_absolute_error(y_test, pred_te))
                    result["test_r2"] = float(r2_score(y_test, pred_te))

            elif model_type == "xgb":
                if task_type == "binary":
                    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                        eval_metric="auc", tree_method="hist", random_state=42, verbosity=0)
                    xgb.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
                    result["val_auc"] = float(roc_auc_score(y_val, xgb.predict_proba(X_va)[:, 1]))
                    result["test_auc"] = float(roc_auc_score(y_test, xgb.predict_proba(X_te)[:, 1]))
                else:
                    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                       eval_metric="mae", tree_method="hist", random_state=42, verbosity=0)
                    xgb.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
                    pred_te = xgb.predict(X_te)
                    result["test_mae"] = float(mean_absolute_error(y_test, pred_te))
                    result["test_r2"] = float(r2_score(y_test, pred_te))

            elif model_type == "mlp":
                model = MLPModel(X_tr.shape[1])
                ds = ActionDataset(X_tr, y_train)
                dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                X_va_t = torch.tensor(X_va, dtype=torch.float32)
                X_te_t = torch.tensor(X_te, dtype=torch.float32)
                train_nn(model, dl, X_va_t, y_val, None, False, task_type)
                metrics = evaluate_model(model, X_te_t, y_test, None, False, task_type)
                result.update(metrics)

            elif model_type == "cnn":
                model = CNNMLPModel(X_tr.shape[1])
                ds = ActionDataset(X_tr, y_train, sm_train)
                dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                X_va_t = torch.tensor(X_va, dtype=torch.float32)
                X_te_t = torch.tensor(X_te, dtype=torch.float32)
                train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                result.update(metrics)

            elif model_type == "gating":
                model = SpatialGatingModel(X_tr.shape[1])
                ds = ActionDataset(X_tr, y_train, sm_train)
                dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                X_va_t = torch.tensor(X_va, dtype=torch.float32)
                X_te_t = torch.tensor(X_te, dtype=torch.float32)
                train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                result.update(metrics)
                # Gate values
                g = model.get_gate_values(X_te_t, sm_test).numpy()
                result["gate_mean"] = float(g.mean())
                result["gate_std"] = float(g.std())

            elapsed = time.time() - t0
            result["time_s"] = round(elapsed, 1)

            if task_type == "binary":
                print(f"test_auc={result.get('test_auc', 'N/A'):.4f}  ({elapsed:.0f}s)")
            else:
                print(f"test_mae={result.get('test_mae', 'N/A'):.4f}  ({elapsed:.0f}s)")

            all_results.append(result)

    # ── Save ──
    output_path = CACHE_DIR / "results_all.json"
    if merge_results and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        retrain_keys = {(r["task"], r["model"]) for r in all_results}
        merged = [r for r in existing if (r["task"], r["model"]) not in retrain_keys]
        merged.extend(all_results)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\n[MERGE] replaced {len(all_results)} entries, kept {len(merged) - len(all_results)} existing")
    else:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──
    total_time = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"ALL DONE: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Results saved to {output_path}")

    # Print results table
    print(f"\n{'Task':<15} {'Model':<15} {'Test AUC/MAE':>12} {'Time':>6}")
    print("-" * 52)
    for r in all_results:
        metric = r.get("test_auc", r.get("test_mae", "N/A"))
        if isinstance(metric, float):
            metric = f"{metric:.4f}"
        print(f"{r['task']:<15} {r['model']:<15} {metric:>12} {r['time_s']:>5.0f}s")


if __name__ == "__main__":
    main()
