#!/usr/bin/env python3
"""
save_pass_predictions: Pass M4 CNN strict holdout predictions (with event_id) for F5.3
=======================================================================================
F5.3 requires comparing our xPass predictions against StatsBomb's native
pass_success_probability on the same set of events.

Inputs:
  data/L1_events_v3.parquet
  data/action_soccermaps_pass.npy + _idx.parquet

Outputs:
  data/predictions/pass_M4_CNN_2ch_strict_with_ids.npz

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/eval/save_pass_predictions.py

"""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, FULL_SEASONS, SB_360_COMMON,
    TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS, pd, train_nn,
)


def main():
    t0 = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading L1_events_v3.parquet ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    for col in ["pass_is_progressive", "pass_cross", "pass_switch",
                "pass_through_ball", "pass_cut_back"]:
        df[col] = df[col].fillna(False).astype(int)

    cfg = TASK_CONFIG["pass"]
    df_t = df[df["type_name"] == cfg["type_name"]].copy()
    if "filter_fn" in cfg:
        df_t = cfg["filter_fn"](df_t)
    labels = cfg["label_fn"](df_t)

    for col, vals in cfg.get("event_onehot", {}).items():
        for v in vals:
            df_t[f"{col}_{v}"] = (df_t[col] == v).astype(int)

    event_feats = list(cfg["event_feats"])
    for col, vals in cfg.get("event_onehot", {}).items():
        event_feats += [f"{col}_{v}" for v in vals]
    all_360 = SB_360_COMMON + cfg.get("extra_360", [])
    event_feats = [f for f in event_feats if f in df_t.columns]
    all_360 = [f for f in all_360 if f in df_t.columns]

    smap_tag = cfg.get("smap_tag", "pass")
    idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
    npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
    idx_df = pd.read_parquet(idx_path)
    smaps = np.load(npy_path, mmap_mode="r")
    id_map = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}
    has = df_t["event_id"].isin(id_map).values
    df_t = df_t[has].reset_index(drop=True)
    labels = labels[has]

    max_s = cfg.get("max_samples")
    if max_s and len(df_t) > max_s:
        idx_sub = np.random.choice(len(df_t), max_s, replace=False)
        idx_sub.sort()
        df_t = df_t.iloc[idx_sub].reset_index(drop=True)
        labels = labels[idx_sub]

    print(f"Pass: {len(df_t):,} samples, pos_rate={labels.mean():.4f}")

    is_train = df_t["season_dir"].isin(TRAIN_SEASONS).values
    is_test = df_t["season_dir"].isin(TEST_SEASONS).values
    train_matches = df_t.loc[is_train, "match_id"].unique()
    np.random.shuffle(train_matches)
    n_t = int(0.8 * len(train_matches))
    train_mask = df_t["match_id"].isin(set(train_matches[:n_t])).values
    val_mask = df_t["match_id"].isin(set(train_matches[n_t:])).values
    test_mask = is_test

    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]

    X_event = df_t[event_feats].fillna(0).values.astype(np.float32)
    X_360 = df_t[all_360].fillna(0).values.astype(np.float32) if all_360 else \
            np.empty((len(df_t), 0), dtype=np.float32)
    X_e360 = np.hstack([X_event, X_360])
    scaler = StandardScaler().fit(X_e360[train_mask])
    Xtr = scaler.transform(X_e360[train_mask])
    Xva = scaler.transform(X_e360[val_mask])
    Xte = scaler.transform(X_e360[test_mask])

    eids = df_t["event_id"].values
    sm_idx = np.array([id_map[e] for e in eids])
    sm_tr = smaps[sm_idx[train_mask]][:, :2, :, :]
    sm_va = torch.tensor(smaps[sm_idx[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
    sm_te = torch.tensor(smaps[sm_idx[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

    print("Training M4 CNN 2ch on Pass (strict holdout) ...")
    torch.manual_seed(42)
    model = CNNMLPModel(Xtr.shape[1], cnn_channels=2)
    ds = ActionDataset(Xtr, y_train, sm_tr)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    train_nn(model, dl, Xva_t, y_val, sm_va, True, "binary")

    model.eval()
    with torch.no_grad():
        z_te = model(Xte_t, sm_te).numpy()
    p_te = 1.0 / (1.0 + np.exp(-z_te))

    PRED_DIR = CACHE_DIR / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / "pass_M4_CNN_2ch_strict_with_ids.npz"
    np.savez(out,
             y_true=y_test.astype(np.float32),
             logits=z_te.astype(np.float32),
             probs=p_te.astype(np.float32),
             event_id=df_t.loc[test_mask, "event_id"].values,
             match_id=df_t.loc[test_mask, "match_id"].values,
             season_dir=df_t.loc[test_mask, "season_dir"].values)

    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(y_test, p_te)
    brier = brier_score_loss(y_test, p_te)
    print(f"Test AUC={auc:.4f}  Brier={brier:.4f}")
    print(f"Wrote {out}")
    print(f"Done: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
