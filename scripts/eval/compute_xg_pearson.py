#!/usr/bin/env python3
"""
compute_xg_pearson: Retrain xG models + compute Pearson correlation
=====================================================================
Main paper L364 reports xG MAE; supp Table S9 reports MAE + R². This
script adds a Pearson correlation column by retraining the 9 single-task
xG models (L1, L2, B1, B2, M1, M2, M3, M4, G1) and pulling U1 xG
predictions from cache (u1_per_task.npz).

Pearson is computed between model prediction and StatsBomb xG label
(shot_statsbomb_xg). Since the training label IS shot_statsbomb_xg,
Pearson is a goodness-of-fit measure complementing R².

Inputs: data/L1_events_v3.parquet
        data/action_soccermaps_shot.npy + _idx.parquet
        data/predictions/u1_per_task.npz  (U1 only)
Outputs: data/results_xg_pearson.json

Usage:
  PYTHONPATH=. python scripts/eval/compute_xg_pearson.py

"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, FULL_SEASONS, MLPModel,
    SB_360_COMMON, SpatialGatingModel, TASK_CONFIG, TEST_SEASONS,
    TRAIN_SEASONS, train_nn,
)


def prepare_xg_data():
    cfg = TASK_CONFIG["xg"]
    print("Loading L1 events ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)

    df_task = df[df["type_name"] == cfg["type_name"]].copy()
    df_task = cfg["filter_fn"](df_task)
    y = cfg["label_fn"](df_task)

    for col, vals in cfg.get("event_onehot", {}).items():
        for v in vals:
            df_task[f"{col}_{v}"] = (df_task[col] == v).astype(int)

    event_feats = list(cfg["event_feats"])
    for col, vals in cfg.get("event_onehot", {}).items():
        event_feats += [f"{col}_{v}" for v in vals]
    all_360 = SB_360_COMMON + cfg.get("extra_360", [])
    event_feats = [f for f in event_feats if f in df_task.columns]
    all_360 = [f for f in all_360 if f in df_task.columns]

    train_mask = df_task["season_dir"].isin(TRAIN_SEASONS).values
    test_mask = df_task["season_dir"].isin(TEST_SEASONS).values
    val_size = int(0.2 * train_mask.sum())
    train_idx = np.where(train_mask)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(train_idx)
    val_idx = set(train_idx[:val_size].tolist())
    train_only = ~np.isin(np.arange(len(df_task)), list(val_idx)) & train_mask
    val_mask_arr = np.zeros(len(df_task), dtype=bool)
    val_mask_arr[list(val_idx)] = True

    smap_tag = cfg.get("smap_tag", "xg")
    idx_df = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet")
    smaps = np.load(CACHE_DIR / f"action_soccermaps_{smap_tag}.npy", mmap_mode="r")
    id_map = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}
    has_smap = df_task["event_id"].isin(id_map).values
    df_task = df_task[has_smap].reset_index(drop=True)
    y = y[has_smap]
    train_only = train_only[has_smap]
    val_mask_arr = val_mask_arr[has_smap]
    test_mask = test_mask[has_smap]

    X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
    X_360 = df_task[all_360].fillna(0).values.astype(np.float32)
    X_e360 = np.hstack([X_event, X_360])

    se = StandardScaler().fit(X_event[train_only])
    se360 = StandardScaler().fit(X_e360[train_only])
    Xe_tr, Xe_va, Xe_te = se.transform(X_event[train_only]), se.transform(X_event[val_mask_arr]), se.transform(X_event[test_mask])
    Xe360_tr, Xe360_va, Xe360_te = se360.transform(X_e360[train_only]), se360.transform(X_e360[val_mask_arr]), se360.transform(X_e360[test_mask])

    eids = df_task["event_id"].values
    sm_idx = np.array([id_map[e] for e in eids])
    sm_tr = smaps[sm_idx[train_only]]
    sm_va = torch.tensor(smaps[sm_idx[val_mask_arr]].copy(), dtype=torch.float32)
    sm_te = torch.tensor(smaps[sm_idx[test_mask]].copy(), dtype=torch.float32)

    return dict(
        y_tr=y[train_only], y_va=y[val_mask_arr], y_te=y[test_mask],
        Xe_tr=Xe_tr, Xe_va=Xe_va, Xe_te=Xe_te,
        Xe360_tr=Xe360_tr, Xe360_va=Xe360_va, Xe360_te=Xe360_te,
        sm_tr=sm_tr, sm_va=sm_va, sm_te=sm_te,
        event_te=df_task.loc[test_mask, "event_id"].values,
        n_train=int(train_only.sum()), n_val=int(val_mask_arr.sum()),
        n_test=int(test_mask.sum()),
    )


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    t0 = time.time()
    np.random.seed(42); torch.manual_seed(42)

    d = prepare_xg_data()
    print(f"  train={d['n_train']:,}  val={d['n_val']:,}  test={d['n_test']:,}")

    y_te = d["y_te"]

    def metrics(pred):
        m = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))
        r, _ = pearsonr(y_te, pred)
        return {"mae": m, "r2": r2, "pearson": float(r)}

    results = []

    # L1: Ridge on event only
    print("\n[L1] Ridge Event ...", end=" ", flush=True); t = time.time()
    pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
    pipe.fit(d["Xe_tr"], d["y_tr"])
    results.append({"model": "L1_LR_Event", **metrics(pipe.predict(d["Xe_te"])),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # L2: Ridge on event+360
    print("[L2] Ridge 360 ...", end=" ", flush=True); t = time.time()
    pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
    pipe.fit(d["Xe360_tr"], d["y_tr"])
    results.append({"model": "L2_LR_360", **metrics(pipe.predict(d["Xe360_te"])),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # B1: XGB event
    print("[B1] XGB Event ...", end=" ", flush=True); t = time.time()
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                       eval_metric="mae", tree_method="hist", random_state=42, verbosity=0)
    xgb.fit(d["Xe_tr"], d["y_tr"], eval_set=[(d["Xe_va"], d["y_va"])], verbose=False)
    results.append({"model": "B1_XGB_Event", **metrics(xgb.predict(d["Xe_te"])),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # B2: XGB 360
    print("[B2] XGB 360 ...", end=" ", flush=True); t = time.time()
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                       eval_metric="mae", tree_method="hist", random_state=42, verbosity=0)
    xgb.fit(d["Xe360_tr"], d["y_tr"], eval_set=[(d["Xe360_va"], d["y_va"])], verbose=False)
    results.append({"model": "B2_XGB_360", **metrics(xgb.predict(d["Xe360_te"])),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # M1: MLP event
    print("[M1] MLP Event ...", end=" ", flush=True); t = time.time()
    torch.manual_seed(42)
    model = MLPModel(d["Xe_tr"].shape[1])
    ds = ActionDataset(d["Xe_tr"], d["y_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(d["Xe_va"], dtype=torch.float32)
    X_te_t = torch.tensor(d["Xe_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, d["y_va"], None, False, "regression")
    model.eval()
    with torch.no_grad():
        pred = sig(model(X_te_t).numpy())
    results.append({"model": "M1_MLP_Event", **metrics(pred),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # M2: MLP 360
    print("[M2] MLP 360 ...", end=" ", flush=True); t = time.time()
    torch.manual_seed(42)
    model = MLPModel(d["Xe360_tr"].shape[1])
    ds = ActionDataset(d["Xe360_tr"], d["y_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(d["Xe360_va"], dtype=torch.float32)
    X_te_t = torch.tensor(d["Xe360_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, d["y_va"], None, False, "regression")
    model.eval()
    with torch.no_grad():
        pred = sig(model(X_te_t).numpy())
    results.append({"model": "M2_MLP_360", **metrics(pred),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # M3: CNN event
    print("[M3] CNN Event ...", end=" ", flush=True); t = time.time()
    torch.manual_seed(42)
    model = CNNMLPModel(d["Xe_tr"].shape[1])
    ds = ActionDataset(d["Xe_tr"], d["y_tr"], d["sm_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(d["Xe_va"], dtype=torch.float32)
    X_te_t = torch.tensor(d["Xe_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, d["y_va"], d["sm_va"], True, "regression")
    model.eval()
    with torch.no_grad():
        pred = sig(model(X_te_t, d["sm_te"]).numpy())
    results.append({"model": "M3_CNN_Event", **metrics(pred),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # M4: CNN full
    print("[M4] CNN Full ...", end=" ", flush=True); t = time.time()
    torch.manual_seed(42)
    model = CNNMLPModel(d["Xe360_tr"].shape[1])
    ds = ActionDataset(d["Xe360_tr"], d["y_tr"], d["sm_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(d["Xe360_va"], dtype=torch.float32)
    X_te_t = torch.tensor(d["Xe360_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, d["y_va"], d["sm_va"], True, "regression")
    model.eval()
    with torch.no_grad():
        pred = sig(model(X_te_t, d["sm_te"]).numpy())
    results.append({"model": "M4_CNN_Full", **metrics(pred),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # G1: Gating
    print("[G1] Gating ...", end=" ", flush=True); t = time.time()
    torch.manual_seed(42)
    model = SpatialGatingModel(d["Xe360_tr"].shape[1])
    ds = ActionDataset(d["Xe360_tr"], d["y_tr"], d["sm_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(d["Xe360_va"], dtype=torch.float32)
    X_te_t = torch.tensor(d["Xe360_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, d["y_va"], d["sm_va"], True, "regression")
    model.eval()
    with torch.no_grad():
        pred = sig(model(X_te_t, d["sm_te"]).numpy())
    results.append({"model": "G1_Gating", **metrics(pred),
                    "train_s": round(time.time() - t, 1)})
    print(f"{time.time()-t:.1f}s")

    # U1: pull from u1_per_task.npz (regression mask)
    print("\n[U1] from u1_per_task.npz cache ...")
    u = np.load(CACHE_DIR / "predictions" / "u1_per_task.npz", allow_pickle=True)
    xg_mask = u["is_regression"] == 1
    u_y = u["y_true"][xg_mask]
    u_p = u["y_prob"][xg_mask]
    u_mae = float(mean_absolute_error(u_y, u_p))
    u_r2 = float(r2_score(u_y, u_p))
    u_r, _ = pearsonr(u_y, u_p)
    results.append({"model": "U1_Unified", "mae": u_mae, "r2": u_r2,
                    "pearson": float(u_r), "n_events_u1": int(xg_mask.sum())})

    out = CACHE_DIR / "results_xg_pearson.json"
    with open(out, "w") as f:
        json.dump({"n_test_single": d["n_test"],
                   "n_test_u1": int(xg_mask.sum()),
                   "models": results}, f, indent=2)
    print(f"\nDone: {time.time()-t0:.0f}s. Saved {out}")

    # Print summary table
    print("\n=== xG Regression: MAE / R² / Pearson (StatsBomb xg) ===")
    print(f"{'Model':<15} {'MAE':>8} {'R²':>8} {'Pearson':>10}")
    for r in results:
        print(f"{r['model']:<15} {r['mae']:>8.4f} {r['r2']:>8.3f} {r['pearson']:>10.3f}")


if __name__ == "__main__":
    main()
