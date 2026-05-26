#!/usr/bin/env python3
"""
relaxed_holdout_dribble_duel: test whether a random 24/25 holdout fixes ECE
==============================================================================
E1 found the drift to be a one-off discrete event at 23/24->24/25 (SB pipeline change).
E3 hypothesis: if the model is given some 24/25 training data it should learn the new
distribution and ECE should return to normal.

Setup:
  Train: all of 22/23 + 23/24 + a random 80% of 24/25 matches (match-level)
  Val:   20% drawn from the 22/23+23/24 portion of train (original logic preserved)
  Test:  the remaining 20% of 24/25 matches

Comparison:
  Original holdout (22/23+23/24 train, 24/25 test) : previously ECE=0.24 for Dribble CNN
  Relaxed holdout (with 80% of 24/25 added)        : ECE expected to be close to val

Runs only Dribble + Duel x M4 + G1, four combinations in total.

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/diagnostic/relaxed_holdout_dribble_duel.py

"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, FULL_SEASONS, SB_360_COMMON,
    SpatialGatingModel, TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS,
    pd, train_nn,
)
from scripts.eval.eval_calibration_sweep import expected_calibration_error


TARGETS = [("dribble", "M4_CNN_2ch"), ("dribble", "G1_Gating_2ch"),
           ("duel", "M4_CNN_2ch"), ("duel", "G1_Gating_2ch")]


def sig(z): return 1.0 / (1.0 + np.exp(-z))


def build_relaxed_split(df_task):
    """Relaxed holdout: 22/23+23/24 all + 80% of 24/25 matches → train pool.
    From train pool, hold 20% of 22/23+23/24 matches as val.
    Test = remaining 20% of 24/25 matches."""
    rng = np.random.default_rng(42)
    is_2425 = df_task["season_dir"].isin(TEST_SEASONS).values
    is_old = df_task["season_dir"].isin(TRAIN_SEASONS).values

    matches_2425 = df_task.loc[is_2425, "match_id"].unique()
    rng.shuffle(matches_2425)
    n_calib = int(0.8 * len(matches_2425))
    calib_matches = set(matches_2425[:n_calib])  # added to train
    test_matches = set(matches_2425[n_calib:])   # strict holdout

    matches_old = df_task.loc[is_old, "match_id"].unique()
    rng.shuffle(matches_old)
    n_tr_old = int(0.8 * len(matches_old))
    train_old = set(matches_old[:n_tr_old])
    val_old = set(matches_old[n_tr_old:])

    mid = df_task["match_id"].values
    train_mask = np.isin(mid, list(train_old | calib_matches))
    val_mask = np.isin(mid, list(val_old))
    test_mask = np.isin(mid, list(test_matches))
    return train_mask, val_mask, test_mask


def prepare_task(df, task_name):
    cfg = TASK_CONFIG[task_name]
    df_task = df[df["type_name"] == cfg["type_name"]].copy()
    if "filter_fn" in cfg:
        df_task = cfg["filter_fn"](df_task)
    labels = cfg["label_fn"](df_task)

    for col, vals in cfg.get("event_onehot", {}).items():
        for v in vals:
            df_task[f"{col}_{v}"] = (df_task[col] == v).astype(int)

    event_feats = list(cfg["event_feats"])
    for col, vals in cfg.get("event_onehot", {}).items():
        event_feats += [f"{col}_{v}" for v in vals]
    all_360 = SB_360_COMMON + cfg.get("extra_360", [])
    event_feats = [f for f in event_feats if f in df_task.columns]
    all_360 = [f for f in all_360 if f in df_task.columns]

    smap_tag = cfg.get("smap_tag", task_name)
    idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
    npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
    idx_df = pd.read_parquet(idx_path)
    smaps = np.load(npy_path, mmap_mode="r")
    id_map = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}
    has_smap = df_task["event_id"].isin(id_map).values
    df_task = df_task[has_smap].reset_index(drop=True)
    labels = labels[has_smap]

    train_mask, val_mask, test_mask = build_relaxed_split(df_task)

    X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
    X_360 = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
    X_e360 = np.hstack([X_event, X_360])

    scaler = StandardScaler().fit(X_e360[train_mask])
    Xe360_tr = scaler.transform(X_e360[train_mask])
    Xe360_va = scaler.transform(X_e360[val_mask])
    Xe360_te = scaler.transform(X_e360[test_mask])

    eids = df_task["event_id"].values
    sm_idx = np.array([id_map[e] for e in eids])
    sm_tr = smaps[sm_idx[train_mask]][:, :2, :, :]
    sm_va = torch.tensor(smaps[sm_idx[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
    sm_te = torch.tensor(smaps[sm_idx[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

    return dict(
        Xe360_tr=Xe360_tr, Xe360_va=Xe360_va, Xe360_te=Xe360_te,
        sm_tr=sm_tr, sm_va=sm_va, sm_te=sm_te,
        y_tr=labels[train_mask], y_va=labels[val_mask], y_te=labels[test_mask],
        match_te=df_task.loc[test_mask, "match_id"].values,
        season_te=df_task.loc[test_mask, "season_dir"].values,
        event_te=df_task.loc[test_mask, "event_id"].values,
    )


def train_and_eval(model_name, d):
    torch.manual_seed(42)
    if model_name == "M4_CNN_2ch":
        model = CNNMLPModel(d["Xe360_tr"].shape[1], cnn_channels=2)
    else:
        model = SpatialGatingModel(d["Xe360_tr"].shape[1], cnn_channels=2)
    ds = ActionDataset(d["Xe360_tr"], d["y_tr"], d["sm_tr"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va = torch.tensor(d["Xe360_va"], dtype=torch.float32)
    X_te = torch.tensor(d["Xe360_te"], dtype=torch.float32)
    train_nn(model, dl, X_va, d["y_va"], d["sm_va"], True, "binary")
    model.eval()
    with torch.no_grad():
        z_va = model(X_va, d["sm_va"]).numpy()
        z_te = model(X_te, d["sm_te"]).numpy()
    p_va = sig(z_va); p_te = sig(z_te)
    return_preds = {"z_va": z_va, "p_va": p_va, "z_te": z_te, "p_te": p_te}
    return return_preds, {
        "val": dict(
            auc=float(roc_auc_score(d["y_va"], p_va)),
            brier=float(brier_score_loss(d["y_va"], p_va)),
            ece=expected_calibration_error(d["y_va"], p_va, 15),
            mean_y=float(d["y_va"].mean()), mean_p=float(p_va.mean()),
        ),
        "test": dict(
            auc=float(roc_auc_score(d["y_te"], p_te)),
            brier=float(brier_score_loss(d["y_te"], p_te)),
            ece=expected_calibration_error(d["y_te"], p_te, 15),
            mean_y=float(d["y_te"].mean()), mean_p=float(p_te.mean()),
        ),
    }


def main():
    t0 = time.time()
    np.random.seed(42); torch.manual_seed(42)

    print("Loading ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)

    data = {}
    for task in sorted({t for t, _ in TARGETS}):
        print(f"\n→ Preparing relaxed split for {task}")
        data[task] = prepare_task(df, task)
        d = data[task]
        print(f"  train={len(d['y_tr'])}  val={len(d['y_va'])}  test={len(d['y_te'])}"
              f"  pos_rate_train={d['y_tr'].mean():.3f}")

    PRED_OUT = CACHE_DIR / "predictions"
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for task, model_name in TARGETS:
        print(f"\n{'='*60}\n{task.upper()} × {model_name}  (relaxed holdout)\n{'='*60}")
        t1 = time.time()
        preds, r = train_and_eval(model_name, data[task])
        dt = time.time() - t1
        print(f"  trained in {dt:.0f}s")
        print(f"  VAL:  AUC={r['val']['auc']:.4f} Brier={r['val']['brier']:.4f} "
              f"ECE={r['val']['ece']:.4f}  mean_y={r['val']['mean_y']:.3f} mean_p={r['val']['mean_p']:.3f}")
        print(f"  TEST: AUC={r['test']['auc']:.4f} Brier={r['test']['brier']:.4f} "
              f"ECE={r['test']['ece']:.4f}  mean_y={r['test']['mean_y']:.3f} mean_p={r['test']['mean_p']:.3f}")
        results.append({"task": task, "model": model_name, "train_s": round(dt, 1), **r})

        d = data[task]
        np.savez(PRED_OUT / f"{task}_{model_name}_relaxed.npz",
                 y_test=d["y_te"].astype(np.float32),
                 logits=preds["z_te"].astype(np.float32),
                 probs=preds["p_te"].astype(np.float32),
                 match_id=d["match_te"],
                 season_dir=d["season_te"],
                 event_id=d["event_te"])

    out = CACHE_DIR / "results_relaxed_holdout.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone: {time.time()-t0:.0f}s. Results → {out}")


if __name__ == "__main__":
    main()
