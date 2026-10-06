#!/usr/bin/env python3
"""
fit_temperature: temperature scaling calibration (Guo et al. 2017)
===================================================================
Applies temperature scaling to the 4 severely miscalibrated combinations
identified by the ECE sweep:
  Dribble x M4 CNN
  Dribble x G1 Gating
  Duel    x M4 CNN
  Duel    x G1 Gating

Procedure:
  1. Retrain the model, saving both val and test logits
  2. Fit the single parameter T on val by minimising NLL (scipy bounded)
  3. p_calib = sigmoid(z_test / T); report T, before/after AUC, Brier, ECE

Why retraining is required:
  The previous eval_calibration_sweep only stored test logits, whereas
  temperature scaling requires val-set fit -> test-set eval, otherwise
  there is data leakage.

Inputs:
  data/L1_events_v3.parquet
  data/action_soccermaps_{task}.npy + _idx.parquet

Outputs:
  data/results_temperature.json
  data/predictions/{task}_{model}_val.npz
  data/predictions/{task}_{model}_test_calib.npz

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/eval/fit_temperature.py

"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, FULL_SEASONS, SB_360_COMMON,
    SpatialGatingModel, TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS,
    pd, train_nn,
)
from scripts.eval.eval_calibration_sweep import (
    expected_calibration_error, PRED_DIR,
)


TARGETS = [
    ("dribble", "M4_CNN_2ch"),
    ("dribble", "G1_Gating_2ch"),
    ("duel", "M4_CNN_2ch"),
    ("duel", "G1_Gating_2ch"),
]


def binary_nll_from_logits(logits, labels):
    """Stable BCE-with-logits mean NLL."""
    # log(sigmoid(z)) = -log1p(exp(-z))
    # log(1 - sigmoid(z)) = -z - log1p(exp(-z))
    # combine stably:
    z = logits
    abs_z = np.abs(z)
    log_1p_exp_neg_abs = np.log1p(np.exp(-abs_z))
    log_p = np.where(z >= 0, -log_1p_exp_neg_abs, z - log_1p_exp_neg_abs)
    log_1mp = np.where(z >= 0, -z - log_1p_exp_neg_abs, -log_1p_exp_neg_abs)
    return float(-np.mean(labels * log_p + (1 - labels) * log_1mp))


def fit_temperature(val_logits, val_labels):
    """Minimize val NLL over T via bounded scalar optimizer (Guo 2017)."""
    val_logits = np.asarray(val_logits, dtype=np.float64).ravel()
    val_labels = np.asarray(val_labels, dtype=np.float64).ravel()

    def obj(T):
        return binary_nll_from_logits(val_logits / T, val_labels)

    res = minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x), float(res.fun)


def metrics_from_probs(y_true, probs):
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs, n_bins=15),
    }


def prepare_task_data(df, task_name):
    """Rebuild the exact task split + features used in eval_calibration_sweep."""
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
    df_idx = pd.read_parquet(idx_path)
    smaps_7ch = np.load(npy_path, mmap_mode="r")
    smap_lookup = {eid: i for i, eid in enumerate(df_idx["event_id"].values)}
    has_smap = df_task["event_id"].isin(smap_lookup).values
    df_task = df_task[has_smap].reset_index(drop=True)
    labels = labels[has_smap]

    max_s = cfg.get("max_samples")
    if max_s and len(df_task) > max_s:
        idx_sub = np.random.choice(len(df_task), max_s, replace=False)
        idx_sub.sort()
        df_task = df_task.iloc[idx_sub].reset_index(drop=True)
        labels = labels[idx_sub]

    is_train_season = df_task["season_dir"].isin(TRAIN_SEASONS).values
    is_test_season = df_task["season_dir"].isin(TEST_SEASONS).values
    train_matches = df_task.loc[is_train_season, "match_id"].unique()
    np.random.shuffle(train_matches)
    n_t = int(0.8 * len(train_matches))
    train_mask = df_task["match_id"].isin(set(train_matches[:n_t])).values
    val_mask = df_task["match_id"].isin(set(train_matches[n_t:])).values
    test_mask = is_test_season

    y_train = labels[train_mask]; y_val = labels[val_mask]; y_test = labels[test_mask]

    X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
    X_360 = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
    X_e360 = np.hstack([X_event, X_360])

    scaler = StandardScaler().fit(X_e360[train_mask])
    Xe360_tr = scaler.transform(X_e360[train_mask])
    Xe360_va = scaler.transform(X_e360[val_mask])
    Xe360_te = scaler.transform(X_e360[test_mask])

    eids = df_task["event_id"].values
    smap_indices = np.array([smap_lookup[eid] for eid in eids])
    sm_train = smaps_7ch[smap_indices[train_mask]][:, :2, :, :]
    sm_val = torch.tensor(smaps_7ch[smap_indices[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
    sm_test = torch.tensor(smaps_7ch[smap_indices[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

    return {
        "Xe360_tr": Xe360_tr, "Xe360_va": Xe360_va, "Xe360_te": Xe360_te,
        "sm_train": sm_train, "sm_val": sm_val, "sm_test": sm_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def train_and_capture_logits(model_name, data):
    torch.manual_seed(42)
    if model_name == "M4_CNN_2ch":
        model = CNNMLPModel(data["Xe360_tr"].shape[1], cnn_channels=2)
    elif model_name == "G1_Gating_2ch":
        model = SpatialGatingModel(data["Xe360_tr"].shape[1], cnn_channels=2)
    else:
        raise ValueError(model_name)

    ds = ActionDataset(data["Xe360_tr"], data["y_train"], data["sm_train"])
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(data["Xe360_va"], dtype=torch.float32)
    X_te_t = torch.tensor(data["Xe360_te"], dtype=torch.float32)
    train_nn(model, dl, X_va_t, data["y_val"], data["sm_val"], True, "binary")
    model.eval()
    with torch.no_grad():
        val_logits = model(X_va_t, data["sm_val"]).numpy()
        test_logits = model(X_te_t, data["sm_test"]).numpy()
    return val_logits, test_logits


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    t0 = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading L1_events_v3.parquet ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)
    for col in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[col] = df[col].fillna(False).astype(int)

    results = []

    # Build data per task once, then train both models on it
    tasks_needed = sorted({t for t, _ in TARGETS})
    task_data = {}
    for task in tasks_needed:
        print(f"\n→ Preparing data for {task}")
        # per-task fresh seed so order doesn't matter
        np.random.seed(42 + hash(task) % 1000)
        task_data[task] = prepare_task_data(df, task)
        d = task_data[task]
        print(f"   train={len(d['y_train'])}  val={len(d['y_val'])}  test={len(d['y_test'])}"
              f"  pos_rate_train={d['y_train'].mean():.3f}")

    for task, model_name in TARGETS:
        print(f"\n{'='*60}\n{task.upper()} × {model_name}\n{'='*60}")
        d = task_data[task]
        t_model = time.time()

        val_logits, test_logits = train_and_capture_logits(model_name, d)
        train_s = time.time() - t_model

        # Before calibration
        probs_test_before = sigmoid(test_logits)
        m_before = metrics_from_probs(d["y_test"], probs_test_before)

        # Fit T on val
        T, val_nll = fit_temperature(val_logits, d["y_val"])

        # After calibration
        probs_test_after = sigmoid(test_logits / T)
        m_after = metrics_from_probs(d["y_test"], probs_test_after)

        print(f"  trained in {train_s:.0f}s. T* = {T:.4f}")
        print(f"  BEFORE: AUC={m_before['auc']:.4f}  Brier={m_before['brier']:.4f}  ECE={m_before['ece']:.4f}")
        print(f"  AFTER:  AUC={m_after['auc']:.4f}  Brier={m_after['brier']:.4f}  ECE={m_after['ece']:.4f}")

        # Save predictions
        np.savez(PRED_DIR / f"{task}_{model_name}_val.npz",
                 y_true=d["y_val"].astype(np.float32),
                 logits=val_logits.astype(np.float32))
        np.savez(PRED_DIR / f"{task}_{model_name}_test_calib.npz",
                 y_true=d["y_test"].astype(np.float32),
                 logits=test_logits.astype(np.float32),
                 T=np.float32(T),
                 probs_before=probs_test_before.astype(np.float32),
                 probs_after=probs_test_after.astype(np.float32))

        results.append({
            "task": task, "model": model_name,
            "T": T, "val_nll_at_T": val_nll,
            "before": m_before, "after": m_after,
            "train_s": round(train_s, 1),
        })

    out = CACHE_DIR / "results_temperature.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*82}\nTEMPERATURE SCALING: BEFORE vs AFTER\n{'='*82}")
    print(f"{'Task':<10}{'Model':<17}{'T':>8}   "
          f"{'AUC_bef':>8}{'AUC_aft':>8}   "
          f"{'Brier_bef':>10}{'Brier_aft':>10}   "
          f"{'ECE_bef':>9}{'ECE_aft':>9}")
    print("-" * 82)
    for r in results:
        b, a = r["before"], r["after"]
        print(f"{r['task']:<10}{r['model']:<17}{r['T']:>8.4f}   "
              f"{b['auc']:>8.4f}{a['auc']:>8.4f}   "
              f"{b['brier']:>10.4f}{a['brier']:>10.4f}   "
              f"{b['ece']:>9.4f}{a['ece']:>9.4f}")

    print(f"\nDone: {time.time()-t0:.0f}s. Results → {out}")


if __name__ == "__main__":
    main()
