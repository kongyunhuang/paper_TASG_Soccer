#!/usr/bin/env python3
"""
eval_calibration_sweep: calibration diagnostics for all models x all tasks
==========================================================================
Extends eval_extended_metrics with ECE (Expected Calibration Error),
and saves the test-set predictions for each model so that subsequent
temperature-scaling / Platt / Isotonic comparison experiments do not
need to retrain.

Inputs:
  data/L1_events_v3.parquet
  data/action_soccermaps_{task}.npy + _idx.parquet

Outputs:
  data/results_calibration.json        (AUC + Brier + ECE per task x model)
  data/predictions/{task}_{model}.npz  (y_true, y_prob, logits [optional])

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/eval/eval_calibration_sweep.py

"""

import json
import time
from pathlib import Path

import numpy as np

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, DataLoader, FULL_SEASONS,
    LogisticRegression, Pipeline, SpatialGatingModel, StandardScaler,
    TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS, XGBClassifier,
    compute_pressure_labels, pd, roc_auc_score,
    torch, train_nn, SB_360_COMMON,
)
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score


PRED_DIR = CACHE_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    ECE with equal-width binning (Guo et al. 2017, eq. 3).

    Weighted average of |accuracy - confidence| across bins.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_prob)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        acc = float(y_true[in_bin].mean())
        conf = float(y_prob[in_bin].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def metrics_binary(y_true, y_prob, y_logits=None, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def save_predictions(task, model, y_true, y_prob, y_logits=None):
    path = PRED_DIR / f"{task}_{model}.npz"
    arrs = {"y_true": y_true.astype(np.float32), "y_prob": y_prob.astype(np.float32)}
    if y_logits is not None:
        arrs["logits"] = y_logits.astype(np.float32)
    np.savez(path, **arrs)


def logit_from_prob(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


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
    for col in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[col] = df[col].fillna(False).astype(int)

    pressure_labels = compute_pressure_labels(df)

    TASKS = ["pass", "dribble", "ball_receipt", "shot", "duel", "interception", "ball_recovery", "pressure"]
    all_results = []

    for task_name in TASKS:
        cfg = TASK_CONFIG[task_name]
        if cfg["task_type"] != "binary":
            continue

        print(f"\n{'=' * 60}\nTASK: {task_name.upper()}\n{'=' * 60}")

        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if "filter_fn" in cfg:
            df_task = cfg["filter_fn"](df_task)

        labels = pressure_labels if task_name == "pressure" else cfg["label_fn"](df_task)

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
        has_smap_cache = idx_path.exists() and npy_path.exists()

        if has_smap_cache:
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

        print(f"  Samples: {len(df_task):,}, pos_rate={labels.mean():.4f}")

        is_train_season = df_task["season_dir"].isin(TRAIN_SEASONS).values
        is_test_season = df_task["season_dir"].isin(TEST_SEASONS).values
        train_matches = df_task.loc[is_train_season, "match_id"].unique()
        np.random.shuffle(train_matches)
        n_t = int(0.8 * len(train_matches))
        train_mask = df_task["match_id"].isin(set(train_matches[:n_t])).values
        val_mask = df_task["match_id"].isin(set(train_matches[n_t:])).values
        test_mask = is_test_season

        y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]

        X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
        X_360 = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
        X_e360 = np.hstack([X_event, X_360])

        scaler_e = StandardScaler().fit(X_event[train_mask])
        scaler_e360 = StandardScaler().fit(X_e360[train_mask])
        Xe_tr = scaler_e.transform(X_event[train_mask])
        Xe_te = scaler_e.transform(X_event[test_mask])
        Xe360_tr = scaler_e360.transform(X_e360[train_mask])
        Xe360_va = scaler_e360.transform(X_e360[val_mask])
        Xe360_te = scaler_e360.transform(X_e360[test_mask])

        sm_train = sm_val = sm_test = None
        if has_smap_cache:
            eids = df_task["event_id"].values
            smap_indices = np.array([smap_lookup[eid] for eid in eids])
            sm_train = smaps_7ch[smap_indices[train_mask]][:, :2, :, :]
            sm_val = torch.tensor(smaps_7ch[smap_indices[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
            sm_test = torch.tensor(smaps_7ch[smap_indices[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

        # ── Majority baseline ──
        p_train = float(y_train.mean())
        maj_prob = np.full(len(y_test), p_train, dtype=np.float32)
        m = metrics_binary(y_test, maj_prob)
        m["auc"] = 0.5  # constant predictor
        all_results.append({"task": task_name, "model": "Majority", "pos_rate": p_train, **m})
        save_predictions(task_name, "Majority", y_test, maj_prob)
        print(f"  Majority:      AUC=0.500 Brier={m['brier']:.4f} ECE={m['ece']:.4f}")

        # ── L1 LR Event ──
        t0 = time.time()
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
        pipe.fit(Xe_tr, y_train)
        prob = pipe.predict_proba(Xe_te)[:, 1]
        m = metrics_binary(y_test, prob)
        m["time_s"] = round(time.time() - t0, 1)
        all_results.append({"task": task_name, "model": "L1_LR_Event", "pos_rate": p_train, **m})
        save_predictions(task_name, "L1_LR_Event", y_test, prob, logit_from_prob(prob))
        print(f"  L1 LR:         AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f}  ({m['time_s']:.0f}s)")

        # ── B2 XGB +360 ──
        t0 = time.time()
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
                            tree_method="hist", random_state=42, verbosity=0)
        xgb.fit(Xe360_tr, y_train, eval_set=[(Xe360_va, y_val)], verbose=False)
        prob = xgb.predict_proba(Xe360_te)[:, 1]
        m = metrics_binary(y_test, prob)
        m["time_s"] = round(time.time() - t0, 1)
        all_results.append({"task": task_name, "model": "B2_XGB_360", "pos_rate": p_train, **m})
        save_predictions(task_name, "B2_XGB_360", y_test, prob, logit_from_prob(prob))
        print(f"  B2 XGB+360:    AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f}  ({m['time_s']:.0f}s)")

        if has_smap_cache:
            X_te_t = torch.tensor(Xe360_te, dtype=torch.float32)

            # ── M4 CNN 2ch ──
            t0 = time.time()
            torch.manual_seed(42)
            model = CNNMLPModel(Xe360_tr.shape[1], cnn_channels=2)
            ds = ActionDataset(Xe360_tr, y_train, sm_train)
            dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
            train_nn(model, dl, torch.tensor(Xe360_va, dtype=torch.float32), y_val, sm_val, True, "binary")
            model.eval()
            with torch.no_grad():
                logits = model(X_te_t, sm_test).numpy()
            prob = 1 / (1 + np.exp(-logits))
            m = metrics_binary(y_test, prob)
            m["time_s"] = round(time.time() - t0, 1)
            all_results.append({"task": task_name, "model": "M4_CNN_2ch", "pos_rate": p_train, **m})
            save_predictions(task_name, "M4_CNN_2ch", y_test, prob, logits)
            print(f"  M4 CNN 2ch:    AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f}  ({m['time_s']:.0f}s)")

            # ── G1 Gating 2ch ──
            t0 = time.time()
            torch.manual_seed(42)
            model = SpatialGatingModel(Xe360_tr.shape[1], cnn_channels=2)
            ds = ActionDataset(Xe360_tr, y_train, sm_train)
            dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
            train_nn(model, dl, torch.tensor(Xe360_va, dtype=torch.float32), y_val, sm_val, True, "binary")
            model.eval()
            with torch.no_grad():
                logits = model(X_te_t, sm_test).numpy()
            prob = 1 / (1 + np.exp(-logits))
            m = metrics_binary(y_test, prob)
            m["time_s"] = round(time.time() - t0, 1)
            all_results.append({"task": task_name, "model": "G1_Gating_2ch", "pos_rate": p_train, **m})
            save_predictions(task_name, "G1_Gating_2ch", y_test, prob, logits)
            print(f"  G1 Gating 2ch: AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f}  ({m['time_s']:.0f}s)")

    out_path = CACHE_DIR / "results_calibration.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 82}")
    print("CALIBRATION SWEEP: AUC / Brier / ECE")
    print(f"{'=' * 82}")
    print(f"{'Task':<14}{'Model':<17}{'AUC':>7}{'Brier':>8}{'ECE':>7}{'F1':>7}{'Pos%':>7}")
    print("-" * 82)
    for r in all_results:
        print(f"{r['task']:<14}{r['model']:<17}{r.get('auc', 0):>7.3f}"
              f"{r.get('brier', 0):>8.4f}{r.get('ece', 0):>7.4f}"
              f"{r.get('f1', 0):>7.3f}{r.get('pos_rate', 0) * 100:>6.1f}%")

    print(f"\nDone: {time.time() - t0_all:.0f}s ({(time.time() - t0_all) / 60:.1f} min)")
    print(f"Results: {out_path}")
    print(f"Predictions: {PRED_DIR}/ ({len(list(PRED_DIR.glob('*.npz')))} files)")


if __name__ == "__main__":
    main()
