#!/usr/bin/env python3
"""
Ablation A1: SoccerMap 2ch vs 7ch
==================================
Run M3/M4/G1 with 2ch SoccerMap (teammate + opponent positions only)
and compare against the existing 7ch results to quantify the marginal
contribution of the distance/angle channels.

2ch = Ch0 (teammates) + Ch1 (opponents); taken as the first two
channels of the 7ch npy.
7ch = full A-G set (existing results in results_all.json).

Runs all tasks that have a SoccerMap cache: pass, dribble, ball_receipt,
shot, duel, interception, ball_recovery, pressure.

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/ablation/ablation_channels.py

"""

from scripts.training.train_all import *


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

    # T8
    pressure_labels = compute_pressure_labels(df)

    all_results = []

    # Run every task that has a SoccerMap cache
    ablation_tasks = ["pass", "dribble", "ball_receipt", "shot", "duel",
                      "interception", "ball_recovery", "pressure"]

    for task_name in ablation_tasks:
        cfg = TASK_CONFIG[task_name]
        print(f"\n{'='*60}")
        print(f"A1 ABLATION: {task_name.upper()}: 2ch SoccerMap")
        print(f"{'='*60}")

        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if "filter_fn" in cfg:
            df_task = cfg["filter_fn"](df_task)

        if task_name == "pressure":
            labels = pressure_labels
        else:
            labels = cfg["label_fn"](df_task)

        # One-hot
        for col, vals in cfg.get("event_onehot", {}).items():
            for v in vals:
                df_task[f"{col}_{v}"] = (df_task[col] == v).astype(int)

        event_feats = list(cfg["event_feats"])
        for col, vals in cfg.get("event_onehot", {}).items():
            event_feats += [f"{col}_{v}" for v in vals]
        all_360 = SB_360_COMMON + cfg.get("extra_360", [])
        event_feats = [f for f in event_feats if f in df_task.columns]
        all_360 = [f for f in all_360 if f in df_task.columns]

        # SoccerMap: load the 7ch npy but keep only the first two channels
        smap_tag = cfg.get("smap_tag", task_name)
        idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
        npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"

        if not idx_path.exists() or not npy_path.exists():
            print(f"  SKIP: no SoccerMap cache")
            continue

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

        print(f"  Samples: {len(df_task):,}")

        # Split
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

        # Features
        X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
        X_360_arr = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
        X_e360 = np.hstack([X_event, X_360_arr])

        scaler_e = StandardScaler().fit(X_event[train_mask])
        scaler_e360 = StandardScaler().fit(X_e360[train_mask])
        Xe_tr = scaler_e.transform(X_event[train_mask])
        Xe_va = scaler_e.transform(X_event[val_mask])
        Xe_te = scaler_e.transform(X_event[test_mask])
        Xe360_tr = scaler_e360.transform(X_e360[train_mask])
        Xe360_va = scaler_e360.transform(X_e360[val_mask])
        Xe360_te = scaler_e360.transform(X_e360[test_mask])

        # 2ch SoccerMap: take only the first two channels
        eids = df_task["event_id"].values
        smap_indices = np.array([smap_lookup[eid] for eid in eids])
        sm_2ch_train = smaps_7ch[smap_indices[train_mask]][:, :2, :, :]  # (N, 2, 8, 12)
        sm_2ch_val = torch.tensor(smaps_7ch[smap_indices[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
        sm_2ch_test = torch.tensor(smaps_7ch[smap_indices[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

        task_type = cfg["task_type"]

        for model_name, ModelClass, X_tr, X_va, X_te in [
            ("M3_CNN_Event_2ch", CNNMLPModel, Xe_tr, Xe_va, Xe_te),
            ("M4_CNN_Full_2ch", CNNMLPModel, Xe360_tr, Xe360_va, Xe360_te),
            ("G1_Gating_2ch", SpatialGatingModel, Xe360_tr, Xe360_va, Xe360_te),
        ]:
            t0 = time.time()
            print(f"\n  {model_name} ...", end=" ", flush=True)
            torch.manual_seed(42)

            # Note: cnn_channels=2
            model = ModelClass(X_tr.shape[1], cnn_channels=2)
            ds = ActionDataset(X_tr, y_train, sm_2ch_train)
            dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
            X_va_t = torch.tensor(X_va, dtype=torch.float32)
            X_te_t = torch.tensor(X_te, dtype=torch.float32)

            train_nn(model, dl, X_va_t, y_val, sm_2ch_val, True, task_type)
            metrics = evaluate_model(model, X_te_t, y_test, sm_2ch_test, True, task_type)

            result = {"task": task_name, "model": model_name, "task_type": task_type,
                      "ablation": "2ch"}
            result.update(metrics)
            result["time_s"] = round(time.time() - t0, 1)

            print(f"test_auc={result.get('test_auc', 'N/A'):.4f}  ({result['time_s']:.0f}s)")
            all_results.append(result)

    # Save to a dedicated file
    output_path = CACHE_DIR / "results_ablation_channels.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*60}")
    print("A1 ABLATION: 2ch vs 7ch COMPARISON")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Model':<20} {'2ch AUC':>10} {'7ch AUC':>10} {'Δ(7ch-2ch)':>12}")
    print("-" * 70)

    with open(CACHE_DIR / "results_all.json") as f:
        main_results = json.load(f)

    for r2 in all_results:
        task = r2["task"]
        base_model = r2["model"].replace("_2ch", "")
        r7 = [r for r in main_results if r["task"] == task and r["model"] == base_model]
        auc_2 = r2.get("test_auc", 0)
        auc_7 = r7[0].get("test_auc", 0) if r7 else 0
        delta = auc_7 - auc_2
        print(f"{task:<15} {base_model:<20} {auc_2:>10.4f} {auc_7:>10.4f} {delta:>+12.4f}")

    total_time = time.time() - t0_all
    print(f"\nDone: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
