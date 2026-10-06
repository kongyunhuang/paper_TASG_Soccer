#!/usr/bin/env python3
"""
Phase 3: cross-league generalisation experiment
================================================
ExpA: train EPL (3 seasons) -> test La Liga (3 seasons)
ExpB: train La Liga (3 seasons) -> test EPL (3 seasons)

Runs L1/L2/B2/M4/G1 on the 8 binary tasks under the cross-league setting
and reports test AUC. Uses 2ch SoccerMap (ablation A1 found 2ch outperforms
7ch). The script supports incremental top-up from the legacy 4-task
results_cross_league.json and writes to results_cross_league_8tasks.json.

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/ablation/ablation_cross_league.py

"""

from scripts.training.train_all import *


EPL_SEASONS = ["2_235_2022_23", "2_281_2023_24", "2_317_2024_25"]
LALIGA_SEASONS = ["11_235_2022_23", "11_281_2023_24", "11_317_2024_25"]

CROSS_LEAGUE_CONFIGS = [
    ("EPL→LaLiga", EPL_SEASONS, LALIGA_SEASONS),
    ("LaLiga→EPL", LALIGA_SEASONS, EPL_SEASONS),
]

# 8 binary tasks. xG is a regression task and does not enter the main
# cross-league AUC table.
EVAL_TASKS = [
    "pass", "dribble", "ball_receipt", "shot",
    "duel", "interception", "ball_recovery", "pressure",
]

MODEL_ORDER = [
    "L1_LR_Event",
    "L2_LR_360",
    "B2_XGB_360",
    "M4_CNN_Full_2ch",
    "G1_Gating_2ch",
]

OUTPUT_PATH = CACHE_DIR / "results_cross_league_8tasks.json"
LEGACY_PATH = CACHE_DIR / "results_cross_league.json"


def result_key(row):
    return row["experiment"], row["task"], row["model"]


def sort_results(rows):
    exp_rank = {name: i for i, (name, _, _) in enumerate(CROSS_LEAGUE_CONFIGS)}
    task_rank = {name: i for i, name in enumerate(EVAL_TASKS)}
    model_rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(
        rows,
        key=lambda r: (
            exp_rank.get(r["experiment"], 99),
            task_rank.get(r["task"], 99),
            model_rank.get(r["model"], 99),
        ),
    )


def load_existing_results():
    seed_path = OUTPUT_PATH if OUTPUT_PATH.exists() else LEGACY_PATH
    if not seed_path.exists():
        return []
    with open(seed_path) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} existing rows from {seed_path}")
    return rows


def save_results(rows):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sort_results(rows), f, indent=2, default=str)


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

    all_results = load_existing_results()
    existing_keys = {result_key(r) for r in all_results}

    for exp_name, train_seasons, test_seasons in CROSS_LEAGUE_CONFIGS:
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"  Train: {train_seasons}")
        print(f"  Test:  {test_seasons}")
        print(f"{'#'*60}")

        for task_name in EVAL_TASKS:
            cfg = TASK_CONFIG[task_name]
            print(f"\n  {'='*50}")
            print(f"  TASK: {task_name.upper()}")

            if all((exp_name, task_name, model_name) in existing_keys for model_name in MODEL_ORDER):
                print("    SKIP: all model results already cached")
                continue

            df_task = df[df["type_name"] == cfg["type_name"]].copy()
            if "filter_fn" in cfg:
                df_task = cfg["filter_fn"](df_task)

            if task_name == "pressure":
                labels = pressure_labels
            else:
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

            # SoccerMap
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

            # Cross-league split: train = all of one league, val = 20% of train, test = all of the other league
            train_mask_all = df_task["season_dir"].isin(train_seasons).values
            test_mask = df_task["season_dir"].isin(test_seasons).values

            train_matches = df_task.loc[train_mask_all, "match_id"].unique()
            np.random.shuffle(train_matches)
            n_t = int(0.8 * len(train_matches))
            train_mask = df_task["match_id"].isin(set(train_matches[:n_t])).values
            val_mask = df_task["match_id"].isin(set(train_matches[n_t:])).values

            y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]
            print(f"    Samples: {len(df_task):,}, Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")

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

            # 2ch SoccerMap (ablation A1 found 2ch performs better)
            eids = df_task["event_id"].values
            smap_indices = np.array([smap_lookup[eid] for eid in eids])
            sm_train = smaps_7ch[smap_indices[train_mask]][:, :2, :, :]
            sm_val = torch.tensor(smaps_7ch[smap_indices[val_mask]][:, :2, :, :].copy(), dtype=torch.float32)
            sm_test = torch.tensor(smaps_7ch[smap_indices[test_mask]][:, :2, :, :].copy(), dtype=torch.float32)

            task_type = cfg["task_type"]

            # Run models
            model_configs = [
                ("L1_LR_Event", "lr", Xe_tr, Xe_va, Xe_te, False),
                ("L2_LR_360", "lr", Xe360_tr, Xe360_va, Xe360_te, False),
                ("B2_XGB_360", "xgb", Xe360_tr, Xe360_va, Xe360_te, False),
                ("M4_CNN_Full_2ch", "cnn", Xe360_tr, Xe360_va, Xe360_te, True),
                ("G1_Gating_2ch", "gating", Xe360_tr, Xe360_va, Xe360_te, True),
            ]

            for model_name, model_type, X_tr, X_va, X_te, use_cnn in model_configs:
                key = (exp_name, task_name, model_name)
                if key in existing_keys:
                    print(f"    {model_name} ... SKIP cached")
                    continue

                t0 = time.time()
                print(f"    {model_name} ...", end=" ", flush=True)
                torch.manual_seed(42)

                result = {"task": task_name, "model": model_name, "experiment": exp_name, "task_type": task_type}

                if model_type == "lr":
                    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
                    pipe.fit(X_tr, y_train)
                    result["test_auc"] = float(roc_auc_score(y_test, pipe.predict_proba(X_te)[:, 1]))

                elif model_type == "xgb":
                    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                        eval_metric="auc", tree_method="hist", random_state=42, verbosity=0)
                    xgb.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
                    result["test_auc"] = float(roc_auc_score(y_test, xgb.predict_proba(X_te)[:, 1]))

                elif model_type == "cnn":
                    model = CNNMLPModel(X_tr.shape[1], cnn_channels=2)
                    ds = ActionDataset(X_tr, y_train, sm_train)
                    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                    X_va_t = torch.tensor(X_va, dtype=torch.float32)
                    X_te_t = torch.tensor(X_te, dtype=torch.float32)
                    train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                    metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                    result.update(metrics)

                elif model_type == "gating":
                    model = SpatialGatingModel(X_tr.shape[1], cnn_channels=2)
                    ds = ActionDataset(X_tr, y_train, sm_train)
                    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                    X_va_t = torch.tensor(X_va, dtype=torch.float32)
                    X_te_t = torch.tensor(X_te, dtype=torch.float32)
                    train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                    metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                    result.update(metrics)

                result["time_s"] = round(time.time() - t0, 1)
                print(f"test_auc={result.get('test_auc', 'N/A'):.4f}  ({result['time_s']:.0f}s)")
                all_results.append(result)
                existing_keys.add(key)
                save_results(all_results)

    # Save
    all_results = sort_results(all_results)
    save_results(all_results)

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-LEAGUE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Exp':<15} {'Task':<15} {'Model':<20} {'Test AUC':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['experiment']:<15} {r['task']:<15} {r['model']:<20} {r.get('test_auc',0):>10.4f}")

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Done: {time.time()-t0_all:.0f}s ({(time.time()-t0_all)/60:.1f} min)")


if __name__ == "__main__":
    main()
