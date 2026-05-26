#!/usr/bin/env python3
"""
A' Full Run: 7ch cross-league for all 8 binary tasks
======================================================
KBS coherence audit (2026-05-15) decision tree: after the Interception
pilot (3.5 min wall-clock) confirmed a single 7ch run takes about 50 s,
launch the full 8 tasks x 2 directions x {M4_7ch, G1_7ch} = 32 runs.
Combined with the existing 2ch cache (results_cross_league_8tasks.json),
this lets Section 5.3 Table 9 expose the channel dimension and removes
the inconsistency between the main benchmark (Table 5, 7ch) and the
cross-league setting.

Tasks: 8 binary (pass, dribble, ball_receipt, shot, duel, interception,
       ball_recovery, pressure)
Directions: EPL->LaLiga, LaLiga->EPL
Models: M4_CNN_Full_7ch, G1_Gating_7ch (cnn_channels=7, no slicing)
Baselines L1/L2/B2: not re-run, independent of channel count - reuse
results_cross_league_8tasks.json directly (already produced during the
2ch run).

Input: reuses the ablation_cross_league.py data pipeline; loads the full
       7ch SoccerMap.
Output: data/results_cross_league_8tasks_7ch.json
Incremental seed: reads results_cross_league_7ch_pilot.json (4 Interception
rows) as the starting point and skips already cached combinations.

Usage:
  PYTHONUNBUFFERED=1 python scripts/ablation/ablation_cross_league_7ch.py

Estimated wall-clock: ~2.5 hours (extrapolated from the 2ch wall-clock
with a x1.95 ratio; the pilot already validated Interception).

"""

from scripts.training.train_all import *


EPL_SEASONS = ["2_235_2022_23", "2_281_2023_24", "2_317_2024_25"]
LALIGA_SEASONS = ["11_235_2022_23", "11_281_2023_24", "11_317_2024_25"]

CROSS_LEAGUE_CONFIGS = [
    ("EPL→LaLiga", EPL_SEASONS, LALIGA_SEASONS),
    ("LaLiga→EPL", LALIGA_SEASONS, EPL_SEASONS),
]

EVAL_TASKS = [
    "pass", "dribble", "ball_receipt", "shot",
    "duel", "interception", "ball_recovery", "pressure",
]

MODEL_ORDER = [
    "M4_CNN_Full_7ch",
    "G1_Gating_7ch",
]

OUTPUT_PATH = CACHE_DIR / "results_cross_league_8tasks_7ch.json"
PILOT_SEED_PATH = CACHE_DIR / "results_cross_league_7ch_pilot.json"


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


def load_seed():
    """Read pilot cache as initial state. Pilot has 4 Interception runs."""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            rows = json.load(f)
        print(f"Loaded {len(rows)} existing rows from {OUTPUT_PATH}")
        return rows
    if PILOT_SEED_PATH.exists():
        with open(PILOT_SEED_PATH) as f:
            rows = json.load(f)
        print(f"Loaded {len(rows)} pilot rows from {PILOT_SEED_PATH}")
        return rows
    return []


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

    pressure_labels = compute_pressure_labels(df)

    all_results = load_seed()
    existing_keys = {result_key(r) for r in all_results}
    save_results(all_results)

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

            train_mask_all = df_task["season_dir"].isin(train_seasons).values
            test_mask = df_task["season_dir"].isin(test_seasons).values

            train_matches = df_task.loc[train_mask_all, "match_id"].unique()
            np.random.shuffle(train_matches)
            n_t = int(0.8 * len(train_matches))
            train_mask = df_task["match_id"].isin(set(train_matches[:n_t])).values
            val_mask = df_task["match_id"].isin(set(train_matches[n_t:])).values

            y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]
            print(f"    Samples: {len(df_task):,}, Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")

            X_event = df_task[event_feats].fillna(0).values.astype(np.float32)
            X_360_arr = df_task[all_360].fillna(0).values.astype(np.float32) if all_360 else np.empty((len(df_task), 0), dtype=np.float32)
            X_e360 = np.hstack([X_event, X_360_arr])

            scaler_e360 = StandardScaler().fit(X_e360[train_mask])
            Xe360_tr = scaler_e360.transform(X_e360[train_mask])
            Xe360_va = scaler_e360.transform(X_e360[val_mask])
            Xe360_te = scaler_e360.transform(X_e360[test_mask])

            # 7ch SoccerMap (matches the main benchmark, no slicing)
            eids = df_task["event_id"].values
            smap_indices = np.array([smap_lookup[eid] for eid in eids])
            sm_train = smaps_7ch[smap_indices[train_mask]]
            sm_val = torch.tensor(smaps_7ch[smap_indices[val_mask]].copy(), dtype=torch.float32)
            sm_test = torch.tensor(smaps_7ch[smap_indices[test_mask]].copy(), dtype=torch.float32)

            task_type = cfg["task_type"]

            model_configs = [
                ("M4_CNN_Full_7ch", "cnn"),
                ("G1_Gating_7ch", "gating"),
            ]

            for model_name, model_type in model_configs:
                key = (exp_name, task_name, model_name)
                if key in existing_keys:
                    print(f"    {model_name} ... SKIP cached")
                    continue

                t0 = time.time()
                print(f"    {model_name} ...", end=" ", flush=True)
                torch.manual_seed(42)

                result = {"task": task_name, "model": model_name, "experiment": exp_name, "task_type": task_type, "cnn_channels": 7}

                if model_type == "cnn":
                    model = CNNMLPModel(Xe360_tr.shape[1], cnn_channels=7)
                    ds = ActionDataset(Xe360_tr, y_train, sm_train)
                    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                    X_va_t = torch.tensor(Xe360_va, dtype=torch.float32)
                    X_te_t = torch.tensor(Xe360_te, dtype=torch.float32)
                    train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                    metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                    result.update(metrics)

                elif model_type == "gating":
                    model = SpatialGatingModel(Xe360_tr.shape[1], cnn_channels=7)
                    ds = ActionDataset(Xe360_tr, y_train, sm_train)
                    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
                    X_va_t = torch.tensor(Xe360_va, dtype=torch.float32)
                    X_te_t = torch.tensor(Xe360_te, dtype=torch.float32)
                    train_nn(model, dl, X_va_t, y_val, sm_val, True, task_type)
                    metrics = evaluate_model(model, X_te_t, y_test, sm_test, True, task_type)
                    result.update(metrics)

                result["time_s"] = round(time.time() - t0, 1)
                print(f"test_auc={result.get('test_auc', 'N/A'):.4f}  ({result['time_s']:.0f}s)")
                all_results.append(result)
                existing_keys.add(key)
                save_results(all_results)

    all_results = sort_results(all_results)
    save_results(all_results)

    print(f"\n{'='*60}")
    print("A' (7ch cross-league, 8 binary tasks) RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Exp':<15} {'Task':<15} {'Model':<22} {'Test AUC':>10} {'Time(s)':>10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['experiment']:<15} {r['task']:<15} {r['model']:<22} {r.get('test_auc',0):>10.4f} {r.get('time_s',0):>10.1f}")

    total_s = time.time() - t0_all
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"A' total wall-clock: {total_s:.0f}s ({total_s/60:.1f} min, {total_s/3600:.2f} h)")


if __name__ == "__main__":
    main()
