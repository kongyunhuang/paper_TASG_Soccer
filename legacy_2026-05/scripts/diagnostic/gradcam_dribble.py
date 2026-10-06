#!/usr/bin/env python3
"""
gradcam_dribble: Grad-CAM interpretability visualisation on the Dribble M4 CNN
================================================================================
Implementation of RESEARCH_PLAN section 4.8 Case 2 (Spatial Attention Visualisation).

Method:
  1. Retrain the Dribble M4 CNN on the relaxed split and save the weights
  2. Select several representative events:
       - Successful dribbles from top dribblers (high P, true=1)
       - Failed dribbles from bottom dribblers (high P, true=0)
       - Boundary cases (P ~ 0.5)
  3. Use Grad-CAM (Selvaraju 2017) to extract activations from the Conv2 layer
  4. Save the heatmaps for subsequent pitch overlay plotting

Inputs:
  data/L1_events_v3.parquet
  data/action_soccermaps_dribble.npy + _idx.parquet
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz (player ranking results)

Outputs:
  data/dribble_cnn_weights.pt
  data/gradcam_dribble_examples.npz  (selected events: inputs + saliency maps + metadata)

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/diagnostic/gradcam_dribble.py

"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.training.train_all import (
    ActionDataset, CACHE_DIR, CNNMLPModel, FULL_SEASONS, SB_360_COMMON,
    TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS, pd, train_nn,
)
from scripts.diagnostic.relaxed_holdout_dribble_duel import build_relaxed_split


def prepare_dribble_relaxed():
    np.random.seed(42); torch.manual_seed(42)
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)

    cfg = TASK_CONFIG["dribble"]
    df_t = df[df["type_name"] == cfg["type_name"]].copy()
    if "filter_fn" in cfg:
        df_t = cfg["filter_fn"](df_t)
    labels = cfg["label_fn"](df_t)

    event_feats = list(cfg["event_feats"])
    all_360 = SB_360_COMMON + cfg.get("extra_360", [])
    event_feats = [f for f in event_feats if f in df_t.columns]
    all_360 = [f for f in all_360 if f in df_t.columns]

    smap_tag = "dribble"
    idx_df = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet")
    smaps = np.load(CACHE_DIR / f"action_soccermaps_{smap_tag}.npy", mmap_mode="r")
    id_map = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}
    has = df_t["event_id"].isin(id_map).values
    df_t = df_t[has].reset_index(drop=True)
    labels = labels[has]

    train_mask, val_mask, test_mask = build_relaxed_split(df_t)

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

    return {
        "df_full": df_t,
        "Xtr": Xtr, "Xva": Xva, "Xte": Xte,
        "sm_tr": sm_tr, "sm_va": sm_va, "sm_te": sm_te,
        "y_tr": labels[train_mask], "y_va": labels[val_mask], "y_te": labels[test_mask],
        "test_mask": test_mask,
        "event_id_te": df_t.loc[test_mask, "event_id"].values,
    }


def gradcam_for_event(model, x_tab, x_smap, target_class=1):
    """
    Compute Grad-CAM on the last conv layer (encoder.4 = Conv2 inside SoccerMapEncoder).

    Returns CAM as np.array of shape (H_conv, W_conv) ~ (4, 6) for our architecture.
    """
    model.eval()
    feats = {}
    grads = {}

    target_layer = model.cnn.encoder[4]  # Conv2d(32, 64), spatial (4, 6)

    def fwd_hook(_, __, out):
        feats["v"] = out

    def bwd_hook(_, grad_in, grad_out):
        grads["v"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x_tab_b = x_tab.unsqueeze(0).clone().requires_grad_(False)
    x_smap_b = x_smap.unsqueeze(0).clone().requires_grad_(True)

    logits = model(x_tab_b, x_smap_b)
    score = logits[0]  # for binary, target class 1
    if target_class == 0:
        score = -score

    model.zero_grad()
    score.backward(retain_graph=False)

    fmap = feats["v"][0].detach().numpy()       # (64, H, W)
    grad = grads["v"][0].detach().numpy()       # (64, H, W)

    weights = grad.mean(axis=(1, 2))            # (64,)
    cam = np.maximum((weights[:, None, None] * fmap).sum(axis=0), 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    h1.remove(); h2.remove()
    return cam


def main():
    t0 = time.time()
    print("Preparing relaxed split for dribble ...")
    d = prepare_dribble_relaxed()

    print(f"  train={len(d['y_tr'])}  val={len(d['y_va'])}  test={len(d['y_te'])}")

    weights_path = CACHE_DIR / "dribble_cnn_weights.pt"
    torch.manual_seed(42)
    model = CNNMLPModel(d["Xtr"].shape[1], cnn_channels=2)
    Xva_t = torch.tensor(d["Xva"], dtype=torch.float32)
    Xte_t = torch.tensor(d["Xte"], dtype=torch.float32)
    if weights_path.exists():
        print(f"Loading existing weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Training M4 CNN 2ch ...")
        ds = ActionDataset(d["Xtr"], d["y_tr"], d["sm_tr"])
        dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
        train_nn(model, dl, Xva_t, d["y_va"], d["sm_va"], True, "binary")
        torch.save(model.state_dict(), weights_path)
        print(f"  weights saved -> {weights_path}")

    # Get test set predictions for selecting events
    model.eval()
    with torch.no_grad():
        z_te = model(Xte_t, d["sm_te"]).numpy()
    p_te = 1.0 / (1.0 + np.exp(-z_te))
    y_te = d["y_te"]

    # Pick exactly 4 representative events (one per quadrant) for the 2x2 figure
    def top_k_by(idx_pool, key_arr, k=1, descending=True):
        if len(idx_pool) == 0:
            return []
        order = np.argsort(key_arr[idx_pool])
        if descending:
            order = order[::-1]
        return list(idx_pool[order[:k]])

    cases = []
    # (a) High P + true positive: confident & correct
    idx_hp_pos = np.where((p_te > 0.75) & (y_te == 1))[0]
    for j in top_k_by(idx_hp_pos, p_te, k=1, descending=True):
        cases.append(("HP_TruePos", j))

    # (b) High P + true negative: confident but WRONG
    idx_hp_neg = np.where((p_te > 0.75) & (y_te == 0))[0]
    for j in top_k_by(idx_hp_neg, p_te, k=1, descending=True):
        cases.append(("HP_FalsePos", j))

    # (c) Low P + true positive: model thought fail but succeeded
    idx_lp_pos = np.where((p_te < 0.30) & (y_te == 1))[0]
    for j in top_k_by(idx_lp_pos, p_te, k=1, descending=False):
        cases.append(("LP_FalseNeg", j))

    # (d) Low P + true negative: confident & correct (failure)
    idx_lp_neg = np.where((p_te < 0.30) & (y_te == 0))[0]
    for j in top_k_by(idx_lp_neg, p_te, k=1, descending=False):
        cases.append(("LP_TrueNeg", j))

    print(f"\nSelected {len(cases)} representative events:")
    for label, i in cases:
        print(f"  {label:<18}  p={p_te[i]:.3f}  y={int(y_te[i])}  event_id={d['event_id_te'][i]}")

    # Compute Grad-CAM for each selected case
    cams_out = []
    for label, i in cases:
        x_tab = torch.tensor(d["Xte"][i], dtype=torch.float32)
        x_smap = d["sm_te"][i].clone()
        cam = gradcam_for_event(model, x_tab, x_smap, target_class=1)
        # Upsample CAM to input grid (8, 12)
        cam_t = torch.tensor(cam[None, None, :, :], dtype=torch.float32)
        cam_up = F.interpolate(cam_t, size=(8, 12), mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze().numpy()
        cams_out.append({
            "label": label,
            "test_idx": int(i),
            "event_id": str(d["event_id_te"][i]),
            "p": float(p_te[i]),
            "y": int(y_te[i]),
            "smap_input": d["sm_te"][i].numpy(),  # (2, 8, 12)
            "cam": cam_up,                        # (8, 12)
        })
        print(f"  {label}: cam max={cam.max():.3f}, sum_norm={cam.sum():.2f}")

    # Save
    out_path = CACHE_DIR / "gradcam_dribble_examples.npz"
    np.savez(out_path,
             labels=np.array([c["label"] for c in cams_out]),
             test_indices=np.array([c["test_idx"] for c in cams_out]),
             event_ids=np.array([c["event_id"] for c in cams_out]),
             ps=np.array([c["p"] for c in cams_out]),
             ys=np.array([c["y"] for c in cams_out]),
             smap_inputs=np.stack([c["smap_input"] for c in cams_out]),
             cams=np.stack([c["cam"] for c in cams_out]))
    print(f"\nWrote {out_path}")
    print(f"Done: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
