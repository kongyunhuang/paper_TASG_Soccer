#!/usr/bin/env python3
"""
ablate_u1_saved.py, test-time ablation of the saved unified models
==================================================================
Measures how much each task metric drops when the player position channels are shuffled.
train_unified_clean.py saves the model weights (data/cache/u1_clean/{tag}_s{seed}.pt). This script rebuilds the data
(the subsampling is fixed, so the test set is the one used in training), loads the weights and computes three sets of
metrics on the 202 clean matches. Unshuffled, the whole tensor shuffled within each task, and only the first two
channels (teammate and opponent positions) shuffled. The output is the drop of every task metric.
For models trained with --channels 2, the json field channels selects the first two channels automatically. Runs that
shuffled the training input are not covered.
For gated models with the pooled encoder two more quantities are computed. drop_gate_zero is the drop when the gate is
forced closed at test time, and drop_hs_mean the drop when the spatial branch output is replaced by its per-task mean vector.

Inputs   data/cache/u1_clean/{tag}_s{seed}.pt and the json of the same name
Outputs  data/cache/u1_clean/{tag}_s{seed}_ablation.json

Usage
  PYTHONPATH=. python -u scripts/eval/ablate_u1_saved.py --tag u1 --seeds 3

Last modified 2026-10-28
"""
import argparse
import json

import numpy as np
import torch

from scripts.eval.summarize_u1_clean import clean_matches
from scripts.training.train_unified_clean import (OUT_DIR, TASK_ID, TASKS, UnifiedGatingClean, UnifiedGatingFCN,
                                                  build_dataset, predict, rich_scalars, task_metrics)


def predict_variant(model, X, S, T, mode, hs_mean=None, bs=8192):
    """Gated models with the pooled encoder only. gate_zero forces the gate closed at test time (scalar branch only); hs_mean
    replaces the spatial branch output by its per-task mean vector. Neither feeds the ball position of other events into the
    model, so both measure whether the model works without the spatial branch, which differs from shuffling the whole tensor."""
    model.eval()
    lb, ld = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, sp, t = X[i:i + bs], S[i:i + bs], T[i:i + bs]
            h_e = model.event_enc(torch.cat([x, model.task_embed(t)], dim=-1))
            if mode == "gate_zero":
                h = h_e
            else:
                h_s = hs_mean[t]
                g = model.gate(torch.cat([h_e, h_s], dim=-1))
                if getattr(model, "hard_gate", False):
                    g = (g > 0.5).float()
                h = g * h_s + (1 - g) * h_e
            z = model.hidden(h)
            lb.append(model.out_bin(z).squeeze(-1))
            ld.append(torch.log_softmax(model.out_dest(z), dim=-1))
    return torch.cat(lb).numpy(), torch.cat(ld).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="u1")
    ap.add_argument("--seeds", default="3")
    a = ap.parse_args()
    data = build_dataset()
    seeds = [int(s) for s in a.seeds.split(",")]
    if json.load(open(OUT_DIR / f"{a.tag}_s{seeds[0]}.json")).get("rich_scalars"):
        # Rich-scalar models need the same widened scalars on the test set (computed from the unshuffled tensors)
        for t in TASKS:
            d = data[t]["test"]
            d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
    clean = clean_matches()
    test = {k: np.concatenate([data[t]["test"][k] for t in TASKS]) for k in ("X", "y", "sm", "match")}
    test["task"] = np.concatenate([np.full(len(data[t]["test"]["y"]), TASK_ID[t]) for t in TASKS])
    dev_sm = {t: torch.tensor(data[t]["dev"]["sm"][:20000]) for t in TASKS}  # first 20,000 training-season events per task, for the mean spatial branch output
    del data
    ok = np.array([int(m) in clean for m in test["match"]])
    for k in test:
        test[k] = test[k][ok]
    S = torch.tensor(test["sm"])
    T = torch.tensor(test["task"], dtype=torch.long)
    rs = np.random.RandomState(4242)
    perms = {t: np.where(test["task"] == TASK_ID[t])[0] for t in TASKS}
    perms = {t: (i, i[rs.permutation(len(i))]) for t, i in perms.items()}
    S_all, S_pl = S.clone(), S.clone()
    for i, sh in perms.values():
        S_all[i] = S[sh]
        S_pl[i, :2] = S[sh, :2]
    for seed in seeds:
        ck = torch.load(OUT_DIR / f"{a.tag}_s{seed}.pt", weights_only=False)
        meta = json.load(open(OUT_DIR / f"{a.tag}_s{seed}.json"))
        Model = UnifiedGatingClean if meta.get("encoder", "pooled") == "pooled" else UnifiedGatingFCN
        X = torch.tensor((test["X"] - ck["scaler_mean"]) / ck["scaler_scale"], dtype=torch.float32)
        ch = meta.get("channels", 7)  # the two-channel model has only the player position channels, so both shuffles coincide
        if meta.get("hard_gate"):
            model = Model(X.shape[1], len(TASKS), fusion=meta["fusion"], cnn_channels=ch, hard_gate=True)
        else:
            model = Model(X.shape[1], len(TASKS), fusion=meta["fusion"], cnn_channels=ch)
        model.load_state_dict(ck["state"])
        out = {}
        base = task_metrics(test["task"], test["y"], *predict(model, X, S[:, :ch], T)[:2])
        m_all = task_metrics(test["task"], test["y"], *predict(model, X, S_all[:, :ch], T)[:2])
        m_pl = task_metrics(test["task"], test["y"], *predict(model, X, S_pl[:, :ch], T)[:2])
        extra = meta.get("encoder", "pooled") == "pooled" and meta["fusion"] == "gate"
        if extra:
            model.eval()
            with torch.no_grad():
                hs_mean = torch.stack([model.spatial_enc(dev_sm[t][:, :ch]).mean(0) for t in TASKS])
            lb_g0, ld_g0 = predict_variant(model, X, S[:, :ch], T, "gate_zero")
            m_g0 = task_metrics(test["task"], test["y"], lb_g0, ld_g0)
            lb_base = predict(model, X, S[:, :ch], T)[0]
            if meta.get("hard_gate"):
                # Raw gate-open probabilities of the hard gate (at test time the model uses them thresholded at 0.5), to check whether open dimensions cluster just above 0.5
                with torch.no_grad():
                    raw_g = torch.cat([model.gate(torch.cat([model.event_enc(torch.cat([X[i:i + 8192], model.task_embed(T[i:i + 8192])], dim=-1)),
                                                             model.spatial_enc(S[i:i + 8192, :ch])], dim=-1)) for i in range(0, len(X), 8192)])
            m_hm = task_metrics(test["task"], test["y"], *predict_variant(model, X, S[:, :ch], T, "hs_mean", hs_mean))
        print(f"\n[{a.tag} seed {seed}] 202 clean matches")
        print(f"{'task':14s} {'metric':>8s} {'drop_all':>9s} {'drop_players':>13s} {'gate_zero':>10s} {'hs_mean':>8s}")
        for t in TASKS:
            out[t] = dict(base=base[t], all_ablated=m_all[t], players_ablated=m_pl[t],
                          drop_all=base[t]["skill"] - m_all[t]["skill"], drop_players=base[t]["skill"] - m_pl[t]["skill"])
            if extra:
                out[t].update(drop_gate_zero=base[t]["skill"] - m_g0[t]["skill"], drop_hs_mean=base[t]["skill"] - m_hm[t]["skill"])
                # Per-event loss increase when the gate is forced closed (cross-entropy for binary tasks, log loss times 1/ln 96 for dest), in the same unit as the gate penalty
                m = test["task"] == TASK_ID[t]
                if t == "dest":
                    out[t]["loss_up_gate_zero"] = float(m_g0[t]["logloss"] - base[t]["logloss"]) / float(np.log(96))
                else:
                    yy = torch.tensor(test["y"][m], dtype=torch.float32)
                    bce = lambda lg: float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg[m]), yy))
                    out[t]["loss_up_gate_zero"] = bce(lb_g0) - bce(lb_base)
                if meta.get("hard_gate"):
                    gi = raw_g[torch.tensor(m)]
                    op = gi > 0.5
                    out[t].update(all_closed_share=float((~op).all(dim=1).float().mean()), open_share=float(op.float().mean()),
                                  open_g_mean=float(gi[op].mean()) if op.any() else None,
                                  open_g_below_06_share=float((gi[op] < 0.6).float().mean()) if op.any() else None)
            print(f"{t:14s} {base[t]['skill']:>8.4f} {out[t]['drop_all']:>9.4f} {out[t]['drop_players']:>13.4f} "
                  f"{out[t].get('drop_gate_zero', float('nan')):>10.4f} {out[t].get('drop_hs_mean', float('nan')):>8.4f}", flush=True)
        json.dump(out, open(OUT_DIR / f"{a.tag}_s{seed}_ablation.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
