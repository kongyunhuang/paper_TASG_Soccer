#!/usr/bin/env python3
"""
对已保存权重的统一模型做测试时消融  只打乱球员位置通道，看各任务指标掉多少
============================
train_unified_clean.py 在 2026-10-02 中午之后的运行会保存权重（data/cache/u1_clean/{tag}_s{seed}.pt）。
本脚本重新组一次数据（抽样固定，所以和训练时的测试集相同），载入权重，在 202 场干净比赛上算三组指标，
不打乱、整个张量在同任务内打乱、只打乱前两个通道（队友和对手位置），输出每个任务的下降量。
两通道版（--channels 2 训练的）按 json 里的 channels 字段自动只取前两个通道。打乱了训练输入的干预运行不适用。
对原编码器的门控模型另算两个量（2026-10-02 事后加的），测试时把门强制关上的下降量 drop_gate_zero，
和把空间分支输出换成同任务平均向量的下降量 drop_hs_mean。

输入  data/cache/u1_clean/{tag}_s{seed}.pt 和同名 json
输出  data/cache/u1_clean/{tag}_s{seed}_ablation.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/ablate_u1_saved.py --tag u1 --seeds 3

Last modified 2026-10-02
"""
import argparse
import json

import numpy as np
import torch

from scripts.eval.summarize_u1_clean import clean_matches
from scripts.training.train_unified_clean import (OUT_DIR, TASK_ID, TASKS, UnifiedGatingClean, UnifiedGatingFCN,
                                                  build_dataset, predict, rich_scalars, task_metrics)


def predict_variant(model, X, S, T, mode, hs_mean=None, bs=8192):
    """只对原编码器的门控模型。gate_zero 是测试时把门强制关上（只用标量分支）；hs_mean 是把空间分支的输出换成同任务的平均向量。
    两者都不往模型里塞别的事件的球位置，量的是“没有空间分支行不行”，和整张量打乱量的东西不同。2026-10-02 事后加的分析。"""
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
        # 加厚标量版，测试集的标量要同样加厚（取自未打乱的张量）
        for t in TASKS:
            d = data[t]["test"]
            d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
    clean = clean_matches()
    test = {k: np.concatenate([data[t]["test"][k] for t in TASKS]) for k in ("X", "y", "sm", "match")}
    test["task"] = np.concatenate([np.full(len(data[t]["test"]["y"]), TASK_ID[t]) for t in TASKS])
    dev_sm = {t: torch.tensor(data[t]["dev"]["sm"][:20000]) for t in TASKS}  # 每任务取训练赛季前两万条，用来算空间分支输出的平均向量
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
        ch = meta.get("channels", 7)  # 两通道版只有球员位置通道，此时两种打乱是同一回事
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
                # 硬门控的原始开门概率（测试时模型用的是它过 0.5 截断后的 0 或 1），用来看开着的维度是不是挤在 0.5 上方
                with torch.no_grad():
                    raw_g = torch.cat([model.gate(torch.cat([model.event_enc(torch.cat([X[i:i + 8192], model.task_embed(T[i:i + 8192])], dim=-1)),
                                                             model.spatial_enc(S[i:i + 8192, :ch])], dim=-1)) for i in range(0, len(X), 8192)])
            m_hm = task_metrics(test["task"], test["y"], *predict_variant(model, X, S[:, :ch], T, "hs_mean", hs_mean))
        print(f"\n[{a.tag} seed {seed}] 202 场干净比赛")
        print(f"{'task':14s} {'metric':>8s} {'drop_all':>9s} {'drop_players':>13s} {'gate_zero':>10s} {'hs_mean':>8s}")
        for t in TASKS:
            out[t] = dict(base=base[t], all_ablated=m_all[t], players_ablated=m_pl[t],
                          drop_all=base[t]["skill"] - m_all[t]["skill"], drop_players=base[t]["skill"] - m_pl[t]["skill"])
            if extra:
                out[t].update(drop_gate_zero=base[t]["skill"] - m_g0[t]["skill"], drop_hs_mean=base[t]["skill"] - m_hm[t]["skill"])
                # 强制关门后每事件损失的上升量（二分类用交叉熵，落点用对数损失乘 1/ln96），和门控代价同单位
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
