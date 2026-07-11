#!/usr/bin/env python3
"""Train the evasion policy/classifier and evaluate it end to end.

Pipeline:
  1. Build a supervised dataset from expert engagements.
  2. Standardise features; split train/val.
  3. Train the MLP with a joint loss: MSE on the two fin commands +
     softmax cross-entropy on the 4-way threat class.
  4. Report held-out regression error and classification accuracy.
  5. Fly the trained network in closed loop and compare its miss distance
     against flying straight (no evasion) and against the expert.
  6. Save artifacts/model.npz (weights + normalisation + arch) and
     artifacts/metrics.json.

Run:  python3 scripts/train.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketnn import data, simulator as sim          # noqa: E402
from rocketnn.nn import MLP                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")

# Network architecture and training hyper-parameters.
HIDDEN = [24, 16]
SEED = 1
N_EPISODES = 2600
EPOCHS = 45
BATCH = 256
LR = 2.0e-3
W_REG = 1.0            # weight on fin-command regression loss
W_CLS = 0.30           # weight on threat-classification loss


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_controller(net, mean, std):
    """Wrap a trained net as a closed-loop controller (features -> fins)."""
    def controller(feat, carry):
        x = (feat - mean) / std
        out = net.forward(x[None, :])[0]
        fp = float(np.clip(out[0], -1.0, 1.0))
        fy = float(np.clip(out[1], -1.0, 1.0))
        return fp, fy
    return controller


def closed_loop_eval(net, mean, std, n_per_class=120, seed=999):
    """Median miss distance per class for none / expert / trained net."""
    rng = np.random.default_rng(seed)
    results = {c: {"none": [], "expert": [], "net": []} for c in range(sim.N_CLASSES)}
    ctrl = make_controller(net, mean, std)
    for kind in range(sim.N_CLASSES):
        for _ in range(n_per_class):
            scn = sim.sample_scenario(rng, kind=kind)
            # Same scenario, three controllers, independent noise draws.
            results[kind]["none"].append(
                sim.run_engagement(scn, "none", np.random.default_rng(rng.integers(1 << 30)))["miss"])
            results[kind]["expert"].append(
                sim.run_engagement(scn, "expert", np.random.default_rng(rng.integers(1 << 30)))["miss"])
            results[kind]["net"].append(
                sim.run_engagement(scn, ctrl, np.random.default_rng(rng.integers(1 << 30)))["miss"])
    summary = {}
    for kind in range(sim.N_CLASSES):
        summary[sim.CLASS_NAMES[kind]] = {
            k: float(np.median(v)) for k, v in results[kind].items()
        }
    return summary


def main():
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)

    print(f"[1/5] Building dataset from {N_EPISODES} expert engagements ...")
    X, Yreg, Ycls, meta = data.build_dataset(n_episodes=N_EPISODES, seed=SEED)
    print(f"      {meta['n_samples']} samples  classes={meta['class_counts']}")
    print(f"      expert median miss (m): {meta['expert_median_miss']}")

    mean, std = data.standardize_stats(X)
    Xn = (X - mean) / std

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xn))
    Xn, Yreg, Ycls = Xn[perm], Yreg[perm], Ycls[perm]
    n_val = len(Xn) // 10
    Xtr, Xva = Xn[n_val:], Xn[:n_val]
    Rtr, Rva = Yreg[n_val:], Yreg[:n_val]
    Ctr, Cva = Ycls[n_val:], Ycls[:n_val]
    onehot_tr = np.eye(sim.N_CLASSES)[Ctr]

    net = MLP([sim.N_FEATURES] + HIDDEN + [sim.N_OUTPUTS], seed=SEED)

    print(f"[2/5] Training  arch={net.sizes}  epochs={EPOCHS}  batch={BATCH}")
    n = len(Xtr)
    for epoch in range(EPOCHS):
        idx = rng.permutation(n)
        lr = LR * (0.5 ** (epoch // 20))          # step decay
        run_reg = run_cls = 0.0
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            xb, rb, ob = Xtr[b], Rtr[b], onehot_tr[b]
            m = len(b)
            cache = {}
            out = net.forward(xb, cache=cache)
            reg_pred, logits = out[:, :2], out[:, 2:]

            d_reg = W_REG * 2.0 * (reg_pred - rb) / m
            p = softmax(logits)
            d_cls = W_CLS * (p - ob) / m
            dOut = np.concatenate([d_reg, d_cls], axis=1)

            dW, db = net.backward(dOut, cache)
            net.adam_step(dW, db, lr=lr)

            run_reg += float(((reg_pred - rb) ** 2).sum())
            run_cls += float(-np.log(p[np.arange(m), Ctr[b]] + 1e-12).sum())
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            va_out = net.forward(Xva)
            va_mse = float(((va_out[:, :2] - Rva) ** 2).mean())
            va_acc = float((va_out[:, 2:].argmax(1) == Cva).mean())
            print(f"      epoch {epoch:2d}  train_mse={run_reg/n:.4f}  "
                  f"val_mse={va_mse:.4f}  val_acc={va_acc:.3f}  lr={lr:.1e}")

    # Final held-out metrics.
    va_out = net.forward(Xva)
    val_mse = float(((va_out[:, :2] - Rva) ** 2).mean())
    val_acc = float((va_out[:, 2:].argmax(1) == Cva).mean())
    conf = np.zeros((sim.N_CLASSES, sim.N_CLASSES), dtype=int)
    pred = va_out[:, 2:].argmax(1)
    for t, pr in zip(Cva, pred):
        conf[t, pr] += 1
    print(f"[3/5] Held-out: fin_MSE={val_mse:.4f}  class_acc={val_acc:.3f}")

    print("[4/5] Closed-loop evaluation (median miss distance, metres) ...")
    cl = closed_loop_eval(net, mean, std)
    for name, d in cl.items():
        print(f"      {name:10s} none={d['none']:6.1f}  "
              f"expert={d['expert']:6.1f}  net={d['net']:6.1f}")

    print("[5/5] Saving artifacts ...")
    net.save(os.path.join(ART, "model.npz"),
             extra=dict(mean=mean, std=std,
                        hidden=np.array(HIDDEN),
                        n_features=sim.N_FEATURES,
                        n_outputs=sim.N_OUTPUTS))
    metrics = dict(
        arch=net.sizes,
        dataset=meta,
        val_fin_mse=val_mse,
        val_class_acc=val_acc,
        confusion=conf.tolist(),
        confusion_labels=sim.CLASS_NAMES,
        closed_loop_median_miss=cl,
        train_seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      wrote artifacts/model.npz and artifacts/metrics.json "
          f"in {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
