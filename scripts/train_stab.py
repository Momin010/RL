#!/usr/bin/env python3
"""Train the fin-stabilization network: clone -> DAgger -> RL refine.

The training cycle (exactly the harness->train->validate->RL loop):

  1. HARNESS DATA — fly the gain-scheduled expert through hundreds of
     randomised flights on real F-motor thrust curves and log
     (noisy features -> expert command) pairs.
  2. TRAIN — behaviour-clone an MLP on that dataset (Adam, MSE).
  3. VALIDATE — held-out MSE plus true closed-loop flights.
  4. DAgger CYCLE — fly the *student*, let the expert relabel every state the
     student actually visits, aggregate, retrain. This kills the compounding-
     error problem of naive cloning.
  5. RL FINE-TUNE — cross-entropy-method search on the network weights,
     maximising a flight reward (small tilt, no loss of control, low servo
     effort) over a fixed batch of flights. This lets the policy beat the
     teacher where the teacher's linear PD is suboptimal.

Artifacts: artifacts/stab_model.npz + stab_metrics.json.
Run:  python3 scripts/train_stab.py         (~2-4 min, pure NumPy)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocketnn import stab_sim as sim      # noqa: E402
from rocketnn.nn import MLP               # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")

ARCH = [sim.N_FEATURES, 20, 16, sim.N_OUTPUTS]
BC_EPOCHS = 60
DAGGER_ROUNDS = 3
DAGGER_FLIGHTS = 80
CEM_ITERS = 12
CEM_POP = 24
CEM_ELITE = 6
CEM_FLIGHTS = 24


def net_policy(net, mean, std):
    def policy(x):
        xn = (x - mean) / std
        return np.clip(net.forward(xn[None, :])[0], -1.0, 1.0)
    return policy


def collect_expert(n_flights, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_flights):
        r = sim.fly(sim.sample_flight(rng), sim.policy_expert, record=True)
        if len(r["feats"]):
            X.append(r["feats"])
            Y.append(r["expert"])
    return np.concatenate(X), np.concatenate(Y)


def train_epochs(net, Xn, Y, epochs, lr=1e-3, batch=256, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(Xn)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s:s + batch]
            cache = {}
            pred = net.forward(Xn[j], cache)
            dOut = 2.0 * (pred - Y[j]) / len(j)
            dW, db = net.backward(dOut, cache)
            net.adam_step(dW, db, lr=lr)
    pred = net.forward(Xn)
    return float(np.mean((pred - Y) ** 2))


def flatten(net):
    return np.concatenate([w.ravel() for w in net.W] + [b for b in net.b])


def unflatten(net, vec):
    i = 0
    for k in range(len(net.W)):
        n = net.W[k].size
        net.W[k] = vec[i:i + n].reshape(net.W[k].shape).copy()
        i += n
    for k in range(len(net.b)):
        n = net.b[k].size
        net.b[k] = vec[i:i + n].copy()
        i += n


def flight_reward(policy, flights):
    """Reward = negative cost over a fixed flight batch. Higher is better."""
    tot = 0.0
    for f in flights:
        r = sim.fly(f, policy)
        cost = r["rms_tilt_deg"] + 0.15 * r["max_tilt_deg"] + (50.0 if r["failed"] else 0.0)
        tot -= cost
    return tot / len(flights)


def main():
    os.makedirs(ART, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---- 1. harness expert data ------------------------------------------
    print("[1/5] flying expert to harness training data ...")
    X, Y = collect_expert(200, seed=7)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mean) / std
    print(f"      {len(X)} state->command pairs from 200 flights, "
          f"{len(set()) or len(sim.MOTOR_NAMES)} real motors")

    # ---- 2. behaviour cloning --------------------------------------------
    print("[2/5] behaviour cloning ...")
    net = MLP(ARCH, seed=3)
    mse = train_epochs(net, Xn, Y, BC_EPOCHS, rng=rng)
    print(f"      train MSE {mse:.5f}")

    # ---- 3. validate ------------------------------------------------------
    Xv, Yv = collect_expert(40, seed=999)
    val_mse = float(np.mean((net.forward((Xv - mean) / std) - Yv) ** 2))
    ev = sim.evaluate(net_policy(net, mean, std), n=60, seed=555)
    print(f"[3/5] validation: held-out MSE {val_mse:.5f}, closed-loop {ev}")

    # ---- 4. DAgger rounds --------------------------------------------------
    for rnd in range(DAGGER_ROUNDS):
        drng = np.random.default_rng(100 + rnd)
        pol = net_policy(net, mean, std)
        nX, nY = [], []
        for _ in range(DAGGER_FLIGHTS):
            r = sim.fly(sim.sample_flight(drng), pol, record=True)
            if len(r["feats"]):
                nX.append(r["feats"])
                nY.append(r["expert"])
        X = np.concatenate([X] + nX)
        Y = np.concatenate([Y] + nY)
        Xn = (X - mean) / std
        mse = train_epochs(net, Xn, Y, 25, lr=5e-4, rng=rng)
        ev = sim.evaluate(net_policy(net, mean, std), n=60, seed=555)
        print(f"[4/5] DAgger round {rnd + 1}/{DAGGER_ROUNDS}: "
              f"dataset {len(X)}, MSE {mse:.5f}, closed-loop "
              f"max {ev['median_max_tilt_deg']:.2f} deg, "
              f"LOC {ev['loss_of_control_pct']:.1f}%")

    # ---- 5. RL fine-tune (CEM on weights, flight reward) -------------------
    print("[5/5] RL fine-tune (cross-entropy method on flight reward) ...")
    frng = np.random.default_rng(2024)
    flights = [sim.sample_flight(frng) for _ in range(CEM_FLIGHTS)]
    mu = flatten(net)
    sigma = np.full_like(mu, 0.02)
    best_vec, best_r = mu.copy(), flight_reward(net_policy(net, mean, std), flights)
    print(f"      reward before RL: {best_r:.3f}")
    crng = np.random.default_rng(77)
    for it in range(CEM_ITERS):
        pop = mu + sigma * crng.standard_normal((CEM_POP, mu.size))
        pop[0] = best_vec  # elitism
        rewards = np.empty(CEM_POP)
        for i, vec in enumerate(pop):
            unflatten(net, vec)
            rewards[i] = flight_reward(net_policy(net, mean, std), flights)
        elite = pop[np.argsort(rewards)[-CEM_ELITE:]]
        mu = elite.mean(0)
        sigma = elite.std(0) + 1e-4
        if rewards.max() > best_r:
            best_r = rewards.max()
            best_vec = pop[int(np.argmax(rewards))].copy()
        print(f"      iter {it + 1:2d}/{CEM_ITERS}: best reward {best_r:.3f}")
    unflatten(net, best_vec)

    # ---- final report -------------------------------------------------------
    pol = net_policy(net, mean, std)
    final = sim.evaluate(pol, n=120, seed=9090)
    base = sim.evaluate(sim.policy_zero, n=120, seed=9090)
    expert = sim.evaluate(sim.policy_expert, n=120, seed=9090)
    per_motor = {m: sim.evaluate(pol, n=40, seed=31, motor_name=m)
                 for m in sim.MOTOR_NAMES}

    print("\n==== closed-loop results (120 held-out flights) ====")
    for name, r in [("no control", base), ("trained net", final), ("PD expert", expert)]:
        print(f"  {name:12s} median max tilt {r['median_max_tilt_deg']:6.2f} deg, "
              f"RMS {r['mean_rms_tilt_deg']:5.2f} deg, "
              f"loss-of-control {r['loss_of_control_pct']:.1f}%")

    n_params = sum(w.size for w in net.W) + sum(b.size for b in net.b)
    net.save(os.path.join(ART, "stab_model.npz"),
             extra={"mean": mean, "std": std})
    with open(os.path.join(ART, "stab_metrics.json"), "w") as f:
        json.dump({"arch": ARCH, "n_params": int(n_params),
                   "val_mse": val_mse, "rl_reward": float(best_r),
                   "baseline_no_control": base, "trained": final,
                   "expert": expert, "per_motor": per_motor}, f, indent=2)
    print(f"\nsaved artifacts/stab_model.npz ({n_params} params) "
          f"+ stab_metrics.json")


if __name__ == "__main__":
    main()
