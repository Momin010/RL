"""Build a supervised dataset by flying the expert through many engagements.

Each timestep of each engagement yields one training sample:
    features (10,)  ->  [expert_fin_pitch, expert_fin_yaw, threat_class]

We also compute per-feature mean/std for standardisation. Those statistics are
baked into the C header so the Teensy normalises its inputs identically.
"""

from __future__ import annotations

import numpy as np

from . import simulator as sim


def build_dataset(n_episodes=3000, seed=0):
    """Fly the expert through ``n_episodes`` engagements and collect samples.

    Returns X (N,10), Yreg (N,2), Ycls (N,), plus a dict of metadata including
    the class balance and the expert's own miss-distance statistics.
    """
    rng = np.random.default_rng(seed)
    X, Yreg, Ycls = [], [], []
    expert_miss = {k: [] for k in range(sim.N_CLASSES)}

    for _ in range(n_episodes):
        scn = sample_scenario_balanced(rng)
        res = sim.run_engagement(scn, "expert", rng, record=True)
        expert_miss[res["kind"]].append(res["miss"])
        if len(res["feats"]) == 0:
            continue
        X.append(res["feats"])
        for fp, fy, k in res["acts"]:
            Yreg.append([fp, fy])
            Ycls.append(k)

    X = np.concatenate(X, axis=0)
    Yreg = np.array(Yreg, dtype=np.float64)
    Ycls = np.array(Ycls, dtype=np.int64)

    meta = dict(
        n_samples=int(X.shape[0]),
        class_counts={sim.CLASS_NAMES[k]: int((Ycls == k).sum())
                      for k in range(sim.N_CLASSES)},
        expert_median_miss={sim.CLASS_NAMES[k]: float(np.median(v)) if v else None
                            for k, v in expert_miss.items()},
    )
    return X, Yreg, Ycls, meta


def sample_scenario_balanced(rng):
    """Uniformly pick a class, then sample a scenario of that class."""
    kind = int(rng.integers(0, sim.N_CLASSES))
    return sim.sample_scenario(rng, kind=kind)


def standardize_stats(X):
    """Per-feature mean/std with a floor on std to avoid divide-by-zero."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std
