"""Cross-entropy search over action sequences.

Phase 1 uses this twice: Pendulum first, then parking-v0. The optimizer does
not know about cars or pendulums. It only sees a cost function.

Vanilla CEM-MPC (sample, elite, refit a diagonal Gaussian, execute the first
action, shift the mean, replan) is Pinneri et al. 2020, Appendix A:
https://arxiv.org/abs/2008.06389
Skip their iCEM extras (colored noise, elite memory) until this version parks.
"""

from __future__ import annotations

import numpy as np


def shift_mean(mean: np.ndarray) -> np.ndarray:
    """Warm-start the next plan: drop the action we just took, repeat the last.

    Receding horizon always commits action 0. The leftover [1:] is still a
    decent guess for the new [0:-1], so CEM does not start from noise each step.
    """
    shifted = np.empty_like(mean)
    shifted[:-1] = mean[1:]
    shifted[-1] = mean[-1]
    return shifted


def optimize(
    cost_fn,
    *,
    horizon: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    n_samples: int = 48,
    n_elites: int = 10,
    n_iters: int = 5,
    init_std: float = 1.0,
    mean: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a Gaussian over plans of shape (horizon, action_dim).

    Returns (mean_plan, best_plan, best_cost). Execute best_plan[0] on the
    real env: that sequence was actually rolled out, the mean might not have
    been (Pinneri §3.3, "best-a").

    ponytail: cost_fn is called in a Python loop. Fine while the "model" is
    env.step. When the model is a batched MLP, score all N sequences in one
    forward and delete this loop.
    """
    if n_elites < 2:
        raise ValueError("n_elites must be >= 2 so std is defined")
    if n_elites > n_samples:
        raise ValueError("n_elites cannot exceed n_samples")

    low = np.asarray(action_low, dtype=np.float64).reshape(action_dim)
    high = np.asarray(action_high, dtype=np.float64).reshape(action_dim)
    rng = rng or np.random.default_rng()

    if mean is None:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
    else:
        mean = np.asarray(mean, dtype=np.float64).reshape(horizon, action_dim)

    # Reset std every optimize() call. Carrying a collapsed std from the
    # previous real step makes later plans unable to explore.
    std = np.full((horizon, action_dim), init_std, dtype=np.float64)
    best_plan = mean.copy()
    best_cost = np.inf

    for _ in range(n_iters):
        noise = rng.normal(size=(n_samples, horizon, action_dim))
        samples = mean + std * noise
        samples = np.clip(samples, low, high)

        costs = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            costs[i] = float(cost_fn(samples[i]))

        elite_idx = np.argpartition(costs, n_elites - 1)[:n_elites]
        elites = samples[elite_idx]
        elite_costs = costs[elite_idx]

        winner = int(np.argmin(elite_costs))
        if elite_costs[winner] < best_cost:
            best_cost = float(elite_costs[winner])
            best_plan = elites[winner].copy()

        mean = elites.mean(axis=0)
        # Floor keeps the next iteration from sampling a delta spike.
        std = np.maximum(elites.std(axis=0), 1e-6)

    return mean, best_plan, best_cost


def _self_check() -> None:
    """1-D quadratic: CEM should sit near x=3 after a few elite refits."""

    def cost(plan: np.ndarray) -> float:
        x = float(plan[0, 0])
        return (x - 3.0) ** 2

    rng = np.random.default_rng(0)
    _mean, best, best_cost = optimize(
        cost,
        horizon=1,
        action_dim=1,
        action_low=np.array([-10.0]),
        action_high=np.array([10.0]),
        n_samples=32,
        n_elites=8,
        n_iters=6,
        init_std=2.0,
        rng=rng,
    )
    assert abs(float(best[0, 0]) - 3.0) < 0.25, best
    assert best_cost < 0.1, best_cost
    print("cem self-check passed", f"best={float(best[0, 0]):.3f}")


if __name__ == "__main__":
    _self_check()
