"""Receding-horizon CEM on Pendulum-v1, using env.step as the world model.

This is cheating on purpose. The planner sees true physics, so if the pendulum
does not swing up, CEM is wrong, not a neural net. Phase 1 pass for this
script: return clearly better than random, angle near 0 (upright) at the end.

Pendulum facts (Gymnasium docs + pendulum.py in this venv):
  obs = [cos(theta), sin(theta), theta_dot], upright is theta = 0
  action = torque in [-2, 2]
  reward = -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
  episode truncates at 200 steps; physics lives in env.unwrapped.state
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from cem import optimize, shift_mean

# Short horizon is enough here: dt = 0.05 s, so 20 steps is 1 second of
# imagination. Parking will use 8-15 for a different reason (collision).
HORIZON = 20
N_SAMPLES = 48
N_ELITES = 10
N_ITERS = 5
INIT_STD = 1.0


def angle_from_obs(obs: np.ndarray) -> float:
    """theta in [-pi, pi] from (cos, sin). 0 is upright."""
    return float(np.arctan2(obs[1], obs[0]))


def snapshot(sim) -> tuple[np.ndarray, float | None]:
    """Copy the two fields step() reads/writes besides the wrappers."""
    return np.array(sim.state, copy=True), sim.last_u


def restore(sim, state: np.ndarray, last_u: float | None) -> None:
    sim.state = np.array(state, copy=True)
    sim.last_u = last_u


def plan_cost(sim, actions: np.ndarray) -> float:
    """Score one candidate by replaying it on true dynamics, then rewind.

    Use sim.step, not env.step. env is wrapped in TimeLimit(200). Planning
    with the wrapper would spend the real episode's step budget on imagined
    rollouts and truncate the life you meant to control.
    """
    state0, last_u0 = snapshot(sim)
    cost = 0.0
    try:
        for action in actions:
            _obs, reward, terminated, truncated, _info = sim.step(action)
            cost += -float(reward)
            if terminated or truncated:
                break
    finally:
        restore(sim, state0, last_u0)
    return cost


def run_episode(env, *, use_cem: bool, seed: int) -> dict[str, np.ndarray | float]:
    obs, _info = env.reset(seed=seed)
    env.action_space.seed(seed)
    sim = env.unwrapped
    rng = np.random.default_rng(seed)

    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    action_dim = int(env.action_space.shape[0])
    mean = np.zeros((HORIZON, action_dim), dtype=np.float64)

    rewards: list[float] = []
    angles: list[float] = [angle_from_obs(obs)]
    torques: list[float] = []

    step = 0
    truncated = False
    terminated = False
    while not (terminated or truncated):
        if use_cem:
            _mean, best, _cost = optimize(
                lambda actions, s=sim: plan_cost(s, actions),
                horizon=HORIZON,
                action_dim=action_dim,
                action_low=low,
                action_high=high,
                n_samples=N_SAMPLES,
                n_elites=N_ELITES,
                n_iters=N_ITERS,
                init_std=INIT_STD,
                mean=mean,
                rng=rng,
            )
            action = best[0].astype(np.float32)
            mean = shift_mean(best)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(float(reward))
        angles.append(angle_from_obs(obs))
        torques.append(float(np.asarray(action).reshape(-1)[0]))
        step += 1
        if use_cem and step % 25 == 0:
            print(f"  cem step {step:3d}  angle={angles[-1]:+.2f}  return={sum(rewards):.1f}")

    return {
        "return": float(np.sum(rewards)),
        "angles": np.asarray(angles),
        "rewards": np.asarray(rewards),
        "torques": np.asarray(torques),
    }


def plot_episode(result: dict[str, np.ndarray | float], path: Path) -> None:
    angles = np.asarray(result["angles"])
    t = np.arange(len(angles))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(t, angles, color="#9f1239", lw=1.4)
    axes[0].axhline(0.0, color="#a8a29e", lw=0.8)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title(f"CEM Pendulum  return={result['return']:.1f}  (0 is up)")
    axes[1].plot(np.arange(len(result["torques"])), result["torques"], color="#1c1917", lw=1.0)
    axes[1].set_ylabel("torque")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "pendulum_cem.png",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Only run CEM (faster when you already believe random is bad).",
    )
    args = parser.parse_args()

    env = gym.make("Pendulum-v1")
    try:
        if not args.skip_random:
            random_result = run_episode(env, use_cem=False, seed=args.seed)
            print(f"random return={random_result['return']:.1f}  (typical: around -1000 to -1600)")

        cem_result = run_episode(env, use_cem=True, seed=args.seed)
    finally:
        env.close()

    tail = np.abs(np.asarray(cem_result["angles"][-50:]))
    print(
        f"cem    return={cem_result['return']:.1f}  "
        f"|theta| last-50 mean={tail.mean():.3f} rad"
    )
    plot_episode(cem_result, args.out)
    print("saved", args.out)
    print(
        "Phase 1 Pendulum pass: CEM return much better than random, "
        "and |theta| near 0 by the end (upright)."
    )


if __name__ == "__main__":
    main()
