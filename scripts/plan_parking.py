"""Receding-horizon CEM on parking-v0, using env.step as the world model.

This is still cheating: true bicycle-model physics, no network. Random actions
never succeed (see data/random_parking.npz), so an f trained only on that npz
has never seen a slow reverse into a stall.

Cost is -reward: weighted distance of achieved_goal to desired_goal, plus the
env's collision penalty. Plan 10 steps, execute 1, replan. Same optimizer as
Pendulum (scripts/cem.py). Pinneri Appendix A.

Pass for this script: info['is_success'] True, a plot at artifacts/parking_cem.png,
and real (s, a, s') tuples at data/cem_parking.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import highway_env
import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from cem import optimize, shift_mean

gym.register_envs(highway_env)

# PLAN.md: 8–15 steps. 10 * 0.2 s = 2 s of imagination. 24*10*3 imagined
# env.steps per real action; each optimize is a couple of seconds on this laptop.
HORIZON = 10
N_SAMPLES = 24
N_ELITES = 6
N_ITERS = 3
INIT_STD = 0.5

# Crash on step 1 would otherwise look cheaper than 10 steps of near-miss
# proximity cost. Charge the remaining horizon as if each leftover step also
# paid collision_reward (-5).
CRASH_REST = 5.0


def snapshot(sim) -> dict:
    """Copy the fields step() mutates. Walls can also get crashed=True."""
    v = sim.vehicle
    objects = []
    for obj in sim.road.objects:
        impact = None if obj.impact is None else np.array(obj.impact, copy=True)
        objects.append((bool(obj.crashed), bool(getattr(obj, "hit", False)), impact))
    return {
        "position": np.array(v.position, copy=True),
        "heading": float(v.heading),
        "speed": float(v.speed),
        "action": dict(v.action),
        "crashed": bool(v.crashed),
        "hit": bool(v.hit),
        "impact": None if v.impact is None else np.array(v.impact, copy=True),
        "time": sim.time,
        "steps": sim.steps,
        "done": sim.done,
        "last_action": np.array(sim.action_type.last_action, copy=True),
        "objects": objects,
    }


def restore(sim, snap: dict) -> None:
    v = sim.vehicle
    v.position = np.array(snap["position"], copy=True)
    v.heading = snap["heading"]
    v.speed = snap["speed"]
    v.action = dict(snap["action"])
    v.crashed = snap["crashed"]
    v.hit = snap["hit"]
    v.impact = None if snap["impact"] is None else np.array(snap["impact"], copy=True)
    # _simulate applies the action only when steps % 3 == 0. If we rewind the
    # car but not the counters, later imagined steps skip throttle/steer.
    sim.time = snap["time"]
    sim.steps = snap["steps"]
    sim.done = snap["done"]
    sim.action_type.last_action = np.array(snap["last_action"], copy=True)
    for obj, (crashed, hit, impact) in zip(sim.road.objects, snap["objects"]):
        obj.crashed = crashed
        obj.hit = hit
        obj.impact = None if impact is None else np.array(impact, copy=True)
    v.on_state_update()


def plan_cost(sim, actions: np.ndarray) -> float:
    """Score one candidate on true dynamics, then rewind.

    Use sim.step (unwrapped). parking-v0 truncates on sim.time, not a
    TimeLimit wrapper, but the counters still belong to the real episode.
    """
    snap = snapshot(sim)
    cost = 0.0
    try:
        for t, action in enumerate(actions):
            _obs, reward, terminated, truncated, info = sim.step(action)
            cost += -float(reward)
            if info.get("crashed"):
                cost += CRASH_REST * (len(actions) - t - 1)
                break
            if terminated or truncated:
                break
    finally:
        restore(sim, snap)
    return cost


def xy_meters(obs: dict, scales: np.ndarray) -> tuple[float, float]:
    """Stored x,y are meters / 100. Plot in meters."""
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    return float(achieved[0] * scales[0]), float(achieved[1] * scales[1])


def goal_xy_meters(obs: dict, scales: np.ndarray) -> tuple[float, float]:
    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
    return float(desired[0] * scales[0]), float(desired[1] * scales[1])


def run_episode(env, *, use_cem: bool, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    sim = env.unwrapped
    rng = np.random.default_rng(seed)
    scales = np.asarray(sim.config["observation"]["scales"], dtype=np.float64)
    features = tuple(sim.config["observation"]["features"])

    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    action_dim = int(env.action_space.shape[0])
    mean = np.zeros((HORIZON, action_dim), dtype=np.float64)

    xs, ys = [xy_meters(obs, scales)[0]], [xy_meters(obs, scales)[1]]
    gx, gy = goal_xy_meters(obs, scales)
    rewards: list[float] = []
    success = False
    crashed = False
    rows: dict[str, list] = {key: [] for key in (
        "observation",
        "achieved_goal",
        "desired_goal",
        "action",
        "next_observation",
        "next_achieved_goal",
        "reward",
        "terminated",
        "truncated",
        "crashed",
        "is_success",
    )}

    step = 0
    terminated = truncated = False
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

        next_obs, reward, terminated, truncated, info = env.step(action)
        # Real transition only. Imagined CEM rollouts stay inside plan_cost.
        rows["observation"].append(np.asarray(obs["observation"], dtype=np.float32))
        rows["achieved_goal"].append(np.asarray(obs["achieved_goal"], dtype=np.float32))
        rows["desired_goal"].append(np.asarray(obs["desired_goal"], dtype=np.float32))
        rows["action"].append(np.asarray(action, dtype=np.float32))
        rows["next_observation"].append(np.asarray(next_obs["observation"], dtype=np.float32))
        rows["next_achieved_goal"].append(np.asarray(next_obs["achieved_goal"], dtype=np.float32))
        rows["reward"].append(float(reward))
        rows["terminated"].append(bool(terminated))
        rows["truncated"].append(bool(truncated))
        rows["crashed"].append(bool(info.get("crashed", False)))
        rows["is_success"].append(bool(info.get("is_success", False)))

        rewards.append(float(reward))
        x, y = xy_meters(next_obs, scales)
        xs.append(x)
        ys.append(y)
        success = bool(info.get("is_success", False))
        crashed = bool(info.get("crashed", False))
        obs = next_obs
        step += 1
        if use_cem and step % 10 == 0:
            print(
                f"  cem step {step:3d}  return={sum(rewards):.1f}  "
                f"success={success} crashed={crashed}"
            )

    transitions = {
        "observation": np.stack(rows["observation"]),
        "achieved_goal": np.stack(rows["achieved_goal"]),
        "desired_goal": np.stack(rows["desired_goal"]),
        "action": np.stack(rows["action"]),
        "next_observation": np.stack(rows["next_observation"]),
        "next_achieved_goal": np.stack(rows["next_achieved_goal"]),
        "reward": np.asarray(rows["reward"], dtype=np.float32),
        "terminated": np.asarray(rows["terminated"], dtype=np.bool_),
        "truncated": np.asarray(rows["truncated"], dtype=np.bool_),
        "crashed": np.asarray(rows["crashed"], dtype=np.bool_),
        "is_success": np.asarray(rows["is_success"], dtype=np.bool_),
        "episode_id": np.zeros(step, dtype=np.int32),
        "feature_names": np.asarray(features),
        "scales": scales.astype(np.float32),
        "action_names": np.asarray(("acceleration", "steering")),
        "policy_frequency": np.asarray(sim.config["policy_frequency"], dtype=np.float32),
        "duration": np.asarray(sim.config["duration"], dtype=np.float32),
        "policy": np.asarray("cem" if use_cem else "random"),
    }
    return {
        "return": float(np.sum(rewards)),
        "xs": np.asarray(xs),
        "ys": np.asarray(ys),
        "goal_xy": np.array([gx, gy]),
        "success": success,
        "crashed": crashed,
        "steps": step,
        "transitions": transitions,
    }


def plot_episode(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.plot(result["xs"], result["ys"], color="#9f1239", lw=1.4, label="ego")
    ax.scatter(result["xs"][0], result["ys"][0], c="#1c1917", s=28, zorder=3, label="start")
    gx, gy = result["goal_xy"]
    ax.scatter([gx], [gy], c="#14532d", s=42, marker="x", zorder=3, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = (
        f"CEM parking  success={result['success']}  crashed={result['crashed']}  "
        f"return={result['return']:.1f}"
    )
    ax.set_title(title)
    # loc="best" sits on the path in an empty lot.
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def concat_transitions(parts: list[dict]) -> dict:
    """Stack episodes and renumber episode_id. Metadata comes from the first."""
    step_keys = (
        "observation",
        "achieved_goal",
        "desired_goal",
        "action",
        "next_observation",
        "next_achieved_goal",
        "reward",
        "terminated",
        "truncated",
        "crashed",
        "is_success",
        "episode_id",
    )
    offset = 0
    shifted = []
    for part in parts:
        row = dict(part)
        n = len(row["reward"])
        row["episode_id"] = np.full(n, offset, dtype=np.int32)
        offset += 1
        shifted.append(row)
    out = {key: np.concatenate([p[key] for p in shifted], axis=0) for key in step_keys}
    for key in ("feature_names", "scales", "action_names", "policy_frequency", "duration", "policy"):
        out[key] = parts[0][key]
    return out


def _rewind_check() -> None:
    """One imagined step must not move the real car."""
    env = gym.make("parking-v0")
    try:
        env.reset(seed=0)
        sim = env.unwrapped
        before = snapshot(sim)
        _ = plan_cost(sim, np.ones((4, 2), dtype=np.float64))
        after = snapshot(sim)
        assert np.allclose(before["position"], after["position"]), (before["position"], after["position"])
        assert before["time"] == after["time"]
        assert before["steps"] == after["steps"]
        assert before["crashed"] == after["crashed"]
    finally:
        env.close()
    print("parking rewind self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "parking_cem.png",
    )
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many CEM parks to collect. Seeds are seed, seed+1, …",
    )
    parser.add_argument(
        "--save-transitions",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "cem_parking.npz",
        help="Real (s, a, s') from the CEM episode. Imagined rollouts are not saved.",
    )
    args = parser.parse_args()

    if not args.skip_check:
        _rewind_check()

    env = gym.make("parking-v0")
    try:
        # Several CEM parks already take minutes. Skip the random baseline then.
        if args.episodes > 1:
            args.skip_random = True
        if not args.skip_random:
            random_result = run_episode(env, use_cem=False, seed=args.seed)
            print(
                f"random return={random_result['return']:.1f}  "
                f"success={random_result['success']} crashed={random_result['crashed']}  "
                f"steps={random_result['steps']}"
            )
        cem_parts = []
        cem_result = None
        for i in range(args.episodes):
            seed = args.seed + i
            result = run_episode(env, use_cem=True, seed=seed)
            cem_parts.append(result["transitions"])
            cem_result = result
            print(
                f"cem seed={seed}  return={result['return']:.1f}  "
                f"success={result['success']} crashed={result['crashed']}  "
                f"steps={result['steps']}"
            )
    finally:
        env.close()

    print(
        f"cem    return={cem_result['return']:.1f}  "
        f"success={cem_result['success']} crashed={cem_result['crashed']}  "
        f"steps={cem_result['steps']}"
    )
    plot_episode(cem_result, args.out)
    print("saved", args.out)
    args.save_transitions.parent.mkdir(parents=True, exist_ok=True)
    saved = concat_transitions(cem_parts)
    np.savez_compressed(args.save_transitions, **saved)
    n_ok = int(saved["is_success"].sum())
    print(
        "saved",
        args.save_transitions,
        f"transitions={len(saved['reward'])} "
        f"episodes={args.episodes} success_flags={n_ok} "
        "(real steps only)",
    )
    print("CEM should reach the stall (is_success True) or at least not crash like random.")


if __name__ == "__main__":
    main()
