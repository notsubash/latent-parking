"""Collect random parking-v0 transitions to data/random_parking.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np

gym.register_envs(highway_env)

# ContinuousAction maps action[0] -> acceleration, action[1] -> steering.
ACTION_NAMES = ("acceleration", "steering")


def collect(steps: int, seed: int) -> dict[str, np.ndarray]:
    env = gym.make("parking-v0")
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    cfg = env.unwrapped.config
    features = tuple(cfg["observation"]["features"])
    scales = np.asarray(cfg["observation"]["scales"], dtype=np.float32)

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
        "episode_id",
    )}

    episode_id = 0
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
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
        rows["episode_id"].append(episode_id)
        if terminated or truncated:
            obs, info = env.reset()
            episode_id += 1
        else:
            obs = next_obs

    env.close()

    out: dict[str, np.ndarray] = {
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
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int32),
        "feature_names": np.asarray(features),
        "scales": scales,
        "action_names": np.asarray(ACTION_NAMES),
        "policy_frequency": np.asarray(cfg["policy_frequency"], dtype=np.float32),
        "duration": np.asarray(cfg["duration"], dtype=np.float32),
    }
    return out


def summarize(data: dict[str, np.ndarray]) -> None:
    n = len(data["reward"])
    n_ep = int(data["episode_id"][-1]) + 1
    ended = data["terminated"] | data["truncated"]
    n_complete = int(ended.sum())
    names = tuple(str(x) for x in data["action_names"])
    features = tuple(str(x) for x in data["feature_names"])
    print(f"transitions={n} episodes={n_ep} completed={n_complete}")
    print(f"action_space=Box(-1, 1, (2,)) names={names}")
    print(f"features={features} scales={data['scales'].tolist()}")
    print(
        "max_episode_actions="
        f"{int(data['duration'] * data['policy_frequency'])} "
        f"(duration={float(data['duration'])}s, policy_frequency={float(data['policy_frequency'])}Hz)"
    )
    if n_complete:
        print(
            f"episode_end crashed={data['crashed'][ended].mean():.3f} "
            f"success={data['is_success'][ended].mean():.3f} "
            f"timeout={data['truncated'][ended].mean():.3f}"
        )
    lengths = np.bincount(data["episode_id"])
    print(f"episode_len mean={lengths.mean():.1f} median={int(np.median(lengths))} max={int(lengths.max())}")


def check(data: dict[str, np.ndarray]) -> None:
    n = len(data["reward"])
    assert n > 0
    assert data["observation"].shape == (n, 6)
    assert data["action"].shape == (n, 2)
    assert data["next_observation"].shape == (n, 6)
    assert np.all(data["action"] >= -1.0) and np.all(data["action"] <= 1.0)
    assert np.array_equal(data["observation"], data["achieved_goal"])
    assert tuple(data["action_names"]) == ACTION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "random_parking.npz",
    )
    args = parser.parse_args()

    data = collect(args.steps, args.seed)
    check(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    summarize(data)
    print("saved", args.out)


if __name__ == "__main__":
    main()
