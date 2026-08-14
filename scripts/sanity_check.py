"""Open parking-v0, step random actions, print spaces, save one frame."""

from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np

gym.register_envs(highway_env)

STEPS = 50
OUT = Path(__file__).resolve().parent.parent / "artifacts"
OUT.mkdir(exist_ok=True)


def main() -> None:
    env = gym.make("parking-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    print("env:", env.spec.id if env.spec else "parking-v0")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("obs type:", type(obs))
    if isinstance(obs, dict):
        for key, value in obs.items():
            arr = np.asarray(value)
            print(f"  obs[{key!r}] shape={arr.shape} dtype={arr.dtype}")
    print("info keys:", sorted(info.keys()))

    frame = env.render()
    assert frame is not None and frame.ndim == 3, "expected an rgb frame"
    print("frame shape:", frame.shape)

    rewards = []
    crashed = False
    for _ in range(STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        crashed = crashed or bool(info.get("crashed", False))
        if terminated or truncated:
            obs, info = env.reset()

    last = env.render()
    np.save(OUT / "sanity_last_frame.npy", last)

    try:
        from matplotlib import pyplot as plt

        plt.imsave(OUT / "sanity_last_frame.png", last)
        print("saved", OUT / "sanity_last_frame.png")
    except Exception as exc:
        print("png save skipped:", exc)

    env.close()
    print(f"steps={STEPS} reward_mean={np.mean(rewards):.4f} crashed={crashed}")
    print("sanity check passed")


if __name__ == "__main__":
    main()
