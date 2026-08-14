# Latent parking planner

Goal-conditioned latent world model that parks in `parking-v0` and refuses the maneuver when imagined rollouts disagree.

This is a solo learning project: world models, JEPA-style latent prediction, model-predictive control, and online uncertainty. No extra hardware. The environment is `highway-env`.

## Setup

Python 3.11+ (3.13 is fine if the sanity check passes).

```bash
cd latent-parking
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Sanity check

Headless (saves a frame under `artifacts/`):

```bash
python scripts/sanity_check.py
```

Interactive window on your machine:

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)
env = gym.make("parking-v0", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

If `highway-env` will not launch on Windows, switch to Gymnasium `Pusher-v5` (`pip install "gymnasium[mujoco]"`). Same project shape, different env. Do not spend a week debugging install.

## Docs

Read [PLAN.md](PLAN.md) for goals, success criteria, and the week-by-week learning plan.

## Hardware

NVIDIA 4070 Ti Super, 16 GB. First models are tiny MLPs on kinematic state. Stay under ~12 GB VRAM so the sim and planner still fit.
