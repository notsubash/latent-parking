# Mission: latent parking planner

## Why
You want a parking assistant that imagines a few seconds ahead in latent space, steers toward a goal pose, and **aborts or replans when it no longer trusts its own imagination**. The skill is model-based control with an honest distrust gate, not a bigger policy network.

## Success looks like
- Short-horizon predictions beat "predict no change."
- Closed-loop MPC parks more often than random on in-distribution goals.
- An online uncertainty signal catches a large fraction of near-collisions **before** the crash, with a precision/recall number you can defend.
- You can point at your own figures and say where the latent model is lying.

## Constraints
- Windows + pip. Environment: Farama `highway-env` `parking-v0`. GPU: NVIDIA 4070 Ti Super (16 GB). Stay under ~12 GB VRAM.
- Intermediate PyTorch. No RL background yet. Do not start the next PLAN.md phase until the current pass/fail artifact exists.
- Level 0 learning PoC: working loop plus diagnostics, not a SOTA table.

## Out of scope
- Habitat, custom 3D assets, a real car, V-JEPA or video diffusion trained from scratch.
- PPO, SAC, or Dreamer as a first algorithm. Pixels before the kinematic version works. Defining success as "zero-shot parking just works."
