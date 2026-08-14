# Plan: latent parking planner

A parking assistant that imagines a few seconds ahead in latent space, steers toward a goal pose, and **aborts or replans when it no longer trusts its own imagination**.

Environment: Farama `highway-env` `parking-v0`. No robot, no custom assets, no Habitat. GPU: NVIDIA 4070 Ti Super (16 GB). Background: intermediate PyTorch, no RL yet.

This is a Level 0 learning PoC. Success is a working loop plus honest diagnostics, not a SOTA table.

---

## The actual problem

Parking is a reach task with a heading constraint. Being a little wrong near another car is a collision. A learned world model will look fine for a short horizon and then lie. The product is the **gate**: act only while imagined rollouts still agree with each other and with the latest real observation.

One-liner for the README:

> Goal-conditioned latent world model that parks in `parking-v0` and refuses the maneuver when imagined rollouts disagree.

---

## Goals

### Learning goals

1. Get fluent in the agent/environment loop (reset, step, transitions as a dataset).
2. Understand MDP vs POMDP enough to know what you are hiding when you drop velocities or switch to pixels.
3. Implement CEM (or MPPI) on **true** dynamics before any neural net, so a later failure is not "maybe CEM is buggy."
4. Train a next-state model, then a JEPA-style latent predictor (encoder + EMA target + predictor, no pixel reconstruction as the main loss).
5. Plan in latent space toward a goal embedding, receding horizon.
6. Diagnose **where** rollouts diverge, and detect that **online** with uncertainty.

You do not need PPO, SAC, or Dreamer as a first algorithm. You need model-based thinking and MPC.

### Project goals (what to ship)

| Priority | Artifact |
|---|---|
| Must | Open-loop prediction error vs horizon plot |
| Must | Latent or state-space MPC that parks more often than random |
| Must | Held-out goal poses (spots / headings not treated as labeled train tasks) |
| Must | Online distrust signal (ensemble disagreement and/or one-step latent residual) |
| Must | Failure cases: when the detector fires too late or not at all |
| Nice | Side-by-side real vs imagined trajectory overlay |
| Later | 64×64 pixels instead of kinematics |
| Not this project | Habitat, V-JEPA from scratch, diffusion video models, a real car |

### Honest success bar

You "solved" this if:

1. Short-horizon predictions beat "predict no change."
2. Closed-loop MPC beats random steering on in-distribution goals.
3. On near-collisions and tight spots, the uncertainty gate catches a large fraction of failures **before** the crash, with a precision/recall number you can defend.
4. You can explain, with your own figures, when the latent model is lying.

You did **not** fail if zero-shot parking is imperfect. Collapse under long horizons and novel layouts **is** the interesting result if you instrument it.

---

## Constraints

- **Windows + pip.** Prefer `highway-env`. Fallback: `gymnasium[mujoco]` `Pusher-v5` (same plan: goal reach + abort).
- **16 GB VRAM.** Kinematic MLP first. Pixels at 64×64 with AMP only after the state version works. Do not load models that need more than ~12 GB.
- **No extra hardware.** Laptop GPU and this repo.
- **Stay in `parking-v0` after a short Pendulum warmup.** Do not collect CartPole for months.

### Environment facts (`parking-v0`)

- Goal-conditioned continuous control: reach a parking pose with the right heading.
- Observation (default `KinematicsGoal`): `x, y, vx, vy, cos_h, sin_h` for achieved and desired goal (dict with `observation`, `achieved_goal`, `desired_goal`).
- Action: continuous acceleration and steering (`ContinuousAction`).
- Default lot is empty. `parking-parked-v0` adds 10 parked cars as obstacles. Use empty lot until MPC works, then parked cars as the hard test.
- Collision reward is large and negative. Success is proximity to the goal under a weighted p-norm.

Docs: [highway-env parking](https://highway-env.farama.org/environments/parking/).

---

## Learning plan

Each phase has a **pass/fail**. Do not start the next one until you have the plot or video from the current one. If a week produces no video or plot of the car, it was too abstract.

### Phase 0 — Gym loop (about 1 week)

**Do**

- Install this repo, run `python scripts/sanity_check.py`.
- Run `parking-v0` with `render_mode="human"` on your machine so you see the lot.
- Optional warmup: Gymnasium Pendulum, random policy, save `(obs, action, next_obs)` tuples.

**Pass:** you can explain observation keys, action bounds, episode length, and you have a dataset of random parking transitions on disk.

**Read (light):** OpenAI Spinning Up "Intro to RL" (concepts only). Sutton & Barto ch. 1–3.

### Phase 1 — Planning with known physics (about 1–2 weeks)

**Do**

- Implement CEM or MPPI on **Pendulum using `env.step` as the model** (cheating, on purpose).
- Then the same planner on `parking-v0` using the real env as the model: cost = distance between `achieved_goal` and `desired_goal` (position + heading). Plan 8–15 steps, execute 1, replan.

**Pass:** Pendulum swings up, or the parking car reaches the spot using true-dynamics MPC, with no learned weights.

**Read:** Sutton & Barto ch. 8 (planning). CEM / MPPI at a high level.

This phase is the most important. If CEM is wrong, every later plot is garbage.

### Phase 2 — Learned world model in true state (about 3 weeks)

**Do**

- Collect data: random actions, plus some true-env CEM so the model sees successful parks, not only noise.
- Train `f(s, a) → s'` as an MLP. MSE on next kinematics.
- Plot **open-loop rollout error vs horizon** (1, 2, 5, 10, 20 steps). This is the core diagnostic of the field.
- Plug learned `f` into the Phase 1 CEM.

**Pass:**

- One-step prediction beats "predict no change."
- Short-horizon MPC beats random.
- You can point to the horizon where rollouts blow up.

**Read:** skip Dreamer internals. Ask only: what is predicted, how is it used to plan?

### Phase 3 — Partial observability (about 1–2 weeks)

**Do**

- Hide `vx, vy` (and maybe heading rate if present). Positions and heading only, or a short history.
- Same MLP or a tiny GRU / frame stack.

**Pass:** prediction and parking get worse in a way you can explain. That is your reason for representation learning.

### Phase 4 — JEPA-style latent model (about 4–6 weeks)

**Do**

- Encoder `E(obs) → z` (MLP on kinematics).
- Target encoder: EMA copy of `E`, stop-grad.
- Predictor `P(z_t, a_t) → z_{t+1}`.
- Loss: distance in latent space to `E_target(obs_{t+1})`. **No decoder as the main objective.** A decoder is allowed only to visualize.
- Multi-step latent rollouts. Same error-vs-horizon plot, now in `z`.
- Optional: PCA of latents colored by `x, y`, heading, or "near obstacle."

**Pixels later, not now:** 64×64 `rgb_array`, small CNN, batch 32–64, AMP. Do not start at 224×224.

**Pass:** multi-step latent error grows slower than a "predict identity" baseline.

**Read:** LeCun on JEPA (why latents, not pixels). After this works, skim V-JEPA 2-AC planning (goal embedding, energy, CEM) as a north star, not as code to copy at full scale. Optionally read TD-MPC2's world model + CEM and ignore value/policy heads.

### Phase 5 — Latent MPC on held-out goals (about 3–4 weeks)

**Do**

- CEM/MPPI minimizes `||z_pred - z_goal||` (encode `desired_goal`, or the full goal features).
- Receding horizon: plan 8–12 steps, execute 1, replan.
- Train on a set of goal poses. Test on **held-out** spots / headings.
- Then try `parking-parked-v0` (other cars) as a harder distribution.

**Pass:** success rate above random and above open-loop (no replan). Keep failure videos.

### Phase 6 — Online divergence detection (about 4–6 weeks)

This is the contribution. The planner is allowed to refuse.

**Signals to implement**

1. One-step residual after acting: `||P(E(o_t), a_t) - E(o_{t+1})||`
2. Ensemble disagreement: 3–5 predictors, std of predicted `z`
3. Optional: imagined-horizon error growth **before** committing the action

**Then**

- Label failures: crash, timeout, or final distance above a threshold.
- Precision/recall of "I should not trust this plan."
- Policy: if uncertain, shorten horizon, replan, or abort.

**Pass:** a plot where uncertainty rises **before** visible failure, plus cases where it does not (slow drift into a parked car, weird heading, etc.).

---

## Weekly rhythm

Every week, one plot or video and one paragraph:

- What I trained
- What I expected
- What diverged
- Next hypothesis

That log *is* the project. Put notes in `notes/` when you start training (not before).

Vectorize CEM: one batched forward for `N` action sequences of length `T`, not a Python loop over candidates.

---

## Reading order (keep it short)

1. Spinning Up intro (concepts)
2. Sutton & Barto 1–3 and 8
3. JEPA motivation (LeCun)
4. TD-MPC2 page/paper: encoder, predictor, CEM only
5. V-JEPA 2 as north star, not an implementation target

Skip for now: PPO details, Rainbow, offline RL theory, hierarchical RL, sim-to-real on hardware.

---

## What not to do

- Do not install Habitat.
- Do not train V-JEPA or a video diffusion model from scratch.
- Do not start on pixels.
- Do not define success as "zero-shot parking just works."
- Do not build a custom pool table or buy hardware.

---

## Fallback

If `parking-v0` cannot be imported on this machine:

```text
pip install "gymnasium[mujoco]"
python -c "import gymnasium as gym; e=gym.make('Pusher-v5'); print(e.reset())"
```

Same phases: learn `s,a → s'`, JEPA, CEM to a held-out target, abort on disagreement. Slightly more contact physics, slightly less obvious "product."

---

## Sanity check (done)

On this machine, `python scripts/sanity_check.py` passed with:

- `parking-v0`
- action: `Box(-1, 1, (2,), float32)` (steering, acceleration)
- obs dict: `observation`, `achieved_goal`, `desired_goal`, each `(6,)` (`x, y, vx, vy, cos_h, sin_h`)
- info: `action`, `crashed`, `is_success`, `speed`
- rgb frame: `300 x 600 x 3`
- 50 random steps, no crash, mean reward about `-0.62`

Frame saved at `artifacts/sanity_last_frame.png`. Open `parking-v0` with `render_mode="human"` locally so you see the lot.

---

## Setup reminder

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python scripts/sanity_check.py
```
