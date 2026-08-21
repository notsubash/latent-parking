# Notes

## Preferences
- Learning and implementation of the current PLAN.md phase happen in the same session.
- Short lessons. One tangible win. Challenge just enough.
- Windows + Git Bash. Venv at `.venv/`. Run with `.venv/Scripts/python`.
- Prefer comments in new code that explain *why* (wrappers, rewind, cheating), not what the line does.
- Do not label lessons, scripts, or teaching replies with PLAN.md phase numbers.
- After every lesson, update `GLOSSARY.md` with each new term (ELI5, `GLOSSARY-FORMAT.md`). Link the lesson to it. Do not wait to be asked.

## Corrections (do not re-teach the wrong version)
- PLAN.md originally labeled the action as `(steering, acceleration)`. highway-env `ContinuousAction` is `[acceleration, steering]` (`action[0]` throttle, `action[1]` steering). Source: `highway_env/envs/common/action.py`.
- `KinematicsGoal` sets `"normalize": False`, but `KinematicsGoalObservation.observe()` still divides by `scales` `[100, 100, 5, 5, 1, 1]`. Saved "x" is meters/100, not meters.

## Teaching
- Phase 0 pass: explain obs keys, action bounds, episode length, and have random parking transitions on disk.
- Glossary: `GLOSSARY.md` is live and ELI5. Update it with every new lesson.
- Phase 1 Pendulum pass: `artifacts/pendulum_cem.png`, CEM return −125.7 vs random −1071.9, `|theta|` last-50 mean 0.012 rad.
- Phase 1 parking / Phase 2 collector: `scripts/plan_parking.py` (H=10, N=24, K=6, 3 iters). Seed 0 parked in 39 steps, ~90 s. Rewind `time` and `steps` or `_simulate` skips actions. Crash remainder cost 5×leftover so early crashes are not cheap.
- Identity overall MSE ≈ 0.0033 on `data/random_parking.npz` (0 / 91 successes). Mixed train of `f` uses that file plus `data/cem_parking.npz` (one 39-step park). More CEM parks still matter before trusting parking-MPC with `f`; the pose-vs-full plot does not.
- User asked not to refer to phases in teaching or code.
- Hide vx, vy and compare pose-error vs horizon to the full six-vector MLP (`scripts/train_dynamics.py`).
