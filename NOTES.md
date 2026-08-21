# Notes

## Preferences
- Learning and implementation of the current PLAN.md phase happen in the same session.
- Short lessons. One tangible win. Challenge just enough.
- Windows + Git Bash. Venv at `.venv/`. Run with `.venv/Scripts/python`.
- Prefer comments in new code that explain *why* (wrappers, rewind, cheating), not what the line does.

## Corrections (do not re-teach the wrong version)
- PLAN.md originally labeled the action as `(steering, acceleration)`. highway-env `ContinuousAction` is `[acceleration, steering]` (`action[0]` throttle, `action[1]` steering). Source: `highway_env/envs/common/action.py`.
- `KinematicsGoal` sets `"normalize": False`, but `KinematicsGoalObservation.observe()` still divides by `scales` `[100, 100, 5, 5, 1, 1]`. Saved "x" is meters/100, not meters.

## Teaching
- Phase 0 pass: explain obs keys, action bounds, episode length, and have random parking transitions on disk.
- Glossary stays empty until the user uses a term correctly.
- Phase 1 Pendulum pass: `artifacts/pendulum_cem.png`, CEM return −125.7 vs random −1071.9, `|theta|` last-50 mean 0.012 rad.
- Phase 1 parking / Phase 2 collector: `scripts/plan_parking.py` (H=10, N=24, K=6, 3 iters). Seed 0 parked in 39 steps, ~90 s. Rewind `time` and `steps` or `_simulate` skips actions. Crash remainder cost 5×leftover so early crashes are not cheap.
- Phase 2 started at identity baseline + CEM parks in the dataset. Do not train `f` until more than one CEM episode is mixed with random. Identity overall MSE ≈ 0.0033 on `data/random_parking.npz` (0 / 91 successes).
- Comments in new code: explain *why* (wrappers, rewind, crash remainder), not what the line does.
