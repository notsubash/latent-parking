# Notes

## Preferences
- Learning and implementation of the current PLAN.md phase happen in the same session.
- Short lessons. One tangible win. Challenge just enough.
- Windows + Git Bash. Venv at `.venv/`. Run with `.venv/Scripts/python`.

## Corrections (do not re-teach the wrong version)
- PLAN.md originally labeled the action as `(steering, acceleration)`. highway-env `ContinuousAction` is `[acceleration, steering]` (`action[0]` throttle, `action[1]` steering). Source: `highway_env/envs/common/action.py`.
- `KinematicsGoal` sets `"normalize": False`, but `KinematicsGoalObservation.observe()` still divides by `scales` `[100, 100, 5, 5, 1, 1]`. Saved "x" is meters/100, not meters.

## Teaching
- Phase 0 pass: explain obs keys, action bounds, episode length, and have random parking transitions on disk.
- Do not start MDP vs POMDP, CEM, or networks until that pass is met.
- Glossary stays empty until the user uses a term correctly.
