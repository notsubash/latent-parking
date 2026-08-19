# Latent parking resources

## Knowledge

- [Gymnasium: Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
  The agent-environment loop in code: `make`, `reset`, `step`, `render`, spaces, and wrappers. Use for: Phase 0 API. Primary source for lesson 0001.
- [Gymnasium: `Env.step` / `Env.reset`](https://gymnasium.farama.org/api/env/)
  Exact return types. `terminated` is an MDP ending (goal or crash). `truncated` is an outside cutoff (time limit). Use for: any code that stores transitions.
- [highway-env: Parking](https://highway-env.farama.org/environments/parking/)
  `parking-v0` config: `KinematicsGoal` features, `ContinuousAction`, duration, frequencies, reward weights, `parking-parked-v0`. Use for: env facts. Confirm against source if a number matters.
- [highway-env: Actions](https://highway-env.farama.org/actions/)
  Continuous control is throttle then steering, each in `[-1, 1]`, mapped to m/s² and radians. Use for: action meaning. Do not trust comments that reverse this order.
- [highway-env: Observations](https://highway-env.farama.org/observations/)
  Feature names (`x, y, vx, vy, cos_h, sin_h`) and `KinematicsGoalObservation`. Use for: decoding saved arrays back into meters.
- [OpenAI Spinning Up: Part 1, Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
  Agent, environment, observation vs state, action space, trajectory, reward/return. Use for: vocabulary. Stop before value functions and Bellman equations until Phase 1.
- [Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
  Canonical textbook, PDF hosted by the authors. Use for: ch. 1–3 after the gym loop feels boring; ch. 8 when you implement planning. Not the Phase 0 primary source.
- [HighwayEnv `parking_env.py` (source)](https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/envs/parking_env.py)
  Ground truth for terminate/truncate, success threshold, and the GoalEnv dict. Use for: any claim about episode length or reward.

## Wisdom (Communities)

- [Farama Foundation Discord](https://discord.com/invite/farama)
  Maintainers of Gymnasium and highway-env. Use for: "is this a wrapper bug or my code?"
- [HighwayEnv GitHub issues](https://github.com/Farama-Foundation/HighwayEnv/issues)
  Env-specific surprises (observation scales, action order, Windows rendering). Use for: matching your stack trace to a known issue.
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
  Mixed quality. Use for: "does this diagnostic plot look sane?" not for algorithm shopping.

## Gaps

- No single canonical notebook that collects `parking-v0` transitions the way this repo will. Lesson 0001 plus `scripts/collect_random.py` fills that.
- Sutton & Barto ch. 1–3 do not mention goal-conditioned dict observations. Keep Gymnasium + highway-env next to the book so the mapping stays explicit.
