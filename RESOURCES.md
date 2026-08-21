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
  Agent, environment, observation vs state, action space, trajectory, reward/return. Use for: vocabulary. Skip value functions and Bellman until they show up in a lesson.
- [Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
  Canonical textbook, PDF hosted by the authors. Use for: ch. 1–3 after the gym loop feels boring; **ch. 8.1 only** when implementing planning (model, sample model, planning as model → policy). Stop before Dyna and MCTS until a later lesson.
- [Gymnasium: Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
  Observation `[cos θ, sin θ, θ̇]`, torque `[-2, 2]`, reward `-(θ² + 0.1 θ̇² + 0.001 τ²)`, 200-step truncation. Use for: Phase 1 warmup env. Confirm physics state against `gymnasium/envs/classic_control/pendulum.py` (`unwrapped.state`).
- [Pinneri et al., “Sample-efficient Cross-Entropy Method for Real-time Planning” (2020)](https://arxiv.org/abs/2008.06389)
  Section 2 and Appendix A: vanilla CEM-MPC (sample action sequences, elite, refit diagonal Gaussian, shift mean, receding horizon). Use for: the planner in `scripts/cem.py`. Skip iCEM extras until vanilla parks.
- [de Boer, Kroese, Mannor, Rubinstein, “A Tutorial on the Cross-Entropy Method” (2005)](https://link.springer.com/article/10.1007/s10479-005-5724-z)
  Original CEM as elite-refit of a sampling distribution. Use for: why it is called cross-entropy, not for the MPC wiring.
- [Nagabandi, Kahn, Fearing, Levine, “Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning” (2018)](https://arxiv.org/abs/1708.02596)
  MLP `f(s, a)` predicts next state (they use a delta), one-step MSE, H-step open-loop validation, then MPC; §IV-D aggregates random data with on-policy MPC rollouts. Use for: Phase 2. Primary source for lesson 0003. Stop before TRPO fine-tuning.
- [Hansen, Su, Wang, TD-MPC2 (2024)](https://arxiv.org/abs/2310.16828) · [project page](https://tdmpc2.com)
  Latent dynamics + MPC. Use for: “what is predicted, how is it used to plan.” Skip Q, policy prior, and SimNorm until Phase 4–5.
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

- No single canonical notebook that collects `parking-v0` transitions the way this repo will. Lesson 0001 plus `scripts/collect_random.py` fills the random half; `scripts/plan_parking.py` fills the CEM half.
- Sutton & Barto ch. 1–3 do not mention goal-conditioned dict observations. Keep Gymnasium + highway-env next to the book so the mapping stays explicit.
- Sutton & Barto ch. 8 never names CEM. Pair 8.1 with Pinneri Appendix A (or de Boer 2005) when teaching the optimizer.
- Nagabandi’s MuJoCo random-only models can walk; parking-v0 random-only never parks. Do not quote their “random is enough” result as if it applied to this lot.
