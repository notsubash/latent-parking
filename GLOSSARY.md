# Latent parking glossary

Words we keep using. Each definition is the kid version. Use these words in lessons, not synonyms.

## The loop

**Agent**:
The thing that picks the next move. In this repo that is a script, not a person.
_Avoid_: player, policy network (we do not train one yet)

**Environment**:
The parking lot simulator. You poke it with an action and it tells you what happened.
_Avoid_: world, game (except when quoting Gym)

**Observation**:
The snapshot the agent is allowed to see. It can hide facts that are still true, like speed.
_Avoid_: input, sensor dump

**State**:
The full truth the physics needs for the next moment. Nothing relevant is missing.
_Avoid_: calling the observation the state when we have just hidden velocity

**Action**:
The move you send: throttle then steering, each in `[-1, 1]`.
_Avoid_: steering first, control vector

**Reward**:
A number the lot gives you after a step. Bigger (less negative) means closer to the stall, unless you crashed.
_Avoid_: score, points

**Return**:
The sum of rewards over a whole episode.
_Avoid_: score, fitness (except inside CEM as a synonym for low cost)

**Transition**:
One triple `(observation, action, next observation)` plus the flags. The brick everything else is built from.
_Avoid_: sample (too vague), tuple

**Trajectory**:
A chain of transitions until the episode stops.
_Avoid_: path (use that for x,y in the lot), rollout when you mean a real episode

**Episode**:
One parking attempt, from `reset` until crash, success, or timeout.
_Avoid_: game, round

**Reset**:
Start a new episode. Must happen before the first `step`, and again after the episode ends.
_Avoid_: restart the process

**Step**:
Send one action, get the next observation. In Gymnasium that is five return values, not four.
_Avoid_: tick, frame

**Terminated**:
The task ended: you parked or you crashed.
_Avoid_: done (the old combined flag)

**Truncated**:
The clock ended. Default lot: 100 simulated seconds.
_Avoid_: done, timeout as a Gym flag name (`truncated` is the flag)

## The lot

**parking-v0**:
The empty Farama lot we use. Kinematics, not pixels, until we say otherwise.
_Avoid_: Habitat, a custom 3D garage

**Bicycle**:
The cartoon physics of our car: pose plus speed, then throttle and steering. Next position is basically velocity times a tiny time step.
_Avoid_: full 3D vehicle, Unity car

**Ego**:
Our car.
_Avoid_: agent (the script is the agent; the car is in the environment)

**Pose**:
Where the car sits and which way it faces: `x, y, cos_h, sin_h`.
_Avoid_: position (that is only x,y)

**Heading**:
Which way the nose points. Stored as `cos_h` and `sin_h` so 359 degrees and 1 degree look close.
_Avoid_: angle theta in the saved arrays (Pendulum uses theta; parking uses cos/sin)

**Velocity**:
How fast, and in which way, the car is sliding: `vx, vy`. The bicycle uses this to move.
_Avoid_: speed as the 2-vector (speed is the scalar the env also reports)

**Achieved goal**:
The ego pose the lot thinks you have. In default parking-v0 this is the same six numbers as `observation`.
_Avoid_: current goal

**Desired goal**:
The stall you want: the same six features, but for the parking spot.
_Avoid_: target as a pixel, waypoint

**Scale**:
What the npz actually stored. `x` is meters/100, `vx` is (m/s)/5. Always compare models in this space.
_Avoid_: treating saved x as meters

**Crash**:
You hit a wall (or later, another car). `terminated` plus a harsh extra cost.
_Avoid_: fail (timeout is a different fail)

**Success**:
Close enough to the stall, with heading, under the env's threshold. `info['is_success']`.
_Avoid_: parked as a vibe check without the flag

## Planning

**Model**:
Anything that answers "if I do this action, what is the next state?" Today that can be `env.step` or a network `f`.
_Avoid_: the PyTorch `nn.Module` only

**Planning**:
Using a model to try pretend futures, then picking an action. Not training a policy network.
_Avoid_: RL as a synonym for planning

**CEM**:
Cross-entropy method. Guess many action sequences, keep the good ones, guess again nearer to them.
_Avoid_: genetic algorithm, random shooting (related, not this refit)

**Elite**:
One of the K cheapest sequences this round. CEM copies their average.
_Avoid_: winner (we keep several)

**Horizon**:
How many pretend steps we look ahead. Parking CEM uses 10 (2 seconds).
_Avoid_: timeout, duration

**Receding horizon**:
Imagine many steps, do only the first, imagine again from the new real observation.
_Avoid_: open-loop plan you never refresh

**MPC**:
Model-predictive control: receding-horizon planning with a model in the loop.
_Avoid_: MPC as a brand of optimizer (CEM is the optimizer; MPC is the loop)

**Cost**:
What CEM minimizes. Here it is `-reward`, plus extra punishment if you crash early.
_Avoid_: loss (that is training `f`)

**Rewind**:
Copy the car, try a pretend sequence, put the car back. Imagination must not move the real episode.
_Avoid_: reset (reset starts a new random episode)

**True dynamics**:
The real simulator as the model. Cheating on purpose, so a later bad plot is not "maybe CEM is buggy."
_Avoid_: ground-truth network

## World model

**World model** (`f`):
A network that maps `(observation, action)` to the next observation.
_Avoid_: policy, value function

**Identity**:
Pretend the next observation equals this one. Ignore the action. The bar `f` must beat on one-step pose error.
_Avoid_: baseline as a trained model

**MSE**:
Mean squared error: how wrong a guess is, averaged. Smaller is better.
_Avoid_: accuracy (that is for classes)

**Open-loop**:
Feed `f` its own last guess, over and over. No peeking at the real car in between.
_Avoid_: closed-loop, imagination with rewind to truth each step

**Closed-loop**:
After each real step, look at the real observation and plan again.
_Avoid_: open-loop with a short horizon (still open-loop if you never look)

**MLP**:
A small stack of linear layers and ReLUs. Our `f` is this, not a CNN.
_Avoid_: deep net, transformer

**Pose MSE**:
Error on `x, y, cos_h, sin_h` only. Use this to compare a pose-only head to a full head.
_Avoid_: overall MSE across masks (dropping vx, vy deletes the large identity terms)

## Hidden bits

**Markov**:
The next state depends only on the latest state and action, not the rest of the tape.
_Avoid_: memoryless as a vibe

**Partially observed**:
The observation is not the state. Two different worlds can look the same.
_Avoid_: noisy, incomplete data (those are cousins, not this)

**POMDP**:
The textbook name for a loop that is Markov in the hidden state but not in what you see.
_Avoid_: using POMDP for "we used pixels"

**Stack**:
Glue the last two poses together and hope the difference is speed. In stored units the difference is tiny, so this often acts like pose-only.
_Avoid_: GRU, LSTM (we did not add one)

**Latent** (`z`):
A learned summary of what you saw, meant to carry the hidden bits. We do not train this yet.
_Avoid_: embedding as a decoder of pixels
