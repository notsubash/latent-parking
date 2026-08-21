# Phase 1 Pendulum CEM passed

The user ran `scripts/plan_pendulum.py`: random return −1071.9, CEM return −125.7, `|θ|` in the last 50 steps about 0.012 rad. Elite-refit CEM and Pendulum rewind do not need re-teaching. The random parking npz still has 0 successes in 91 finished episodes, so Phase 2 cannot be “fit an MLP on random only.”
