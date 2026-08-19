# Prior knowledge at Phase 0 start

The user has intermediate PyTorch and no RL yet. `python scripts/sanity_check.py` already passed on this machine: `parking-v0`, dict observations of shape `(6,)`, action `Box(-1, 1, (2,), float32)`. Do not re-teach pip, tensors, or how to import the env. Teach the meaning of the gym loop, the six features, action order, and episode cutoffs, then save transitions.
