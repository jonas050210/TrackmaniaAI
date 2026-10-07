"""Training loop, evaluation, checkpointing and object wiring.

* :mod:`~tmai.training.factory` -- the single place where configuration becomes live objects,
  and the gate that refuses the simulated driver unless explicitly allowed.
* :mod:`~tmai.training.trainer` -- the collect/learn loop.
* :mod:`~tmai.training.evaluate` -- deterministic policy evaluation.
* :mod:`~tmai.training.checkpoint` -- atomic, resumable checkpoints.
"""
