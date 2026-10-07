"""RL algorithms.

Answers "how do we train the model in :mod:`tmai.models`". The trainer only ever sees the
:class:`~tmai.agents.base.Learner` protocol, so swapping algorithms is additive.

* :mod:`~tmai.agents.base` -- ``Learner``, ``Transition``, ``Batch``.
* :mod:`~tmai.agents.replay` -- off-policy experience storage.
* :mod:`~tmai.agents.sac` -- Soft Actor-Critic, the default algorithm. See
  ``docs/ALGORITHM.md`` for why.
"""
