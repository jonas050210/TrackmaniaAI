"""The reinforcement-learning environment.

:class:`~tmai.env.tm_env.TrackmaniaEnv` is a thin gymnasium environment that delegates its
three pieces of logic to dedicated modules so each can be reasoned about and tested alone:

* :mod:`~tmai.env.observation` -- track-relative observation encoding.
* :mod:`~tmai.env.reward` -- progress-based reward and its anti-exploit clamps.
* :mod:`~tmai.env.termination` -- episode termination with machine-readable reasons.
"""
