"""TrackmaniaAI: reinforcement learning that drives the real Trackmania game.

The package is layered so that the real game integration stays narrow and swappable::

    tmai.game      GameDriver protocol + the TMInterface bridge to the real game
    tmai.tracks    track-relative geometry (centreline, corridor, projection)
    tmai.env       the gymnasium environment: observations, reward, termination
    tmai.models    the neural network policy/value model
    tmai.agents    RL algorithms (SAC today) that train that model
    tmai.training  trainer loop, evaluation, checkpointing, run logging
    tmai.viz       simplified track visualisation for debugging

Nothing above ``tmai.game`` imports a game-specific module, which is what allows the whole
stack to be unit-tested without Windows or the game.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
