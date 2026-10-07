"""The neural network policy/value model.

Contains representation only -- no loss functions, no optimisers, no RL. The algorithm that
trains these networks lives in :mod:`tmai.agents`. That separation is what allows a different
algorithm (PPO, TD3, REDQ) to reuse the same model unchanged.
"""
