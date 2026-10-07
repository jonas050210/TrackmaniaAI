"""Simplified track visualisation.

Renders the geometry the agent actually uses -- centreline, drivable corridor, curvature, car
state -- rather than Trackmania's rendering, and can export the same geometry as Wavefront
``.obj`` for an external viewer or the future GUI. Headless-safe (matplotlib ``Agg``).
"""
