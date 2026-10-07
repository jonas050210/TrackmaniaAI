"""Track representation.

The centreline is the only piece of map knowledge phase 1 needs, and it is what makes
track-relative observations -- and therefore generalisation to unseen maps -- possible.

* :mod:`~tmai.tracks.centerline` -- arc-length parameterised centreline with a drivable
  corridor: projection, lateral offset, curvature, edge distances.
* :mod:`~tmai.tracks.recording` -- build a centreline by driving the real map once.
* :mod:`~tmai.tracks.synthetic` -- procedural tracks for tests and CI. **Not real maps.**
"""
