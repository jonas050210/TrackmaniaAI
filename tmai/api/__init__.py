"""Read-only data APIs for external consumers (dashboards, notebooks, tooling).

Nothing here renders anything or talks to the game. These modules turn the artefacts a
training run already writes into plain, JSON-serialisable data, so a GUI can be built on top
later without touching the trainer.
"""
