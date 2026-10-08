# Demonstrations

Human-driven laps recorded with `tmai record-demo`, stored as JSON Lines (one transition per
line: observation, game-reported action, position, speed, reward, race time).

```bash
tmai record-demo -c tmai/configs/default.yaml --out data/demos/my_lap.jsonl
tmai pretrain  -c tmai/configs/default.yaml --demo data/demos/my_lap.jsonl --out models/pretrained.pt
tmai train     -c tmai/configs/default.yaml --resume models/pretrained.pt
```

or enable `bc:` in the config to pretrain automatically at the start of a training run.

`.jsonl` files are git-ignored artefacts (they are data, not source); this README is kept so
the directory exists in a fresh checkout.
