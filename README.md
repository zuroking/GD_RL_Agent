# GD RL Agent

Reinforcement learning agent that learns to play Geometry Dash (Steam) via screen capture and keyboard injection. Educational project — no RL frameworks, everything built from scratch on PyTorch.

## Key constraints

- CPU-only (Intel i7-1225U, no GPU required)
- Real-time environment — no simulation speedup, no parallelization
- Binary action space: jump / no_jump (cube mode only)
- Training on Stereo Madness 0–29% (cube segment, no portals)

## Quick start

```bash
pip install -e .
python cli.py calibrate        # find window, measure latency, save config
python cli.py debug            # verify vision pipeline visually
python cli.py train --episodes 500 --run-name vanilla
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt
```

## DQN variants

```bash
python cli.py train --no-double --no-dueling --run-name vanilla
python cli.py train --double    --no-dueling --run-name double
python cli.py train --no-double --dueling    --run-name dueling
```

## Architecture

```
gd-rl-agent/
├── capture/       # mss screen capture, window detection
├── input/         # pydirectinput jump injection, latency measurement
├── vision/        # HSV obstacle/pit/death detection, state extractor
├── env/           # custom env loop: reset() / step() / close()
├── agent/         # QNetwork, ReplayBuffer, Trainer
├── configs/       # Pydantic v2 config models
├── cli.py         # Typer CLI: calibrate / debug / train / watch / eval
└── tests/         # unit tests for agent/ (no real game needed)
```

## Reward formula

```
r_t = +0.01   per step survived
    + (-1.0)  on death
    + (+5.0)  on segment completion
    + (-0.02) if useless jump (grounded, no obstacle within threshold)
γ = 0.99
```

## Feature vector (4 floats)

| Feature | Range | Sentinel |
|---|---|---|
| `dist_obstacle` | [0, 1] | 1.0 |
| `dist_pit` | [0, 1] | 1.0 |
| `obstacle_height` | [0, 1] | 0.0 |
| `is_grounded` | {0, 1} | — |

## Tests

```bash
python -m pytest tests/ -v
```

See `INSTRUCTION_en.md` for full setup and calibration guide.
