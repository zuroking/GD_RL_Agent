# GD RL Agent

Reinforcement learning agent that learns to play Geometry Dash (Steam version) via screen capture and input injection. Educational project — no RL frameworks, everything from scratch on PyTorch.

## Project structure

```
gd-rl-agent/
├── capture/          # mss screen capture + window detection
├── input/            # pydirectinput jump injection
├── vision/           # pixel-based state extraction (Phase 2)
├── env/              # custom env loop (Phase 3)
├── agent/            # DQN + replay buffer (Phase 4)
├── configs/          # Pydantic config models
├── cli.py            # Typer CLI entry point
└── tests/
```

## Setup

```bash
pip install -e .
```

Requires Python 3.12+, Windows (for pydirectinput).

## Usage

### Phase 1 — Calibration

```bash
python cli.py calibrate
```

1. Finds the GD window
2. Measures capture FPS and input latency
3. Saves calibrated config to `configs/agent_config.json`

### Phase 4 — Training (not yet implemented)

```bash
python cli.py train
python cli.py watch
```

## Design decisions

All architecture decisions (feature vector, reward formula, HSV thresholds) were locked in via `grill-me` session before implementation. See project instructions for full rationale.

**Key constraints:**
- CPU-only (Intel i7-1225U)
- Real-time env (no simulation speedup, no parallelization)
- Binary action space: jump / no_jump
- Training on Stereo Madness 0–29% (cube segment only)

**Reward formula (from grill-me Q6–Q8):**
```
r_t = +0.01 per step
    + (-1.0 if death)
    + (+5.0 if completion)
    + (-0.02 if useless jump)
γ = 0.99
```

**Feature vector (4 floats):**
- `dist_obstacle` — distance to nearest obstacle above ground
- `dist_pit` — distance to nearest gap in ground
- `obstacle_height` — height of nearest obstacle
- `is_grounded` — boolean (cube on ground or in air)
