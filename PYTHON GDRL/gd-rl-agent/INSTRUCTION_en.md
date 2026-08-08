# GD RL Agent — Setup & Usage Guide

A reinforcement learning agent that learns to play Geometry Dash (Steam) via screen capture and keyboard injection. No RL frameworks — everything built from scratch on PyTorch.

## Requirements

- Windows 10/11 (pydirectinput requires Windows)
- Python 3.12 or newer
- Geometry Dash (Steam version 2.2)
- CPU-only machine is fine (no GPU required)

## Installation

```bash
git clone <repo-url>
cd gd-rl-agent
pip install -e .
```

## Step 1 — Calibration

Launch Geometry Dash in **windowed mode** and start Stereo Madness in **practice mode**.

```bash
python cli.py calibrate
```

This will:
1. Find the GD window automatically
2. Measure capture FPS and input latency
3. Take a screenshot — open `calibration_frame.png` and enter pixel coordinates when prompted
4. Save the config to `configs/agent_config.json`

**Pixel values to enter:**
- `player_right_x` — X pixel at the right edge of the cube sprite
- `ground_y` — Y pixel at the top of the ground tiles
- `cube_height_px` — height of the cube in pixels (measure in the screenshot)

## Step 2 — Verify the vision pipeline

```bash
python cli.py debug --duration 10
```

Watch the printed feature vectors and check `debug_frame.png`. The green rectangle is the scan ROI, the red vertical line marks the detected obstacle, orange marks a pit. If nothing is detected where it should be, the HSV thresholds in `vision/obstacle_detector.py` need empirical tuning.

## Step 3 — Train

Enable **Auto-Retry** in GD settings. Start Stereo Madness in practice mode and place a checkpoint at the very beginning of the level (press `Z`). Then run:

```bash
# Vanilla DQN
python cli.py train --episodes 500 --run-name vanilla

# Double DQN
python cli.py train --episodes 500 --double --run-name double

# Dueling DQN
python cli.py train --episodes 500 --dueling --run-name dueling
```

Training logs are saved to `runs/<run-name>/train_log.csv`. Checkpoints are saved every 50 episodes.

**Important:** After the first training run, check the actual capture FPS in the log and recalibrate `max_episode_steps` in `configs/agent_config.json` if it differs significantly from the calibration value.

## Step 4 — Watch / Evaluate

```bash
# Watch the agent play (no weight updates)
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 5

# Evaluate mean reward and success rate
python cli.py eval --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 20
```

## Configuration reference

All settings live in `configs/agent_config.json` (created by `calibrate`).

| Field | Default | Description |
|---|---|---|
| `episode.max_episode_steps` | 250 | Steps per episode — calibrate with DQN running |
| `episode.auto_retry` | true | Set true if GD Auto-Retry is enabled |
| `reward.r_step` | 0.01 | Reward per step survived |
| `reward.r_death` | -1.0 | Penalty on death |
| `reward.r_completion` | 5.0 | Bonus for reaching episode end |
| `reward.penalize_useless_jumps` | true | Penalise jumps with no obstacle ahead |
| `reward.useless_jump_threshold` | 0.35 | Distance threshold for "useless" jump |
| `hp.use_double_dqn` | false | Enable Double DQN |
| `hp.use_dueling_dqn` | false | Enable Dueling DQN |
| `hp.use_obstacle_lookahead` | false | Add second obstacle to feature vector |

## Project structure

```
gd-rl-agent/
├── capture/          # mss screen capture, window detection
├── input/            # pydirectinput jump injection
├── vision/           # HSV obstacle/pit/death detection, state extractor
├── env/              # Custom env loop (reset/step/close)
├── agent/            # QNetwork, ReplayBuffer, Trainer
├── configs/          # Pydantic config models
├── cli.py            # Typer CLI (calibrate/debug/train/watch/eval)
└── tests/            # Unit tests for agent/ (no real game needed)
```

## Running tests

```bash
python -m pytest tests/ -v
```
