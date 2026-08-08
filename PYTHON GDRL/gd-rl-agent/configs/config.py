"""Pydantic v2 configuration models for the GD RL agent."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class CaptureRegion(BaseModel):
    """Screen region to capture, in absolute monitor coordinates."""

    left: int
    top: int
    width: int
    height: int


class PlayerCalibration(BaseModel):
    """Pixel-space constants for the player cube, set during calibrate."""

    # X pixel of the cube's right edge (start of obstacle scan ROI)
    player_right_x: int = 0
    # Y pixel of the ground line (bottom of cube when landed)
    ground_y: int = 0
    # Pixel that is "orange" on the cube top when grounded (for is_grounded check)
    grounded_check_y: int = 0
    # Guard pixels added after player_right_x to exclude particle effects
    guard_px: int = 12
    # Horizontal scan depth in pixels (how far ahead to look)
    scan_depth: int = 300
    # Cube height in pixels (used to normalise obstacle_height feature)
    cube_height_px: int = 36


class RewardConfig(BaseModel):
    """Reward shaping constants (all values from grill-me session)."""

    r_step: float = 0.01
    r_death: float = -1.0
    r_completion: float = 5.0

    # useless-jump penalty (intentionally 2× r_step; see reward design rationale)
    penalize_useless_jumps: bool = True
    useless_jump_threshold: float = 0.35
    useless_jump_penalty: float = -0.02

    gamma: float = 0.99


class EpisodeConfig(BaseModel):
    """Episode / env loop configuration."""

    # Calibrated with real DQN forward pass in Phase 1, re-checked in Phase 4.
    # Based on: Stereo Madness 0-29% at measured capture FPS.
    max_episode_steps: int = 250
    # Capture rate used when max_episode_steps was calibrated.
    calibration_fps: float = 10.0
    # If True, GD restarts automatically (Auto-Retry enabled) — reset() just waits.
    # If False, reset() sends _RESTART_KEY to trigger manual restart.
    auto_retry: bool = True


class HyperParams(BaseModel):
    """DQN training hyperparameters."""

    lr: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 10_000
    target_update_freq: int = 200  # steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    hidden_dim: int = 128
    use_double_dqn: bool = False
    use_dueling_dqn: bool = False
    use_obstacle_lookahead: bool = False


class AgentConfig(BaseSettings):
    """Top-level config; loaded from env vars or a TOML/JSON file."""

    model_config = {"env_prefix": "GD_", "extra": "ignore"}

    capture_region: CaptureRegion = Field(
        default_factory=lambda: CaptureRegion(left=0, top=0, width=1920, height=1080)
    )
    player: PlayerCalibration = Field(default_factory=PlayerCalibration)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    hp: HyperParams = Field(default_factory=HyperParams)

    log_dir: Path = Path("runs")
    config_path: Path = Path("configs/agent_config.json")

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "AgentConfig":
        return cls.model_validate_json(path.read_text())
