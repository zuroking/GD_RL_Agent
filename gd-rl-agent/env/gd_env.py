"""Custom environment loop for Geometry Dash (no Gymnasium dependency).

Implements reset() / step(action) / close() — the minimal interface
the trainer needs. All RL logic lives in agent/; this file is purely
the interface to the real game.

Action space: 0 = no_op, 1 = jump (binary, cube mode only).

Reward formula (grill-me Q6-Q8):
    r_t = +0.01 per step survived
        + (-1.0 if death)
        + (+5.0 if success / episode timeout reached)
        + (-0.02 if useless jump, when penalize_useless_jumps=True)

useless_jump: action==1 AND dist_obstacle >= threshold AND dist_pit >= threshold
  AND is_grounded==1 (jump must be physically possible to count as useless;
  a jump in air is already a no-op by physics and should NOT incur extra penalty).

success: step_count >= max_episode_steps (step-count proxy for X-progress;
  valid because scroll speed is constant in Stereo Madness — grill-me Q8).
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from capture.screen_capture import ScreenCapture
from capture.window_utils import bring_to_foreground, find_gd_window
from configs.config import AgentConfig
from input.controller import Controller
from vision.state_extractor import StateExtractor


# How long to wait after restart keypress before the level is running (seconds).
_RESTART_SETTLE = 2.5
# Key used by GD to restart from the death screen (default: R in practice mode).
_RESTART_KEY = "r"


@dataclass
class StepResult:
    obs: NDArray[np.float32]
    reward: float
    terminal: bool
    success: bool
    info: dict = field(default_factory=dict)


class GDEnv:
    """Real-time Geometry Dash environment.

    Args:
        config: Full agent config with calibrated pixel constants and reward params.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._reward_cfg = config.reward
        self._ep_cfg = config.episode

        win = find_gd_window()
        bring_to_foreground(win.hwnd)

        self._cap = ScreenCapture(config.capture_region)
        self._ctrl = Controller(win)
        self._extractor = StateExtractor(config)

        self._step_count: int = 0
        self._hwnd = win.hwnd

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> NDArray[np.float32]:
        """Restart the level and return the initial observation.

        If auto_retry is enabled in config, GD restarts automatically after death
        — we just wait for the settle time. Otherwise, sends the restart key.

        Returns:
            Initial observation (4-float feature vector).
        """
        bring_to_foreground(self._hwnd)
        time.sleep(0.1)

        if not self._ep_cfg.auto_retry:
            import pydirectinput
            pydirectinput.PAUSE = 0.0
            pydirectinput.keyDown(_RESTART_KEY)
            time.sleep(0.05)
            pydirectinput.keyUp(_RESTART_KEY)

        # Wait for level to start (auto-retry or manual restart animation to finish)
        time.sleep(_RESTART_SETTLE)

        self._step_count = 0
        frame = self._cap.grab()
        return self._extractor.extract(frame)

    def step(self, action: int) -> StepResult:
        """Execute one action and return the transition.

        Args:
            action: 0 = no_op, 1 = jump.

        Returns:
            StepResult with obs, reward, terminal, success, info.
        """
        # Grab pre-action frame to read state before the action resolves
        frame = self._cap.grab()
        obs = self._extractor.extract(frame)

        is_grounded = bool(obs[3] > 0.5)
        dist_obs = float(obs[0])
        dist_pit = float(obs[1])

        # Execute action
        if action == 1:
            self._ctrl.jump()
        else:
            self._ctrl.no_op()

        self._step_count += 1

        # Small sleep to let the game advance ~1 physics frame before next grab
        time.sleep(0.02)

        # Grab post-action frame
        next_frame = self._cap.grab()
        next_obs = self._extractor.extract(next_frame)

        # ── terminal detection ──────────────────────────────────────────
        dead = self._extractor.is_dead(next_frame)
        success = self._step_count >= self._ep_cfg.max_episode_steps
        terminal = dead or success

        # ── reward ──────────────────────────────────────────────────────
        reward = self._reward_cfg.r_step

        if dead:
            reward += self._reward_cfg.r_death
        elif success:
            reward += self._reward_cfg.r_completion

        # useless-jump penalty: only when action=jump was physically possible
        # (is_grounded==True) and no obstacle/pit is within the threshold.
        # A jump in air is already a physics no-op; penalising it twice is wrong.
        if (
            action == 1
            and self._reward_cfg.penalize_useless_jumps
            and is_grounded
            and dist_obs >= self._reward_cfg.useless_jump_threshold
            and dist_pit >= self._reward_cfg.useless_jump_threshold
        ):
            reward += self._reward_cfg.useless_jump_penalty

        return StepResult(
            obs=next_obs,
            reward=reward,
            terminal=terminal,
            success=success,
            info={
                "step": self._step_count,
                "dead": dead,
                "dist_obs": dist_obs,
                "dist_pit": dist_pit,
                "is_grounded": is_grounded,
                "action": action,
            },
        )

    def close(self) -> None:
        """Release the screen capture resource."""
        self._cap.close()

    def __enter__(self) -> "GDEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
